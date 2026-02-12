//! Namespace registry — runtime lookup and validation for hierarchical GIDs.

use std::collections::HashMap;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::hash::hierarchical_gid;
use crate::layout::{gid_is_descendant_of as gid_is_descendant_of, LEVEL_MASKS, MAX_DEPTH};
use crate::traits::IntoGid;
use crate::GID;

/// Definition of a namespace node (used for registry building from macro).
#[derive(Clone, Copy, Debug)]
pub struct NamespaceDef {
    pub path: &'static str,
    pub parent: Option<&'static str>,
}

impl NamespaceDef {
    pub const fn new(path: &'static str, parent: Option<&'static str>) -> Self {
        Self { path, parent }
    }
}

/// Runtime entry for a registered namespace node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamespaceEntry {
    pub gid: GID,
    pub path: String,
    /// True if this tag was registered at runtime (not from macro).
    pub is_dynamic: bool,
}

/// Registry for namespace tags.
///
/// Provides:
/// - Path ↔ GID bidirectional lookup
/// - O(1) subtree membership test via bitmask
/// - Dynamic tag registration at runtime
/// - DFS-ordered iteration (for cases that need sequential traversal)
#[derive(Clone, Debug, PartialEq)]
pub struct NamespaceRegistry {
    /// Maximum tree depth encountered (0 = empty, 1 = only root nodes, etc.).
    max_depth: usize,
    entries: Vec<NamespaceEntry>,
    path_to_idx: HashMap<String, usize>,
    gid_to_idx: HashMap<GID, usize>,
    dfs_order: Vec<GID>,
    /// Dynamic metadata storage: GID → (key → bytes)
    /// User is responsible for serialization/deserialization.
    metadata: HashMap<GID, HashMap<String, Vec<u8>>>,
}

impl Default for NamespaceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl NamespaceRegistry {
    pub fn new() -> Self {
        Self {
            max_depth: 0,
            entries: Vec::new(),
            path_to_idx: HashMap::new(),
            gid_to_idx: HashMap::new(),
            dfs_order: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Build a registry from namespace definitions (from macro).
    ///
    /// Uses the fixed static layout for GID computation.
    pub fn build(defs: &[NamespaceDef]) -> Result<Self, String> {
        if defs.is_empty() {
            return Ok(Self::new());
        }

        // 1. Validate
        Self::validate_defs(defs)?;

        // 2. Build tree structure
        let tree = TreeBuilder::from_defs(defs)?;

        // 3. Record max depth
        let max_depth = tree.max_depth as usize + 1;

        // 4. Assign hierarchical GIDs
        let mut entries = Vec::with_capacity(defs.len());
        let mut gid_set: HashMap<GID, &'static str> = HashMap::new();

        for node in &tree.nodes {
            let segments = Self::path_segments(node.path);
            let seg_bytes: Vec<&[u8]> = segments.iter().map(|s| s.as_bytes()).collect();

            let gid = hierarchical_gid(&seg_bytes);

            // 5. Collision detection
            if let Some(&existing) = gid_set.get(&gid) {
                return Err(format!(
                    "GID collision: '{}' and '{}' produce the same hierarchical hash {:#034x}. \
                     Consider renaming one of them.",
                    node.path, existing, gid
                ));
            }
            gid_set.insert(gid, node.path);

            entries.push(NamespaceEntry {
                gid,
                path: node.path.to_string(),
                is_dynamic: false,
            });
        }

        // 6. Build indices
        let path_to_idx: HashMap<String, usize> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.path.clone(), i))
            .collect();

        let gid_to_idx: HashMap<GID, usize> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.gid, i))
            .collect();

        // 7. DFS order (entries are already in DFS order from TreeBuilder)
        let dfs_order: Vec<GID> = entries.iter().map(|e| e.gid).collect();

        Ok(Self {
            max_depth,
            entries,
            path_to_idx,
            gid_to_idx,
            dfs_order,
            metadata: HashMap::new(),
        })
    }

    /// Path → GID
    #[inline]
    pub fn gid_of(&self, path: &str) -> Option<GID> {
        self.path_to_idx.get(path).map(|&i| self.entries[i].gid)
    }

    /// GID → Path
    ///
    /// Accepts both raw `GID` and `Tag` types.
    #[inline]
    pub fn path_of(&self, gid: impl IntoGid) -> Option<&str> {
        self.gid_to_idx
            .get(&gid.into_gid())
            .map(|&i| self.entries[i].path.as_str())
    }

    /// Get the current maximum tree depth (0 = empty, 1 = only root nodes, etc.).
    ///
    /// This value grows dynamically as deeper tags are registered.
    #[inline]
    pub fn tree_depth(&self) -> usize {
        self.max_depth
    }

    /// Total number of registered nodes.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All entries in DFS order.
    #[inline]
    pub fn dfs_order(&self) -> &[GID] {
        &self.dfs_order
    }

    /// Iterate all entries.
    pub fn entries(&self) -> &[NamespaceEntry] {
        &self.entries
    }

    /// Register a new tag at runtime.
    ///
    /// The path must be a valid dot-separated path (e.g., "Combat.Special.Fireball").
    /// Parent nodes are automatically created if they don't exist.
    ///
    /// Returns the GID of the registered tag.
    ///
    /// # Errors
    ///
    /// - Returns error if path is empty
    /// - Returns error if path depth exceeds MAX_DEPTH (16)
    /// - Returns error if path already exists (no-op, returns existing GID via Ok)
    pub fn register(&mut self, path: &str) -> Result<GID, String> {
        if path.is_empty() {
            return Err("empty path is not allowed".into());
        }

        // Check if already exists
        if let Some(&idx) = self.path_to_idx.get(path) {
            return Ok(self.entries[idx].gid);
        }

        let segments: Vec<&str> = path.split('.').collect();
        let depth = segments.len() - 1;

        if depth >= MAX_DEPTH {
            return Err(format!(
                "path '{}' has depth {} which exceeds MAX_DEPTH ({})",
                path, depth, MAX_DEPTH
            ));
        }

        // Ensure all parent nodes exist (auto-create)
        for i in 0..segments.len() - 1 {
            let parent_path: String = segments[..=i].join(".");
            if self.path_to_idx.contains_key(&parent_path) {
                continue;
            }
            // Auto-create parent
            let parent_segs: Vec<&[u8]> = segments[..=i].iter().map(|s| s.as_bytes()).collect();
            let gid = hierarchical_gid(&parent_segs);

            let idx = self.entries.len();
            self.entries.push(NamespaceEntry {
                gid,
                path: parent_path.clone(),
                is_dynamic: true,
            });
            self.path_to_idx.insert(parent_path, idx);
            self.gid_to_idx.insert(gid, idx);
            self.dfs_order.push(gid);
        }

        // Register the actual node
        let seg_bytes: Vec<&[u8]> = segments.iter().map(|s| s.as_bytes()).collect();
        let gid = hierarchical_gid(&seg_bytes);

        // Check for GID collision
        if let Some(&existing_idx) = self.gid_to_idx.get(&gid) {
            let existing_path = &self.entries[existing_idx].path;
            return Err(format!(
                "GID collision: '{}' and '{}' produce the same hash {:#034x}",
                path, existing_path, gid
            ));
        }

        let idx = self.entries.len();
        self.entries.push(NamespaceEntry {
            gid,
            path: path.to_string(),
            is_dynamic: true,
        });
        self.path_to_idx.insert(path.to_string(), idx);
        self.gid_to_idx.insert(gid, idx);
        self.dfs_order.push(gid);

        // Update max depth if needed
        if depth >= self.max_depth {
            self.max_depth = depth + 1;
        }

        Ok(gid)
    }

    /// Check if a path is registered.
    #[inline]
    pub fn contains(&self, path: &str) -> bool {
        self.path_to_idx.contains_key(path)
    }

    /// Check if a GID is registered.
    ///
    /// Accepts both raw `GID` and `Tag` types.
    #[inline]
    pub fn contains_gid(&self, gid: impl IntoGid) -> bool {
        self.gid_to_idx.contains_key(&gid.into_gid())
    }

    /// Set typed metadata for a GID.
    ///
    /// The type must implement `zerocopy::IntoBytes + Immutable`.
    /// Use `#[derive(IntoBytes, Immutable)]` on your type.
    ///
    /// Accepts both raw `GID` and `Tag` types.
    ///
    /// Returns the previous raw bytes if any.
    pub fn set_meta<G: IntoGid, T: IntoBytes + Immutable>(
        &mut self,
        gid: G,
        key: impl Into<String>,
        value: &T,
    ) -> Option<Vec<u8>> {
        self.metadata
            .entry(gid.into_gid())
            .or_default()
            .insert(key.into(), value.as_bytes().to_vec())
    }

    /// Get typed metadata for a GID.
    ///
    /// The type must implement `zerocopy::FromBytes + Immutable`.
    /// Use `#[derive(FromBytes, Immutable)]` on your type.
    ///
    /// Accepts both raw `GID` and `Tag` types.
    ///
    /// Returns `None` if key doesn't exist or bytes don't match type layout.
    #[inline]
    pub fn get_meta<T: FromBytes + KnownLayout + Immutable>(
        &self,
        gid: impl IntoGid,
        key: &str,
    ) -> Option<&T> {
        let bytes = self.metadata.get(&gid.into_gid())?.get(key)?;
        T::ref_from_bytes(bytes).ok()
    }

    /// Set raw bytes metadata for a GID.
    ///
    /// Accepts both raw `GID` and `Tag` types.
    ///
    /// Use this when you need manual serialization control.
    pub fn set_meta_raw(
        &mut self,
        gid: impl IntoGid,
        key: impl Into<String>,
        value: Vec<u8>,
    ) -> Option<Vec<u8>> {
        self.metadata
            .entry(gid.into_gid())
            .or_default()
            .insert(key.into(), value)
    }

    /// Get raw bytes metadata for a GID.
    ///
    /// Accepts both raw `GID` and `Tag` types.
    #[inline]
    pub fn get_meta_raw(&self, gid: impl IntoGid, key: &str) -> Option<&[u8]> {
        self.metadata
            .get(&gid.into_gid())?
            .get(key)
            .map(|v| v.as_slice())
    }

    /// Check if a GID has a specific metadata key.
    ///
    /// Accepts both raw `GID` and `Tag` types.
    #[inline]
    pub fn has_meta(&self, gid: impl IntoGid, key: &str) -> bool {
        self.metadata
            .get(&gid.into_gid())
            .is_some_and(|m| m.contains_key(key))
    }

    /// Remove metadata for a GID.
    ///
    /// Accepts both raw `GID` and `Tag` types.
    ///
    /// Returns the removed raw bytes if any.
    pub fn remove_meta(&mut self, gid: impl IntoGid, key: &str) -> Option<Vec<u8>> {
        self.metadata.get_mut(&gid.into_gid())?.remove(key)
    }

    /// Get all metadata keys for a GID.
    ///
    /// Accepts both raw `GID` and `Tag` types.
    pub fn meta_keys(&self, gid: impl IntoGid) -> Option<impl Iterator<Item = &str>> {
        self.metadata
            .get(&gid.into_gid())
            .map(|m| m.keys().map(|s| s.as_str()))
    }

    /// Get all metadata for a GID as (key, bytes) pairs.
    ///
    /// Accepts both raw `GID` and `Tag` types.
    pub fn meta_iter(&self, gid: impl IntoGid) -> Option<impl Iterator<Item = (&str, &[u8])>> {
        self.metadata
            .get(&gid.into_gid())
            .map(|m| m.iter().map(|(k, v)| (k.as_str(), v.as_slice())))
    }

    /// Check if `candidate` path is a descendant of (or equal to) `ancestor` path.
    ///
    /// Returns `None` if either path is not found in the registry.
    ///
    /// For GID-based checks without registry lookup, use `is_descendant_of` directly.
    ///
    /// ```text
    /// registry.is_descendant_of_path("Movement.Idle", "Movement") → Some(true)
    /// registry.is_descendant_of_path("Combat.Attack", "Movement") → Some(false)
    /// registry.is_descendant_of_path("Unknown", "Movement") → None
    /// ```
    pub fn is_descendant_of_path(&self, candidate: &str, ancestor: &str) -> Option<bool> {
        let candidate_gid = self.gid_of(candidate)?;
        let ancestor_gid = self.gid_of(ancestor)?;
        Some(gid_is_descendant_of(candidate_gid, ancestor_gid))
    }

    /// Check if `candidate` path is a descendant of (or equal to) `ancestor`.
    ///
    /// ```text
    /// registry.is_descendant_of_path(movement::Idle, Movement) → true
    /// registry.is_descendant_of_path(combat::Attack, Movement) → false
    /// ```
    pub fn is_descendant_of(&self, candidate: impl IntoGid, ancestor: impl IntoGid) -> bool {
        gid_is_descendant_of(candidate.into_gid(), ancestor.into_gid())
    }

    /// Collect all registered descendants of `ancestor` (including itself).
    ///
    /// Not O(1) — iterates all entries. Use `is_descendant_of` for single checks.
    ///
    /// Accepts both raw `GID` and `Tag` types.
    pub fn descendants_of(&self, ancestor: impl IntoGid) -> Vec<GID> {
        let ancestor_gid = ancestor.into_gid();
        let ancestor_depth = crate::layout::depth_of(ancestor_gid) as usize;

        // Only compare payload bits (exclude depth bits)
        let mask = if ancestor_depth < MAX_DEPTH {
            LEVEL_MASKS[ancestor_depth] & !crate::layout::DEPTH_MASK
        } else {
            return vec![];
        };
        let prefix = ancestor_gid & mask;

        self.entries
            .iter()
            .filter(|e| (e.gid & mask) == prefix)
            .map(|e| e.gid)
            .collect()
    }

    fn validate_defs(defs: &[NamespaceDef]) -> Result<(), String> {
        let mut paths = std::collections::HashSet::new();
        for def in defs {
            if def.path.is_empty() {
                return Err("empty namespace path is not allowed".into());
            }
            if !paths.insert(def.path) {
                return Err(format!("duplicate namespace path: {}", def.path));
            }
        }
        for def in defs {
            if let Some(parent) = def.parent
                && !paths.contains(parent)
            {
                return Err(format!("missing parent for '{}': '{}'", def.path, parent));
            }
        }
        Ok(())
    }

    /// Split "A.B.C" into ["A", "B", "C"].
    fn path_segments(path: &str) -> Vec<&str> {
        path.split('.').collect()
    }
}

// =============================================================================
// Tree builder — reconstructs tree from flat NamespaceDef slice
// =============================================================================

#[derive(Debug)]
struct TreeNode {
    path: &'static str,
}

#[derive(Debug)]
struct TreeBuilder {
    nodes: Vec<TreeNode>,
    max_depth: u8,
}

impl TreeBuilder {
    fn from_defs(defs: &[NamespaceDef]) -> Result<Self, String> {
        // Build children map
        let mut children: HashMap<Option<&str>, Vec<&NamespaceDef>> = HashMap::new();
        for def in defs {
            children.entry(def.parent).or_default().push(def);
        }
        // Sort children by path for deterministic DFS order
        for list in children.values_mut() {
            list.sort_by_key(|d| d.path);
        }

        // Compute depth for each node
        let mut depth_map: HashMap<&str, u8> = HashMap::new();
        let mut max_depth: u8 = 0;

        // Process in topological order (roots first, then children)
        let mut queue = std::collections::VecDeque::new();
        if let Some(roots) = children.get(&None) {
            for root in roots {
                depth_map.insert(root.path, 0);
                queue.push_back(root.path);
            }
        }

        while let Some(path) = queue.pop_front() {
            let d = depth_map[path];
            if d > max_depth {
                max_depth = d;
            }
            if let Some(kids) = children.get(&Some(path)) {
                for kid in kids {
                    let child_depth = d + 1;
                    if child_depth as usize >= MAX_DEPTH {
                        return Err(format!(
                            "tree depth exceeds maximum ({}) at path '{}'",
                            MAX_DEPTH, kid.path
                        ));
                    }
                    depth_map.insert(kid.path, child_depth);
                    queue.push_back(kid.path);
                }
            }
        }

        if depth_map.len() != defs.len() {
            return Err("disconnected tree — some nodes are unreachable from roots".into());
        }

        // DFS traversal for output ordering
        let mut nodes = Vec::with_capacity(defs.len());
        Self::dfs_collect(None, &children, &mut nodes);

        Ok(Self { nodes, max_depth })
    }

    fn dfs_collect(
        parent: Option<&'static str>,
        children: &HashMap<Option<&str>, Vec<&NamespaceDef>>,
        out: &mut Vec<TreeNode>,
    ) {
        let key = parent;
        if let Some(kids) = children.get(&key) {
            for kid in kids {
                out.push(TreeNode { path: kid.path });
                Self::dfs_collect(Some(kid.path), children, out);
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_DEFS: &[NamespaceDef] = &[
        NamespaceDef::new("Movement", None),
        NamespaceDef::new("Movement.Idle", Some("Movement")),
        NamespaceDef::new("Movement.Running", Some("Movement")),
        NamespaceDef::new("Movement.Jumping", Some("Movement")),
        NamespaceDef::new("Combat", None),
        NamespaceDef::new("Combat.Attack", Some("Combat")),
        NamespaceDef::new("Combat.Block", Some("Combat")),
    ];

    fn sample_defs() -> &'static [NamespaceDef] {
        SAMPLE_DEFS
    }

    #[test]
    fn build_and_lookup() {
        let reg = NamespaceRegistry::build(sample_defs()).unwrap();

        // Every path has a GID
        assert!(reg.gid_of("Movement").is_some());
        assert!(reg.gid_of("Movement.Idle").is_some());
        assert!(reg.gid_of("Combat.Attack").is_some());

        // Round-trip
        let gid = reg.gid_of("Movement.Running").unwrap();
        assert_eq!(reg.path_of(gid).unwrap(), "Movement.Running");
    }

    #[test]
    fn gid_is_stable_regardless_of_def_order() {
        let defs_a = &[
            NamespaceDef::new("A", None),
            NamespaceDef::new("A.B", Some("A")),
            NamespaceDef::new("X", None),
        ];
        let defs_b = &[
            // Reversed order
            NamespaceDef::new("X", None),
            NamespaceDef::new("A.B", Some("A")),
            NamespaceDef::new("A", None),
        ];

        let reg_a = NamespaceRegistry::build(defs_a).unwrap();
        let reg_b = NamespaceRegistry::build(defs_b).unwrap();

        // Same paths produce same GIDs regardless of definition order
        assert_eq!(reg_a.gid_of("A"), reg_b.gid_of("A"));
        assert_eq!(reg_a.gid_of("A.B"), reg_b.gid_of("A.B"));
        assert_eq!(reg_a.gid_of("X"), reg_b.gid_of("X"));
    }

    #[test]
    fn gid_is_stable_after_adding_sibling() {
        let defs_v1 = &[
            NamespaceDef::new("A", None),
            NamespaceDef::new("A.B", Some("A")),
            NamespaceDef::new("X", None),
        ];
        let defs_v2 = &[
            NamespaceDef::new("A", None),
            NamespaceDef::new("A.B", Some("A")),
            NamespaceDef::new("A.NEW", Some("A")), // added!
            NamespaceDef::new("X", None),
        ];

        let reg_v1 = NamespaceRegistry::build(defs_v1).unwrap();
        let reg_v2 = NamespaceRegistry::build(defs_v2).unwrap();

        // Existing GIDs unchanged after adding a sibling
        assert_eq!(reg_v1.gid_of("A"), reg_v2.gid_of("A"));
        assert_eq!(reg_v1.gid_of("A.B"), reg_v2.gid_of("A.B"));
        assert_eq!(reg_v1.gid_of("X"), reg_v2.gid_of("X"));

        // New node has its own GID
        assert!(reg_v2.gid_of("A.NEW").is_some());
    }

    #[test]
    fn subtree_check_o1() {
        let reg = NamespaceRegistry::build(sample_defs()).unwrap();

        let movement = reg.gid_of("Movement").unwrap();
        let idle = reg.gid_of("Movement.Idle").unwrap();
        let running = reg.gid_of("Movement.Running").unwrap();
        let combat = reg.gid_of("Combat").unwrap();
        let attack = reg.gid_of("Combat.Attack").unwrap();

        // Movement.Idle is under Movement
        assert!(gid_is_descendant_of(idle, movement));
        assert!(gid_is_descendant_of(running, movement));

        // Combat.Attack is NOT under Movement
        assert!(!gid_is_descendant_of(attack, movement));

        // Combat.Attack IS under Combat
        assert!(gid_is_descendant_of(attack, combat));

        // A node is its own descendant
        assert!(gid_is_descendant_of(movement, movement));

        // String-based convenience function
        assert_eq!(reg.is_descendant_of_path("Movement.Idle", "Movement"), Some(true));
        assert_eq!(reg.is_descendant_of_path("Combat.Attack", "Movement"), Some(false));
        assert_eq!(reg.is_descendant_of_path("Unknown", "Movement"), None);
    }

    #[test]
    fn descendants_of_collects_subtree() {
        let reg = NamespaceRegistry::build(sample_defs()).unwrap();
        let movement = reg.gid_of("Movement").unwrap();

        let desc = reg.descendants_of(movement);
        let desc_paths: Vec<&str> = desc.iter().filter_map(|&gid| reg.path_of(gid)).collect();

        assert!(desc_paths.contains(&"Movement"));
        assert!(desc_paths.contains(&"Movement.Idle"));
        assert!(desc_paths.contains(&"Movement.Running"));
        assert!(desc_paths.contains(&"Movement.Jumping"));
        assert!(!desc_paths.contains(&"Combat"));
        assert!(!desc_paths.contains(&"Combat.Attack"));
    }

    #[test]
    fn depth_tracking() {
        use crate::layout::depth_of;
        let reg = NamespaceRegistry::build(sample_defs()).unwrap();

        assert_eq!(depth_of(reg.gid_of("Movement").unwrap()), 0);
        assert_eq!(depth_of(reg.gid_of("Movement.Idle").unwrap()), 1);
        assert_eq!(depth_of(reg.gid_of("Combat.Attack").unwrap()), 1);
    }

    #[test]
    fn parent_tracking() {
        use crate::layout::parent_of;
        let reg = NamespaceRegistry::build(sample_defs()).unwrap();

        let movement = reg.gid_of("Movement").unwrap();
        let idle = reg.gid_of("Movement.Idle").unwrap();

        assert_eq!(parent_of(movement), None);
        assert_eq!(parent_of(idle), Some(movement));
    }

    #[test]
    fn empty_build_returns_empty_registry() {
        let reg = NamespaceRegistry::build(&[]).unwrap();
        assert!(reg.is_empty());
    }

    #[test]
    fn rejects_duplicate_path() {
        let defs = &[NamespaceDef::new("A", None), NamespaceDef::new("A", None)];
        assert!(NamespaceRegistry::build(defs).is_err());
    }

    #[test]
    fn rejects_missing_parent() {
        let defs = &[NamespaceDef::new("A.B", Some("A"))];
        assert!(NamespaceRegistry::build(defs).is_err());
    }

    #[test]
    fn deep_tree_works() {
        // 4 levels deep - should work fine with static layout
        let defs = &[
            NamespaceDef::new("A", None),
            NamespaceDef::new("A.B", Some("A")),
            NamespaceDef::new("A.B.C", Some("A.B")),
            NamespaceDef::new("A.B.C.D", Some("A.B.C")),
        ];

        let reg = NamespaceRegistry::build(defs).unwrap();

        let a = reg.gid_of("A").unwrap();
        let ab = reg.gid_of("A.B").unwrap();
        let abc = reg.gid_of("A.B.C").unwrap();
        let abcd = reg.gid_of("A.B.C.D").unwrap();

        // Every deeper node is descendant of every shallower ancestor
        assert!(gid_is_descendant_of(abcd, abc));
        assert!(gid_is_descendant_of(abcd, ab));
        assert!(gid_is_descendant_of(abcd, a));
        assert!(gid_is_descendant_of(abc, ab));
        assert!(gid_is_descendant_of(abc, a));
        assert!(gid_is_descendant_of(ab, a));

        // Not the other way
        assert!(!gid_is_descendant_of(a, ab));

        // Depths are correct (use standalone function)
        use crate::layout::depth_of;
        assert_eq!(depth_of(a), 0);
        assert_eq!(depth_of(ab), 1);
        assert_eq!(depth_of(abc), 2);
        assert_eq!(depth_of(abcd), 3);
    }

    #[test]
    fn very_deep_tree_works() {
        // Test 8 levels deep (MAX_DEPTH)
        let defs: Vec<NamespaceDef> = (0..8)
            .map(|i| {
                let path: &'static str = match i {
                    0 => "L0",
                    1 => "L0.L1",
                    2 => "L0.L1.L2",
                    3 => "L0.L1.L2.L3",
                    4 => "L0.L1.L2.L3.L4",
                    5 => "L0.L1.L2.L3.L4.L5",
                    6 => "L0.L1.L2.L3.L4.L5.L6",
                    7 => "L0.L1.L2.L3.L4.L5.L6.L7",
                    _ => unreachable!(),
                };
                let parent: Option<&'static str> = if i == 0 {
                    None
                } else {
                    Some(match i {
                        1 => "L0",
                        2 => "L0.L1",
                        3 => "L0.L1.L2",
                        4 => "L0.L1.L2.L3",
                        5 => "L0.L1.L2.L3.L4",
                        6 => "L0.L1.L2.L3.L4.L5",
                        7 => "L0.L1.L2.L3.L4.L5.L6",
                        _ => unreachable!(),
                    })
                };
                NamespaceDef::new(path, parent)
            })
            .collect();

        let reg = NamespaceRegistry::build(&defs).unwrap();

        // Verify depths (use standalone function)
        use crate::layout::depth_of;
        for i in 0..8 {
            let path = defs[i].path;
            assert_eq!(depth_of(reg.gid_of(path).unwrap()), i as u8);
        }

        // Verify descendant relationships
        let root = reg.gid_of("L0").unwrap();
        let leaf = reg.gid_of("L0.L1.L2.L3.L4.L5.L6.L7").unwrap();
        assert!(gid_is_descendant_of(leaf, root));
    }

    // =========================================================================
    // Dynamic registration tests
    // =========================================================================

    #[test]
    fn dynamic_register_basic() {
        use crate::layout::depth_of;
        let mut reg = NamespaceRegistry::new();

        let gid = reg.register("Combat").unwrap();
        assert!(reg.contains("Combat"));
        assert!(reg.contains_gid(gid));
        assert_eq!(reg.path_of(gid).unwrap(), "Combat");
        assert_eq!(depth_of(gid), 0);
    }

    #[test]
    fn dynamic_register_nested() {
        use crate::layout::depth_of;
        let mut reg = NamespaceRegistry::new();

        // Register nested path - parents should auto-create
        let gid = reg.register("Combat.Special.Fireball").unwrap();

        // All parents should exist
        assert!(reg.contains("Combat"));
        assert!(reg.contains("Combat.Special"));
        assert!(reg.contains("Combat.Special.Fireball"));

        // Depths should be correct
        assert_eq!(depth_of(reg.gid_of("Combat").unwrap()), 0);
        assert_eq!(depth_of(reg.gid_of("Combat.Special").unwrap()), 1);
        assert_eq!(depth_of(gid), 2);

        // Descendant check should work
        let combat = reg.gid_of("Combat").unwrap();
        assert!(gid_is_descendant_of(gid, combat));
    }

    #[test]
    fn dynamic_register_idempotent() {
        let mut reg = NamespaceRegistry::new();

        let gid1 = reg.register("Combat").unwrap();
        let gid2 = reg.register("Combat").unwrap();

        // Same GID returned
        assert_eq!(gid1, gid2);
        // Only one entry
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn dynamic_register_all_marked_dynamic() {
        let mut reg = NamespaceRegistry::new();
        reg.register("A.B.C").unwrap();

        for entry in reg.entries() {
            assert!(entry.is_dynamic);
        }
    }

    #[test]
    fn dynamic_mixed_with_static() {
        // Build from static defs
        let defs = &[
            NamespaceDef::new("Movement", None),
            NamespaceDef::new("Movement.Idle", Some("Movement")),
        ];
        let mut reg = NamespaceRegistry::build(defs).unwrap();

        // Static entries are not dynamic
        assert!(!reg.entries().iter().any(|e| e.is_dynamic));

        // Add dynamic
        reg.register("Movement.Running").unwrap();
        reg.register("Combat.Attack").unwrap();

        // Check mix
        assert!(
            !reg.entries()
                .iter()
                .find(|e| e.path == "Movement")
                .unwrap()
                .is_dynamic
        );
        assert!(
            !reg.entries()
                .iter()
                .find(|e| e.path == "Movement.Idle")
                .unwrap()
                .is_dynamic
        );
        assert!(
            reg.entries()
                .iter()
                .find(|e| e.path == "Movement.Running")
                .unwrap()
                .is_dynamic
        );
        assert!(
            reg.entries()
                .iter()
                .find(|e| e.path == "Combat")
                .unwrap()
                .is_dynamic
        );
        assert!(
            reg.entries()
                .iter()
                .find(|e| e.path == "Combat.Attack")
                .unwrap()
                .is_dynamic
        );

        // Descendant checks work across static/dynamic
        let movement = reg.gid_of("Movement").unwrap();
        let running = reg.gid_of("Movement.Running").unwrap();
        assert!(gid_is_descendant_of(running, movement));
    }

    #[test]
    fn dynamic_rejects_empty_path() {
        let mut reg = NamespaceRegistry::new();
        assert!(reg.register("").is_err());
    }

    #[test]
    fn dynamic_gid_stable() {
        // Same path should produce same GID whether registered first or later
        let mut reg1 = NamespaceRegistry::new();
        let mut reg2 = NamespaceRegistry::new();

        reg1.register("A.B.C").unwrap();
        reg2.register("X.Y").unwrap();
        reg2.register("A.B.C").unwrap();

        assert_eq!(reg1.gid_of("A.B.C"), reg2.gid_of("A.B.C"));
    }

    #[test]
    fn tree_depth_grows_dynamically() {
        let mut reg = NamespaceRegistry::new();

        // Empty registry has depth 0
        assert_eq!(reg.tree_depth(), 0);

        // Register root node (depth 0) -> tree_depth becomes 1
        reg.register("Root").unwrap();
        assert_eq!(reg.tree_depth(), 1);

        // Register depth 1 node -> tree_depth becomes 2
        reg.register("Root.Child").unwrap();
        assert_eq!(reg.tree_depth(), 2);

        // Register depth 3 node -> tree_depth becomes 4
        reg.register("Root.Child.Grand.Great").unwrap();
        assert_eq!(reg.tree_depth(), 4);

        // Register shallower node -> tree_depth stays at 4
        reg.register("Other").unwrap();
        assert_eq!(reg.tree_depth(), 4);

        // Register another deep node at same depth -> stays at 4
        reg.register("Other.A.B.C").unwrap();
        assert_eq!(reg.tree_depth(), 4);

        // Register deeper -> grows to 5
        reg.register("Other.A.B.C.D").unwrap();
        assert_eq!(reg.tree_depth(), 5);
    }

    #[test]
    fn tree_depth_with_static_defs() {
        let defs = &[
            NamespaceDef::new("A", None),
            NamespaceDef::new("A.B", Some("A")),
            NamespaceDef::new("A.B.C", Some("A.B")),
        ];

        let mut reg = NamespaceRegistry::build(defs).unwrap();
        assert_eq!(reg.tree_depth(), 3);

        // Dynamic registration can grow it further
        reg.register("A.B.C.D.E").unwrap();
        assert_eq!(reg.tree_depth(), 5);
    }

    // =========================================================================
    // Metadata tests
    // =========================================================================

    #[test]
    fn metadata_typed_set_get() {
        let mut reg = NamespaceRegistry::new();
        let gid = reg.register("Combat").unwrap();

        // Set typed metadata (i32 implements IntoBytes)
        reg.set_meta(gid, "damage", &50i32);
        reg.set_meta(gid, "range", &100u16);

        // Get typed metadata
        assert_eq!(reg.get_meta::<i32>(gid, "damage"), Some(&50i32));
        assert_eq!(reg.get_meta::<u16>(gid, "range"), Some(&100u16));
        assert_eq!(reg.get_meta::<i32>(gid, "nonexistent"), None);

        // Wrong type returns None (size mismatch)
        assert_eq!(reg.get_meta::<u64>(gid, "damage"), None);
    }

    #[test]
    fn metadata_raw_set_get() {
        let mut reg = NamespaceRegistry::new();
        let gid = reg.register("Combat").unwrap();

        // Set raw bytes
        reg.set_meta_raw(gid, "data", vec![1, 2, 3, 4]);

        // Get raw bytes
        assert_eq!(reg.get_meta_raw(gid, "data"), Some(&[1, 2, 3, 4][..]));
    }

    #[test]
    fn metadata_has_and_remove() {
        let mut reg = NamespaceRegistry::new();
        let gid = reg.register("Combat").unwrap();

        reg.set_meta(gid, "damage", &50i32);

        assert!(reg.has_meta(gid, "damage"));
        assert!(!reg.has_meta(gid, "nonexistent"));

        let removed = reg.remove_meta(gid, "damage");
        assert!(removed.is_some());
        assert!(!reg.has_meta(gid, "damage"));
    }

    #[test]
    fn metadata_keys_and_iter() {
        let mut reg = NamespaceRegistry::new();
        let gid = reg.register("Combat").unwrap();

        reg.set_meta(gid, "damage", &50i32);
        reg.set_meta(gid, "range", &10i32);

        let keys: Vec<_> = reg.meta_keys(gid).unwrap().collect();
        assert!(keys.contains(&"damage"));
        assert!(keys.contains(&"range"));

        let pairs: Vec<_> = reg.meta_iter(gid).unwrap().collect();
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn metadata_overwrite() {
        let mut reg = NamespaceRegistry::new();
        let gid = reg.register("Combat").unwrap();

        let old = reg.set_meta(gid, "damage", &50i32);
        assert!(old.is_none());

        let old = reg.set_meta(gid, "damage", &100i32);
        assert!(old.is_some()); // previous bytes

        assert_eq!(reg.get_meta::<i32>(gid, "damage"), Some(&100i32));
    }

    // =========================================================================
    // Standalone is_descendant_of with dynamic GIDs
    // =========================================================================

    #[test]
    fn standalone_is_descendant_of_works_with_dynamic_gids() {
        use crate::layout::{depth_of, gid_is_descendant_of};

        let mut reg = NamespaceRegistry::new();

        // Register a hierarchy dynamically
        let root = reg.register("DynRoot").unwrap();
        let child = reg.register("DynRoot.Child").unwrap();
        let grandchild = reg.register("DynRoot.Child.Grandchild").unwrap();
        let other = reg.register("OtherRoot").unwrap();

        // Verify depths are embedded correctly
        assert_eq!(depth_of(root), 0);
        assert_eq!(depth_of(child), 1);
        assert_eq!(depth_of(grandchild), 2);
        assert_eq!(depth_of(other), 0);

        // Standalone is_descendant_of should work WITHOUT registry
        assert!(gid_is_descendant_of(child, root));
        assert!(gid_is_descendant_of(grandchild, root));
        assert!(gid_is_descendant_of(grandchild, child));

        // Self is descendant of self
        assert!(gid_is_descendant_of(root, root));
        assert!(gid_is_descendant_of(child, child));

        // Not descendant of different root
        assert!(!gid_is_descendant_of(other, root));
        assert!(!gid_is_descendant_of(root, other));

        // Parent is not descendant of child
        assert!(!gid_is_descendant_of(root, child));
        assert!(!gid_is_descendant_of(child, grandchild));
    }
}
