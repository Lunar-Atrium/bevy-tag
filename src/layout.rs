use crate::GID;

/// Maximum supported tree depth (0-7, encoded in 3 bits).
pub const MAX_DEPTH: usize = 8;

/// Bits reserved for depth encoding (documentation constant).
#[allow(dead_code)]
pub const DEPTH_BITS: u8 = 3;

/// Bit position where depth is stored (bits 127:125).
pub const DEPTH_SHIFT: u8 = 125;

/// Mask to extract depth from GID.
pub const DEPTH_MASK: u128 = 0b111 << DEPTH_SHIFT;

/// Fixed bit widths per level (after depth bits).
///
/// Distribution rationale (8 levels, 125 bits total after 3 depth bits):
/// - Level 0: 21 bits (2M slots) - top-level categories
/// - Level 1: 18 bits (256K slots) - major subcategories
/// - Level 2: 16 bits (64K slots) - common nesting
/// - Level 3: 16 bits (64K slots) - typical depth
/// - Level 4: 14 bits (16K slots) - detailed classification
/// - Level 5: 14 bits (16K slots) - detailed classification
/// - Level 6: 13 bits (8K slots) - deep nesting
/// - Level 7: 13 bits (8K slots) - deepest level
///
/// Total: 21 + 18 + 16 + 16 + 14 + 14 + 13 + 13 = 125 bits
pub const LEVEL_WIDTHS: [u8; MAX_DEPTH] = [
    21, // level 0: 2M nodes
    18, // level 1: 256K nodes
    16, // level 2: 64K nodes
    16, // level 3: 64K nodes
    14, // level 4: 16K nodes
    14, // level 5: 16K nodes
    13, // level 6: 8K nodes
    13, // level 7: 8K nodes
];

/// Precomputed cumulative bit offsets per level (from bit 0).
/// These are offsets within the 125-bit payload area (after 3 depth bits).
pub const LEVEL_OFFSETS: [u8; MAX_DEPTH] = {
    let mut offsets = [0u8; MAX_DEPTH];
    let mut acc = 0u8;
    let mut i = 0;
    // Build from the end (level 7 is at lowest bits)
    while i < MAX_DEPTH {
        let level = MAX_DEPTH - 1 - i;
        offsets[level] = acc;
        acc += LEVEL_WIDTHS[level];
        i += 1;
    }
    // Now reverse to get correct order
    let mut result = [0u8; MAX_DEPTH];
    let mut j = 0;
    while j < MAX_DEPTH {
        // Level 7 starts at bit 0, level 0 starts highest (after 3 depth bits)
        result[j] = 125 - offsets[j] - LEVEL_WIDTHS[j];
        j += 1;
    }
    result
};

/// Precomputed masks for O(1) subtree checks.
/// `LEVEL_MASKS[d]` masks out everything below level d, preserving depth + levels 0..=d.
pub const LEVEL_MASKS: [u128; MAX_DEPTH] = {
    let mut masks = [0u128; MAX_DEPTH];
    let mut i = 0;
    while i < MAX_DEPTH {
        // Start with depth bits
        let mut mask = DEPTH_MASK;
        // Add bits for levels 0..=i
        let mut j = 0;
        while j <= i {
            let width = LEVEL_WIDTHS[j];
            let offset = LEVEL_OFFSETS[j];
            mask |= ((1u128 << width) - 1) << offset;
            j += 1;
        }
        masks[i] = mask;
        i += 1;
    }
    masks
};

/// Static assertion: total bits must equal 125 (128 - 3 depth bits).
const _: () = {
    let mut total: u16 = 0;
    let mut i = 0;
    while i < MAX_DEPTH {
        total += LEVEL_WIDTHS[i] as u16;
        i += 1;
    }
    assert!(
        total == 125,
        "LEVEL_WIDTHS must sum to exactly 125 bits (128 - 3 depth bits)"
    );
};

// =============================================================================
// Standalone GID operations
// =============================================================================

/// Extract the depth (0-7) from a GID.
#[inline]
pub const fn depth_of(gid: GID) -> u8 {
    ((gid >> DEPTH_SHIFT) & 0b111) as u8
}

/// Create a GID with depth encoded.
///
/// `payload` is the hierarchical hash (125 bits), `depth` is 0-7.
#[inline]
pub const fn encode_gid(payload: u128, depth: u8) -> GID {
    debug_assert!(depth < MAX_DEPTH as u8, "depth must be < 8");
    debug_assert!(
        payload & DEPTH_MASK == 0,
        "payload must not overlap with depth bits"
    );
    payload | ((depth as u128) << DEPTH_SHIFT)
}

/// O(1) subtree test: is `candidate` a descendant of (or equal to) `ancestor`?
///
/// This is completely self-contained - extracts depth from the ancestor GID
/// and performs a single mask comparison on the payload bits only.
///
/// ```text
/// is_descendant_of(Movement.Idle.GID, Movement.GID) → true
/// is_descendant_of(Combat.Attack.GID, Movement.GID) → false
/// ```
#[inline]
pub fn gid_is_descendant_of(candidate: GID, ancestor: GID) -> bool {
    let ancestor_depth = depth_of(ancestor) as usize;
    if ancestor_depth >= MAX_DEPTH {
        return false;
    }
    // Only compare payload bits (exclude depth bits)
    // LEVEL_MASKS includes DEPTH_MASK, so we need to exclude it
    let payload_mask = LEVEL_MASKS[ancestor_depth] & !DEPTH_MASK;
    (candidate & payload_mask) == (ancestor & payload_mask)
}

/// Check if two GIDs share the same parent at a given depth.
#[inline]
pub const fn is_sibling(a: GID, b: GID) -> bool {
    let depth_a = depth_of(a);
    let depth_b = depth_of(b);
    if depth_a != depth_b || depth_a == 0 {
        return depth_a == depth_b && depth_a == 0; // Both are root level
    }
    // Same parent means same bits at parent's depth
    let parent_depth = (depth_a - 1) as usize;
    let parent_mask = LEVEL_MASKS[parent_depth];
    (a & parent_mask) == (b & parent_mask)
}

/// Get the parent GID of a given GID.
///
/// Returns `None` if the GID is at root level (depth 0).
#[inline]
pub const fn parent_of(gid: GID) -> Option<GID> {
    let depth = depth_of(gid);
    if depth == 0 {
        return None;
    }
    let parent_depth = depth - 1;
    // Mask out current level and update depth
    let parent_mask = LEVEL_MASKS[parent_depth as usize];
    let parent_payload = gid & parent_mask & !DEPTH_MASK;
    Some(encode_gid(parent_payload, parent_depth))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_constants_are_valid() {
        // Total bits should be 125 (128 - 3 for depth)
        let total: u16 = LEVEL_WIDTHS.iter().map(|&w| w as u16).sum();
        assert_eq!(total, 125);

        // Depth bits should not overlap with level bits
        for i in 0..MAX_DEPTH {
            let level_mask = ((1u128 << LEVEL_WIDTHS[i]) - 1) << LEVEL_OFFSETS[i];
            assert_eq!(
                level_mask & DEPTH_MASK,
                0,
                "level {} overlaps with depth bits",
                i
            );
        }
    }

    #[test]
    fn depth_extraction_works() {
        for d in 0..8u8 {
            let gid = encode_gid(0x123456, d);
            assert_eq!(depth_of(gid), d);
        }
    }

    #[test]
    fn is_descendant_works() {
        // Simulate parent at depth 0, child at depth 1
        let parent_payload = 0x100000u128 << LEVEL_OFFSETS[0]; // Some bits in level 0
        let parent = encode_gid(parent_payload, 0);

        let child_payload = parent_payload | (0x20000u128 << LEVEL_OFFSETS[1]); // Same level 0, different level 1
        let child = encode_gid(child_payload, 1);

        let other_payload = 0x200000u128 << LEVEL_OFFSETS[0]; // Different level 0
        let other = encode_gid(other_payload, 0);

        assert!(gid_is_descendant_of(child, parent), "child should be descendant of parent");
        assert!(!gid_is_descendant_of(other, parent), "other should not be descendant of parent");
        assert!(gid_is_descendant_of(parent, parent), "node is descendant of itself");
    }

    #[test]
    fn masks_include_depth() {
        // All masks should include the depth bits
        for i in 0..MAX_DEPTH {
            assert_eq!(
                LEVEL_MASKS[i] & DEPTH_MASK,
                DEPTH_MASK,
                "mask {} should include depth bits",
                i
            );
        }
    }

    #[test]
    fn parent_of_works() {
        let level0 = encode_gid(0x100000u128 << LEVEL_OFFSETS[0], 0);
        assert!(parent_of(level0).is_none(), "root has no parent");

        let level1 = encode_gid(
            (0x100000u128 << LEVEL_OFFSETS[0]) | (0x20000u128 << LEVEL_OFFSETS[1]),
            1,
        );
        let parent = parent_of(level1).unwrap();
        assert_eq!(depth_of(parent), 0);
        assert!(gid_is_descendant_of(level1, parent));
    }

    #[test]
    fn deep_hierarchy_fits() {
        // Build a GID with all 8 levels populated
        let mut payload: u128 = 0;
        for level in 0..MAX_DEPTH {
            let width = LEVEL_WIDTHS[level];
            let offset = LEVEL_OFFSETS[level];
            let max_val = (1u128 << width) - 1;
            payload |= max_val << offset;
        }

        // Should not overflow into depth bits
        assert_eq!(payload & DEPTH_MASK, 0, "payload should not touch depth bits");

        let gid = encode_gid(payload, 7);
        assert_eq!(depth_of(gid), 7);

        // Verify descendant chain works
        let ancestor = encode_gid(payload & LEVEL_MASKS[0] & !DEPTH_MASK, 0);
        assert!(gid_is_descendant_of(gid, ancestor));
    }
}
