//! Bevy integration for namespace tags.
//!
//! Provides:
//! - `NamespacePlugin` — builder-pattern plugin to initialize the registry as a Resource
//! - `TagContainer` — multi-tag component with O(1) membership checks
//!
//! # Example
//!
//! ```ignore
//! use bevy::prelude::*;
//! use bevy_tag::bevy::*;
//! use bevy_tag_macro::namespace;
//!
//! namespace! {
//!     pub mod Tags {
//!         Movement { Idle; Running; }
//!         Combat { Attack; Block; }
//!     }
//! }
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(NamespacePlugin::from_definitions(Tags::DEFINITIONS))
//!         .add_systems(Startup, spawn_entities)
//!         .run();
//! }
//!
//! fn spawn_entities(mut commands: Commands) {
//!     commands.spawn(
//!         TagContainer::new()
//!             .with(Tags::movement::Idle::GID)
//!             .with(Tags::combat::Block::GID)
//!     );
//! }
//! ```

use bevy::prelude::*;
use std::collections::HashSet;

use crate::{
    gid_is_descendant_of,
    registry::{NamespaceDef, NamespaceRegistry},
    GID,
};

// =============================================================================
// Plugin
// =============================================================================

/// Bevy plugin for namespace tag system.
///
/// Use the builder pattern to configure:
///
/// ```ignore
/// App::new()
///     .add_plugins(
///         NamespacePlugin::from_definitions(Tags::DEFINITIONS)
///     )
/// ```
#[derive(Default)]
pub struct NamespacePlugin {
    definitions: Option<&'static [NamespaceDef]>,
}

impl NamespacePlugin {
    /// Create a new plugin with no initial definitions.
    ///
    /// The registry will be empty until tags are dynamically registered.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a plugin from static namespace definitions (from `namespace!` macro).
    ///
    /// This is the most common way to initialize the plugin:
    ///
    /// ```ignore
    /// NamespacePlugin::from_definitions(Tags::DEFINITIONS)
    /// ```
    pub fn from_definitions(definitions: &'static [NamespaceDef]) -> Self {
        Self {
            definitions: Some(definitions),
        }
    }
}

impl Plugin for NamespacePlugin {
    fn build(&self, app: &mut App) {
        let registry = if let Some(defs) = self.definitions {
            NamespaceRegistry::build(defs).expect("Failed to build NamespaceRegistry from definitions")
        } else {
            NamespaceRegistry::new()
        };

        app.insert_resource(registry);
    }
}

// =============================================================================
// TagContainer Component
// =============================================================================

/// A container for multiple namespace tags.
///
/// Use this when an entity can have multiple tags simultaneously.
/// Provides O(1) membership checks via `HashSet`.
///
/// # Example
///
/// ```ignore
/// // Builder pattern
/// let tags = TagContainer::new()
///     .with(Tags::movement::Idle::GID)
///     .with(Tags::combat::Block::GID);
///
/// commands.spawn(tags);
///
/// // Query and check
/// fn system(query: Query<&TagContainer>) {
///     for container in query.iter() {
///         if container.has(Tags::movement::Idle::GID) {
///             // Entity has the Idle tag
///         }
///         if container.has_descendant_of(Tags::Combat::GID) {
///             // Entity has some Combat-related tag
///         }
///     }
/// }
/// ```
#[derive(Component, Clone, Debug, Default, PartialEq, Eq)]
pub struct TagContainer {
    tags: HashSet<GID>,
}

impl TagContainer {
    /// Create an empty tag container.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a container with a single tag.
    #[inline]
    pub fn single(gid: GID) -> Self {
        let mut tags = HashSet::new();
        tags.insert(gid);
        Self { tags }
    }

    /// Builder method: add a tag and return self.
    #[inline]
    pub fn with(mut self, gid: GID) -> Self {
        self.tags.insert(gid);
        self
    }

    /// Add a tag to the container.
    ///
    /// Returns `true` if the tag was newly inserted.
    #[inline]
    pub fn insert(&mut self, gid: GID) -> bool {
        self.tags.insert(gid)
    }

    /// Remove a tag from the container.
    ///
    /// Returns `true` if the tag was present.
    #[inline]
    pub fn remove(&mut self, gid: GID) -> bool {
        self.tags.remove(&gid)
    }

    /// Check if the container has a specific tag (O(1)).
    #[inline]
    pub fn has(&self, gid: GID) -> bool {
        self.tags.contains(&gid)
    }

    /// Check if any tag in the container is a descendant of the given ancestor.
    ///
    /// This is O(n) where n is the number of tags in the container.
    /// For frequent checks, consider caching results or using a different data structure.
    #[inline]
    pub fn has_descendant_of(&self, ancestor: GID) -> bool {
        self.tags.iter().any(|&gid| gid_is_descendant_of(gid, ancestor))
    }

    /// Get all tags that are descendants of the given ancestor.
    pub fn descendants_of(&self, ancestor: GID) -> impl Iterator<Item = GID> + '_ {
        self.tags
            .iter()
            .copied()
            .filter(move |&gid| gid_is_descendant_of(gid, ancestor))
    }

    /// Iterate over all tags in the container.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = GID> + '_ {
        self.tags.iter().copied()
    }

    /// Get the number of tags in the container.
    #[inline]
    pub fn len(&self) -> usize {
        self.tags.len()
    }

    /// Check if the container is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// Clear all tags from the container.
    #[inline]
    pub fn clear(&mut self) {
        self.tags.clear();
    }
}

impl FromIterator<GID> for TagContainer {
    fn from_iter<T: IntoIterator<Item = GID>>(iter: T) -> Self {
        Self {
            tags: iter.into_iter().collect(),
        }
    }
}

impl Extend<GID> for TagContainer {
    fn extend<T: IntoIterator<Item = GID>>(&mut self, iter: T) {
        self.tags.extend(iter);
    }
}

// =============================================================================
// Resource impl for NamespaceRegistry
// =============================================================================

impl Resource for NamespaceRegistry {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_container_builder() {
        let container = TagContainer::new()
            .with(1)
            .with(2)
            .with(3);

        assert_eq!(container.len(), 3);
        assert!(container.has(1));
        assert!(container.has(2));
        assert!(container.has(3));
        assert!(!container.has(4));
    }

    #[test]
    fn tag_container_insert_remove() {
        let mut container = TagContainer::new();

        assert!(container.insert(1));
        assert!(!container.insert(1)); // duplicate
        assert_eq!(container.len(), 1);

        assert!(container.remove(1));
        assert!(!container.remove(1)); // already removed
        assert!(container.is_empty());
    }

    #[test]
    fn tag_container_from_iter() {
        let container: TagContainer = [1, 2, 3].into_iter().collect();
        assert_eq!(container.len(), 3);
    }

    #[test]
    fn tag_container_extend() {
        let mut container = TagContainer::single(1);
        container.extend([2, 3]);
        assert_eq!(container.len(), 3);
    }

    #[test]
    fn tag_container_clear() {
        let mut container = TagContainer::new().with(1).with(2);
        container.clear();
        assert!(container.is_empty());
    }
}
