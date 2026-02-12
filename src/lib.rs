//! # Hierarchical Namespace ID System (bevy-tag)
//!
//! Provides stable, hierarchical identifiers (GID) for tree-structured namespaces.
//! Inspired by UE5 GameplayTags with O(1) subtree membership checks.
//!
//! ## Design
//!
//! A `GID` is a `u128` with embedded depth and fixed bit allocation across 8 levels:
//!
//! ```text
//! ┌─────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
//! │ Depth   │ Level 0  │ Level 1  │ Level 2  │ Level 3  │ Level 4  │ Level 5  │ Level 6  │ Level 7  │
//! │ 3 bits  │ 21 bits  │ 18 bits  │ 16 bits  │ 16 bits  │ 14 bits  │ 14 bits  │ 13 bits  │ 13 bits  │
//! └─────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
//! ```
//!
//! ## Self-Contained Operations
//!
//! Because depth is embedded in the GID, subtree checks are completely self-contained if you cannot acces registery:
//!
//! ```ignore
//! use bevy_tag::{gid_is_descendant_of, depth_of};
//!
//! // No registry needed!
//! if gid_is_descendant_of(entity_tag, Movement::GID) {
//!     // entity is under Movement subtree
//! }
//!
//! let depth = depth_of(tag); // Extract depth directly from GID
//! ```
//!
//! ## Quick Start
//!
//! ```ignore
//! use bevy_tag::*;
//! use bevy_tag_macro::namespace;
//!
//! namespace! {
//!     pub mod Tags {
//!         Movement { Idle; Running; }
//!         Combat { Attack; Block; }
//!     }
//! }
//!
//! // Compile-time GID access
//! let gid = Tags::movement::Idle::GID;
//!
//! // O(1) subtree check
//! assert!(gid_is_descendant_of(gid, Tags::Movement::GID));
//! let registry = NamespaceRegistry::build(Tags::DEFINITIONS).unwrap();
//! assert!(is_descendant_of(registery, gid, Tags::Movement));
//!
//! // Runtime registry for path ↔ GID lookup
//! let registry = NamespaceRegistry::build(Tags::DEFINITIONS).unwrap();
//! assert_eq!(registry.path_of(gid), Some("Movement.Idle"));
//! ```

pub(crate) mod hash;
pub(crate) mod layout;
mod registry;
mod traits;

// =============================================================================
// Core Types
// =============================================================================

/// Global Identifier — a stable, hierarchical hash packed into u128.
///
/// The top 3 bits encode the depth (0-7), remaining 125 bits are
/// partitioned by tree level with a fixed layout.
///
/// Subtree membership is a single bitmask comparison (O(1)),
/// and requires no external registry lookup.
pub type GID = u128;

/// Root GID constant (all zeros, depth 0).
pub const ROOT_GID: GID = 0;

/// Maximum supported tree depth (0-7, 8 levels total).
pub use layout::MAX_DEPTH;

pub use traits::{HasData, IntoGid, IntoGids, IntoGidWithRegistry, NamespaceTag, Redirect};
pub use layout::{depth_of, gid_is_descendant_of, is_sibling, parent_of};
pub use registry::{NamespaceDef, NamespaceEntry, NamespaceRegistry};

/// Compute a full hierarchical GID from path segments.
///
/// This is primarily used by the `namespace!` macro. Users typically don't
/// need to call this directly — use the generated `Tag::GID` constants instead.
#[doc(hidden)]
pub use hash::hierarchical_gid;

