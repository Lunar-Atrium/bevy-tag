//! O(1) subtree membership checks.
//!
//! This example shows how to:
//! - Check if a tag is a descendant of another tag
//! - Understand how hierarchical GIDs enable O(1) checks
//! - Use `gid_is_descendant_of` for game logic (e.g., damage type filtering)
//! - Use tuples with `IntoGids` for ergonomic GID collection

use bevy_tag::{gid_is_descendant_of, GID, IntoGid, IntoGids, NamespaceRegistry};
use bevy_tag_macro::namespace;

namespace! {
    pub mod DamageTags {
        Physical {
            Blunt;
            Slash;
            Pierce;
        }
        Magical {
            Fire;
            Ice;
            Lightning;
        }
        True;  // Ignores all resistances
    }
}

/// Simulated entity with damage resistance
struct Entity {
    name: &'static str,
    /// GIDs of damage types this entity resists
    resistances: Vec<GID>,
}

fn main() {
    let registry = NamespaceRegistry::build(DamageTags::DEFINITIONS).unwrap();

    // 1. Basic descendant check - no registry needed!
    println!(
        "  Is Slash a descendant of Physical? {}",
        gid_is_descendant_of(DamageTags::physical::Slash::GID, DamageTags::Physical::GID)
    );
    println!();

    // 2. Collect all descendants
    println!("All descendants of 'Physical':");
    for gid in registry.descendants_of(DamageTags::Physical) {
        if let Some(path) = registry.path_of(gid) {
            println!("  - {}", path);
        }
    }
    println!();

    println!("All descendants of 'Magical':");
    for gid in registry.descendants_of(DamageTags::Magical) {
        if let Some(path) = registry.path_of(gid) {
            println!("  - {}", path);
        }
    }
    println!();

    let entities = vec![
        Entity {
            name: "Stone Golem",
            resistances: vec![DamageTags::Physical.into_gid()],
        },
        Entity {
            name: "Fire Elemental",
            resistances: vec![DamageTags::magical::Fire.into_gid()],
        },
        Entity {
            name: "Ghost",
            // Multiple tags - use tuple.into_gids()
            resistances: (DamageTags::Physical, DamageTags::Magical).into_gids(),
        },
    ];

    let damage_types: [(&str, u128); 4] = [
        ("Sword (Slash)", DamageTags::physical::Slash.into_gid()),
        ("Fireball (Fire)", DamageTags::magical::Fire.into_gid()),
        ("Ice Shard (Ice)", DamageTags::magical::Ice.into_gid()),
        ("Divine Smite (True)", DamageTags::True.into_gid()),
    ];

    for entity in &entities {
        println!("{}:", entity.name);
        for (name, damage_gid) in &damage_types {
            // gid_is_descendant_of is O(1) and needs no registry!
            let resisted = entity
                .resistances
                .iter()
                .any(|&resistance| gid_is_descendant_of(*damage_gid, resistance));
            let status = if resisted { "RESISTED" } else { "takes damage" };
            println!("  {} -> {}", name, status);
        }
        println!();
    }
}
