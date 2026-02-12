//! Dynamic tag registration at runtime.
//!
//! This example shows how to:
//! - Register tags at runtime (not via macro)
//! - Mix static (macro) and dynamic tags
//! - Auto-create parent nodes when registering nested paths

use bevy_tag::{NamespaceRegistry, depth_of, gid_is_descendant_of};
use bevy_tag_macro::namespace;

// Static tags defined at compile time
namespace! {
    pub mod StaticTags {
        Item {
            Weapon;
            Armor;
        }
    }
}

fn main() {
    println!("=== Dynamic Registration Example ===\n");

    // 1. Start with static definitions
    let mut registry = NamespaceRegistry::build(StaticTags::DEFINITIONS).unwrap();

    println!("Initial static tags:");
    for entry in registry.entries() {
        println!("  {} (dynamic={})", entry.path, entry.is_dynamic);
    }
    println!();

    // 2. Register dynamic tags - parents auto-created
    println!("Registering 'Item.Weapon.Sword.Longsword'...");
    let longsword = registry.register("Item.Weapon.Sword.Longsword").unwrap();
    println!("  GID: {:#034x}", longsword);
    println!();

    // Note: "Item.Weapon.Sword" was auto-created as a parent
    println!("After registration:");
    for entry in registry.entries() {
        let marker = if entry.is_dynamic {
            "[dynamic]"
        } else {
            "[static]"
        };
        println!("  {} {}", marker, entry.path);
    }
    println!();

    // 3. Idempotent registration - same path returns same GID
    println!("Registering same path again...");
    let longsword2 = registry.register("Item.Weapon.Sword.Longsword").unwrap();
    assert_eq!(longsword, longsword2);
    println!("  Same GID returned: {}", longsword == longsword2);
    println!();

    // 4. Register completely new branch
    println!("Registering new branch 'Skill.Combat.Slash'...");
    registry.register("Skill.Combat.Slash").unwrap();

    println!("Final registry:");
    for entry in registry.entries() {
        let marker = if entry.is_dynamic {
            "[dynamic]"
        } else {
            "[static]"
        };
        let depth_indent = "  ".repeat(depth_of(entry.gid) as usize);
        println!("  {}{} {}", depth_indent, marker, entry.path);
    }
    println!();

    // 5. Descendant checks work across static/dynamic (no registry needed!)
    println!("Descendant checks:");
    let item = StaticTags::Item::GID;
    let weapon = StaticTags::item::Weapon::GID;

    println!(
        "  Is Longsword under Item.Weapon? {}",
        gid_is_descendant_of(longsword, weapon)
    );
    println!(
        "  Is Longsword under Item? {}",
        gid_is_descendant_of(longsword, item)
    );

    // String-based lookup when you only have path strings
    let is_under_skill = registry
        .is_descendant_of_path("Item.Weapon.Sword.Longsword", "Skill")
        .unwrap_or(false);
    println!("  Is Longsword under Skill? {}", is_under_skill);
    println!();

    // 6. GID stability - same path always produces same GID
    println!("GID stability test:");
    let mut fresh_registry = NamespaceRegistry::new();
    let fresh_longsword = fresh_registry
        .register("Item.Weapon.Sword.Longsword")
        .unwrap();
    println!("  Original registry GID: {:#034x}", longsword);
    println!("  Fresh registry GID:    {:#034x}", fresh_longsword);
    println!("  GIDs match: {}", longsword == fresh_longsword);
}
