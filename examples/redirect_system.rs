//! Redirect system demonstration (UE5-style redirectors).
//!
//! This example shows how to:
//! - Use `#[redirect = "Target.Path"]` to redirect old paths to new canonical paths
//! - The redirected type becomes `Redirect<TargetType>` automatically
//! - GID matches the target, PATH returns canonical path
//!
//! ## Use Cases
//!
//! 1. **Renaming tags**: Redirect old path to new canonical location
//! 2. **Restructuring**: Move tags while preserving serialized GIDs
//! 3. **Type-level documentation**: `Redirect<T>` makes redirects explicit in code

use bevy_tag::*;
use bevy_tag_macro::namespace;

namespace! {
    pub mod Tags {
        // Current canonical paths
        Equipment {
            Weapon {
                Blade;
                Bow;
            }
            Armor {
                Helmet;
                Chestplate;
            }
        }

        Ability {
            Combat {
                Strike;
                Parry;
            }
        }

        // Old paths redirected to new locations
        // These generate: pub type OldSword = Redirect<equipment::weapon::Blade>;
        Legacy {
            #[redirect = "Equipment.Weapon.Blade"]
            OldSword;

            #[redirect = "Equipment.Weapon.Bow"]
            OldBow;

            #[redirect = "Equipment.Armor.Helmet"]
            OldHelmet;

            #[redirect = "Ability.Combat.Strike"]
            OldAttack;
        }
    }
}

fn main() {
    println!("=== Redirect System Example (UE5-style) ===\n");

    // -------------------------------------------------------------------------
    // 1. Demonstrate that redirects share the same GID as their target
    // -------------------------------------------------------------------------
    println!("1. Redirect GID equality:");

    let blade_gid = Tags::equipment::weapon::Blade::GID;
    // Tags::legacy::OldSword is now Redirect<equipment::weapon::Blade>
    #[allow(deprecated)]
    let redirected_gid = Tags::legacy::OldSword::GID;

    println!("   Equipment.Weapon.Blade GID: {:#034x}", blade_gid);
    println!("   Legacy.OldSword GID:        {:#034x}", redirected_gid);
    println!("   Are they equal? {}", blade_gid == redirected_gid);
    println!();

    // -------------------------------------------------------------------------
    // 2. PATH returns canonical path (UE5 behavior)
    // -------------------------------------------------------------------------
    println!("2. PATH returns canonical location:");

    #[allow(deprecated)]
    let redirected_path = Tags::legacy::OldSword::PATH;
    let canonical_path = Tags::equipment::weapon::Blade::PATH;

    println!("   Legacy.OldSword::PATH:          {}", redirected_path);
    println!("   Equipment.Weapon.Blade::PATH:   {}", canonical_path);
    println!("   (Both return the canonical path)");
    println!();

    // -------------------------------------------------------------------------
    // 3. Type shows the redirect relationship
    // -------------------------------------------------------------------------
    println!("3. Type-level redirect information:");

    // The type itself is Redirect<equipment::weapon::Blade>
    fn show_type<T: NamespaceTag>(_: T) {
        println!("   Type PATH: {}", T::PATH);
        println!("   Type GID:  {:#034x}", T::GID);
    }

    println!("   Calling show_type with redirected OldSword:");
    #[allow(deprecated)]
    show_type(Tags::legacy::OldSword::default());
    println!();

    // -------------------------------------------------------------------------
    // 4. Subtree checks work correctly
    // -------------------------------------------------------------------------
    println!("4. Subtree membership:");

    let equipment = Tags::Equipment::GID;

    println!(
        "   Is Blade under Equipment? {}",
        gid_is_descendant_of(blade_gid, equipment)
    );
    #[allow(deprecated)]
    let old_sword_gid = Tags::legacy::OldSword::GID;
    println!(
        "   Is redirected OldSword under Equipment? {}",
        gid_is_descendant_of(old_sword_gid, equipment)
    );
    println!();

    // -------------------------------------------------------------------------
    // 5. Registry resolves to canonical path
    // -------------------------------------------------------------------------
    println!("5. Registry compatibility:");

    let registry = NamespaceRegistry::build(Tags::DEFINITIONS).unwrap();

    // Note: DEFINITIONS doesn't include redirect nodes, only canonical paths
    #[allow(deprecated)]
    if let Some(path) = registry.path_of(Tags::legacy::OldSword::GID) {
        println!("   Redirected OldSword resolves to: '{}'", path);
    }
    println!();

    // -------------------------------------------------------------------------
    // 6. Summary
    // -------------------------------------------------------------------------
    println!("=== Summary ===");
    println!();
    println!("Use `#[redirect = \"Target.Path\"]` when:");
    println!("  - Renaming a tag but need to preserve GID for serialized data");
    println!("  - Moving tags to a new location in the hierarchy");
    println!("  - You want compile-time deprecation warnings on old paths");
    println!("  - The type `Redirect<T>` documents the relationship");
    println!();
    println!("Key behaviors:");
    println!("  - `OldPath::GID` == `NewPath::GID` (same GID)");
    println!("  - `OldPath::PATH` returns the canonical path (NewPath::PATH)");
    println!("  - Using old paths triggers deprecation warnings");
}