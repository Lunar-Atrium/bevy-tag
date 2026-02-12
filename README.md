# bevy-tag

> **Still under heavy development** - API may change significantly before 1.0 release.

A hierarchical namespace tag system for Rust, inspired by Unreal Engine 5's GameplayTags. Provides stable, compile-time identifiers with O(1) subtree membership checks.

## Features

- **Compile-time GIDs**: Zero-cost tag identifiers computed at compile time
- **O(1) Subtree Checks**: Instant hierarchy membership tests via bit manipulation
- **Type-safe Tags**: Each tag is a unique zero-sized type with associated constants
- **Runtime Registry**: Path ↔ GID bidirectional lookup and dynamic registration
- **UE5-style Redirects**: Rename tags while preserving serialized GIDs
- **Build-time Code Generation**: Generate tags from TOML configuration

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
bevy-tag = "0.1"
bevy-tag-macro = "0.1"
```

Define your tags:

```rust
use bevy_tag::*;
use bevy_tag_macro::namespace;

namespace! {
    pub mod Tags {
        Movement {
            Idle;
            Running;
            Jumping;
        }
        Combat {
            Attack;
            Block;
        }
    }
}

fn main() {
    // Compile-time GID access
    let idle_gid = Tags::movement::Idle::GID;

    // O(1) subtree check - no registry needed
    let dominated  = Tags::movement::Running::GID;
    assert!(gid_is_descendant_of(dominated, Tags::Movement::GID));
    assert!(!gid_is_descendant_of(dominated, Tags::Combat::GID));

    // Runtime registry for path lookup
    let registry = NamespaceRegistry::build(Tags::DEFINITIONS).unwrap();
    assert_eq!(registry.path_of(idle_gid), Some("Movement.Idle"));
    assert_eq!(registry.gid_of("Combat.Attack"), Some(Tags::combat::Attack::GID));
}
```

## GID Layout

A `GID` is a `u128` with embedded depth and fixed bit allocation across 8 levels:

```
┌─────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ Depth   │ Level 0  │ Level 1  │ Level 2  │ Level 3  │ Level 4  │ Level 5  │ Level 6  │ Level 7  │
│ 3 bits  │ 21 bits  │ 18 bits  │ 16 bits  │ 16 bits  │ 14 bits  │ 14 bits  │ 13 bits  │ 13 bits  │
└─────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

## Core Operations

### Subtree Membership

```rust
use bevy_tag::gid_is_descendant_of;

// Check if a tag is under a parent - single bitmask comparison
if gid_is_descendant_of(entity_tag, Tags::Movement::GID) {
    // entity has a Movement-related tag
}
```

### Hierarchy Navigation

```rust
use bevy_tag::{depth_of, parent_of, is_sibling};

let tag = Tags::movement::Idle::GID;

// Get depth (0 = root level)
assert_eq!(depth_of(tag), 1);

// Get parent GID
assert_eq!(parent_of(tag), Some(Tags::Movement::GID));

// Check if two tags share a parent
assert!(is_sibling(
    Tags::movement::Idle::GID,
    Tags::movement::Running::GID
));
```

### Registry Operations

```rust
let mut registry = NamespaceRegistry::build(Tags::DEFINITIONS).unwrap();

// Bidirectional lookup
let gid = registry.gid_of("Movement.Idle").unwrap();
let path = registry.path_of(gid).unwrap();

// Dynamic registration
let custom_gid = registry.register("Combat.Special.Fireball").unwrap();

// Collect all descendants
let movement_tags = registry.descendants_of(Tags::Movement::GID);
```

## Redirects

When renaming tags, use redirects to preserve serialized GIDs:

```rust
namespace! {
    pub mod Tags {
        // New canonical location
        Equipment {
            Weapon {
                Blade;
            }
        }

        // Old path redirects to new location
        Legacy {
            #[redirect = "Equipment.Weapon.Blade"]
            OldSword;  // Same GID as Equipment.Weapon.Blade
        }
    }
}

// Using old path triggers deprecation warning
#[allow(deprecated)]
let old = Tags::legacy::OldSword::GID;
let new = Tags::equipment::weapon::Blade::GID;
assert_eq!(old, new);  // Same GID!
```

## Build-time Generation (Optional)

For large tag sets, generate from TOML:

```toml
# tags.toml
module_name = "GameTags"
on_remove = "warn"  # or "error" (default)

[tags]
paths = [
    "Item.Weapon.Sword",
    "Item.Weapon.Axe",
    "Item.Armor.Helmet",
    "Skill.Combat.Strike",
]

[redirects]
"Legacy.OldSword" = "Item.Weapon.Sword"
```

```rust
// build.rs
fn main() {
    println!("cargo:rerun-if-changed=tags.toml");
    bevy_tag_build::generate("tags.toml", "src/generated_tags.rs")
        .expect("Failed to generate tags");
}
```

## Static Metadata

Attach compile-time constants to tags:

```rust
namespace! {
    pub mod Abilities {
        #[mana_cost = 10]
        #[cooldown = 5.0]
        Fireball;

        #[mana_cost = 25]
        IceBlast;
    }
}

// Access metadata as associated constants
assert_eq!(Abilities::Fireball::MANA_COST, 10);
assert_eq!(Abilities::Fireball::COOLDOWN, 5.0);
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| `bevy-tag` | Core library with GID operations and registry |
| `bevy-tag-macro` | `namespace!` procedural macro |
| `bevy-tag-build` | Build-time TOML parsing and code generation |

## License

MIT OR Apache-2.0