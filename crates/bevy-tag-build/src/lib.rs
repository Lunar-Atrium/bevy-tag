//! Build-time utilities for bevy-tag.
//!
//! This crate provides tools for:
//! - Parsing `tags.toml` configuration files
//! - Managing `tags.lock.toml` lock files for change detection
//! - Generating Rust code with the `namespace!` macro
//!
//! # Usage in build.rs
//!
//! ```ignore
//! // build.rs
//! fn main() {
//!     bevy_tag_build::generate("tags.toml", "src/generated_tags.rs")
//!         .expect("Failed to generate tags");
//! }
//! ```
//!
//! # Lock File Mechanism
//!
//! The lock file ensures that changes to `tags.toml` are intentional:
//!
//! - First build: generates `tags.lock.toml` with all paths and GIDs
//! - Subsequent builds: compares against lock file
//! - Mismatch (path removed/renamed): **compile error** (default) or **warning** (with `on_remove = "warn"`)
//! - New paths added: automatically appended to lock
//!
//! # Migration Modes
//!
//! Configure `on_remove` in `tags.toml`:
//!
//! ```toml
//! # Default: error on removed paths
//! on_remove = "error"
//!
//! # Or: warn (generates #[deprecated] for removed paths)
//! on_remove = "warn"
//! ```
//!
//! To intentionally break compatibility, delete the lock file and rebuild.

mod codegen;
mod lock;
mod toml_parser;

pub use codegen::{generate_namespace_code, generate_namespace_code_from_lock};
pub use lock::{LockFile, LockFileError};
pub use toml_parser::{OnRemove, RedirectEntry, TagsConfig, TagsConfigError};

use std::path::Path;

/// Main entry point for build.rs integration.
///
/// Reads `tags.toml`, compares with `tags.lock.toml`, and generates Rust code.
///
/// # Arguments
///
/// * `config_path` - Path to `tags.toml`
/// * `output_path` - Path to output Rust file (e.g., `src/generated_tags.rs`)
///
/// # Errors
///
/// Returns an error if:
/// - `tags.toml` cannot be read or parsed
/// - Lock file mismatch detected (paths removed) and `on_remove = "error"`
/// - Output file cannot be written
///
/// # Example
///
/// ```ignore
/// // build.rs
/// fn main() {
///     println!("cargo:rerun-if-changed=tags.toml");
///     bevy_tag_build::generate("tags.toml", "src/generated_tags.rs")
///         .expect("Failed to generate tags");
/// }
/// ```
pub fn generate(
    config_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<(), GenerateError> {
    let config_path = config_path.as_ref();
    let output_path = output_path.as_ref();

    // Derive lock file path from config path
    let lock_path = config_path.with_extension("lock.toml");

    generate_with_lock(config_path, &lock_path, output_path)
}

/// Generate with explicit lock file path.
pub fn generate_with_lock(
    config_path: impl AsRef<Path>,
    lock_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<(), GenerateError> {
    let config_path = config_path.as_ref();
    let lock_path = lock_path.as_ref();
    let output_path = output_path.as_ref();

    // 1. Parse tags.toml
    let config = TagsConfig::from_file(config_path)?;

    // 2. Load or create lock file
    let (lock, diff) = if lock_path.exists() {
        let existing_lock = LockFile::from_file(lock_path)?;
        let diff = existing_lock.diff(&config);
        (existing_lock, Some(diff))
    } else {
        (LockFile::from_config(&config), None)
    };

    // 3. Handle removed paths based on on_remove strategy
    let mut updated_lock = lock;
    if let Some(ref diff) = diff
        && !diff.removed.is_empty()
    {
        match config.on_remove {
            OnRemove::Error => {
                return Err(GenerateError::LockMismatch(format_lock_error(diff)));
            }
            OnRemove::Warn => {
                // Mark removed paths as deprecated instead of erroring
                for path in &diff.removed {
                    updated_lock.mark_deprecated(path);
                }
                // Emit cargo warning
                for path in &diff.removed {
                    println!(
                        "cargo:warning=bevy-tag: Path '{}' was removed from tags.toml and is now deprecated",
                        path
                    );
                }
            }
        }
    }

    // 4. Update lock file with new entries
    if let Some(ref diff) = diff {
        for path in &diff.added {
            if let Some(entry) = config.entries().find(|e| &e.path == path) {
                updated_lock.add_entry(entry.clone());
            }
        }
    }

    // 5. Write updated lock file
    updated_lock.write_to_file(lock_path)?;

    // 6. Generate Rust code (include deprecated entries from lock)
    let code = generate_namespace_code_from_lock(&config, &updated_lock);
    std::fs::write(output_path, code)?;

    Ok(())
}

fn format_lock_error(diff: &lock::LockDiff) -> String {
    let mut msg = String::new();
    msg.push_str("bevy-tag: Lock file mismatch!\n\n");
    msg.push_str("  Missing in tags.toml (existed in lock):\n");
    for path in &diff.removed {
        msg.push_str(&format!("    - {}\n", path));
    }
    msg.push_str("\n  To fix:\n");
    msg.push_str("    1. Add the path(s) back to tags.toml, OR\n");
    msg.push_str("    2. Set `on_remove = \"warn\"` in tags.toml to deprecate instead, OR\n");
    msg.push_str("    3. Delete tags.lock.toml to regenerate (BREAKING CHANGE!)\n");
    msg
}

/// Errors that can occur during generation.
#[derive(Debug)]
pub enum GenerateError {
    /// Failed to parse tags.toml
    ConfigError(TagsConfigError),
    /// Failed to read/write lock file
    LockError(LockFileError),
    /// Lock file mismatch (paths removed)
    LockMismatch(String),
    /// IO error
    Io(std::io::Error),
}

impl std::fmt::Display for GenerateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConfigError(e) => write!(f, "Config error: {}", e),
            Self::LockError(e) => write!(f, "Lock file error: {}", e),
            Self::LockMismatch(msg) => write!(f, "{}", msg),
            Self::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for GenerateError {}

impl From<TagsConfigError> for GenerateError {
    fn from(e: TagsConfigError) -> Self {
        Self::ConfigError(e)
    }
}

impl From<LockFileError> for GenerateError {
    fn from(e: LockFileError) -> Self {
        Self::LockError(e)
    }
}

impl From<std::io::Error> for GenerateError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
