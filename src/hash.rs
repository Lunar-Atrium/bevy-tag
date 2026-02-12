//! Hash functions for hierarchical GID computation.
//!
//! Uses FNV-1a for fast, const-compatible hashing with good distribution.
//! The depth is automatically encoded into the GID (bits 127:125).

use crate::layout::{encode_gid, DEPTH_MASK, LEVEL_OFFSETS, LEVEL_WIDTHS, MAX_DEPTH};

/// FNV-1a 64-bit hash — simple, fast, const-compatible.
pub const fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        i += 1;
    }
    hash
}

/// Hash a path segment into `width` bits.
///
/// The result is guaranteed to be non-zero (reserves 0 for "no node at this level").
#[inline]
pub const fn segment_hash(segment: &[u8], width: u8) -> u128 {
    debug_assert!(width > 0 && width <= 64, "width must be in 1..=64");
    let full = fnv1a_64(segment);
    // Mix bits for better distribution
    let mixed = full ^ (full >> 32) ^ (full >> 17);
    let mask = (1u128 << width) - 1;
    // Avoid 0 — reserve 0 for "no node at this level"
    let val = (mixed as u128) & mask;
    if val == 0 {
        1
    } else {
        val
    }
}

/// Compute a full hierarchical GID from path segments.
///
/// The depth is automatically encoded into the top 3 bits.
/// Uses the fixed static layout (LEVEL_WIDTHS and LEVEL_OFFSETS).
///
/// # Panics
///
/// Panics at compile time if `segments.len() > MAX_DEPTH`.
pub const fn hierarchical_gid(segments: &[&[u8]]) -> u128 {
    assert!(
        segments.len() <= MAX_DEPTH,
        "tree depth exceeds MAX_DEPTH (8)"
    );
    assert!(!segments.is_empty(), "segments cannot be empty");

    let depth = (segments.len() - 1) as u8; // depth is 0-indexed

    let mut payload: u128 = 0;
    let mut i = 0;
    while i < segments.len() {
        let seg = segment_hash(segments[i], LEVEL_WIDTHS[i]);
        payload |= seg << LEVEL_OFFSETS[i];
        i += 1;
    }

    // Verify payload doesn't overlap with depth bits
    debug_assert!(
        payload & DEPTH_MASK == 0,
        "payload should not touch depth bits"
    );

    encode_gid(payload, depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{depth_of, gid_is_descendant_of};

    #[test]
    fn fnv_basic_sanity() {
        assert_ne!(fnv1a_64(b"hello"), fnv1a_64(b"world"));
        assert_eq!(fnv1a_64(b"hello"), fnv1a_64(b"hello"));
    }

    #[test]
    fn segment_hash_never_zero() {
        // Test a bunch of inputs — none should produce 0
        let inputs = [b"a" as &[u8], b"b", b"foo", b"bar", b"Movement"];
        for width in [4, 8, 10, 12, 13, 14, 16, 18, 21] {
            for input in &inputs {
                assert_ne!(
                    segment_hash(input, width),
                    0,
                    "got 0 for {:?} at width {}",
                    input,
                    width
                );
            }
        }
    }

    #[test]
    fn hierarchical_gid_has_correct_depth() {
        let gid1 = hierarchical_gid(&[b"Movement"]);
        assert_eq!(depth_of(gid1), 0, "single segment should have depth 0");

        let gid2 = hierarchical_gid(&[b"Movement", b"Idle"]);
        assert_eq!(depth_of(gid2), 1, "two segments should have depth 1");

        let gid3 = hierarchical_gid(&[b"A", b"B", b"C", b"D"]);
        assert_eq!(depth_of(gid3), 3, "four segments should have depth 3");
    }

    #[test]
    fn hierarchical_gid_parent_is_prefix() {
        let parent = hierarchical_gid(&[b"Movement"]);
        let child = hierarchical_gid(&[b"Movement", b"Idle"]);

        // Child should be descendant of parent
        assert!(
            gid_is_descendant_of(child, parent),
            "child should be descendant of parent"
        );

        // But full GIDs differ
        assert_ne!(parent, child);
    }

    #[test]
    fn gid_stability() {
        // Same segments should always produce same GID
        let gid1 = hierarchical_gid(&[b"A", b"B", b"C"]);
        let gid2 = hierarchical_gid(&[b"A", b"B", b"C"]);
        assert_eq!(gid1, gid2);

        // Different segments produce different GIDs
        let gid3 = hierarchical_gid(&[b"A", b"B", b"D"]);
        assert_ne!(gid1, gid3);
    }

    #[test]
    fn deep_hierarchy_works() {
        // Test with many levels (up to MAX_DEPTH = 8)
        let segments: [&[u8]; MAX_DEPTH] = [b"L0", b"L1", b"L2", b"L3", b"L4", b"L5", b"L6", b"L7"];

        let gid = hierarchical_gid(&segments);

        // Should have depth 7 (0-indexed)
        assert_eq!(depth_of(gid), 7);

        // Verify each level has non-zero contribution
        for level in 0..MAX_DEPTH {
            let width = LEVEL_WIDTHS[level];
            let offset = LEVEL_OFFSETS[level];
            let mask = ((1u128 << width) - 1) << offset;
            let level_bits = gid & mask;
            assert_ne!(level_bits, 0, "level {} should have non-zero bits", level);
        }
    }

    #[test]
    fn descendant_chain_works() {
        let l0 = hierarchical_gid(&[b"A"]);
        let l1 = hierarchical_gid(&[b"A", b"B"]);
        let l2 = hierarchical_gid(&[b"A", b"B", b"C"]);
        let l3 = hierarchical_gid(&[b"A", b"B", b"C", b"D"]);

        // All are descendants of l0
        assert!(gid_is_descendant_of(l1, l0));
        assert!(gid_is_descendant_of(l2, l0));
        assert!(gid_is_descendant_of(l3, l0));

        // l2, l3 are descendants of l1
        assert!(gid_is_descendant_of(l2, l1));
        assert!(gid_is_descendant_of(l3, l1));

        // l3 is descendant of l2
        assert!(gid_is_descendant_of(l3, l2));

        // Not the other way
        assert!(!gid_is_descendant_of(l0, l1));
        assert!(!gid_is_descendant_of(l1, l2));

        // Different root is not a descendant
        let other = hierarchical_gid(&[b"X"]);
        assert!(!gid_is_descendant_of(other, l0));
        assert!(!gid_is_descendant_of(l0, other));
    }
}
