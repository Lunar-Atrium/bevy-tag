use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{braced, token, Expr, Ident, Result, Token, Type, Visibility};

use proc_macro_crate::{crate_name, FoundCrate};

// =============================================================================
// Constants - must match bevy-tag's layout.rs
// =============================================================================

/// Maximum supported tree depth (0-7, encoded in 3 bits).
const MAX_DEPTH: usize = 8;

// =============================================================================
// Parsing
// =============================================================================

/// Metadata attribute: #[key = value]
#[derive(Clone)]
struct MetaAttr {
    key: Ident,
    value: Expr,
}

/// Deprecation attribute: #[deprecated(note = "...")]
#[derive(Clone, Default)]
struct DeprecationAttr {
    /// Whether the node is deprecated
    is_deprecated: bool,
    /// Optional deprecation note
    note: Option<String>,
}

/// Parsed attributes for a node.
#[derive(Clone, Default)]
struct NodeAttrs {
    /// User-defined metadata attributes (#[key = value])
    meta: Vec<MetaAttr>,
    /// Deprecation attribute (#[deprecated] or #[deprecated(note = "...")])
    deprecation: DeprecationAttr,
    /// Redirect target path (#[redirect = "Path.To.Target"])
    redirect_to: Option<String>,
}

struct Node {
    name: Ident,
    /// All parsed attributes
    attrs: NodeAttrs,
    /// Optional: Node<DataType>
    data_type: Option<Type>,
    children: Vec<Node>,
}

struct NamespaceInput {
    vis: Visibility,
    root: Ident,
    nodes: Vec<Node>,
}

impl Parse for NamespaceInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let vis: Visibility = input.parse()?;
        input.parse::<Token![mod]>()?;
        let root: Ident = input.parse()?;
        let content;
        braced!(content in input);
        let nodes = parse_nodes(&content)?;
        Ok(Self { vis, root, nodes })
    }
}

fn parse_nodes(input: ParseStream) -> Result<Vec<Node>> {
    let mut nodes = Vec::new();
    while !input.is_empty() {
        // Parse attributes
        let attrs = parse_all_attrs(input)?;

        // Parse node name
        let name: Ident = input.parse()?;

        // Parse optional type parameter: Node<Type>
        let data_type = if input.peek(Token![<]) {
            input.parse::<Token![<]>()?;
            let ty: Type = input.parse()?;
            input.parse::<Token![>]>()?;
            Some(ty)
        } else {
            None
        };

        // Parse children or semicolon
        if input.peek(token::Brace) {
            let content;
            braced!(content in input);
            let children = parse_nodes(&content)?;
            nodes.push(Node {
                name,
                attrs,
                data_type,
                children,
            });
        } else {
            input.parse::<Token![;]>()?;
            nodes.push(Node {
                name,
                attrs,
                data_type,
                children: Vec::new(),
            });
        }
    }
    Ok(nodes)
}

/// Parse all attributes into NodeAttrs.
///
/// Handles:
/// - `#[deprecated]` or `#[deprecated(note = "...")]`
/// - `#[redirect = "Path.To.Target"]`
/// - `#[key = value]` (metadata)
fn parse_all_attrs(input: ParseStream) -> Result<NodeAttrs> {
    let mut result = NodeAttrs::default();

    while input.peek(Token![#]) {
        input.parse::<Token![#]>()?;
        let content;
        syn::bracketed!(content in input);

        let key: Ident = content.parse()?;

        if key == "deprecated" {
            result.deprecation.is_deprecated = true;

            // Check for (note = "...")
            if content.peek(syn::token::Paren) {
                let inner;
                syn::parenthesized!(inner in content);

                // Parse note = "..."
                if !inner.is_empty() {
                    let note_key: Ident = inner.parse()?;
                    if note_key == "note" {
                        inner.parse::<Token![=]>()?;
                        let note_value: syn::LitStr = inner.parse()?;
                        result.deprecation.note = Some(note_value.value());
                    }
                }
            }
        } else if key == "redirect" {
            // #[redirect = "Path.To.Target"]
            content.parse::<Token![=]>()?;
            let target: syn::LitStr = content.parse()?;
            result.redirect_to = Some(target.value());
        } else {
            // Regular metadata attribute: #[key = value]
            content.parse::<Token![=]>()?;
            let value: Expr = content.parse()?;
            result.meta.push(MetaAttr { key, value });
        }
    }

    Ok(result)
}

// =============================================================================
// Tree analysis (runs at macro expansion time)
// =============================================================================

/// Flattened node with computed metadata.
struct FlatNode {
    /// Path segments: ["Movement", "Idle"]
    segments: Vec<String>,
    /// Depth: 0 for roots, 1 for children, etc.
    depth: u8,
}

/// Flatten the parsed tree into a list with depth/path info.
/// Skips redirect nodes (they don't have their own GID).
fn flatten_nodes(nodes: &[Node], prefix: &str, depth: u8, out: &mut Vec<FlatNode>) {
    for node in nodes {
        // Skip redirect nodes - they use target's GID
        if node.attrs.redirect_to.is_some() {
            continue;
        }

        let path = if prefix.is_empty() {
            node.name.to_string()
        } else {
            format!("{}.{}", prefix, node.name)
        };

        let segments: Vec<String> = path.split('.').map(String::from).collect();

        out.push(FlatNode { segments, depth });

        flatten_nodes(&node.children, &path, depth + 1, out);
    }
}

// =============================================================================
// Crate path resolution
// =============================================================================

fn namespace_crate_path() -> TokenStream2 {
    match crate_name("bevy-tag") {
        Ok(FoundCrate::Itself) => {
            quote!(::bevy_tag)
        }
        Ok(FoundCrate::Name(name)) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!(::#ident)
        }
        Err(_) => quote!(::bevy_tag),
    }
}

// =============================================================================
// Code generation
// =============================================================================

/// Generate tag struct and its implementations.
fn generate_tag_impl(
    node_ident: &Ident,
    path_lit: &syn::LitStr,
    depth_lit: u8,
    seg_count: usize,
    seg_lits: &[syn::LitByteStr],
    metadata: &TokenStream2,
    ns_crate: &TokenStream2,
) -> TokenStream2 {
    quote! {
        /// Zero-sized tag type for this namespace node.
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub struct #node_ident;

        impl #node_ident {
            /// Full dot-separated path.
            pub const PATH: &'static str = #path_lit;

            /// Depth in the namespace tree (0 = top-level).
            pub const DEPTH: u8 = #depth_lit;

            /// Stable hierarchical GID, computed at compile time.
            pub const GID: #ns_crate::GID = {
                const SEGS: [&[u8]; #seg_count] = [#(#seg_lits),*];
                #ns_crate::hierarchical_gid(&SEGS)
            };

            /// Get the GID (convenience method).
            #[inline]
            pub const fn gid() -> #ns_crate::GID {
                Self::GID
            }

            #metadata
        }

        impl core::fmt::Display for #node_ident {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(Self::PATH)
            }
        }

        impl core::fmt::LowerHex for #node_ident {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::LowerHex::fmt(&Self::GID, f)
            }
        }

        impl core::fmt::UpperHex for #node_ident {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::UpperHex::fmt(&Self::GID, f)
            }
        }

        impl #ns_crate::NamespaceTag for #node_ident {
            const PATH: &'static str = #path_lit;
            const DEPTH: u8 = #depth_lit;
            const GID: #ns_crate::GID = #node_ident::GID;
        }
    }
}

/// Convert CamelCase to snake_case.
fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }
    result
}

/// Recursively generate tag types using CamelCase struct + snake_case module pattern.
///
/// Strategy:
/// - All nodes get a CamelCase struct (the tag type)
/// - Branch nodes also get a snake_case module containing re-exports of children
///
/// Example:
/// ```ignore
/// namespace! {
///     pub mod Tags {
///         Movement {       // Branch: has children
///             Idle;        // Leaf
///             Running;     // Leaf
///         }
///         Simple;          // Leaf at root
///     }
/// }
///
/// // Generates:
/// pub mod Tags {
///     pub struct Movement;  // CamelCase struct
///     impl Movement { pub const PATH, GID, ... }
///
///     pub mod movement {    // snake_case module for children
///         pub use super::Idle;
///         pub use super::Running;
///     }
///
///     pub struct Idle;
///     impl Idle { pub const PATH = "Movement.Idle", ... }
///
///     pub struct Running;
///     impl Running { ... }
///
///     pub struct Simple;
///     impl Simple { ... }
/// }
///
/// // Usage:
/// Tags::Movement              // The Movement tag (type)
/// Tags::Movement::GID         // Movement's GID
/// Tags::movement::Idle        // Child via snake_case module
/// Tags::movement::Idle::GID   // Child's GID
/// Tags::Simple                // Leaf at root
/// ```
/// Convert a dot-separated path to a Rust type path relative to the module root.
///
/// Example: "Equipment.Weapon.Blade" -> equipment::weapon::Blade
fn path_to_rust_type_path(path: &str) -> TokenStream2 {
    let segments: Vec<&str> = path.split('.').collect();
    if segments.len() == 1 {
        // Root level: just the type name
        let ident = Ident::new(segments[0], Span::call_site());
        quote! { #ident }
    } else {
        // Nested: snake_case modules + CamelCase type
        let modules: Vec<Ident> = segments[..segments.len() - 1]
            .iter()
            .map(|s| Ident::new(&to_snake_case(s), Span::call_site()))
            .collect();
        let type_name = Ident::new(segments.last().unwrap(), Span::call_site());
        quote! { #(#modules::)* #type_name }
    }
}

fn generate_tags_recursive(
    nodes: &[Node],
    prefix: &str,
    depth: u8,
    ns_crate: &TokenStream2,
) -> Vec<TokenStream2> {
    if depth as usize >= MAX_DEPTH {
        panic!(
            "namespace tree depth ({}) exceeds maximum ({})",
            depth, MAX_DEPTH
        );
    }

    let mut output = Vec::new();

    for node in nodes {
        let node_ident = &node.name;
        let path = if prefix.is_empty() {
            node.name.to_string()
        } else {
            format!("{}.{}", prefix, node.name)
        };

        // Generate deprecation attribute if present
        let deprecation_attr = if node.attrs.deprecation.is_deprecated {
            if let Some(ref note) = node.attrs.deprecation.note {
                let note_lit = syn::LitStr::new(note, Span::call_site());
                quote! { #[deprecated(note = #note_lit)] }
            } else {
                quote! { #[deprecated] }
            }
        } else {
            quote! {}
        };

        // Check if this node is a redirect
        if let Some(ref target_path) = node.attrs.redirect_to {
            // Generate type alias: pub type OldName = Redirect<target::path::Type>;
            let target_type = path_to_rust_type_path(target_path);

            // Add deprecation note about redirect if not already deprecated
            let redirect_deprecation = if node.attrs.deprecation.is_deprecated {
                deprecation_attr.clone()
            } else {
                let note = format!("redirected to {}", target_path);
                let note_lit = syn::LitStr::new(&note, Span::call_site());
                quote! { #[deprecated(note = #note_lit)] }
            };

            output.push(quote! {
                #redirect_deprecation
                pub type #node_ident = #ns_crate::Redirect<#target_type>;
            });

            // Redirects cannot have children
            if !node.children.is_empty() {
                panic!(
                    "Node '{}' has #[redirect] but also has children. Redirects must be leaf nodes.",
                    path
                );
            }

            continue;
        }

        // Regular node generation (not a redirect)
        let path_lit = syn::LitStr::new(&path, Span::call_site());

        // Build segments as byte string literals for const fn call
        let segments: Vec<&str> = path.split('.').collect();
        let seg_count = segments.len();
        let seg_lits: Vec<syn::LitByteStr> = segments
            .iter()
            .map(|s| syn::LitByteStr::new(s.as_bytes(), Span::call_site()))
            .collect();

        let depth_lit = depth;

        // Generate metadata constants from attributes
        let metadata = generate_metadata_consts(&node.attrs.meta);

        // Generate data type association if present
        let data_type_def = if let Some(ref ty) = node.data_type {
            quote! {
                impl #ns_crate::HasData for #node_ident {
                    type Data = #ty;
                }
            }
        } else {
            quote! {}
        };

        // Generate tag implementation
        let tag_impl = generate_tag_impl(
            node_ident,
            &path_lit,
            depth_lit,
            seg_count,
            &seg_lits,
            &metadata,
            ns_crate,
        );

        // Generate children recursively (they are flat siblings)
        let children_output = if !node.children.is_empty() {
            generate_tags_recursive(&node.children, &path, depth + 1, ns_crate)
        } else {
            Vec::new()
        };

        // Generate snake_case module with re-exports for branch nodes
        let child_module = if !node.children.is_empty() {
            let snake_name = to_snake_case(&node.name.to_string());
            let mod_ident = Ident::new(&snake_name, node.name.span());

            // Collect all descendant names for re-export (direct children only)
            let reexports: Vec<TokenStream2> = node
                .children
                .iter()
                .map(|child| {
                    let child_ident = &child.name;
                    let child_deprecation = if child.attrs.deprecation.is_deprecated {
                        if let Some(ref note) = child.attrs.deprecation.note {
                            let note_lit = syn::LitStr::new(note, Span::call_site());
                            quote! { #[deprecated(note = #note_lit)] }
                        } else {
                            quote! { #[deprecated] }
                        }
                    } else if child.attrs.redirect_to.is_some() {
                        // Redirects are implicitly deprecated
                        quote! { #[deprecated] }
                    } else {
                        quote! {}
                    };

                    // If child has children, also re-export its snake_case module
                    if !child.children.is_empty() {
                        let child_snake = to_snake_case(&child.name.to_string());
                        let child_mod_ident = Ident::new(&child_snake, child.name.span());
                        quote! {
                            #child_deprecation
                            pub use super::#child_ident;
                            #child_deprecation
                            pub use super::#child_mod_ident;
                        }
                    } else {
                        quote! {
                            #child_deprecation
                            pub use super::#child_ident;
                        }
                    }
                })
                .collect();

            quote! {
                #deprecation_attr
                #[allow(non_camel_case_types)]
                pub mod #mod_ident {
                    #(#reexports)*
                }
            }
        } else {
            quote! {}
        };

        // Output: first all children (flat), then this node, then child module
        output.extend(children_output);
        output.push(quote! {
            #deprecation_attr
            #tag_impl
            #data_type_def
        });
        if !node.children.is_empty() {
            output.push(child_module);
        }
    }

    output
}

/// Generate const fields from metadata attributes.
fn generate_metadata_consts(attrs: &[MetaAttr]) -> TokenStream2 {
    let consts: Vec<TokenStream2> = attrs
        .iter()
        .map(|attr| {
            let key = &attr.key;
            let value = &attr.value;

            // Convert ident to SCREAMING_SNAKE_CASE for const name
            let const_name = Ident::new(&key.to_string().to_uppercase(), key.span());

            // Try to infer type from expression
            let ty = infer_type_from_expr(value);

            quote! {
                #[doc = concat!("Metadata: ", stringify!(#key))]
                pub const #const_name: #ty = #value;
            }
        })
        .collect();

    quote! { #(#consts)* }
}

/// Infer Rust type from expression (best-effort).
fn infer_type_from_expr(expr: &Expr) -> TokenStream2 {
    match expr {
        Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Int(_) => quote!(i32),
            syn::Lit::Float(_) => quote!(f64),
            syn::Lit::Bool(_) => quote!(bool),
            syn::Lit::Str(_) => quote!(&'static str),
            syn::Lit::Char(_) => quote!(char),
            _ => quote!(i32), // fallback
        },
        _ => quote!(i32), // fallback for complex expressions
    }
}

/// Generate `NamespaceDef` entries.
/// Skips redirect nodes (they don't have their own definition).
fn collect_defs(
    nodes: &[Node],
    prefix: &str,
    parent: Option<&str>,
    ns_crate: &TokenStream2,
    out: &mut Vec<TokenStream2>,
) {
    for node in nodes {
        // Skip redirect nodes - they point to another definition
        if node.attrs.redirect_to.is_some() {
            continue;
        }

        let path = if prefix.is_empty() {
            node.name.to_string()
        } else {
            format!("{}.{}", prefix, node.name)
        };
        let path_lit = syn::LitStr::new(&path, Span::call_site());

        let parent_tokens = match parent {
            Some(pp) => {
                let parent_lit = syn::LitStr::new(pp, Span::call_site());
                quote!(Some(#parent_lit))
            }
            None => quote!(None),
        };

        out.push(quote! {
            #ns_crate::NamespaceDef {
                path: #path_lit,
                parent: #parent_tokens,
            },
        });

        collect_defs(&node.children, &path, Some(&path), ns_crate, out);
    }
}

/// Generate compile-time collision detection with detailed error messages.
fn generate_collision_check(flat: &[FlatNode], ns_crate: &TokenStream2) -> TokenStream2 {
    // Generate individual collision checks for each pair with specific error messages
    let mut checks = Vec::new();

    for i in 0..flat.len() {
        for j in (i + 1)..flat.len() {
            let path_i = flat[i].segments.join(".");
            let path_j = flat[j].segments.join(".");

            let seg_count_i = flat[i].segments.len();
            let seg_lits_i: Vec<syn::LitByteStr> = flat[i]
                .segments
                .iter()
                .map(|s| syn::LitByteStr::new(s.as_bytes(), Span::call_site()))
                .collect();

            let seg_count_j = flat[j].segments.len();
            let seg_lits_j: Vec<syn::LitByteStr> = flat[j]
                .segments
                .iter()
                .map(|s| syn::LitByteStr::new(s.as_bytes(), Span::call_site()))
                .collect();

            let error_msg = format!(
                "GID collision detected: '{}' and '{}' hash to the same value",
                path_i, path_j
            );

            checks.push(quote! {
                const _: () = {
                    const GID_A: #ns_crate::GID = {
                        const SEGS: [&[u8]; #seg_count_i] = [#(#seg_lits_i),*];
                        #ns_crate::hierarchical_gid(&SEGS)
                    };
                    const GID_B: #ns_crate::GID = {
                        const SEGS: [&[u8]; #seg_count_j] = [#(#seg_lits_j),*];
                        #ns_crate::hierarchical_gid(&SEGS)
                    };
                    assert!(GID_A != GID_B, #error_msg);
                };
            });
        }
    }

    quote! {
        #(#checks)*
    }
}

// =============================================================================
// Entry point
// =============================================================================

#[proc_macro]
pub fn namespace(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as NamespaceInput);
    let ns_crate = namespace_crate_path();

    // 1. Flatten tree and analyze shape
    let mut flat = Vec::new();
    flatten_nodes(&input.nodes, "", 0, &mut flat);

    // Validate depth
    let max_depth = flat.iter().map(|n| n.depth).max().unwrap_or(0);
    if max_depth as usize >= MAX_DEPTH {
        panic!(
            "namespace tree depth ({}) exceeds maximum ({})",
            max_depth + 1,
            MAX_DEPTH
        );
    }

    let tree_depth = (max_depth + 1) as usize;
    let node_count = flat.len();

    // 2. Generate tags
    let tags = generate_tags_recursive(&input.nodes, "", 0, &ns_crate);

    // 3. Generate NamespaceDef entries
    let mut defs = Vec::new();
    collect_defs(&input.nodes, "", None, &ns_crate, &mut defs);

    // 4. Generate collision detection
    let collision_check = generate_collision_check(&flat, &ns_crate);

    // 5. Assemble
    let vis = input.vis;
    let root = input.root;

    let expanded = quote! {
        #[allow(non_snake_case, non_camel_case_types)]
        #vis mod #root {
            /// Number of tree levels in this namespace.
            pub const TREE_DEPTH: usize = #tree_depth;

            /// Total number of namespace nodes.
            pub const NODE_COUNT: usize = #node_count;

            /// Flat NamespaceDef table (for runtime registry).
            pub const DEFINITIONS: &'static [#ns_crate::NamespaceDef] = &[
                #(#defs)*
            ];

            #collision_check

            #(#tags)*
        }
    };

    expanded.into()
}
