use quote::{quote, format_ident};
use syn::{parse_macro_input, DeriveInput, Data, Ident, Token, Visibility};
use syn::parse::{Parse, ParseStream};

#[proc_macro_attribute]
pub fn multi_allocation(_attr: proc_macro::TokenStream, input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let DeriveInput { vis, ident, generics, data, .. } = parse_macro_input!(input as DeriveInput);
    let data = match data {
        Data::Struct(data) => data,
        _ => panic!("#[derive(MultiType)] only supports structs"),
    };
    if !generics.params.is_empty() {
        panic!("#[derive(MultiType)] only does not support generics");
    }

    let item_count = data.fields.len();

    assert!(item_count <= 12, "#[derive(MultiType)] only supports up to 12 fields");

    let item_types = data.fields.iter().map(|f| f.ty.clone()).collect::<Vec<_>>();
    let item_names = data.fields.iter().map(|f| f.ident.clone()
        .ok_or_else(|| panic!("#[derive(MultiType)] struct field missing identifier")).unwrap()).collect::<Vec<_>>();
    let item_indices = 0..item_names.len();
    let count_struct_ident = format_ident!("{ident}Counts");

    quote! {
        #vis struct #ident<'a> {
            __allocation: gpu::Allocation<'a, u8>,
            #(pub #item_names: gpu::MultiAllocationPart<'a, #item_types>,)*
        }

        #[derive(Default, Copy, Clone, Debug)]
        #vis struct #count_struct_ident {
            #(pub #item_names: usize,)*
        }

        impl From<#count_struct_ident> for [usize; #item_count] {
            fn from(value: #count_struct_ident) -> Self {
                [#(value.#item_names,)*]
            }
        }

        impl<'a> #ident<'a> {
            #vis fn new(gpu: &'a gpu::Gpu, lengths: impl Into<[usize; #item_count]>) -> anyhow::Result<Self> {
                Self::new_mem(gpu, lengths, gpu::Memory::Default)
            }
            #vis fn new_mem(gpu: &'a gpu::Gpu, lengths: impl Into<[usize; #item_count]>, memory: gpu::Memory) -> anyhow::Result<Self> {
                let __allocation = gpu::MultiAllocation::<(#(#item_types,)*)>::new_mem(gpu, lengths.into(), memory)?;
                #(
                    let #item_names = {
                        let (device, host, count) = __allocation.part::<#item_indices, #item_types>().to_raw_parts();
                        unsafe { gpu::MultiAllocationPart::from_raw_parts(device, host, count) }
                    };
                )*
                Ok(Self {
                    __allocation: __allocation.into_inner(),
                    #(#item_names,)*
                })
            }
        }
    }.into()
}

struct WinitToImguiKeyInput {
    vis: Visibility,
    ident: Ident,
    custom_pairs: Vec<(Ident, Ident)>,
}

impl Parse for WinitToImguiKeyInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let vis = input.parse().unwrap_or(Visibility::Inherited);
        let ident = input.parse()?;
        input.parse::<Token![;]>()?;
        let custom_pairs = input.parse_terminated(|input| {
            let dst = input.parse::<Ident>()?;
            if let Ok(_) = input.parse::<Token![=]>() {
                let src = input.parse()?;
                Ok((dst, src))
            } else {
                Ok((dst.clone(), dst))
            }
        }, Token![,])?.into_iter().collect();

        Ok(Self {
            vis,
            ident,
            custom_pairs,
        })
    }
}

#[proc_macro]
pub fn winit_to_imgui_key_fn(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let WinitToImguiKeyInput {
        vis,
        ident,
        custom_pairs,
    } = parse_macro_input!(input as WinitToImguiKeyInput);

    let src_enum = quote!(winit::keyboard::KeyCode);
    let dst_enum = quote!(dear_imgui_rs::Key);

    let mut dsts = vec![];
    let mut srcs = vec![];

    for (dst, src) in custom_pairs {
        dsts.push(dst);
        srcs.push(src);
    }

    for n in 0..10 {
        dsts.push(format_ident!("Key{n}"));
        srcs.push(format_ident!("Digit{n}"));

        dsts.push(format_ident!("Keypad{n}"));
        srcs.push(format_ident!("Numpad{n}"));
    }

    for c in 'A'..='Z' {
        dsts.push(format_ident!("{c}"));
        srcs.push(format_ident!("Key{c}"));
    }

    for n in 1..=24 {
        dsts.push(format_ident!("F{n}"));
        srcs.push(format_ident!("F{n}"));
    }

    quote! {
        #vis fn #ident(key: #src_enum) -> Option<#dst_enum> {
            Some(match key {
                #(#src_enum::#srcs => #dst_enum::#dsts,)*
                _ => return None,
            })
        }
    }.into()
}
