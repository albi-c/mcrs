use quote::{quote, format_ident, ToTokens, TokenStreamExt};
use proc_macro2::TokenStream;
use syn::{parse_macro_input, DeriveInput, Data, Type, Ident};

struct NameAllocationPartPair<'a>(&'a Ident, &'a Type);

impl<'a> ToTokens for NameAllocationPartPair<'a> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let Self(ident, ty) = self;
        tokens.append_all(quote! {
            pub #ident: gpu::MultiAllocationPart<'a, #ty>
        });
    }
}

struct GenerateMultiAllocation<'a>(&'a Ident, &'a Type, usize);

impl<'a> ToTokens for GenerateMultiAllocation<'a> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let Self(ident, ty, index) = self;
        tokens.append_all(quote! {
            let #ident = {
                let (device, host, count) = __allocation.part::<#index, #ty>().to_raw_parts();
                unsafe { gpu::MultiAllocationPart::from_raw_parts(device, host, count) }
            };
        });
    }
}

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
    let pairs = item_names.iter().zip(item_types.iter())
        .map(|(n, t)| NameAllocationPartPair(n, t)).collect::<Vec<_>>();
    let generate_ma = pairs.iter().enumerate()
        .map(|(i, p)| GenerateMultiAllocation(p.0, p.1, i));
    let count_struct_ident = format_ident!("{ident}Counts");

    quote! {
        #vis struct #ident<'a> {
            __allocation: gpu::Allocation<'a, u8>,
            #(#pairs,)*
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
                let __allocation = gpu::MultiAllocation::<(#(#item_types,)*)>::new(gpu, lengths.into())?;
                #(#generate_ma)*
                Ok(Self {
                    __allocation: __allocation.into_inner(),
                    #(#item_names,)*
                })
            }
        }
    }.into()
}
