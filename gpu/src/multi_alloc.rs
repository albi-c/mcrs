use std::alloc::Layout;
use std::any::TypeId;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use bytemuck::Pod;
use crate::{Allocation, DevicePointer, Gpu, Memory, MemoryAllocation};

// TODO: derive macro, instead of Tuples (associated type) have structs
pub trait MultiType {
    const N: usize;
    const SIZES: Self::Array<usize>;
    const ALIGNS: Self::Array<usize>;
    const TYPES: Self::ArrayNoDefault<TypeId>;

    type Array<T: Default>: Default + Index<usize, Output = T> + IndexMut<usize, Output = T>;
    type ArrayNoDefault<T: Copy>: Copy + const Index<usize, Output = T>;
    type DevicePtrs;
    type UntypedPtrs;
    type Ptrs;
    type Slices<'a> where Self: 'a;
    type SlicesMut<'a> where Self: 'a;

    fn get_device_ptrs(a: &MultiAllocation<'_, Self>) -> Self::DevicePtrs;
    fn get_untyped_ptrs(a: &MultiAllocation<'_, Self>) -> Self::UntypedPtrs;
    fn get_ptrs(a: &MultiAllocation<'_, Self>) -> Self::Ptrs;
    fn get_slices<'a>(a: &'a MultiAllocation<'_, Self>) -> Self::Slices<'a>;
    fn get_slices_mut<'a>(a: &'a mut MultiAllocation<'_, Self>) -> Self::SlicesMut<'a>;
}

macro_rules! multi_type_for_tuple {
    ($count:tt; $($n:tt, $t:ident),*) => {
        impl<$($t: bytemuck::Pod,)*> MultiType for ($($t,)*) {
            const N: usize = $count;
            const SIZES: Self::Array<usize> = [$(std::mem::size_of::<$t>(),)*];
            const ALIGNS: Self::Array<usize> = [$(std::mem::align_of::<$t>(),)*];
            const TYPES: Self::ArrayNoDefault<std::any::TypeId> = [$(std::any::TypeId::of::<$t>(),)*];

            type Array<T: Default> = [T; $count];
            type ArrayNoDefault<T: Copy> = [T; $count];
            type DevicePtrs = [DevicePointer; $count];
            type UntypedPtrs = [*mut u8; $count];
            type Ptrs = ($(*mut $t,)*);
            type Slices<'a> = ($(&'a [$t],)*);
            type SlicesMut<'a> = ($(&'a mut [$t],)*);

            fn get_device_ptrs(a: &MultiAllocation<'_, Self>) -> Self::DevicePtrs {
                [$(a.device::<$n>(),)*]
            }
            fn get_untyped_ptrs(a: &MultiAllocation<'_, Self>) -> Self::UntypedPtrs {
                [$(a.host_raw_untyped::<$n>(),)*]
            }
            fn get_ptrs(a: &MultiAllocation<'_, Self>) -> Self::Ptrs {
                ($(a.host_raw::<$n, $t>(),)*)
            }
            fn get_slices<'a>(a: &'a MultiAllocation<'_, Self>) -> Self::Slices<'a> {
                ($(a.host::<$n, $t>(),)*)
            }
            fn get_slices_mut<'a>(a: &'a mut MultiAllocation<'_, Self>) -> Self::SlicesMut<'a> {
                unsafe { ($(a.host_mut_disjoint_unchecked::<$n, $t>(),)*) }
            }
        }
    };
}

multi_type_for_tuple!( 1; 0, A);
multi_type_for_tuple!( 2; 0, A, 1, B);
multi_type_for_tuple!( 3; 0, A, 1, B, 2, C);
multi_type_for_tuple!( 4; 0, A, 1, B, 2, C, 3, D);
multi_type_for_tuple!( 5; 0, A, 1, B, 2, C, 3, D, 4, E);
multi_type_for_tuple!( 6; 0, A, 1, B, 2, C, 3, D, 4, E, 5, F);
multi_type_for_tuple!( 7; 0, A, 1, B, 2, C, 3, D, 4, E, 5, F, 6, G);
multi_type_for_tuple!( 8; 0, A, 1, B, 2, C, 3, D, 4, E, 5, F, 6, G, 7, H);
multi_type_for_tuple!( 9; 0, A, 1, B, 2, C, 3, D, 4, E, 5, F, 6, G, 7, H, 8, I);
multi_type_for_tuple!(10; 0, A, 1, B, 2, C, 3, D, 4, E, 5, F, 6, G, 7, H, 8, I, 9, J);
multi_type_for_tuple!(11; 0, A, 1, B, 2, C, 3, D, 4, E, 5, F, 6, G, 7, H, 8, I, 9, J, 10, K);
multi_type_for_tuple!(12; 0, A, 1, B, 2, C, 3, D, 4, E, 5, F, 6, G, 7, H, 8, I, 9, J, 10, K, 11, L);
multi_type_for_tuple!(13; 0, A, 1, B, 2, C, 3, D, 4, E, 5, F, 6, G, 7, H, 8, I, 9, J, 10, K, 11, L, 12, M);
multi_type_for_tuple!(14; 0, A, 1, B, 2, C, 3, D, 4, E, 5, F, 6, G, 7, H, 8, I, 9, J, 10, K, 11, L, 12, M, 13, N);
multi_type_for_tuple!(15; 0, A, 1, B, 2, C, 3, D, 4, E, 5, F, 6, G, 7, H, 8, I, 9, J, 10, K, 11, L, 12, M, 13, N, 14, O);
multi_type_for_tuple!(16; 0, A, 1, B, 2, C, 3, D, 4, E, 5, F, 6, G, 7, H, 8, I, 9, J, 10, K, 11, L, 12, M, 13, N, 14, O, 15, P);

struct MultiAllocationContainer<'a> {
    ma: Allocation<'a, u8>,
    pub part1: MultiAllocationPart<'a, usize>,
    pub part2: MultiAllocationPart<'a, usize>,
}

impl<'a> MultiAllocationContainer<'a> {
    pub fn new(&self, gpu: &'a Gpu, lengths: [usize; 2]) -> anyhow::Result<Self> {
        let ma = MultiAllocation::<(usize, usize)>::new(gpu, lengths)?;
        let part1 = {
            let MultiAllocationPart { device, host, count, .. } = ma.part::<0, usize>();
            MultiAllocationPart { device, host, count, pd: PhantomData }
        };
        let part2 = {
            let MultiAllocationPart { device, host, count, .. } = ma.part::<1, usize>();
            MultiAllocationPart { device, host, count, pd: PhantomData }
        };
        Ok(Self {
            ma: ma.allocation,
            part1,
            part2,
        })
    }
}

pub struct MultiAllocationPart<'a, T: Pod> {
    device: DevicePointer,
    host: *mut T,
    count: usize,
    pd: PhantomData<&'a T>,
}

impl<'a, T: Pod> MultiAllocationPart<'a, T> {
    pub unsafe fn from_raw_parts(device: DevicePointer, host: *mut T, count: usize) -> Self {
        Self {
            device,
            host,
            count,
            pd: PhantomData,
        }
    }

    pub fn to_raw_parts(self) -> (DevicePointer, *mut T, usize) {
        (self.device, self.host, self.count)
    }
}

impl<'a, T: Pod> MemoryAllocation for MultiAllocationPart<'a, T> {
    type Type = T;

    fn host(&self) -> &[Self::Type] {
        unsafe { std::slice::from_raw_parts(self.host, self.count) }
    }
    fn host_mut(&mut self) -> &mut [Self::Type] {
        unsafe { std::slice::from_raw_parts_mut(self.host, self.count) }
    }
    fn host_raw(&self) -> *mut Self::Type {
        self.host
    }

    fn len(&self) -> usize {
        self.count
    }

    fn device(&self) -> DevicePointer {
        self.device
    }
}

#[derive(Debug)]
pub struct MultiAllocation<'a, T: ?Sized + MultiType> {
    allocation: Allocation<'a, u8>,
    offsets_counts: T::Array<(usize, usize)>,
    pd: PhantomData<T>,
}

impl<'a, T: ?Sized + MultiType> MultiAllocation<'a, T> {
    pub type Types = T;

    pub type DevicePtrs = T::DevicePtrs;
    pub type UntypedPtrs = T::UntypedPtrs;
    pub type Ptrs = T::Ptrs;
    pub type Slices<'b> = T::Slices<'b> where T: 'b;
    pub type SlicesMut<'b> = T::SlicesMut<'b> where T: 'b;

    pub fn new_mem(gpu: &'a Gpu, counts: T::Array<usize>, memory: Memory) -> anyhow::Result<Self> {
        let mut layout = Layout::new::<()>();
        let mut offsets_counts = T::Array::<(usize, usize)>::default();

        for i in 0..T::N {
            let size = T::SIZES[i];
            let align = T::ALIGNS[i];
            let count = counts[i];
            let (l, stride) = Layout::from_size_align(size, align)?.repeat(count)?;
            assert_eq!(stride, size, "element stride must match its size to avoid padding");
            let (l, offset) = layout.extend(l)?;
            layout = l;
            offsets_counts[i] = (offset, count);
        }

        let allocation = gpu.alloc_mem_aligned::<u8>(layout.size(), layout.align(), memory)?;

        Ok(Self {
            allocation,
            offsets_counts,
            pd: PhantomData,
        })
    }

    pub fn new(gpu: &'a Gpu, counts: T::Array<usize>) -> anyhow::Result<Self> {
        Self::new_mem(gpu, counts, Memory::Default)
    }

    pub fn into_inner(self) -> Allocation<'a, u8> {
        self.allocation
    }

    pub fn device_i(&self, i: usize) -> DevicePointer {
        assert!(i < T::N);
        self.allocation.device().add(self.offsets_counts[i].0)
    }

    pub fn host_raw_untyped_i(&self, i: usize) -> *mut u8 {
        assert!(i < T::N);
        unsafe { self.allocation.host_raw().add(self.offsets_counts[i].0) }
    }

    fn check_i_u<U: Pod>(i: usize) {
        assert!(i < T::N);
        assert_eq!(TypeId::of::<U>(), T::TYPES[i]);
    }

    pub fn host_raw_i<U: Pod>(&self, i: usize) -> *mut U {
        Self::check_i_u::<U>(i);
        unsafe { self.allocation.host_raw().add(self.offsets_counts[i].0).cast() }
    }
    pub fn host_i<U: Pod>(&self, i: usize) -> &[U] {
        Self::check_i_u::<U>(i);
        let (offset, count) = self.offsets_counts[i];
        unsafe { std::slice::from_raw_parts(self.allocation.host_raw().add(offset).cast::<U>(), count) }
    }
    pub fn host_mut_i<U: Pod>(&mut self, i: usize) -> &mut [U] {
        Self::check_i_u::<U>(i);
        let (offset, count) = self.offsets_counts[i];
        unsafe { std::slice::from_raw_parts_mut(self.allocation.host_raw().add(offset).cast::<U>(), count) }
    }
    pub unsafe fn host_mut_disjoint_unchecked_i<U: Pod>(&self, i: usize) -> &mut [U] {
        Self::check_i_u::<U>(i);
        let (offset, count) = self.offsets_counts[i];
        unsafe { std::slice::from_raw_parts_mut(self.allocation.host_raw().add(offset).cast::<U>(), count) }
    }

    pub fn part_i<U: Pod>(&self, i: usize) -> MultiAllocationPart<'_, U> {
        let host = self.host_i::<U>(i);
        MultiAllocationPart {
            count: host.len(),
            host: host.as_ptr() as *mut _,
            device: self.device_i(i),
            pd: PhantomData,
        }
    }

    pub fn device<const I: usize>(&self) -> DevicePointer {
        const {
            assert!(I < T::N);
        }
        self.allocation.device().add(self.offsets_counts[I].0)
    }

    pub fn host_raw_untyped<const I: usize>(&self) -> *mut u8 {
        const {
            assert!(I < T::N);
        }
        unsafe { self.allocation.host_raw().add(self.offsets_counts[I].0) }
    }

    const fn check_const_i_u<const I: usize, U: Pod>() {
        const {
            assert!(I < T::N);
            // RustRover causes error even though the function is safe
            #[allow(unused_unsafe)]
            unsafe { assert!(TypeId::of::<U>().eq(&T::TYPES[I])); }
        }
    }

    pub fn host_raw<const I: usize, U: Pod>(&self) -> *mut U {
        const {
            Self::check_const_i_u::<I, U>();
        }
        unsafe { self.allocation.host_raw().add(self.offsets_counts[I].0).cast() }
    }
    pub fn host<const I: usize, U: Pod>(&self) -> &[U] {
        const {
            Self::check_const_i_u::<I, U>();
        }
        let (offset, count) = self.offsets_counts[I];
        unsafe { std::slice::from_raw_parts(self.allocation.host_raw().add(offset).cast::<U>(), count) }
    }
    pub fn host_mut<const I: usize, U: Pod>(&mut self) -> &mut [U] {
        const {
            Self::check_const_i_u::<I, U>();
        }
        let (offset, count) = self.offsets_counts[I];
        unsafe { std::slice::from_raw_parts_mut(self.allocation.host_raw().add(offset).cast::<U>(), count) }
    }
    pub unsafe fn host_mut_disjoint_unchecked<const I: usize, U: Pod>(&self) -> &mut [U] {
        const {
            Self::check_const_i_u::<I, U>();
        }
        let (offset, count) = self.offsets_counts[I];
        unsafe { std::slice::from_raw_parts_mut(self.allocation.host_raw().add(offset).cast::<U>(), count) }
    }

    pub fn part<const I: usize, U: Pod>(&self) -> MultiAllocationPart<'_, U> {
        let host = self.host::<I, U>();
        MultiAllocationPart {
            count: host.len(),
            host: host.as_ptr() as *mut _,
            device: self.device::<I>(),
            pd: PhantomData,
        }
    }

    pub fn device_n(&self) -> Self::DevicePtrs {
        T::get_device_ptrs(self)
    }
    pub fn host_raw_untyped_n(&self) -> Self::UntypedPtrs {
        T::get_untyped_ptrs(self)
    }
    pub fn host_raw_n(&self) -> Self::Ptrs {
        T::get_ptrs(self)
    }
    pub fn host_n(&self) -> Self::Slices<'_> {
        T::get_slices(self)
    }
    pub fn host_mut_n(&mut self) -> Self::SlicesMut<'_> {
        T::get_slices_mut(self)
    }
}
