use std::alloc::Layout;
use std::any::TypeId;
use std::intrinsics::type_id_eq;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use bytemuck::Pod;
use crate::{Allocation, DevicePointer, Gpu, Memory, MemoryAllocation};

// TODO: derive macro, instead of Tuples (associated type) have structs
pub trait MultiType {
    const N: usize;
    const SIZES: &'static [usize];
    const ALIGNS: &'static [usize];
    const TYPES: &'static [TypeId];

    type Array<T: Default>: Default + Index<usize, Output = T> + IndexMut<usize, Output = T>;
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

impl<A: Pod, B: Pod> MultiType for (A, B) {
    const N: usize = 2;
    const SIZES: &'static [usize] = &[size_of::<A>(), size_of::<B>()];
    const ALIGNS: &'static [usize] = &[align_of::<A>(), align_of::<B>()];
    const TYPES: &'static [TypeId] = &[TypeId::of::<A>(), TypeId::of::<B>()];

    type Array<T: Default> = [T; 2];
    type DevicePtrs = [DevicePointer; 2];
    type UntypedPtrs = [*mut u8; 2];
    type Ptrs = (*mut A, *mut B);
    type Slices<'a> = (&'a [A], &'a [B]);
    type SlicesMut<'a> = (&'a mut [A], &'a mut [B]);

    fn get_device_ptrs(a: &MultiAllocation<'_, Self>) -> Self::DevicePtrs {
        [a.device::<0>(), a.device::<1>()]
    }
    fn get_untyped_ptrs(a: &MultiAllocation<'_, Self>) -> Self::UntypedPtrs {
        [a.host_raw_untyped::<0>(), a.host_raw_untyped::<1>()]
    }
    fn get_ptrs(a: &MultiAllocation<'_, Self>) -> Self::Ptrs {
        (a.host_raw::<0, A>(), a.host_raw::<1, B>())
    }
    fn get_slices<'a>(a: &'a MultiAllocation<'_, Self>) -> Self::Slices<'a> {
        (a.host::<0, A>(), a.host::<1, B>())
    }
    fn get_slices_mut<'a>(a: &'a mut MultiAllocation<'_, Self>) -> Self::SlicesMut<'a> {
        unsafe { (a.host_mut_disjoint_unchecked::<0, A>(), a.host_mut_disjoint_unchecked::<1, B>()) }
    }
}

pub struct MultiAllocation<'a, T: ?Sized + MultiType> {
    allocation: Allocation<'a, u8>,
    offsets_counts: T::Array<(usize, usize)>,
    pd: PhantomData<T>,
}

impl<'a, T: MultiType> MultiAllocation<'a, T> {
    pub fn new_mem(gpu: &'a Gpu, counts: &[usize], memory: Memory) -> anyhow::Result<Self> {
        const {
            assert!(T::N == T::SIZES.len());
            assert!(T::N == T::ALIGNS.len());
            assert!(T::N == T::TYPES.len());
        }
        assert_eq!(counts.len(), T::N, "number of element counts must match type count");

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

    pub fn new(gpu: &'a Gpu, counts: &[usize]) -> anyhow::Result<Self> {
        Self::new_mem(gpu, counts, Memory::Default)
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
            unsafe { assert!(type_id_eq(TypeId::of::<U>(), T::TYPES[I])); }
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

    pub fn device_n(&self) -> T::DevicePtrs {
        T::get_device_ptrs(self)
    }
    pub fn host_raw_untyped_n(&self) -> T::UntypedPtrs {
        T::get_untyped_ptrs(self)
    }
    pub fn host_raw_n(&self) -> T::Ptrs {
        T::get_ptrs(self)
    }
    pub fn host_n(&self) -> T::Slices<'_> {
        T::get_slices(self)
    }
    pub fn host_mut_n(&mut self) -> T::SlicesMut<'_> {
        T::get_slices_mut(self)
    }
}
