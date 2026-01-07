use std::cell::Cell;
use std::marker::PhantomData;
use bytemuck::Pod;
use anyhow::{anyhow, Result};
use crate::{Allocation, DevicePointer, MemoryAllocation, MemoryAllocator};

pub struct Arena<'a> {
    pub(crate) allocation: Allocation<'a, u8>,
    pub(crate) offset: Cell<usize>,
}

impl<'a> Arena<'a> {
    pub fn reset(&self) {
        self.offset.set(0);
    }
}

impl<'a> MemoryAllocator for Arena<'a> {
    type Allocation<T: Pod> = ArenaAllocation<'a, T>;

    fn alloc_aligned<T: Pod>(&self, n: usize, align: usize) -> Result<Self::Allocation<T>> {
        let align = align.max(align_of::<T>());
        assert!(align.is_power_of_two(), "alignment must be a power of two");
        self.offset.set((self.offset.get() + align - 1) & !(align - 1));

        let size = n * size_of::<T>();
        let offset = self.offset.get();
        let new_offset = offset + size;
        if new_offset > self.allocation.count {
            return Err(anyhow!("arena out of memory"));
        }
        self.offset.set(new_offset);

        Ok(ArenaAllocation {
            host: unsafe { self.allocation.host_raw().add(offset) } as *mut _,
            device: self.allocation.device().add(offset),
            count: n,
            pd: PhantomData,
        })
    }
}

pub struct ArenaAllocation<'a, T: Pod> {
    host: *mut T,
    device: DevicePointer,
    count: usize,
    pd: PhantomData<&'a Arena<'a>>,
}

impl<'a, T: Pod> MemoryAllocation for ArenaAllocation<'a, T> {
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
