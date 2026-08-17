use std::fmt::Debug;
use std::marker::PhantomData;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use vulkanalia::vk;
use vulkanalia::vk::{ExtDescriptorBufferExtensionDeviceCommands, Handle, HasBuilder, KhrAccelerationStructureExtensionDeviceCommands};
use crate::{Allocation, CommandBuffer, DevicePointer, Gpu, HasIndexType, HazardFlags, Memory, MemoryAllocation, MemoryAllocator, Stage};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum VertexFormat {
    XYZ32Float = vk::Format::R32G32B32_SFLOAT.as_raw(),
}

fn build<'b, 'a: 'b>(
    gpu: &'a Gpu, cmd_buf: &mut CommandBuffer<'b>,
    geometries: &[vk::AccelerationStructureGeometryKHR], primitive_counts: &[u32],
    build_ranges: &[vk::AccelerationStructureBuildRangeInfoKHR],
    ty: vk::AccelerationStructureTypeKHR,
) -> Result<(vk::AccelerationStructureKHR, DevicePointer, Allocation<'a, u8>)> {
    assert!(!geometries.is_empty());

    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .type_(ty)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(geometries);

    let mut build_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe { gpu.device.get_acceleration_structure_build_sizes_khr(
        vk::AccelerationStructureBuildTypeKHR::DEVICE, &build_info, primitive_counts, &mut build_sizes) };

    let buffer = gpu.alloc_mem_aligned::<u8>(
        build_sizes.acceleration_structure_size as usize, 256, Memory::AccelerationStructureStorage)?;
    let create_info = vk::AccelerationStructureCreateInfoKHR::builder()
        .buffer(buffer.buffer)
        .size(build_sizes.acceleration_structure_size)
        .type_(ty);
    let structure = unsafe {
        gpu.device.create_acceleration_structure_khr(&create_info, None)? };
    let addr_info = vk::AccelerationStructureDeviceAddressInfoKHR::builder()
        .acceleration_structure(structure);
    let device_address = unsafe {
        gpu.device.get_acceleration_structure_device_address_khr(&addr_info) };
    let scratch = gpu.alloc_mem_aligned::<u8>(
        build_sizes.build_scratch_size as usize,
        gpu.acceleration_structure_info.properties
            .min_acceleration_structure_scratch_offset_alignment as usize,
        Memory::Gpu)?;

    let build_info = build_info
        .dst_acceleration_structure(structure)
        .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch.device().0 });

    unsafe { gpu.device.cmd_build_acceleration_structures_khr(
        cmd_buf.buffer, &[build_info], &[build_ranges]) };

    cmd_buf.give_ownership(scratch);

    Ok((structure, DevicePointer(device_address), buffer))
}

#[derive(Debug, Default)]
pub struct BottomLevelASBuilder<'a> {
    geometries: Vec<vk::AccelerationStructureGeometryKHR>,
    triangle_counts: Vec<u32>,
    build_ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
    buffers: Vec<Box<dyn Debug + 'a>>,
    pd: PhantomData<&'a Gpu>,
}

impl<'a> BottomLevelASBuilder<'a> {
    pub fn geometry<T: Pod + Debug, I: HasIndexType>(
        &mut self, vertices: Allocation<'a, T>, vertex_format: VertexFormat, indices: Allocation<'a, I>,
        transform: Allocation<'a, [[f32; 4]; 3]>, transform_index: u32,
    ) {
        let vert_count = u32::try_from(vertices.len()).expect("too many vertices");
        let idx_count = u32::try_from(indices.len()).expect("too many indices");
        assert_eq!(idx_count % 3, 0);
        let data = vk::AccelerationStructureGeometryDataKHR {
            triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                .vertex_format(vk::Format::from_raw(vertex_format as i32))
                .max_vertex(vert_count - 1)
                .vertex_stride(size_of::<T>() as vk::DeviceSize)
                .index_type(vk::IndexType::from_raw(I::INDEX_TYPE as i32))
                .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: vertices.device().0 })
                .index_data(vk::DeviceOrHostAddressConstKHR { device_address: indices.device().0 })
                .transform_data(vk::DeviceOrHostAddressConstKHR { device_address: transform.device().0 })
                .build()
        };
        let geom = vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(data)
            .build();
        let range = vk::AccelerationStructureBuildRangeInfoKHR::builder()
            .primitive_count(idx_count / 3)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(transform_index * size_of::<[[f32; 4]; 3]>() as u32)
            .build();
        self.geometries.push(geom);
        self.triangle_counts.push(idx_count / 3);
        self.build_ranges.push(range);

        self.buffers.push(Box::new(vertices));
        self.buffers.push(Box::new(indices));
        self.buffers.push(Box::new(transform));
    }

    pub fn build<'b>(self, gpu: &'a Gpu, cmd_buf: &mut CommandBuffer<'b>) -> Result<BottomLevelAS<'a>> where 'a: 'b {
        let (structure, device, buffer) = build(
            gpu, cmd_buf, &self.geometries, &self.triangle_counts, &self.build_ranges,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)?;

        let info = vk::AccelerationStructureDeviceAddressInfoKHR::builder()
            .acceleration_structure(structure);
        let device_address = unsafe { gpu.device.get_acceleration_structure_device_address_khr(&info) };

        cmd_buf.give_ownership(self.buffers);

        Ok(BottomLevelAS {
            gpu,
            structure,
            device,
            buffer,
            device_address,
        })
    }
}

#[derive(Debug)]
pub struct BottomLevelAS<'a> {
    gpu: &'a Gpu,
    structure: vk::AccelerationStructureKHR,
    device: DevicePointer,
    buffer: Allocation<'a, u8>,
    device_address: vk::DeviceAddress,
}

impl<'a> BottomLevelAS<'a> {
    pub fn builder() -> BottomLevelASBuilder<'a> {
        Default::default()
    }
}

impl<'a> Drop for BottomLevelAS<'a> {
    fn drop(&mut self) {
        unsafe { self.gpu.device.destroy_acceleration_structure_khr(self.structure, None) };
    }
}

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq)]
struct AccelerationStructureInstanceWrapper(pub vk::AccelerationStructureInstanceKHR);
unsafe impl Pod for AccelerationStructureInstanceWrapper {}
unsafe impl Zeroable for AccelerationStructureInstanceWrapper {}

#[derive(Debug, Default)]
pub struct TopLevelASBuilder<'a> {
    instances: Vec<AccelerationStructureInstanceWrapper>,
    pd: PhantomData<&'a Gpu>,
}

impl<'a> TopLevelASBuilder<'a> {
    pub fn add(&mut self, bottom_level_as: &BottomLevelAS<'a>, transform: &[[f32; 4]; 3]) {
        let instance = vk::AccelerationStructureInstanceKHR::builder()
            .transform(vk::TransformMatrixKHR { matrix: *transform })
            .instance_custom_index::<u32>(0)
            .mask::<u32>(0xff)
            .instance_shader_binding_table_record_offset::<u32>(0)
            .flags::<u32>(vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.bits())
            .acceleration_structure_reference(bottom_level_as.structure.as_raw())
            // .acceleration_structure_reference(bottom_level_as.device_address)
            .build();
        self.instances.push(AccelerationStructureInstanceWrapper(instance));
    }

    pub fn build<'b>(self, gpu: &'a Gpu, cmd_buf: &mut CommandBuffer<'b>) -> Result<TopLevelAS<'a>> where 'a: 'b {
        cmd_buf.barrier(
            Stage::AccelerationStructureBuild, Stage::AccelerationStructureBuild,
            HazardFlags::AccelerationStructure);

        let instance_buffer = gpu
            .allocator_mem(Memory::AccelerationStructureInput)
            .alloc_data(&self.instances)?;

        let data = vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                .array_of_pointers(false)
                .data(vk::DeviceOrHostAddressConstKHR { device_address: instance_buffer.device().0 })
                .build()
        };
        let geometry = vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(data)
            .build();
        let primitive_count = self.instances.len() as u32;
        let range = vk::AccelerationStructureBuildRangeInfoKHR::builder()
            .primitive_count(primitive_count)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0)
            .build();

        let (structure, device, buffer) = build(
            gpu, cmd_buf, &[geometry], &[primitive_count], &[range],
            vk::AccelerationStructureTypeKHR::TOP_LEVEL)?;

        cmd_buf.give_ownership(instance_buffer);

        Ok(TopLevelAS {
            gpu,
            structure,
            device,
            buffer,
        })
    }
}

#[derive(Debug)]
pub struct TopLevelAS<'a> {
    gpu: &'a Gpu,
    structure: vk::AccelerationStructureKHR,
    device: DevicePointer,
    buffer: Allocation<'a, u8>,
}

impl<'a> TopLevelAS<'a> {
    pub fn builder() -> TopLevelASBuilder<'a> {
        Default::default()
    }

    pub fn descriptor(&self, descriptor: &mut [u8]) {
        assert_eq!(descriptor.len(), self.gpu.descriptor_sizes.accel_struct,
                   "incorrect buffer size for acceleration structure descriptor");

        let info = vk::DescriptorGetInfoEXT::builder()
            .type_(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .data(vk::DescriptorDataEXT {
                acceleration_structure: self.device.0,
            });
        unsafe { self.gpu.device.get_descriptor_ext(&info, descriptor) };
    }
}

impl<'a> Drop for TopLevelAS<'a> {
    fn drop(&mut self) {
        unsafe { self.gpu.device.destroy_acceleration_structure_khr(self.structure, None) };
    }
}
