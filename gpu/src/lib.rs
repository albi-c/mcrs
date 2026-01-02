#![feature(try_find)]
#![feature(linked_list_cursors)]

mod vulkan;

use std::alloc::Layout;
use std::cell::RefCell;
use anyhow::Result;
use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};
use smart_default::SmartDefault;
use vulkanalia::{vk, Device, Instance, Version};
use vulkanalia::vk::{DeviceV1_0, ExtShaderObjectExtensionDeviceCommands, Handle, HasBuilder, KhrDynamicRenderingExtensionDeviceCommands, KhrSurfaceExtensionInstanceCommands, KhrSwapchainExtensionDeviceCommands, KhrTimelineSemaphoreExtensionDeviceCommands};
use crate::vulkan::{create_logical_device, create_semaphore, create_shader, create_swapchain, find_suitable_device, CommandBufferPool, PooledCommandBuffer, QueueFamilies, Queues, Swapchain};

pub const VALIDATION_LAYER: vk::ExtensionName = vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

pub fn need_portability_ext(version: Version) -> bool {
    const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);
    cfg!(target_os = "macos") && version >= PORTABILITY_MACOS_VERSION
}

pub fn validation_enabled() -> bool {
    cfg!(debug_assertions)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Memory {
    Default,
    Gpu,
    Readback,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Cull {
    None,
    CCW,
    CW,
    All,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct DepthFlags(u32);
bitflags! {
    impl DepthFlags : u32 {
        const None = 0x0;
        const Read = 0x1;
        const Write = 0x2;
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Op {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StencilOp {
    Keep,
    Zero,
    Replace,
    Invert,
    IncrementClamp,
    DecrementClamp,
    IncrementWrap,
    DecrementWrap,
}

// #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
// pub enum Load {
//     Load,
//     Clear,
//     DontCare,
// }
//
// #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
// pub enum Store {
//     Store,
//     DontCare,
// }

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Blend {
    Add,
    Subtract,
    RevSubtract,
    Min,
    Max,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Factor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    ConstantColor,
    OneMinusConstantColor,
    ConstantAlpha,
    OneMinusConstantAlpha,
    SrcAlphaSaturate,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Topology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
    TriangleFan,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TextureType {
    Tex1D,
    Tex2D,
    Tex3D,
    Cube,
    Arr2D,
    CubeArr,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Format {
    None,
    RGBA8UNorm,
    Depth32Float,
    RG11B10Float,
    RGB10A2UNorm,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct UsageFlags(u32);
bitflags! {
    impl UsageFlags : u32 {
        const None = 0x0;
        const TransferSrc = 0x1;
        const TransferDst = 0x2;
        const Sampled = 0x4;
        const Storage = 0x8;
        const ColorAttachment = 0x10;
        const DepthStencilAttachment = 0x20;
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Stage {
    Transfer,
    Compute,
    RasterColorOut,
    PixelShader,
    VertexShader,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct HazardFlags(u32);
bitflags! {
    impl HazardFlags : u32 {
        const None = 0x0;
        const DrawArguments = 0x1;
        const Descriptors = 0x2;
        const DepthStencil = 0x4;
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Signal {
    AtomicSet,
    AtomicMax,
    AtomicOr,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum QueueType {
    Graphics,
    Compute,
    Transfer,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ColorComponents(u32);
bitflags! {
    impl ColorComponents : u32 {
        const None = 0x0;
        const R = 0x1;
        const G = 0x2;
        const B = 0x4;
        const A = 0x8;
        const All = 0xf;
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    Vertex,
    Pixel,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, SmartDefault)]
pub struct Stencil {
    #[default(Op::Always)]
    pub test: Op,
    #[default(StencilOp::Keep)]
    pub fail: StencilOp,
    #[default(StencilOp::Keep)]
    pub pass: StencilOp,
    #[default(StencilOp::Keep)]
    pub depth_fail: StencilOp,
    #[default = 0]
    pub reference: u8,
}

#[derive(Debug, Clone, PartialEq, SmartDefault)]
pub struct DepthStencilDesc {
    #[default(DepthFlags::None)]
    pub depth_mode: DepthFlags,
    #[default(Op::Always)]
    pub depth_test: Op,
    #[default = 0.0]
    pub depth_bias: f32,
    #[default = 0.0]
    pub depth_bias_slope_factor: f32,
    #[default = 0.0]
    pub depth_bias_clamp: f32,
    #[default = 0xff]
    pub stencil_read_mask: u8,
    #[default = 0xff]
    pub stencil_write_mask: u8,
    pub front: Stencil,
    pub back: Stencil,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, SmartDefault)]
pub struct BlendDesc {
    #[default(Blend::Add)]
    pub color_op: Blend,
    #[default(Factor::One)]
    pub src_color_factor: Factor,
    #[default(Factor::Zero)]
    pub dst_color_factor: Factor,
    #[default(Blend::Add)]
    pub alpha_op: Blend,
    #[default(Factor::One)]
    pub src_alpha_factor: Factor,
    #[default(Factor::Zero)]
    pub dst_alpha_factor: Factor,
    #[default(ColorComponents::All)]
    pub color_write_mask: ColorComponents,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, SmartDefault)]
pub struct ColorTarget {
    #[default(Format::None)]
    pub format: Format,
    #[default = 0xf]
    pub write_mask: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, SmartDefault)]
pub struct RasterDesc<'a> {
    #[default(Topology::TriangleList)]
    pub topology: Topology,
    #[default = false]
    pub primitive_restart: bool,
    #[default(Cull::None)]
    pub cull: Cull,
    // TODO: DepthStencilDesc
    // #[default = false]
    // pub alpha_to_coverage: bool,
    // #[default = false]
    // pub dual_source_blending: bool,
    #[default = 1]
    pub sample_count: u8,
    // #[default(Format::None)]
    // pub depth_format: Format,
    // #[default(Format::None)]
    // pub stencil_format: Format,
    // #[default(&[])]
    // pub color_targets: &'a [ColorTarget],
    pub blend_state: Option<&'a BlendDesc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, SmartDefault)]
pub struct RenderPassDesc {
    #[default((0, 0))]
    pub render_area_offset: (i32, i32),
    #[default((0, 0))]
    pub render_area_size: (u32, u32),
    #[default = 1]
    pub layer_count: u32,
    #[default = 0]
    pub view_mask: u32,
    // TODO: attachments
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, SmartDefault)]
pub struct TextureDesc {
    #[default(TextureType::Tex2D)]
    pub ty: TextureType,
    #[default((0, 0, 0))]
    pub dimensions: (u32, u32, u32),
    #[default = 1]
    pub mip_count: u32,
    #[default = 1]
    pub layer_count: u32,
    #[default = 1]
    pub sample_count: u32,
    #[default(Format::None)]
    pub format: Format,
    #[default(UsageFlags::None)]
    pub usage: UsageFlags,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, SmartDefault)]
pub struct ViewDesc {
    #[default(Format::None)]
    pub format: Format,
    #[default = 0]
    pub base_mip: u8,
    #[default = 0xff]
    pub mip_count: u8,
    #[default(0)]
    pub base_layer: u16,
    #[default = 0xffff]
    pub layer_count: u16,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct TextureSizeAlign {
    pub size: usize,
    pub align: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct TextureDescriptor {
    pub data: [u64; 4],
}

unsafe impl Zeroable for TextureDescriptor {}
unsafe impl Pod for TextureDescriptor {}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct DevicePointer(u64);

impl DevicePointer {
    pub fn to_raw(self) -> u64 {
        self.0
    }

    pub fn add(self, offset: usize) -> DevicePointer {
        DevicePointer(self.0 + offset as u64)
    }
}

impl From<DevicePointer> for u64 {
    fn from(value: DevicePointer) -> Self {
        value.to_raw()
    }
}

#[derive(Debug)]
pub struct Allocation<'a, T: Zeroable> {
    host: *mut T,
    device: DevicePointer,
    count: usize,
    gpu: &'a Gpu,
}

impl<'a, T: Zeroable> Allocation<'a, T> {
    pub fn host(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.host, self.count) }
    }
    pub fn host_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.host, self.count) }
    }
    pub fn host_raw(&self) -> *mut T {
        self.host
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn device(&self) -> DevicePointer {
        self.device
    }
}

#[derive(Debug)]
pub struct Texture<'a> {
    gpu: &'a Gpu,
}

impl<'a> Texture<'a> {
    pub fn view_descriptor(&self) -> TextureDescriptor {
        todo!()
    }
    pub fn rw_view_descriptor(&self) -> TextureDescriptor {
        todo!()
    }
}

#[derive(Debug)]
pub struct DepthStencilState<'a> {
    gpu: &'a Gpu,
}

#[derive(Debug)]
pub struct BlendState<'a> {
    gpu: &'a Gpu,
}

#[derive(Debug)]
pub struct Queue<'a> {
    queue: vk::Queue,
    command_pool: &'a CommandBufferPool,
    gpu: Option<&'a Gpu>,
}

impl<'a> Queue<'a> {
    pub fn create_buffer(&self) -> Result<CommandBuffer<'_>> {
        self.create_buffer_gpu(self.gpu.unwrap())
    }

    fn create_buffer_gpu(&self, gpu: &'a Gpu) -> Result<CommandBuffer<'_>> {
        let buffer = self.command_pool.acquire(&gpu.device)?;
        Ok(CommandBuffer {
            buffer: buffer.data().0,
            item: buffer,
            gpu,
        })
    }

    pub fn submit(&self, buffer:CommandBuffer<'a>, signal_semaphore: &Semaphore<'a>,
                  signal_value: u64) -> Result<()> {
        self.submit_gpu(self.gpu.unwrap(), buffer, signal_semaphore, signal_value)
    }

    fn buffer_prepare_submit(buffer: &mut CommandBuffer<'a>) -> (vk::Semaphore, u64) {
        let buffer_data = buffer.item.data_mut();
        buffer_data.2 += 1;
        (buffer_data.1, buffer_data.2)
    }

    fn submit_gpu(&self, gpu: &Gpu, mut buffer: CommandBuffer<'a>, signal_semaphore: &Semaphore<'a>,
                  signal_value: u64) -> Result<()> {
        let (buffer_semaphore, buffer_value) = Self::buffer_prepare_submit(&mut buffer);
        self.custom_submit(gpu, buffer,
                           &[signal_semaphore.semaphore, buffer_semaphore],
                           &[signal_value, buffer_value],
                           &[], &[], &[])
    }

    fn custom_submit(&self, gpu: &Gpu, buffer: CommandBuffer<'a>,
                     signal_semaphores: &[vk::Semaphore], signal_values: &[u64],
                     wait_semaphores: &[vk::Semaphore], wait_values: &[u64],
                     wait_dst_stage_mask: &[vk::PipelineStageFlags]) -> Result<()> {
        let mut semaphore_info = vk::TimelineSemaphoreSubmitInfo::builder()
            .wait_semaphore_values(wait_values)
            .signal_semaphore_values(signal_values);

        let command_buffers = [buffer.buffer];
        let info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_dst_stage_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_semaphores)
            .push_next(&mut semaphore_info);

        unsafe { gpu.device.queue_submit(self.queue, &[info], vk::Fence::null())? };

        self.command_pool.release(&gpu.device, buffer.item);

        Ok(())
    }
}

#[derive(Debug)]
pub struct CommandBuffer<'a> {
    buffer: vk::CommandBuffer,
    item: PooledCommandBuffer,
    gpu: &'a Gpu,
}

impl<'a> CommandBuffer<'a> {
    pub fn begin_recording(&mut self) -> Result<()> {
        let info = vk::CommandBufferBeginInfo::builder();
        unsafe { self.gpu.device.begin_command_buffer(self.buffer, &info)? };
        Ok(())
    }
    pub fn end_recording(&mut self) -> Result<()> {
        unsafe { self.gpu.device.end_command_buffer(self.buffer)? };
        Ok(())
    }

    pub fn copy(&mut self, dst: DevicePointer, src: DevicePointer) {
        todo!()
    }
    pub fn copy_to_texture(&mut self, dst: DevicePointer, src: DevicePointer, tex: &Texture<'_>) {
        todo!()
    }
    pub fn copy_from_texture(&mut self, dst: DevicePointer, src: DevicePointer, tex: &Texture<'_>) {
        todo!()
    }

    pub fn set_active_texture_heap_pointer(&mut self, ptr: DevicePointer) {
        todo!()
    }

    pub fn barrier(&mut self, before: Stage, after: Stage, hazards: HazardFlags) {
        todo!()
    }
    pub fn signal_after(&mut self, before: Stage, ptr: DevicePointer, value: u64, signal: Signal) {
        todo!()
    }
    pub fn wait_before_masked(&mut self, after: Stage, ptr: DevicePointer, value: u64,
                              op: Op, hazards: HazardFlags, mask: u64) {
        todo!()
    }
    pub fn wait_before(&mut self, after: Stage, ptr: DevicePointer, value: u64, op: Op, hazards: HazardFlags) {
        self.wait_before_masked(after, ptr, value, op, hazards, u64::MAX)
    }

    // pub fn set_pipeline(&mut self, pipeline: &Pipeline<'_>) {
    //     unsafe { self.gpu.device.cmd_bind_pipeline(self.buffer, pipeline.bind_point, pipeline.pipeline) };
    // }
    pub fn set_depth_stencil_state(&mut self, state: &DepthStencilState<'a>) {
        todo!()
    }
    pub fn set_blend_state(&mut self, state: &BlendState<'_>) {
        todo!()
    }

    pub fn bind_shaders<const N: usize>(&mut self, shaders: [&Shader<'_>; N]) {
        self.unbind_shaders_raw(&[vk::ShaderStageFlags::GEOMETRY]);

        let stages = shaders.map(|s| s.stage);
        let handles = shaders.map(|s| s.shader);

        unsafe { self.gpu.device.cmd_bind_shaders_ext(self.buffer, &stages, &handles) };
    }
    pub fn unbind_shaders<const N: usize>(&mut self, stages: [ShaderStage; N]) {
        self.unbind_shaders_raw(&stages.map(vulkan::get_stage));
    }
    fn unbind_shaders_raw(&mut self, stages: &[vk::ShaderStageFlags]) {
        unsafe {
            (self.gpu.device.commands().cmd_bind_shaders_ext)(
                self.buffer,
                stages.len() as u32,
                stages.as_ptr(),
                std::ptr::null()
            );
        }
    }

    pub fn dispatch(&mut self, data: DevicePointer, dimensions: (u32, u32, u32)) {
        todo!()
    }
    pub fn dispatch_indirect(&mut self, data: DevicePointer, dimensions: DevicePointer) {
        todo!()
    }

    fn image_barrier(&mut self, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout, image: vk::Image) {
        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let barrier = vk::ImageMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(image)
            .subresource_range(subresource_range);

        let barriers = [barrier];
        unsafe {
            self.gpu.device.cmd_pipeline_barrier(
                self.buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[] as &[vk::MemoryBarrier],
                &[] as &[vk::BufferMemoryBarrier],
                &barriers,
            );
        }
    }

    pub fn begin_render_pass(&mut self, desc: &RenderPassDesc, framebuffer: usize) {
        let render_area_size = if desc.render_area_size == (0, 0) {
            self.gpu.swapchain.borrow().extent
        } else {
            vk::Extent2D { width: desc.render_area_size.0, height: desc.render_area_size.1 }
        };

        self.image_barrier(
            vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            self.gpu.swapchain.borrow().images[framebuffer]);

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: desc.render_area_offset.0, y: desc.render_area_offset.1 })
            .extent(render_area_size);

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 1.0, 0.0, 1.0],
            },
        };

        let color_attachment = vk::RenderingAttachmentInfoKHR::builder()
            .image_view(self.gpu.swapchain.borrow().image_views[framebuffer])
            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(color_clear_value);

        let color_attachments = [color_attachment];
        let info = vk::RenderingInfoKHR::builder()
            .render_area(render_area)
            .layer_count(desc.layer_count.max(1))
            .view_mask(desc.view_mask)
            .color_attachments(&color_attachments);

        let dev = &self.gpu.device;

        unsafe { dev.cmd_begin_rendering_khr(self.buffer, &info) };

        unsafe {
            dev.cmd_set_stencil_test_enable_ext(self.buffer, false);
            dev.cmd_set_color_blend_enable_ext(self.buffer, 0, &[vk::FALSE]);
            dev.cmd_set_color_write_mask_ext(self.buffer, 0, &[vk::ColorComponentFlags::all()]);

            dev.cmd_set_depth_compare_op_ext(self.buffer, vk::CompareOp::LESS);
            dev.cmd_set_depth_test_enable_ext(self.buffer, false);
            dev.cmd_set_depth_write_enable_ext(self.buffer, false);
            dev.cmd_set_depth_bias_enable_ext(self.buffer, false);
            // dev.cmd_set_depth_clip_enable_ext(self.buffer, false);

            let viewport = vk::Viewport::builder()
                .x(render_area.offset.x as f32)
                .y(render_area.offset.y as f32)
                .width(render_area.extent.width as f32)
                .height(render_area.extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0);
            dev.cmd_set_viewport_with_count_ext(self.buffer, &[viewport]);
            dev.cmd_set_scissor_with_count_ext(self.buffer, &[render_area]);
            dev.cmd_set_rasterizer_discard_enable_ext(self.buffer, false);

            dev.cmd_set_vertex_input_ext(
                self.buffer,
                &[] as &[vk::VertexInputBindingDescription2EXT],
                &[] as &[vk::VertexInputAttributeDescription2EXT],
            );
            dev.cmd_set_rasterization_samples_ext(self.buffer, vk::SampleCountFlags::_1);
            dev.cmd_set_primitive_topology_ext(self.buffer, vk::PrimitiveTopology::TRIANGLE_LIST);
            dev.cmd_set_primitive_restart_enable_ext(self.buffer, false);

            dev.cmd_set_sample_mask_ext(self.buffer, vk::SampleCountFlags::_1, Some(&1));
            dev.cmd_set_alpha_to_coverage_enable_ext(self.buffer, false);
            dev.cmd_set_polygon_mode_ext(self.buffer, vk::PolygonMode::FILL);
            dev.cmd_set_cull_mode_ext(self.buffer, vk::CullModeFlags::NONE);
            dev.cmd_set_front_face_ext(self.buffer, vk::FrontFace::CLOCKWISE);
        }
    }
    pub fn end_render_pass(&mut self, framebuffer: usize) {
        unsafe { self.gpu.device.cmd_end_rendering_khr(self.buffer) };

        self.image_barrier(
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR,
            self.gpu.swapchain.borrow().images[framebuffer]);
    }

    pub fn draw_instanced(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
        unsafe { self.gpu.device.cmd_draw(self.buffer, vertex_count, instance_count, first_vertex, first_instance) };
    }

    pub fn draw_indexed_instanced(&mut self, vertex_data: DevicePointer, pixel_data: DevicePointer,
                                  indices: DevicePointer, index_count: u32, instance_count: u32) {
        todo!()
    }
    pub fn draw_indexed_instanced_indirect(&mut self, vertex_data: DevicePointer, pixel_data: DevicePointer,
                                           indices: DevicePointer, args: DevicePointer) {
        todo!()
    }
    pub fn draw_indexed_instanced_indirect_multi(&mut self,
                                                 vertex_data: DevicePointer, vertex_stride: usize,
                                                 pixel_data: DevicePointer, pixel_stride: usize,
                                                 args: DevicePointer, draw_count: DevicePointer) {
        todo!()
    }

    pub fn draw_meshlets(&mut self, meshlet_data: DevicePointer, pixel_data: DevicePointer,
                         dimensions: (u32, u32, u32)) {
        todo!()
    }
    pub fn draw_meshlets_indirect(&mut self, meshlet_data: DevicePointer, pixel_data: DevicePointer,
                                  dimensions: DevicePointer) {
        todo!()
    }
}

#[derive(Debug)]
pub struct Semaphore<'a> {
    semaphore: vk::Semaphore,
    gpu: &'a Gpu,
}

impl<'a> Drop for Semaphore<'a> {
    fn drop(&mut self) {
        unsafe { self.gpu.device.destroy_semaphore(self.semaphore, None) };
    }
}

impl<'a> Semaphore<'a> {
    pub fn wait(&self, value: u64) -> Result<()> {
        let semaphores = [self.semaphore];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::builder()
            .semaphores(&semaphores)
            .values(&values);
        unsafe { self.gpu.device.wait_semaphores_khr(&info, u64::MAX)? };
        Ok(())
    }
}

#[derive(Debug)]
pub struct Shader<'a> {
    shader: vk::ShaderEXT,
    stage: vk::ShaderStageFlags,
    gpu: &'a Gpu,
}

impl<'a> Drop for Shader<'a> {
    fn drop(&mut self) {
        unsafe { self.gpu.device.destroy_shader_ext(self.shader, None) };
    }
}

#[derive(Debug)]
pub struct Gpu {
    pub(crate) queue_families: QueueFamilies,
    pub(crate) queues: Queues,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) device: Device,
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) swapchain: RefCell<Swapchain>,
}

impl Gpu {
    pub fn new(instance: &Instance, surface: vk::SurfaceKHR, window_size: (u32, u32)) -> Result<Self> {
        let (physical_device, queue_families) = find_suitable_device(instance, surface)?;
        let logical_device = create_logical_device(instance, physical_device, queue_families)?;
        let internal_queues = Queues::new(&logical_device, queue_families)?;
        let swapchain = create_swapchain(
            instance, physical_device, surface, window_size, queue_families, &logical_device)?;
        Ok(Self {
            queue_families,
            queues: internal_queues,
            physical_device,
            device: logical_device,
            surface,
            swapchain: RefCell::new(swapchain),
        })
    }

    pub fn recreate_swapchain(&self, instance: &Instance, window_size: (u32, u32)) -> Result<()> {
        let swapchain = create_swapchain(
            instance, self.physical_device, self.surface, window_size, self.queue_families, &self.device)?;
        unsafe { self.device.device_wait_idle()? };
        let mut old_swapchain = self.swapchain.replace(swapchain);
        unsafe { old_swapchain.destroy(&self.device) };
        Ok(())
    }

    pub fn wait_before_destroy(&self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
        }
    }

    pub unsafe fn destroy(&mut self, instance: &Instance) {
        unsafe {
            self.swapchain.borrow_mut().destroy(&self.device);
            self.queues.destroy(&self.device);
            self.device.destroy_device(None);
            instance.destroy_surface_khr(self.surface, None);
        }
    }

    unsafe fn alloc_layout<T: Zeroable>(&self, layout: Layout, memory: Memory) -> Allocation<'_, T> {
        todo!()
    }

    pub fn alloc<T: Zeroable>(&self, n: usize, memory: Memory) -> Allocation<'_, T> {
        unsafe { self.alloc_layout(Layout::array::<T>(n).unwrap(), memory) }
    }

    pub fn alloc_aligned<T: Zeroable>(&self, n: usize, align: usize, memory: Memory) -> Allocation<'_, T> {
        unsafe { self.alloc_layout(Layout::array::<T>(n).unwrap().align_to(align).unwrap(), memory) }
    }

    pub fn create_texture(&self, desc: TextureDesc, data: DevicePointer) -> Texture<'_> {
        todo!()
    }

    pub fn create_shader(&self, spirv: &[u8], stage: ShaderStage) -> Result<Shader<'_>> {
        create_shader(self, spirv, stage)
    }

    // pub fn create_compute_pipeline(&self, spirv: &[u8]) -> Pipeline<'_> {
    //     todo!()
    // }
    // pub fn create_graphics_pipeline(&self, vertex_spirv: &[u8], pixel_spirv: &[u8],
    //                                 raster_desc: RasterDesc) -> Result<Pipeline<'_>> {
    //     create_graphics_pipeline(self, vertex_spirv, pixel_spirv, raster_desc)
    // }
    // pub fn create_graphics_meshlet_pipeline(&self, meshlet_spirv: &[u8], pixel_spirv: &[u8],
    //                                         raster_desc: RasterDesc) -> Pipeline<'_> {
    //     todo!()
    // }

    pub fn create_depth_stencil_state(&self, desc: DepthStencilDesc) -> DepthStencilState<'_> {
        todo!()
    }
    pub fn create_blend_state(&self, desc: BlendDesc) -> BlendState<'_> {
        todo!()
    }

    pub fn create_queue(&self, ty: QueueType) -> Result<Queue<'_>> {
        Ok(match ty {
            QueueType::Graphics => self.queues.graphics(self),
            _ => panic!("unsupported queue type: {:?}", ty),
        })
    }

    pub fn create_semaphore(&self, value: u64) -> Result<Semaphore<'_>> {
        Ok(Semaphore {
            semaphore: create_semaphore(&self.device, value)?,
            gpu: self,
        })
    }

    pub fn next_swapchain_image(&self) -> Result<usize> {
        let fence_info = vk::FenceCreateInfo::builder();
        let fence = unsafe { self.device.create_fence(&fence_info, None)? };

        let mut swapchain = self.swapchain.borrow_mut();

        let next_image_index = unsafe {
            self.device.acquire_next_image_khr(
                swapchain.swapchain,
                u64::MAX,
                vk::Semaphore::null(),
                fence,
            )?.0 as usize
        };
        swapchain.image_index = next_image_index;

        unsafe {
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.destroy_fence(fence, None);
        }

        // TODO: move image layout transition here

        Ok(next_image_index)
    }

    pub fn swapchain_present(&self, wait_semaphore: &Semaphore<'_>, wait_value: u64) -> Result<()> {
        let swapchain = self.swapchain.borrow();

        let image_index = swapchain.image_index;
        let present_semaphore = swapchain.present_semaphores[image_index];

        let graphics_queue = self.queues.graphics(self);
        let mut cmd_buf = graphics_queue.create_buffer()?;
        cmd_buf.begin_recording()?;

        // TODO: move image layout transition here

        cmd_buf.end_recording()?;

        // let wait_semaphore_values = [wait_value];
        // let mut semaphore_info = vk::TimelineSemaphoreSubmitInfo::builder()
        //     .wait_semaphore_values(&wait_semaphore_values);
        //
        // let command_buffers = [cmd_buf.buffer];
        // let wait_semaphores = [wait_semaphore.semaphore];
        // let wait_stage_flags = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        // let signal_semaphores = [present_semaphore];
        // let submit_info = vk::SubmitInfo::builder()
        //     .command_buffers(&command_buffers)
        //     .wait_semaphores(&wait_semaphores)
        //     .wait_dst_stage_mask(&wait_stage_flags)
        //     .signal_semaphores(&signal_semaphores)
        //     .push_next(&mut semaphore_info);

        let wait_semaphores = [present_semaphore];
        let swapchains = [swapchain.swapchain];
        let image_indices = [image_index as u32];
        let info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let (buffer_semaphore, buffer_value) = Queue::buffer_prepare_submit(&mut cmd_buf);
        graphics_queue.custom_submit(self, cmd_buf,
                                     &[buffer_semaphore, present_semaphore],
                                     &[buffer_value, 0],
                                     &[wait_semaphore.semaphore],
                                     &[wait_value],
                                     &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])?;

        unsafe {
            // self.device.queue_submit(self.queues.raw_graphics(), &[submit_info], vk::Fence::null())?;
            self.device.queue_present_khr(self.queues.present(), &info)?;
        }

        Ok(())
    }
}

pub use vulkan::create_debug_info_callback;
