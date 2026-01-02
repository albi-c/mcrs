#![feature(try_find)]

mod vulkan;

use std::alloc::Layout;
use anyhow::Result;
use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};
use smart_default::SmartDefault;
use vulkanalia::{vk, Device, Instance, Version};
use vulkanalia::vk::{DeviceV1_0, Handle, HasBuilder, KhrSurfaceExtensionInstanceCommands, KhrSwapchainExtensionDeviceCommands};
use crate::vulkan::{create_framebuffers, create_graphics_pipeline, create_logical_device, create_render_pass, find_suitable_device, QueueFamilies, Swapchain};

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
    Present,
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
pub struct RenderPass<'a> {
    pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    gpu: &'a Gpu,
}

impl<'a> Drop for RenderPass<'a> {
    fn drop(&mut self) {
        unsafe {
            for fb in self.framebuffers.drain(..) {
                self.gpu.device.destroy_framebuffer(fb, None);
            }
            self.gpu.device.destroy_render_pass(self.pass, None);
        }
    }
}

#[derive(Debug)]
pub struct Pipeline<'a> {
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    bind_point: vk::PipelineBindPoint,
    gpu: &'a Gpu,
}

impl<'a> Drop for Pipeline<'a> {
    fn drop(&mut self) {
        unsafe {
            self.gpu.device.destroy_pipeline(self.pipeline, None);
            self.gpu.device.destroy_pipeline_layout(self.layout, None);
        }
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
    command_pool: vk::CommandPool,
    gpu: &'a Gpu,
}

impl<'a> Queue<'a> {
    pub fn create_buffer<'b>(&'b self) -> Result<CommandBuffer<'a>> {
        let info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        Ok(CommandBuffer {
            buffer: unsafe { self.gpu.device.allocate_command_buffers(&info)?[0] },
            gpu: self.gpu,
        })
    }

    pub fn submit<'b>(&self, buffer: &'b CommandBuffer<'a>, wait_semaphore: &Semaphore<'a>,
                      signal_semaphore: &Semaphore<'a>) -> Result<()> where 'a: 'b {
        let wait_semaphores = [wait_semaphore.semaphore];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [buffer.buffer];
        let signal_semaphores = [signal_semaphore.semaphore];
        let info = vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        unsafe { self.gpu.device.queue_submit(self.queue, &[info], vk::Fence::null())? };
        Ok(())
    }
}

impl<'a> Drop for Queue<'a> {
    fn drop(&mut self) {
        unsafe { self.gpu.device.destroy_command_pool(self.command_pool, None); }
    }
}

#[derive(Debug)]
pub struct CommandBuffer<'a> {
    buffer: vk::CommandBuffer,
    gpu: &'a Gpu,
}

impl<'a> Drop for CommandBuffer<'a> {
    fn drop(&mut self) {
        // TODO: destroy
    }
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

    pub fn set_pipeline(&mut self, pipeline: &Pipeline<'_>) {
        unsafe { self.gpu.device.cmd_bind_pipeline(self.buffer, pipeline.bind_point, pipeline.pipeline) };
    }
    pub fn set_depth_stencil_state(&mut self, state: &DepthStencilState<'a>) {
        todo!()
    }
    pub fn set_blend_state(&mut self, state: &BlendState<'_>) {
        todo!()
    }

    pub fn dispatch(&mut self, data: DevicePointer, dimensions: (u32, u32, u32)) {
        todo!()
    }
    pub fn dispatch_indirect(&mut self, data: DevicePointer, dimensions: DevicePointer) {
        todo!()
    }

    // pub fn begin_render_pass(&mut self, desc: &RasterDesc) {
    pub fn begin_render_pass(&mut self, pass: &RenderPass<'_>, framebuffer: usize) {
        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(self.gpu.swapchain.extent);
        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let clear_values = [color_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(pass.pass)
            .framebuffer(pass.framebuffers[framebuffer])
            .render_area(render_area)
            .clear_values(&clear_values);
        unsafe { self.gpu.device.cmd_begin_render_pass(self.buffer, &info, vk::SubpassContents::INLINE) };
    }
    pub fn end_render_pass(&mut self) {
        unsafe { self.gpu.device.cmd_end_render_pass(self.buffer) };
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
    pub fn wait(&self, value: u64) {
        todo!()
    }
}

#[derive(Debug)]
pub struct Gpu {
    pub(crate) queue_families: QueueFamilies,
    pub(crate) device: Device,
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) swapchain: Swapchain,
}

impl Gpu {
    pub fn new(instance: &Instance, surface: vk::SurfaceKHR, window_size: (u32, u32)) -> Result<Self> {
        let (physical_device, queue_families) = find_suitable_device(instance, surface)?;
        let (logical_device, swapchain) = create_logical_device(
            instance, physical_device, queue_families, surface, window_size)?;
        Ok(Self {
            queue_families,
            device: logical_device,
            surface,
            swapchain,
        })
    }

    pub unsafe fn destroy(&mut self, instance: &Instance) {
        unsafe {
            // self.device.device_wait_idle().unwrap();
            self.swapchain.destroy(&self.device);
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

    pub fn create_render_pass(&self) -> Result<RenderPass<'_>> {
        let pass = create_render_pass(self)?;
        Ok(RenderPass {
            pass,
            framebuffers: create_framebuffers(self, pass)?,
            gpu: self,
        })
    }

    pub fn create_compute_pipeline(&self, spirv: &[u8]) -> Pipeline<'_> {
        todo!()
    }
    pub fn create_graphics_pipeline(&self, vertex_spirv: &[u8], pixel_spirv: &[u8],
                                    raster_desc: RasterDesc, render_pass: &RenderPass<'_>) -> Result<Pipeline<'_>> {
        create_graphics_pipeline(self, vertex_spirv, pixel_spirv, raster_desc, render_pass.pass)
    }
    pub fn create_graphics_meshlet_pipeline(&self, meshlet_spirv: &[u8], pixel_spirv: &[u8],
                                            raster_desc: RasterDesc) -> Pipeline<'_> {
        todo!()
    }

    pub fn create_depth_stencil_state(&self, desc: DepthStencilDesc) -> DepthStencilState<'_> {
        todo!()
    }
    pub fn create_blend_state(&self, desc: BlendDesc) -> BlendState<'_> {
        todo!()
    }

    pub fn create_queue(&self, ty: QueueType, index: u32) -> Result<Queue<'_>> {
        let family = match ty {
            QueueType::Graphics => self.queue_families.graphics,
            QueueType::Present => self.queue_families.present,
            _ => panic!("unsupported queue type: {:?}", ty),
        };
        let queue = unsafe { self.device.get_device_queue(family, index) };

        let pool_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(family);
        let command_pool = unsafe { self.device.create_command_pool(&pool_info, None)? };

        Ok(Queue {
            queue,
            command_pool,
            gpu: self,
        })
    }

    // pub fn create_semaphore(&self, value: u64) -> Result<Semaphore<'_>> {
    pub fn create_semaphore(&self) -> Result<Semaphore<'_>> {
        let info = vk::SemaphoreCreateInfo::builder();
        Ok(Semaphore {
            semaphore: unsafe { self.device.create_semaphore(&info, None) }?,
            gpu: self,
        })
    }

    pub fn next_swapchain_image(&self, image_available_semaphore: &Semaphore<'_>) -> Result<usize> {
        Ok(unsafe { self.device.acquire_next_image_khr(
            self.swapchain.swapchain,
            u64::MAX,
            image_available_semaphore.semaphore,
            vk::Fence::null(),
        )?.0 as usize })
    }

    pub fn swapchain_present(&self, image_index: usize, present_queue: &Queue<'_>, render_finished_semaphore: &Semaphore<'_>) -> Result<()> {
        let wait_semaphores = [render_finished_semaphore.semaphore];
        let swapchains = [self.swapchain.swapchain];
        let image_indices = [image_index as u32];
        let info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        unsafe { self.device.queue_present_khr(present_queue.queue, &info)? };
        Ok(())
    }
}
