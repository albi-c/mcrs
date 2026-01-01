#![feature(try_find)]

mod vulkan;

use std::alloc::Layout;
use std::marker::PhantomData;
use anyhow::Result;
use bitflags::bitflags;
use bytemuck::Zeroable;
use smart_default::SmartDefault;
use vulkanalia::{vk, Device, Instance, Version};
use vulkanalia::vk::{DeviceV1_0, KhrSurfaceExtensionInstanceCommands};
use crate::vulkan::{create_logical_device, find_suitable_device, QueueFamilies, Swapchain};

pub const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
pub const VALIDATION_LAYER: vk::ExtensionName = vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

pub fn need_portability_ext(version: Version) -> bool {
    const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);
    cfg!(target_os = "macos") && version >= PORTABILITY_MACOS_VERSION
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
    #[default = 0xf]
    pub color_write_mask: u8,
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
    #[default(Cull::None)]
    pub cull: Cull,
    #[default = false]
    pub alpha_to_coverage: bool,
    #[default = false]
    pub dual_source_blending: bool,
    #[default = 1]
    pub sample_count: u8,
    #[default(Format::None)]
    pub depth_format: Format,
    #[default(Format::None)]
    pub stencil_format: Format,
    #[default(&[])]
    pub color_targets: &'a [ColorTarget],
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
pub struct TextureSizeAlign {
    pub size: usize,
    pub align: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TextureDescriptor {
    pub data: [u64; 4],
}

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
pub struct Pipeline<'a> {
    gpu: &'a Gpu,
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
    pd: PhantomData<&'a Gpu>,
}

impl<'a> Queue<'a> {
    pub fn start_recording(&self) -> CommandBuffer<'_> {
        todo!()
    }

    pub fn submit<'b>(&'b self, buffers: impl IntoIterator<Item = CommandBuffer<'b>>) {
        todo!()
    }
}

#[derive(Debug)]
pub struct CommandBuffer<'a> {
    queue: &'a Queue<'a>,
}

impl<'a> CommandBuffer<'a> {
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
        todo!()
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

    pub fn begin_render_pass(&mut self, desc: &RasterDesc) {
        todo!()
    }
    pub fn end_render_pass(&mut self) {
        todo!()
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
    gpu: &'a Gpu,
}

impl<'a> Semaphore<'a> {
    pub fn wait(&self, value: u64) {
        todo!()
    }
}

#[derive(Debug)]
pub struct Gpu {
    queue_families: QueueFamilies,
    device: Device,
    surface: vk::SurfaceKHR,
    swapchain: Swapchain,
}

impl Gpu {
    pub fn new(instance: &Instance, surface: vk::SurfaceKHR, window_size: (u32, u32)) -> Result<Self> {
        let (physical_device, queue_families, swapchain_support) = find_suitable_device(
            instance, surface)?;
        let (logical_device, swapchain) = create_logical_device(
            instance, physical_device, queue_families, swapchain_support, surface, window_size)?;
        Ok(Self {
            queue_families,
            device: logical_device,
            surface,
            swapchain,
        })
    }

    pub unsafe fn destroy(&mut self, instance: &Instance) {
        unsafe {
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

    pub fn create_compute_pipeline(&self, ir: &[u8]) -> Pipeline<'_> {
        todo!()
    }
    pub fn create_graphics_pipeline(&self, vertex_ir: &[u8], pixel_ir: &[u8],
                                    raster_desc: RasterDesc) -> Pipeline<'_> {
        todo!()
    }
    pub fn create_graphics_meshlet_pipeline(&self, meshlet_ir: &[u8], pixel_ir: &[u8],
                                            raster_desc: RasterDesc) -> Pipeline<'_> {
        todo!()
    }

    pub fn create_depth_stencil_state(&self, desc: DepthStencilDesc) -> DepthStencilState<'_> {
        todo!()
    }
    pub fn create_blend_state(&self, desc: BlendDesc) -> BlendState<'_> {
        todo!()
    }

    pub fn create_queue(&self, ty: QueueType, index: u32) -> Queue<'_> {
        let family = match ty {
            QueueType::Graphics => self.queue_families.graphics,
            QueueType::Present => self.queue_families.present,
            _ => panic!("unsupported queue type: {:?}", ty),
        };
        let queue = unsafe { self.device.get_device_queue(family, index) };

        Queue {
            queue,
            pd: PhantomData,
        }
    }

    pub fn create_semaphore(&self, value: u64) -> Semaphore<'_> {
        todo!()
    }
}
