#![feature(try_find)]
#![feature(linked_list_cursors)]
#![feature(btree_cursors)]

mod vulkan;
mod arena;

use std::alloc::Layout;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Bound, Index, IndexMut};
use anyhow::Result;
use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};
use smart_default::SmartDefault;
use vulkanalia::{vk, Device, Instance, Version};
use vulkanalia::vk::{DeviceV1_0, ExtDescriptorBufferExtensionDeviceCommands, ExtShaderObjectExtensionDeviceCommands, Handle, HasBuilder, KhrBufferDeviceAddressExtensionDeviceCommands, KhrDynamicRenderingExtensionDeviceCommands, KhrSurfaceExtensionInstanceCommands, KhrSwapchainExtensionDeviceCommands, KhrSynchronization2ExtensionDeviceCommands, KhrTimelineSemaphoreExtensionDeviceCommands};
use vulkanalia_vma as vma;
use vulkanalia_vma::Alloc;
use crate::vulkan::{create_logical_device, create_semaphore, create_shader, create_swapchain, find_suitable_device, get_sample_count_flag, CommandBufferPool, DescriptorSizes, PipelineLayout, PooledCommandBuffer, QueueFamilies, Queues, Swapchain};

pub use vulkan::create_debug_info_callback;
pub use crate::arena::Arena;

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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Load {
    Load = vk::AttachmentLoadOp::LOAD.as_raw(),
    Clear = vk::AttachmentLoadOp::CLEAR.as_raw(),
    DontCare = vk::AttachmentLoadOp::DONT_CARE.as_raw(),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Store {
    Store = vk::AttachmentStoreOp::STORE.as_raw(),
    DontCare = vk::AttachmentStoreOp::DONT_CARE.as_raw(),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Blend {
    Add = vk::BlendOp::ADD.as_raw(),
    Subtract = vk::BlendOp::SUBTRACT.as_raw(),
    RevSubtract = vk::BlendOp::REVERSE_SUBTRACT.as_raw(),
    Min = vk::BlendOp::MIN.as_raw(),
    Max = vk::BlendOp::MAX.as_raw(),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Factor {
    Zero = vk::BlendFactor::ZERO.as_raw(),
    One = vk::BlendFactor::ONE.as_raw(),
    SrcColor = vk::BlendFactor::SRC_COLOR.as_raw(),
    OneMinusSrcColor = vk::BlendFactor::ONE_MINUS_SRC_COLOR.as_raw(),
    DstColor = vk::BlendFactor::DST_COLOR.as_raw(),
    OneMinusDstColor = vk::BlendFactor::ONE_MINUS_DST_COLOR.as_raw(),
    SrcAlpha = vk::BlendFactor::SRC_ALPHA.as_raw(),
    OneMinusSrcAlpha = vk::BlendFactor::ONE_MINUS_SRC_ALPHA.as_raw(),
    DstAlpha = vk::BlendFactor::DST_ALPHA.as_raw(),
    OneMinusDstAlpha = vk::BlendFactor::ONE_MINUS_DST_ALPHA.as_raw(),
    ConstantColor = vk::BlendFactor::CONSTANT_COLOR.as_raw(),
    OneMinusConstantColor = vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR.as_raw(),
    ConstantAlpha = vk::BlendFactor::CONSTANT_ALPHA.as_raw(),
    OneMinusConstantAlpha = vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA.as_raw(),
    SrcAlphaSaturate = vk::BlendFactor::SRC_ALPHA_SATURATE.as_raw(),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Topology {
    PointList = vk::PrimitiveTopology::POINT_LIST.as_raw(),
    LineList = vk::PrimitiveTopology::LINE_LIST.as_raw(),
    LineStrip = vk::PrimitiveTopology::LINE_STRIP.as_raw(),
    TriangleList = vk::PrimitiveTopology::TRIANGLE_LIST.as_raw(),
    TriangleStrip = vk::PrimitiveTopology::TRIANGLE_STRIP.as_raw(),
    TriangleFan = vk::PrimitiveTopology::TRIANGLE_FAN.as_raw(),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum TextureType {
    Tex1D = vk::ImageType::_1D.as_raw(),
    Tex2D = vk::ImageType::_2D.as_raw(),
    Tex3D = vk::ImageType::_3D.as_raw(),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Format {
    RGBA8UNorm = vk::Format::R8G8B8A8_UNORM.as_raw(),
    Depth32Float = vk::Format::D32_SFLOAT.as_raw(),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TextureUsageFlags(u32);
bitflags! {
    impl TextureUsageFlags : u32 {
        const None = vk::ImageUsageFlags::empty().bits();
        const TransferSrc = vk::ImageUsageFlags::TRANSFER_SRC.bits();
        const TransferDst = vk::ImageUsageFlags::TRANSFER_DST.bits();
        const Sampled = vk::ImageUsageFlags::SAMPLED.bits();
        const Storage = vk::ImageUsageFlags::STORAGE.bits();
        const ColorAttachment = vk::ImageUsageFlags::COLOR_ATTACHMENT.bits();
        const DepthStencilAttachment = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT.bits();
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Stage(u32);
bitflags! {
    impl Stage : u32 {
        const Top = vk::PipelineStageFlags::TOP_OF_PIPE.bits();
        const Transfer = vk::PipelineStageFlags::TRANSFER.bits();
        const Compute = vk::PipelineStageFlags::COMPUTE_SHADER.bits();
        const RasterColorOut = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT.bits();
        const VertexShader = vk::PipelineStageFlags::VERTEX_SHADER.bits();
        const PixelShader = vk::PipelineStageFlags::FRAGMENT_SHADER.bits();
        const Bottom = vk::PipelineStageFlags::BOTTOM_OF_PIPE.bits();
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Access(u32);
bitflags! {
    impl Access : u32 {
        const ColorTargetRead = vk::AccessFlags::COLOR_ATTACHMENT_READ.bits();
        const ColorTargetWrite = vk::AccessFlags::COLOR_ATTACHMENT_WRITE.bits();
        const DepthStencilTargetRead = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ.bits();
        const DepthStencilTargetWrite = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE.bits();
    }
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum IndexType {
    U16 = vk::IndexType::UINT16.as_raw(),
    U32 = vk::IndexType::UINT32.as_raw(),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Filter {
    Linear = vk::Filter::LINEAR.as_raw(),
    Nearest = vk::Filter::NEAREST.as_raw(),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum MipmapMode {
    Linear = vk::SamplerMipmapMode::LINEAR.as_raw(),
    Nearest = vk::SamplerMipmapMode::NEAREST.as_raw(),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum TextureWrap {
    Repeat = vk::SamplerAddressMode::REPEAT.as_raw(),
    MirroredRepeat = vk::SamplerAddressMode::MIRRORED_REPEAT.as_raw(),
    ClampToEdge = vk::SamplerAddressMode::CLAMP_TO_EDGE.as_raw(),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum TextureLayout {
    Undefined = vk::ImageLayout::UNDEFINED.as_raw(),
    General = vk::ImageLayout::GENERAL.as_raw(),
    ColorAttachmentOptimal = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL.as_raw(),
    DepthStencilAttachmentOptimal = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL.as_raw(),
    DepthStencilReadOnlyOptimal = vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL.as_raw(),
    ShaderReadOnlyOptimal = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL.as_raw(),
    TransferSrcOptimal = vk::ImageLayout::TRANSFER_SRC_OPTIMAL.as_raw(),
    TransferDstOptimal = vk::ImageLayout::TRANSFER_DST_OPTIMAL.as_raw(),
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

#[derive(Debug, Clone, PartialEq)]
pub enum ClearValue {
    Color([f32; 4]),
    ColorI([i32; 4]),
    ColorU([u32; 4]),
    DepthStencil(f32, u32),
}

impl Eq for ClearValue {}
impl Hash for ClearValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::Color(arr) => bytemuck::cast_slice::<_, u32>(arr).hash(state),
            Self::ColorI(arr) => arr.hash(state),
            Self::ColorU(arr) => arr.hash(state),
            Self::DepthStencil(d, s) => {
                bytemuck::cast::<_, u32>(*d).hash(state);
                s.hash(state);
            }
        }
    }
}
impl Default for ClearValue {
    fn default() -> Self {
        Self::Color([0.0, 0.0, 0.0, 1.0])
    }
}

impl ClearValue {
    fn to_vulkan(&self) -> vk::ClearValue {
        match self {
            Self::Color(arr) => vk::ClearValue { color: vk::ClearColorValue {
                float32: *arr,
            } },
            Self::ColorI(arr) => vk::ClearValue { color: vk::ClearColorValue {
                int32: *arr,
            } },
            Self::ColorU(arr) => vk::ClearValue { color: vk::ClearColorValue {
                uint32: *arr,
            } },
            Self::DepthStencil(d, s) => vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue {
                depth: *d,
                stencil: *s,
            } },
        }
    }
}

#[derive(Debug, SmartDefault)]
pub struct Target<'a> {
    pub view: TextureView<'a>,
    #[default(Load::Load)]
    pub load_op: Load,
    #[default(Store::Store)]
    pub store_op: Store,
    pub clear_value: ClearValue,
}

#[derive(Debug, Clone, SmartDefault)]
pub struct RenderPassDesc<'a> {
    #[default((0, 0))]
    pub render_area_offset: (i32, i32),
    #[default((0, 0))]
    pub render_area_size: (u32, u32),
    #[default = 1]
    pub layer_count: u32,
    #[default = 0]
    pub view_mask: u32,
    #[default(Topology::TriangleList)]
    pub topology: Topology,
    #[default = false]
    pub primitive_restart: bool,
    #[default(Cull::None)]
    pub cull: Cull,
    #[default = false]
    pub alpha_to_coverage: bool,
    #[default = false]
    pub dual_source_blending: bool,
    #[default = 1]
    pub sample_count: u8,
    pub blend_state: Option<&'a BlendDesc>,
    pub color_targets: &'a [Target<'a>],
    pub depth_target: Option<&'a Target<'a>>,
    pub stencil_target: Option<&'a Target<'a>>,
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
    #[default(Format::RGBA8UNorm)]
    pub format: Format,
    #[default(TextureUsageFlags::None)]
    pub usage: TextureUsageFlags,
    #[default(TextureLayout::General)]
    pub layout: TextureLayout,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, SmartDefault)]
pub struct SamplerDesc {
    #[default(Filter::Linear)]
    pub min_filter: Filter,
    #[default(Filter::Linear)]
    pub mag_filter: Filter,
    #[default(MipmapMode::Linear)]
    pub mip_mode: MipmapMode,
    #[default(TextureWrap::Repeat)]
    pub wrap_u: TextureWrap,
    #[default(TextureWrap::Repeat)]
    pub wrap_v: TextureWrap,
    #[default(TextureWrap::Repeat)]
    pub wrap_w: TextureWrap,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, SmartDefault)]
pub struct ViewDesc {
    #[default(Format::RGBA8UNorm)]
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub struct DevicePointer(vk::DeviceAddress);

impl DevicePointer {
    pub fn null() -> Self {
        Self(0)
    }
    pub fn is_null(self) -> bool {
        self.0 == 0
    }

    pub fn to_raw(self) -> vk::DeviceAddress {
        self.0
    }

    pub fn add(self, offset: usize) -> DevicePointer {
        DevicePointer(self.0 + vk::DeviceAddress::try_from(offset).expect("offset too large"))
    }
}

impl From<DevicePointer> for u64 {
    fn from(value: DevicePointer) -> Self {
        value.to_raw()
    }
}

unsafe impl Zeroable for DevicePointer {}
unsafe impl Pod for DevicePointer {}

pub trait MemoryAllocation {
    type Type: Zeroable;

    fn host(&self) -> &[Self::Type];
    fn host_mut(&mut self) -> &mut [Self::Type];
    fn host_raw(&self) -> *mut Self::Type;

    fn len(&self) -> usize;
    fn len_bytes(&self) -> usize {
        self.len() * size_of::<Self::Type>()
    }

    fn device(&self) -> DevicePointer;
}

pub trait MemoryAllocator {
    type Allocation<T: Pod>: MemoryAllocation<Type = T>;

    fn alloc<T: Pod>(&self, n: usize) -> Result<Self::Allocation<T>> {
        self.alloc_aligned(n, 1)
    }
    fn alloc_aligned<T: Pod>(&self, n: usize, align: usize) -> Result<Self::Allocation<T>>;

    fn alloc_data<T: Pod>(&self, data: &[T]) -> Result<Self::Allocation<T>> {
        let mut alloc = self.alloc(data.len())?;
        alloc.host_mut().copy_from_slice(data);
        Ok(alloc)
    }
}

#[derive(Debug)]
pub struct Allocation<'a, T: Pod> {
    host: *mut T,
    device: DevicePointer,
    count: usize,
    buffer: vk::Buffer,
    allocation: vma::Allocation,
    gpu: &'a Gpu,
}

impl<'a, T: Pod> Drop for Allocation<'a, T> {
    fn drop(&mut self) {
        unsafe { self.gpu.dealloc(self) };
    }
}

impl<'a, T: Pod> MemoryAllocation for Allocation<'a, T> {
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
pub struct DescriptorHeap<'a> {
    allocation: Allocation<'a, u8>,
    count: usize,
    element_size: usize,
}

impl<'a> DescriptorHeap<'a> {
    pub fn get(&self, index: usize) -> &[u8] {
        assert!(index < self.count, "descriptor index out of bounds");
        let offset = index * self.element_size;
        &self.allocation.host()[offset..offset + self.element_size]
    }
    pub fn get_mut(&mut self, index: usize) -> &mut [u8] {
        assert!(index < self.count, "descriptor index out of bounds");
        let offset = index * self.element_size;
        &mut self.allocation.host_mut()[offset..offset + self.element_size]
    }

    pub fn get_range(&self, index: usize, count: usize) -> &[u8] {
        assert!(index + count <= self.count, "descriptor index out of bounds");
        let offset = index * self.element_size;
        &self.allocation.host()[offset..offset + count * self.element_size]
    }
    pub fn get_range_mut(&mut self, index: usize, count: usize) -> &mut [u8] {
        assert!(index + count <= self.count, "descriptor index out of bounds");
        let offset = index * self.element_size;
        &mut self.allocation.host_mut()[offset..offset + count * self.element_size]
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn device(&self) -> DevicePointer {
        self.allocation.device()
    }
}

impl<'a> Index<usize> for DescriptorHeap<'a> {
    type Output = [u8];

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
    }
}
impl<'a> IndexMut<usize> for DescriptorHeap<'a> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index)
    }
}

#[derive(Debug, Default)]
pub struct TextureView<'a> {
    view: vk::ImageView,
    pd: PhantomData<&'a Texture<'a>>,
}

#[derive(Debug)]
pub struct Texture<'a> {
    dimensions: (u32, u32, u32),
    format: Format,
    ty: TextureType,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    allocation: vma::Allocation,
    gpu: &'a Gpu,
    view: Cell<vk::ImageView>,
}

impl<'a> Texture<'a> {
    pub fn dimensions(&self) -> (u32, u32, u32) {
        self.dimensions
    }

    fn view_type(ty: TextureType) -> vk::ImageViewType {
        match ty {
            TextureType::Tex1D => vk::ImageViewType::_1D,
            TextureType::Tex2D => vk::ImageViewType::_2D,
            TextureType::Tex3D => vk::ImageViewType::_3D,
        }
    }

    fn aspect_flags(format: Format) -> vk::ImageAspectFlags {
        match format {
            Format::RGBA8UNorm => vk::ImageAspectFlags::COLOR,
            Format::Depth32Float => vk::ImageAspectFlags::DEPTH,
        }
    }

    fn get_view(&self) -> Result<vk::ImageView> {
        let view = self.view.get();
        if !view.is_null() {
            Ok(view)
        } else {
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(self.aspect)
                .level_count(1)
                .layer_count(1);
            let view_info = vk::ImageViewCreateInfo::builder()
                .image(self.image)
                .view_type(Self::view_type(self.ty))
                .format(vk::Format::from_raw(self.format as i32))
                .subresource_range(subresource_range);
            let view = unsafe { self.gpu.device.create_image_view(&view_info, None)? };
            self.view.set(view);
            Ok(view)
        }
    }

    pub fn view(&self) -> Result<TextureView<'_>> {
        Ok(TextureView {
            view: self.get_view()?,
            pd: PhantomData,
        })
    }

    pub fn view_descriptor_size(&self) -> usize {
        self.gpu.descriptor_sizes.sampled_texture
    }

    pub fn view_descriptor(&self, descriptor: &mut [u8]) -> Result<()> {
        assert_eq!(descriptor.len(), self.view_descriptor_size(),
                   "incorrect buffer size for texture descriptor");

        let view = self.view()?;

        let image_info = vk::DescriptorImageInfo::builder()
            .sampler(vk::Sampler::null())
            .image_view(view.view)
            .image_layout(vk::ImageLayout::GENERAL)
            .build();
        let info = vk::DescriptorGetInfoEXT::builder()
            .type_(vk::DescriptorType::SAMPLED_IMAGE)
            .data(vk::DescriptorDataEXT {
                sampled_image: &raw const image_info,
            });
        unsafe { self.gpu.device.get_descriptor_ext(&info, descriptor) };

        Ok(())
    }
    pub fn rw_view_descriptor(&self) -> Result<()> {
        todo!()
    }
}

impl<'a> Drop for Texture<'a> {
    fn drop(&mut self) {
        unsafe {
            let view = self.view.get();
            if !view.is_null() {
                self.gpu.device.destroy_image_view(view, None);
            }
            self.gpu.allocator.as_ref().unwrap().destroy_image(self.image, self.allocation);
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
pub struct SubmitWait<'a>(vk::Semaphore, u64, &'a Gpu);

impl<'a> SubmitWait<'a> {
    pub fn wait(self) {
        let semaphores = [self.0];
        let values = [self.1];
        let info = vk::SemaphoreWaitInfo::builder()
            .semaphores(&semaphores)
            .values(&values);
        unsafe { self.2.device.wait_semaphores_khr(&info, u64::MAX).expect("semaphore wait failed") };
    }
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

    pub fn submit(&self, buffer: CommandBuffer<'a>, signal_semaphore: &Semaphore<'a>,
                  signal_value: u64) -> Result<SubmitWait<'_>> {
        self.submit_gpu(self.gpu.unwrap(), buffer, signal_semaphore, signal_value)
    }

    pub fn submit_no_signal(&self, mut buffer: CommandBuffer<'a>) -> Result<SubmitWait<'_>> {
        let (buffer_semaphore, buffer_value) = Self::buffer_prepare_submit(&mut buffer);
        self.custom_submit(self.gpu.unwrap(), buffer,
                           &[buffer_semaphore],
                           &[buffer_value],
                           &[], &[], &[])?;
        Ok(SubmitWait(buffer_semaphore, buffer_value, self.gpu.unwrap()))
    }

    fn buffer_prepare_submit(buffer: &mut CommandBuffer<'a>) -> (vk::Semaphore, u64) {
        let buffer_data = buffer.item.data_mut();
        buffer_data.2 += 1;
        (buffer_data.1, buffer_data.2)
    }

    fn submit_gpu<'b>(&self, gpu: &'b Gpu, mut buffer: CommandBuffer<'a>, signal_semaphore: &Semaphore<'a>,
                      signal_value: u64) -> Result<SubmitWait<'b>> {
        let (buffer_semaphore, buffer_value) = Self::buffer_prepare_submit(&mut buffer);
        self.custom_submit(gpu, buffer,
                           &[signal_semaphore.semaphore, buffer_semaphore],
                           &[signal_value, buffer_value],
                           &[], &[], &[])?;
        Ok(SubmitWait(buffer_semaphore, buffer_value, gpu))
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
    pub fn copy_to_texture(&mut self, src: DevicePointer, tex: &Texture<'_>) {
        let (buffer, offset) = self.gpu.device_addr_to_buffer_offset(src)
            .expect("invalid device pointer");
        let img_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(tex.aspect)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1);
        let img_copy = vk::BufferImageCopy::builder()
            .buffer_offset(offset)
            .buffer_row_length(tex.dimensions.0)
            .buffer_image_height(tex.dimensions.1)
            .image_offset(vk::Offset3D::default())
            .image_extent(vk::Extent3D { width: tex.dimensions.0, height: tex.dimensions.1, depth: tex.dimensions.2 })
            .image_subresource(img_subresource);
        unsafe { self.gpu.device.cmd_copy_buffer_to_image(
            self.buffer, buffer, tex.image, vk::ImageLayout::GENERAL, &[img_copy]) };
    }
    pub fn copy_from_texture(&mut self, dst: DevicePointer, tex: &Texture<'_>) {
        todo!()
    }

    pub fn set_texture_heap(&mut self, textures: Option<&DescriptorHeap<'_>>,
                            textures_rw: Option<&DescriptorHeap<'_>>, samplers: Option<&DescriptorHeap<'_>>) {
        let pointers = [
            textures.map(|heap| heap.device()).unwrap_or(DevicePointer::null()),
            textures_rw.map(|heap| heap.device()).unwrap_or(DevicePointer::null()),
            samplers.map(|heap| heap.device()).unwrap_or(DevicePointer::null()),
        ];
        let mut infos = [vk::DescriptorBufferBindingInfoEXT::default(); 3];
        let mut index = 0;
        for &pointer in &pointers {
            if pointer.is_null() {
                continue;
            }
            infos[index] = vk::DescriptorBufferBindingInfoEXT::builder()
                .address(pointer.to_raw())
                .usage(vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC)
                .build();
            index += 1;
        }
        unsafe { self.gpu.device.cmd_bind_descriptor_buffers_ext(self.buffer, &infos[..index]) };

        let mut index = 0;
        for (i, &pointer) in pointers.iter().enumerate() {
            if pointer.is_null() {
                continue;
            }
            unsafe { self.gpu.device.cmd_set_descriptor_buffer_offsets_ext(
                self.buffer, vk::PipelineBindPoint::GRAPHICS, self.gpu.pipeline_layout.layout,
                i as u32, &[index], &[0]) };
            index += 1;
        }
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

    fn image_barrier_raw(&mut self, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout, image: vk::Image,
                         src_stage: vk::PipelineStageFlags, dst_stage: vk::PipelineStageFlags,
                         src_access: vk::AccessFlags, dst_access: vk::AccessFlags,
                         aspect: vk::ImageAspectFlags) {
        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(aspect)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(image)
            .subresource_range(subresource_range)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access);

        let barriers = [barrier];
        unsafe {
            self.gpu.device.cmd_pipeline_barrier(
                self.buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[] as &[vk::MemoryBarrier],
                &[] as &[vk::BufferMemoryBarrier],
                &barriers,
            );
        }
    }

    fn image_barrier(&mut self, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout, image: vk::Image,
                     src_stage: vk::PipelineStageFlags, dst_stage: vk::PipelineStageFlags,
                     before_render: bool) {
        self.image_barrier_raw(
            old_layout, new_layout, image,
            src_stage, dst_stage,
            if before_render {
                vk::AccessFlags::empty()
            } else {
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            },
            if before_render {
                vk::AccessFlags::COLOR_ATTACHMENT_READ
            } else {
                vk::AccessFlags::empty()
            },
            vk::ImageAspectFlags::COLOR,
        );
    }

    pub fn texture_barrier(&mut self, texture: &Texture<'_>,
                           old_layout: TextureLayout, new_layout: TextureLayout,
                           src_stage: Stage, dst_stage: Stage,
                           src_access: Access, dst_access: Access) {
        self.image_barrier_raw(
            vk::ImageLayout::from_raw(old_layout as i32),
            vk::ImageLayout::from_raw(new_layout as i32),
            texture.image,
            vk::PipelineStageFlags::from_bits(src_stage.bits()).unwrap(),
            vk::PipelineStageFlags::from_bits(dst_stage.bits()).unwrap(),
            vk::AccessFlags::from_bits(src_access.bits()).unwrap(),
            vk::AccessFlags::from_bits(dst_access.bits()).unwrap(),
            texture.aspect,
        )
    }

    fn create_attachment(attachment: &Target<'_>) -> vk::RenderingAttachmentInfoBuilder<'static> {
        vk::RenderingAttachmentInfo::builder()
            .image_view(attachment.view.view)
            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)  // TODO: move to struct
            .load_op(vk::AttachmentLoadOp::from_raw(attachment.load_op as i32))
            .store_op(vk::AttachmentStoreOp::from_raw(attachment.store_op as i32))
            .clear_value(attachment.clear_value.to_vulkan())
    }

    pub fn begin_render_pass(&mut self, desc: &RenderPassDesc<'_>) {
        let render_area_size = if desc.render_area_size == (0, 0) {
            self.gpu.swapchain.borrow().extent
        } else {
            vk::Extent2D { width: desc.render_area_size.0, height: desc.render_area_size.1 }
        };

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: desc.render_area_offset.0, y: desc.render_area_offset.1 })
            .extent(render_area_size);

        let color_attachments = desc.color_targets.iter()
            .map(Self::create_attachment)
            .collect::<Vec<_>>();
        let depth_attachment;
        let stencil_attachment;
        let mut info = vk::RenderingInfoKHR::builder()
            .render_area(render_area)
            .layer_count(desc.layer_count.max(1))
            .view_mask(desc.view_mask)
            .color_attachments(&color_attachments);
        if let Some(a) = desc.depth_target {
            depth_attachment = Self::create_attachment(a);
            info = info.depth_attachment(&depth_attachment);
        }
        if let Some(a) = desc.stencil_target {
            stencil_attachment = Self::create_attachment(a);
            info = info.stencil_attachment(&stencil_attachment);
        }

        let dev = &self.gpu.device;

        unsafe { dev.cmd_begin_rendering_khr(self.buffer, &info) };

        unsafe {
            // TODO: use values from desc

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
    pub fn end_render_pass(&mut self) {
        unsafe { self.gpu.device.cmd_end_rendering_khr(self.buffer) };
    }

    pub fn draw_instanced(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
        unsafe { self.gpu.device.cmd_draw(self.buffer, vertex_count, instance_count, first_vertex, first_instance) };
    }

    pub fn draw_indexed(&mut self, vertex_data: DevicePointer, pixel_data: DevicePointer,
                        indices: DevicePointer, index_count: u32, index_type: IndexType) {
        self.draw_indexed_instanced(vertex_data, pixel_data, indices, index_count,
                                    0, index_type, 0, 1, 0);
    }
    pub fn draw_indexed_instanced(&mut self, vertex_data: DevicePointer, pixel_data: DevicePointer,
                                  indices: DevicePointer, index_count: u32, first_index: u32,
                                  index_type: IndexType, vertex_offset: i32, instance_count: u32,
                                  first_instance: u32) {
        let (buffer, offset) = self.gpu.device_addr_to_buffer_offset(indices)
            .expect("invalid index buffer");
        let push_constants = [vertex_data, pixel_data];
        unsafe {
            self.gpu.device.cmd_push_constants(self.buffer, self.gpu.pipeline_layout.layout,
                                               vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                                               0, bytemuck::cast_slice(&push_constants));
            self.gpu.device.cmd_bind_index_buffer(self.buffer, buffer, offset,
                                                  vk::IndexType::from_raw(index_type as i32));
            self.gpu.device.cmd_draw_indexed(self.buffer, index_count, instance_count, first_index,
                                             vertex_offset, first_instance);
        }
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

pub trait SwapchainContext {
    fn get_instance(&self) -> &Instance;
    fn get_window_size(&self) -> (u32, u32);
}

impl SwapchainContext for (&Instance, (u32, u32)) {
    fn get_instance(&self) -> &Instance {
        self.0
    }
    fn get_window_size(&self) -> (u32, u32) {
        self.1
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
    pub(crate) allocator: Option<vma::Allocator>,
    pub(crate) buffers: RefCell<BTreeMap<vk::DeviceAddress, (vk::Buffer, vk::DeviceSize)>>,
    pub(crate) pipeline_layout: PipelineLayout,
    pub(crate) descriptor_sizes: DescriptorSizes,
}

impl Gpu {
    pub fn new(ctx: &dyn SwapchainContext, surface: vk::SurfaceKHR) -> Result<Self> {
        let instance = ctx.get_instance();
        let window_size = ctx.get_window_size();
        let (physical_device, queue_families) = find_suitable_device(instance, surface)?;
        let logical_device = create_logical_device(instance, physical_device, queue_families)?;
        let internal_queues = Queues::new(&logical_device, queue_families)?;
        let swapchain = create_swapchain(
            instance, physical_device, surface, window_size, queue_families,
            &logical_device, vk::SwapchainKHR::null())?;
        let mut allocator_options = vma::AllocatorOptions::new(
            instance, &logical_device, physical_device);
        allocator_options.flags |= vma::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
        Ok(Self {
            queue_families,
            queues: internal_queues,
            physical_device,
            surface,
            swapchain: RefCell::new(swapchain),
            allocator: Some(unsafe { vma::Allocator::new(&allocator_options)? }),
            pipeline_layout: PipelineLayout::new(&logical_device)?,
            device: logical_device,
            buffers: RefCell::new(BTreeMap::new()),
            descriptor_sizes: DescriptorSizes::new(instance, physical_device)?,
        })
    }

    pub fn recreate_swapchain(&self, ctx: &dyn SwapchainContext) -> Result<()> {
        unsafe { self.device.device_wait_idle()? };
        let instance = ctx.get_instance();
        let window_size = ctx.get_window_size();
        let swapchain = create_swapchain(
            instance, self.physical_device, self.surface, window_size, self.queue_families,
            &self.device, self.swapchain.borrow().swapchain)?;
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
            drop(self.allocator.take().unwrap());
            self.pipeline_layout.destroy(&self.device);
            self.swapchain.borrow_mut().destroy(&self.device);
            self.queues.destroy(&self.device);
            self.device.destroy_device(None);
            instance.destroy_surface_khr(self.surface, None);
        }
    }

    unsafe fn alloc_layout<T: Pod>(&self, count: usize, layout: Layout, memory: Memory) -> Result<Allocation<'_, T>> {
        let (properties, usage) = match memory {
            Memory::Default => (
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vma::MemoryUsage::AutoPreferDevice
            ),
            Memory::Gpu => (
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vma::MemoryUsage::AutoPreferDevice,
            ),
            Memory::Readback => (
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_CACHED,
                vma::MemoryUsage::AutoPreferDevice,
            )
        };
        let buffer_usage = match memory {
            Memory::Default | Memory::Readback =>
                vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC,
            Memory::Gpu => vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
        };

        let length = vk::DeviceSize::try_from(layout.size()).expect("buffer size too large");
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(length)
            .usage(buffer_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let alloc_options = vma::AllocationOptions {
            flags: vma::AllocationCreateFlags::HOST_ACCESS_RANDOM,
            usage,
            required_flags: properties,
            ..Default::default()
        };

        let allocator = self.allocator.as_ref().expect("allocator dropped");
        let (buffer, alloc) = unsafe {
            allocator.create_buffer_with_alignment(
                buffer_info, &alloc_options, layout.align().try_into()
                    .expect("buffer alignment too large"))?
        };
        let mapping = unsafe { allocator.map_memory(alloc)? };

        let addr_info = vk::BufferDeviceAddressInfo::builder()
            .buffer(buffer);
        let addr = unsafe { self.device.get_buffer_device_address_khr(&addr_info) };

        assert!(self.buffers.borrow_mut().insert(addr, (buffer, length)).is_none(),
                "device address returned twice");

        Ok(Allocation {
            host: mapping as *mut _,
            device: DevicePointer(addr),
            count,
            buffer,
            allocation: alloc,
            gpu: self,
        })
    }

    unsafe fn dealloc<T: Pod>(&self, alloc: &mut Allocation<'_, T>) {
        self.buffers.borrow_mut().remove(&alloc.device.0);
        let allocator = self.allocator.as_ref().expect("allocator dropped");
        unsafe {
            allocator.unmap_memory(alloc.allocation);
            allocator.destroy_buffer(alloc.buffer, alloc.allocation);
        }
    }

    fn device_addr_to_buffer_offset(&self, addr: DevicePointer) -> Option<(vk::Buffer, vk::DeviceSize)> {
        let buffers = self.buffers.borrow();
        let mut cur = buffers.lower_bound(Bound::Included(&addr.0));
        if let Some((&base, &(buffer, size))) = cur.next() {
            if base <= addr.0 && addr.0 < base + size {
                return Some((buffer, addr.0 - base));
            }
        }
        None
    }

    pub fn alloc<T: Pod>(&self, n: usize) -> Result<Allocation<'_, T>> {
        self.alloc_mem(n, Memory::Default)
    }
    pub fn alloc_aligned<T: Pod>(&self, n: usize, align: usize) -> Result<Allocation<'_, T>> {
        self.alloc_mem_aligned(n, align, Memory::Default)
    }
    pub fn alloc_mem<T: Pod>(&self, n: usize, memory: Memory) -> Result<Allocation<'_, T>> {
        unsafe { self.alloc_layout(n, Layout::array::<T>(n)?, memory) }
    }
    pub fn alloc_mem_aligned<T: Pod>(&self, n: usize, align: usize, memory: Memory) -> Result<Allocation<'_, T>> {
        unsafe { self.alloc_layout(n, Layout::array::<T>(n)?.align_to(align)?, memory) }
    }

    fn alloc_descriptor_heap(&self, count: u32, element_size: usize) -> Result<DescriptorHeap<'_>> {
        Ok(DescriptorHeap {
            allocation: self.alloc(count as usize * element_size)?,
            count: count as usize,
            element_size,
        })
    }

    pub fn alloc_texture_descriptor_heap(&self) -> Result<DescriptorHeap<'_>> {
        self.alloc_descriptor_heap(PipelineLayout::MAX_TEXTURES, self.descriptor_sizes.sampled_texture)
    }
    pub fn alloc_texture_rw_descriptor_heap(&self) -> Result<DescriptorHeap<'_>> {
        self.alloc_descriptor_heap(PipelineLayout::MAX_TEXTURES_RW, self.descriptor_sizes.storage_texture)
    }
    pub fn alloc_sampler_descriptor_heap(&self) -> Result<DescriptorHeap<'_>> {
        self.alloc_descriptor_heap(PipelineLayout::MAX_SAMPLERS, self.descriptor_sizes.sampler)
    }

    pub fn allocator(&self) -> impl MemoryAllocator {
        struct GpuMemoryAllocator<'a>(&'a Gpu);
        impl<'a> MemoryAllocator for GpuMemoryAllocator<'a> {
            type Allocation<T: Pod> = Allocation<'a, T>;

            fn alloc<T: Pod>(&self, n: usize) -> Result<Self::Allocation<T>> {
                self.0.alloc(n)
            }
            fn alloc_aligned<T: Pod>(&self, n: usize, align: usize) -> Result<Self::Allocation<T>> {
                self.0.alloc_aligned(n, align)
            }
        }
        GpuMemoryAllocator(self)
    }
    pub fn allocator_mem(&self, memory: Memory) -> impl MemoryAllocator {
        struct GpuMemoryAllocatorMem<'a>(&'a Gpu, Memory);
        impl<'a> MemoryAllocator for GpuMemoryAllocatorMem<'a> {
            type Allocation<T: Pod> = Allocation<'a, T>;

            fn alloc<T: Pod>(&self, n: usize) -> Result<Self::Allocation<T>> {
                self.0.alloc_mem(n, self.1)
            }
            fn alloc_aligned<T: Pod>(&self, n: usize, align: usize) -> Result<Self::Allocation<T>> {
                self.0.alloc_mem_aligned(n, align, self.1)
            }
        }
        GpuMemoryAllocatorMem(self, memory)
    }

    pub fn create_arena(&self, size: usize) -> Result<Arena<'_>> {
        self.create_arena_mem(size, Memory::Default)
    }
    pub fn create_arena_mem(&self, size: usize, memory: Memory) -> Result<Arena<'_>> {
        self.create_arena_aligned(size, 16, memory)
    }
    pub fn create_arena_aligned(&self, size: usize, align: usize, memory: Memory) -> Result<Arena<'_>> {
        let allocation = self.alloc_mem_aligned::<u8>(size, align, memory)?;
        Ok(Arena {
            allocation,
            offset: Cell::new(0),
        })
    }

    pub fn create_texture(&self, desc: TextureDesc, cmd_buf: &mut CommandBuffer<'_>) -> Result<Texture<'_>> {
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::from_raw(desc.ty as i32))
            .format(vk::Format::from_raw(desc.format as i32))
            .extent(vk::Extent3D { width: desc.dimensions.0, height: desc.dimensions.1, depth: desc.dimensions.2 })
            .mip_levels(desc.mip_count)
            .array_layers(desc.layer_count)
            .samples(get_sample_count_flag(desc.sample_count.try_into()
                .expect("too many samples specified for texture")))
            .usage(vk::ImageUsageFlags::from_bits(desc.usage.bits()).expect("invalid texture usage flags")
                | vk::ImageUsageFlags::TRANSFER_DST)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let alloc_info = vma::AllocationOptions {
            usage: vma::MemoryUsage::AutoPreferDevice,
            preferred_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ..Default::default()
        };
        let (image, allocation) = unsafe {
            self.allocator.as_ref().unwrap().create_image(image_info, &alloc_info)? };

        let aspect = Texture::aspect_flags(desc.format);
        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(aspect)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let transition = vk::ImageMemoryBarrier2::builder()
            .image(image)
            .subresource_range(subresource_range)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::from_raw(desc.layout as i32))
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let transitions = [transition];
        let dep_info = vk::DependencyInfoKHR::builder()
            .image_memory_barriers(&transitions);
        unsafe { self.device.cmd_pipeline_barrier2_khr(cmd_buf.buffer, &dep_info) };

        Ok(Texture {
            dimensions: desc.dimensions,
            format: desc.format,
            aspect,
            ty: desc.ty,
            image,
            allocation,
            gpu: self,
            view: Cell::new(vk::ImageView::null()),
        })
    }

    pub fn sampler_descriptor_size(&self) -> usize {
        self.descriptor_sizes.sampler
    }
    pub fn sampler_descriptor(&self, desc: SamplerDesc, descriptor: &mut [u8]) -> Result<()> {
        assert_eq!(descriptor.len(), self.descriptor_sizes.sampler,
                   "incorrect buffer size for sampler descriptor");

        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::from_raw(desc.mag_filter as i32))
            .min_filter(vk::Filter::from_raw(desc.min_filter as i32))
            .mipmap_mode(vk::SamplerMipmapMode::from_raw(desc.mip_mode as i32))
            .address_mode_u(vk::SamplerAddressMode::from_raw(desc.wrap_u as i32))
            .address_mode_v(vk::SamplerAddressMode::from_raw(desc.wrap_v as i32))
            .address_mode_w(vk::SamplerAddressMode::from_raw(desc.wrap_w as i32));
        let sampler = unsafe { self.device.create_sampler(&sampler_info, None) }?;

        let image_info = vk::DescriptorImageInfo::builder()
            .sampler(sampler)
            .image_view(vk::ImageView::null())
            .image_layout(vk::ImageLayout::GENERAL)
            .build();
        let info = vk::DescriptorGetInfoEXT::builder()
            .type_(vk::DescriptorType::SAMPLER)
            .data(vk::DescriptorDataEXT {
                sampled_image: &raw const image_info,
            });
        unsafe { self.device.get_descriptor_ext(&info, descriptor) };

        unsafe { self.device.destroy_sampler(sampler, None) };

        Ok(())
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

    pub fn next_swapchain_image(&self, ctx: &dyn SwapchainContext,
                                cmd_buf: &mut CommandBuffer<'_>) -> Result<Target<'_>> {
        let fence_info = vk::FenceCreateInfo::builder();
        let fence = unsafe { self.device.create_fence(&fence_info, None)? };

        let mut swapchain = self.swapchain.borrow_mut();

        let acquire_result = unsafe {
            self.device.acquire_next_image_khr(
                swapchain.swapchain,
                u64::MAX,
                vk::Semaphore::null(),
                fence,
            )
        };
        let next_image_index = match acquire_result {
            Ok((_, vk::SuccessCode::SUBOPTIMAL_KHR)) | Err(vk::ErrorCode::OUT_OF_DATE_KHR) => {
                unsafe { self.device.destroy_fence(fence, None) };
                self.recreate_swapchain(ctx)?;
                return self.next_swapchain_image(ctx, cmd_buf);
            },
            Ok((index, _)) => index as usize,
            Err(e) => Err(e)?,
        };

        swapchain.image_index = next_image_index;

        unsafe {
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.destroy_fence(fence, None);
        }

        cmd_buf.image_barrier(
            vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            swapchain.images[next_image_index],
            vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            true
        );

        Ok(Target {
            view: TextureView {
                view: swapchain.image_views[next_image_index],
                pd: PhantomData,
            },
            load_op: Load::Clear,
            store_op: Store::Store,
            clear_value: ClearValue::Color([0.0, 0.0, 0.0, 1.0]),
        })
    }

    pub fn swapchain_present(&self, wait_semaphore: &Semaphore<'_>, wait_value: u64) -> Result<()> {
        let swapchain = self.swapchain.borrow();

        let image_index = swapchain.image_index;
        let present_semaphore = swapchain.present_semaphores[image_index];

        let graphics_queue = self.queues.graphics(self);
        let mut cmd_buf = graphics_queue.create_buffer()?;
        cmd_buf.begin_recording()?;

        cmd_buf.image_barrier(
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR,
            swapchain.images[image_index],
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            false
        );

        cmd_buf.end_recording()?;

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
            self.device.queue_present_khr(self.queues.present(), &info)?;
        }

        Ok(())
    }
}
