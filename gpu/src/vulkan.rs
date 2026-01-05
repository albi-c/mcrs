use std::alloc::Layout;
use std::cell::RefCell;
use std::collections::{HashSet, LinkedList};
use std::ffi::{c_void, CStr};
use anyhow::{anyhow, Result};
use vulkanalia::{vk, Device, Instance};
use vulkanalia::vk::{DeviceV1_0, ExtShaderObjectExtensionDeviceCommands, HasBuilder, InstanceV1_0, InstanceV1_1, KhrSurfaceExtensionInstanceCommands, KhrSwapchainExtensionDeviceCommands, KhrTimelineSemaphoreExtensionDeviceCommands};
use crate::{need_portability_ext, validation_enabled, Gpu, Queue, Shader, ShaderStage, VALIDATION_LAYER};

const FEATURE_REQUIREMENTS: &[(fn(&vk::PhysicalDeviceFeatures) -> vk::Bool32, &str)] = &[
    (|f| f.shader_int64, "shader int64"),
    (|f| f.multi_draw_indirect, "multi draw indirect"),
    (|f| f.sampler_anisotropy, "sampler anisotropy"),
];

const EXTENSION_REQUIREMENTS: &[vk::ExtensionName] = &[
    vk::KHR_SYNCHRONIZATION2_EXTENSION.name,
    vk::KHR_DEVICE_GROUP_EXTENSION.name,
    vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION.name,
    vk::KHR_SWAPCHAIN_EXTENSION.name,
    vk::KHR_MULTIVIEW_EXTENSION.name,
    vk::KHR_MAINTENANCE2_EXTENSION.name,
    vk::KHR_CREATE_RENDERPASS2_EXTENSION.name,
    vk::KHR_DEPTH_STENCIL_RESOLVE_EXTENSION.name,
    vk::KHR_DYNAMIC_RENDERING_EXTENSION.name,
    vk::EXT_SHADER_OBJECT_EXTENSION.name,
    vk::KHR_TIMELINE_SEMAPHORE_EXTENSION.name,
    vk::KHR_MAINTENANCE3_EXTENSION.name,
    vk::EXT_DESCRIPTOR_INDEXING_EXTENSION.name,
    vk::EXT_DESCRIPTOR_BUFFER_EXTENSION.name,
    vk::KHR_GET_MEMORY_REQUIREMENTS2_EXTENSION.name,
];

pub fn create_semaphore(device: &Device, value: u64) -> Result<vk::Semaphore> {
    let mut init_info = vk::SemaphoreTypeCreateInfo::builder()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(value);
    let info = vk::SemaphoreCreateInfo::builder()
        .push_next(&mut init_info);
    Ok( unsafe { device.create_semaphore(&info, None) }?)
}

#[derive(Debug)]
pub struct PooledCommandBuffer {
    item: LinkedList<(vk::CommandBuffer, vk::Semaphore, u64)>,
}

impl PooledCommandBuffer {
    pub fn data(&self) -> &(vk::CommandBuffer, vk::Semaphore, u64) {
        self.item.front().unwrap()
    }

    pub fn data_mut(&mut self) -> &mut (vk::CommandBuffer, vk::Semaphore, u64) {
        self.item.front_mut().unwrap()
    }
}

#[derive(Debug)]
pub struct CommandBufferPool {
    pool: vk::CommandPool,
    buffers: RefCell<LinkedList<(vk::CommandBuffer, vk::Semaphore, u64)>>,
    length: usize,
}

impl CommandBufferPool {
    const GROW_BUFFERS: usize = 8;
    const MAX_BUFFERS: usize = 128;

    fn create_buffers(device: &Device, pool: vk::CommandPool) -> Result<LinkedList<(vk::CommandBuffer, vk::Semaphore, u64)>> {
        let info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(Self::GROW_BUFFERS as u32);
        Ok(unsafe { device.allocate_command_buffers(&info)? }
            .into_iter()
            .map(|buffer| {
                Ok((buffer, create_semaphore(device, 0)?, 0))
            })
            .collect::<Result<LinkedList<_>>>()?)
    }

    pub fn new(device: &Device, queue_family: u32) -> Result<Self> {
        let info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family);
        let pool = unsafe { device.create_command_pool(&info, None)? };
        let buffers = Self::create_buffers(device, pool)?;
        Ok(Self {
            pool,
            length: buffers.len(),
            buffers: RefCell::new(buffers),
        })
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            for (buffer, semaphore, _) in self.buffers.take() {
                device.reset_command_buffer(buffer, vk::CommandBufferResetFlags::RELEASE_RESOURCES)
                    .expect("failed to reset command buffer");
                device.free_command_buffers(self.pool, &[buffer]);
                device.destroy_semaphore(semaphore, None);
            }

            device.destroy_command_pool(self.pool, None);
        }
    }

    fn next_command_buffer(&self, device: &Device) -> Result<PooledCommandBuffer> {
        let mut buffers = self.buffers.borrow_mut();
        let mut cur = buffers.cursor_front_mut();
        loop {
            let Some((_, semaphore, count)) = cur.current() else { break };
            if unsafe { device.get_semaphore_counter_value_khr(*semaphore)? } >= *count {
                let item = cur.remove_current_as_list().unwrap();
                return Ok(PooledCommandBuffer {
                    item,
                });
            }
            cur.move_next();
        }
        if self.length >= Self::MAX_BUFFERS {
            return Err(anyhow!("out of command buffers"));
        }
        let mut new_buffers = Self::create_buffers(device, self.pool)?;
        let item = new_buffers.cursor_front_mut().remove_current_as_list().unwrap();
        buffers.append(&mut new_buffers);
        Ok(PooledCommandBuffer {
            item,
        })
    }

    pub fn acquire(&self, device: &Device) -> Result<PooledCommandBuffer> {
        let buffer = self.next_command_buffer(device)?;
        unsafe { device.reset_command_buffer(buffer.data().0, vk::CommandBufferResetFlags::empty())? };
        Ok(buffer)
    }

    pub fn release(&self, _device: &Device, buffer: PooledCommandBuffer) {
        self.buffers.borrow_mut().cursor_back_mut().splice_after(buffer.item);
    }
}

#[derive(Debug, Copy, Clone)]
pub struct QueueFamilies {
    pub graphics: u32,
    pub present: u32,
}

#[derive(Debug)]
pub struct Queues {
    graphics: vk::Queue,
    present: vk::Queue,
    graphics_pool: CommandBufferPool,
}

impl Queues {
    pub fn new(device: &Device, families: QueueFamilies) -> Result<Self> {
        unsafe {
            Ok(Self {
                graphics: device.get_device_queue(families.graphics, 0),
                present: device.get_device_queue(families.present, 0),
                graphics_pool: CommandBufferPool::new(device, families.graphics)?,
            })
        }
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe { self.graphics_pool.destroy(device) };
    }

    pub fn graphics<'a>(&'a self, gpu: &'a Gpu) -> Queue<'a> {
        Queue {
            queue: self.graphics,
            command_pool: &self.graphics_pool,
            gpu: Some(gpu),
        }
    }

    pub fn present(&self) -> vk::Queue {
        self.present
    }
}

#[derive(Debug)]
pub struct PipelineLayout {
    pub layout: vk::PipelineLayout,
    pub set_layouts: [vk::DescriptorSetLayout; 3],
}

impl PipelineLayout {
    pub const MAX_TEXTURES: u32 = 65536;
    pub const MAX_TEXTURES_RW: u32 = 8192;
    pub const MAX_SAMPLERS: u32 = 1024;

    fn create_descriptor_set_layout(device: &Device, ty: vk::DescriptorType, count: u32) -> Result<vk::DescriptorSetLayout> {
        let binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(ty)
            .descriptor_count(count)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);
        let bindings = [binding];
        let info = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(vk::DescriptorSetLayoutCreateFlags::DESCRIPTOR_BUFFER_EXT)
            .bindings(&bindings);
        Ok(unsafe { device.create_descriptor_set_layout(&info, None)? })
    }

    pub fn new(device: &Device) -> Result<Self> {
        let push_constant_ranges = [
            vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .size(16),
        ];
        let set_layouts = [
            Self::create_descriptor_set_layout(device, vk::DescriptorType::SAMPLED_IMAGE, Self::MAX_TEXTURES)?,
            Self::create_descriptor_set_layout(device, vk::DescriptorType::STORAGE_IMAGE, Self::MAX_TEXTURES_RW)?,
            Self::create_descriptor_set_layout(device, vk::DescriptorType::SAMPLER, Self::MAX_SAMPLERS)?,
        ];
        let info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);
        let layout = unsafe { device.create_pipeline_layout(&info, None)? };
        Ok(Self {
            layout,
            set_layouts,
        })
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            for layout in self.set_layouts {
                device.destroy_descriptor_set_layout(layout, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

#[derive(Debug)]
pub struct DescriptorSizes {
    pub sampled_texture: usize,
    pub storage_texture: usize,
    pub sampler: usize,
}

impl DescriptorSizes {
    pub fn new(instance: &Instance, device: vk::PhysicalDevice) -> Result<Self> {
        let prop = {
            let mut properties_ext = vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
            let mut properties = vk::PhysicalDeviceProperties2::builder()
                .push_next(&mut properties_ext);
            unsafe { instance.get_physical_device_properties2(device, &mut properties) };
            properties_ext
        };
        Ok(Self {
            sampled_texture: prop.sampled_image_descriptor_size,
            storage_texture: prop.storage_image_descriptor_size,
            sampler: prop.sampler_descriptor_size,
        })
    }
}

fn rank_device(instance: &Instance, device: vk::PhysicalDevice) -> usize {
    let properties = unsafe { instance.get_physical_device_properties(device) };
    match properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 1,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 3,
        vk::PhysicalDeviceType::CPU => 4,
        vk::PhysicalDeviceType::OTHER => 5,
        _ => 6,
    }
}

#[derive(Debug)]
pub struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    fn get(instance: &Instance, device: vk::PhysicalDevice, surface: vk::SurfaceKHR) -> Result<Self> {
        unsafe { Ok(Self {
            capabilities: instance.get_physical_device_surface_capabilities_khr(device, surface)?,
            formats: instance.get_physical_device_surface_formats_khr(device, surface)?,
            present_modes: instance.get_physical_device_surface_present_modes_khr(device, surface)?,
        }) }
    }
}

pub fn create_swapchain(instance: &Instance, physical_device: vk::PhysicalDevice, surface: vk::SurfaceKHR,
                        window_size: (u32, u32), queue_families: QueueFamilies, device: &Device,
                        old_swapchain: vk::SwapchainKHR) -> Result<Swapchain> {
    let support = SwapchainSupport::get(instance, physical_device, surface)?;

    let surface_format = support.formats
        .iter()
        .find(|f| f.format == vk::Format::B8G8R8A8_SRGB && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .unwrap_or(&support.formats[0]);

    let present_mode = support.present_modes
        .iter()
        .copied()
        .find(|&m| m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO);

    let extent = if support.capabilities.current_extent.width != u32::MAX {
        support.capabilities.current_extent
    } else {
        vk::Extent2D::builder()
            .width(window_size.0.clamp(
                support.capabilities.min_image_extent.width,
                support.capabilities.max_image_extent.width,
            ))
            .height(window_size.1.clamp(
                support.capabilities.min_image_extent.height,
                support.capabilities.max_image_extent.height,
            ))
            .build()
    };

    let image_count = if support.capabilities.max_image_count == 0 {
        support.capabilities.min_image_count + 1
    } else {
        (support.capabilities.min_image_count + 1).min(support.capabilities.max_image_count)
    };

    let (family_indices, sharing_mode) = if queue_families.graphics != queue_families.present {
        (vec![queue_families.graphics, queue_families.present], vk::SharingMode::CONCURRENT)
    } else {
        (vec![], vk::SharingMode::EXCLUSIVE)
    };

    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(sharing_mode)
        .queue_family_indices(&family_indices)
        .pre_transform(support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(old_swapchain);

    let swapchain = unsafe { device.create_swapchain_khr(&info, None)? };

    let images = unsafe { device.get_swapchain_images_khr(swapchain)? };
    let image_views = create_swapchain_image_views(device, &images, surface_format.format)?;

    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let present_semaphores = images.iter()
        .map(|_| unsafe { device.create_semaphore(&semaphore_info, None) })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Swapchain {
        swapchain,
        format: surface_format.format,
        extent,
        images,
        image_views,
        present_semaphores,
        image_index: 0,
    })
}

fn create_swapchain_image_views(device: &Device, images: &[vk::Image], format: vk::Format) -> Result<Vec<vk::ImageView>> {
    let views = images.iter()
        .map(|i| {
            let components = vk::ComponentMapping::builder()
                .r(vk::ComponentSwizzle::IDENTITY)
                .g(vk::ComponentSwizzle::IDENTITY)
                .b(vk::ComponentSwizzle::IDENTITY)
                .a(vk::ComponentSwizzle::IDENTITY);
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            let info = vk::ImageViewCreateInfo::builder()
                .image(*i)
                .view_type(vk::ImageViewType::_2D)
                .format(format)
                .components(components)
                .subresource_range(subresource_range);
            unsafe { device.create_image_view(&info, None) }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(views)
}

fn check_device(instance: &Instance, device: vk::PhysicalDevice, surface: vk::SurfaceKHR) -> Result<QueueFamilies> {
    let device_properties = unsafe { instance.get_physical_device_properties(device) };

    let features = unsafe { instance.get_physical_device_features(device) };
    for (test, msg) in FEATURE_REQUIREMENTS {
        if test(&features) != vk::TRUE {
            return Err(anyhow!("gpu '{}' missing {} support", device_properties.device_name, msg))
        }
    }

    let extensions = unsafe { instance.enumerate_device_extension_properties(device, None)? }
        .into_iter()
        .map(|ext| ext.extension_name)
        .collect::<HashSet<_>>();

    for ext in EXTENSION_REQUIREMENTS {
        if !extensions.contains(ext) {
            return Err(anyhow!("gpu '{}' missing extension: {}", device_properties.device_name, ext.to_string_lossy()));
        }
    }

    let properties = unsafe { instance.get_physical_device_queue_family_properties(device) };
    let graphics = properties.iter()
        .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER))
        .ok_or_else(|| anyhow!("missing gpu graphics queue"))? as u32;
    let present = properties.iter()
        .enumerate()
        .try_find(|&(i, _)| unsafe { instance.get_physical_device_surface_support_khr(device, i as u32, surface) })?
        .ok_or_else(|| anyhow!("missing gpu present queue"))?.0 as u32;

    let swapchain_support = SwapchainSupport::get(instance, device, surface)?;
    if swapchain_support.formats.is_empty() || swapchain_support.present_modes.is_empty() {
        return Err(anyhow!("insufficient swapchain capabilities"));
    }

    let queue_families = QueueFamilies {
        graphics,
        present,
    };

    Ok(queue_families)
}

pub fn find_suitable_device(instance: &Instance, surface: vk::SurfaceKHR) -> Result<(vk::PhysicalDevice, QueueFamilies)> {
    let mut devices = unsafe { instance.enumerate_physical_devices()? };
    devices.sort_unstable_by_key(|&device| rank_device(instance, device));

    let mut first_error = None;
    for device in devices {
        match check_device(instance, device, surface) {
            Ok(queue_families) => return Ok((device, queue_families)),
            Err(err) => {
                first_error.get_or_insert(err);
            },
        }
    }
    Err(first_error.unwrap_or_else(|| anyhow!("no gpu found")))
}

#[derive(Debug)]
pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub present_semaphores: Vec<vk::Semaphore>,
    pub image_index: usize,
}

impl Swapchain {
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            for view in self.image_views.drain(..) {
                device.destroy_image_view(view, None);
            }
            for sem in self.present_semaphores.drain(..) {
                device.destroy_semaphore(sem, None);
            }
            device.destroy_swapchain_khr(self.swapchain, None);
        }
    }
}

extern "system" fn vulkan_debug_callback(severity: vk::DebugUtilsMessageSeverityFlagsEXT,
                                         ty: vk::DebugUtilsMessageTypeFlagsEXT,
                                         data: *const vk::DebugUtilsMessengerCallbackDataEXT,
                                         _: *mut c_void) -> vk::Bool32 {
    let data = unsafe { &*data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        log::error!("[{:?}] {}", ty, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        log::warn!("[{:?}] {}", ty, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        log::debug!("[{:?}] {}", ty, message);
    } else {
        log::trace!("[{:?}] {}", ty, message);
    }

    vk::FALSE
}

pub fn create_debug_info_callback() -> vk::DebugUtilsMessengerCreateInfoEXTBuilder<'static> {
    let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE)
        .user_callback(Some(vulkan_debug_callback));
    debug_info
}

pub fn create_logical_device(instance: &Instance, physical_device: vk::PhysicalDevice,
                             queue_families: QueueFamilies) -> Result<Device> {
    let unique_indices = HashSet::from([queue_families.graphics, queue_families.present]);
    let queue_priorities = [1.0];
    let queue_infos = unique_indices
        .into_iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(i)
                .queue_priorities(&queue_priorities)
        })
        .collect::<Vec<_>>();

    let layers = if validation_enabled() {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

    let mut extensions = EXTENSION_REQUIREMENTS
        .iter()
        .map(|ext| ext.as_ptr())
        .collect::<Vec<_>>();

    if need_portability_ext(instance.version()) {
        extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
    }

    let available_layers = unsafe { instance.enumerate_device_layer_properties(physical_device)? }
        .iter()
        .map(|layer| layer.layer_name)
        .collect::<HashSet<_>>();

    if validation_enabled() && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("validation layer not supported"));
    }

    let features = vk::PhysicalDeviceFeatures::builder()
        .shader_int64(true)
        .multi_draw_indirect(true)
        .sampler_anisotropy(true);

    let mut info_12 = vk::PhysicalDeviceVulkan12Features::builder()
        .runtime_descriptor_array(true)
        .buffer_device_address(true)
        .timeline_semaphore(true)
        .descriptor_indexing(true)
        .shader_sampled_image_array_non_uniform_indexing(true);

    let mut info_13 = vk::PhysicalDeviceVulkan13Features::builder()
        .dynamic_rendering(true)
        .synchronization2(true);

    let mut info_shader_object = vk::PhysicalDeviceShaderObjectFeaturesEXT::builder()
        .shader_object(true);

    let mut info_descriptor_buffer = vk::PhysicalDeviceDescriptorBufferFeaturesEXT::builder()
        .descriptor_buffer(true);

    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features)
        .push_next(&mut info_12)
        .push_next(&mut info_13)
        .push_next(&mut info_shader_object)
        .push_next(&mut info_descriptor_buffer);
    let device = unsafe { instance.create_device(physical_device, &info, None)? };

    Ok(device)
}

// fn get_topology(topology: Topology) -> vk::PrimitiveTopology {
//     match topology {
//         Topology::PointList => vk::PrimitiveTopology::POINT_LIST,
//         Topology::LineList => vk::PrimitiveTopology::LINE_LIST,
//         Topology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
//         Topology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
//         Topology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
//         Topology::TriangleFan => vk::PrimitiveTopology::TRIANGLE_FAN,
//     }
// }
//
// fn get_cull_mode(cull: Cull) -> vk::CullModeFlags {
//     match cull {
//         Cull::None => vk::CullModeFlags::NONE,
//         Cull::CCW => vk::CullModeFlags::BACK,
//         Cull::CW => vk::CullModeFlags::BACK,
//         Cull::All => vk::CullModeFlags::FRONT_AND_BACK,
//     }
// }
//
// fn get_front_face(cull: Cull) -> vk::FrontFace {
//     match cull {
//         Cull::None => vk::FrontFace::CLOCKWISE,
//         Cull::CCW => vk::FrontFace::CLOCKWISE,
//         Cull::CW => vk::FrontFace::COUNTER_CLOCKWISE,
//         Cull::All => vk::FrontFace::CLOCKWISE,
//     }
// }

pub fn get_sample_count_flag(count: u8) -> vk::SampleCountFlags {
    match count {
        1 => vk::SampleCountFlags::_1,
        2 => vk::SampleCountFlags::_2,
        4 => vk::SampleCountFlags::_4,
        8 => vk::SampleCountFlags::_8,
        16 => vk::SampleCountFlags::_16,
        32 => vk::SampleCountFlags::_32,
        64 => vk::SampleCountFlags::_64,
        _ => panic!("invalid multisample count")
    }
}

pub fn get_stage(stage: ShaderStage) -> vk::ShaderStageFlags {
    match stage {
        ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
        ShaderStage::Pixel => vk::ShaderStageFlags::FRAGMENT,
    }
}

fn get_next_stage(stage: ShaderStage) -> vk::ShaderStageFlags {
    match stage {
        ShaderStage::Vertex => vk::ShaderStageFlags::FRAGMENT,
        ShaderStage::Pixel => vk::ShaderStageFlags::empty(),
    }
}

pub fn create_shader<'a>(gpu: &'a Gpu, spirv: &[u8], stage: ShaderStage) -> Result<Shader<'a>> {
    let length = (spirv.len() + 3) & !3;
    let layout = Layout::array::<u8>(length)?.align_to(4)?;
    let mem = unsafe { std::alloc::alloc(layout) };
    let buffer = unsafe { std::slice::from_raw_parts_mut(mem, length) };
    buffer[..spirv.len()].copy_from_slice(spirv);

    let vk_stage = get_stage(stage);
    let push_constant_ranges = [
        vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(16),
    ];
    let info = vk::ShaderCreateInfoEXT::builder()
        .stage(vk_stage)
        .next_stage(get_next_stage(stage))
        .code(buffer)
        .code_type(vk::ShaderCodeTypeEXT::SPIRV)
        .push_constant_ranges(&push_constant_ranges)
        .name(b"main\0")
        .set_layouts(&gpu.pipeline_layout.set_layouts);

    let infos = [info];
    let shader = unsafe { gpu.device.create_shaders_ext(&infos, None)?.0[0] };

    unsafe { std::alloc::dealloc(mem, layout) };

    Ok(Shader {
        shader,
        stage: vk_stage,
        gpu,
    })
}
