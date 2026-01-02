use std::collections::HashSet;
use anyhow::{anyhow, Result};
use vulkanalia::{vk, Device, Instance};
use vulkanalia::bytecode::Bytecode;
use vulkanalia::vk::{DeviceV1_0, Handle, HasBuilder, InstanceV1_0, KhrSurfaceExtensionInstanceCommands, KhrSwapchainExtensionDeviceCommands};
use crate::{need_portability_ext, validation_enabled, BlendDesc, Cull, Gpu, Pipeline, RasterDesc, Topology, VALIDATION_LAYER};

const FEATURE_REQUIREMENTS: &[(fn(&vk::PhysicalDeviceFeatures) -> vk::Bool32, &str)] = &[
    (|f| f.shader_int64, "shader int64"),
    (|f| f.multi_draw_indirect, "multi draw indirect"),
    (|f| f.sampler_anisotropy, "sampler anisotropy"),
];

const EXTENSION_REQUIREMENTS: &[vk::ExtensionName] = &[
    vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION.name,
    vk::KHR_SWAPCHAIN_EXTENSION.name,
    vk::KHR_DYNAMIC_RENDERING_EXTENSION.name,
];

#[derive(Debug, Copy, Clone)]
pub struct QueueFamilies {
    pub graphics: u32,
    pub present: u32,
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

fn create_swapchain(instance: &Instance, physical_device: vk::PhysicalDevice, surface: vk::SurfaceKHR,
                    window_size: (u32, u32), queue_families: QueueFamilies, device: &Device) -> Result<Swapchain> {
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
        .old_swapchain(vk::SwapchainKHR::null());

    let swapchain = unsafe { device.create_swapchain_khr(&info, None)? };

    let images = unsafe { device.get_swapchain_images_khr(swapchain)? };
    let image_views = create_swapchain_image_views(device, &images, surface_format.format)?;

    Ok(Swapchain {
        swapchain,
        format: surface_format.format,
        extent,
        images,
        image_views,
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
    let features = unsafe { instance.get_physical_device_features(device) };
    for (test, msg) in FEATURE_REQUIREMENTS {
        if test(&features) != vk::TRUE {
            return Err(anyhow!("gpu missing {} support", msg))
        }
    }

    let extensions = unsafe { instance.enumerate_device_extension_properties(device, None)? }
        .into_iter()
        .map(|ext| ext.extension_name)
        .collect::<HashSet<_>>();

    for ext in EXTENSION_REQUIREMENTS {
        if !extensions.contains(ext) {
            return Err(anyhow!("gpu missing extension: {}", ext.to_string_lossy()));
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
}

impl Swapchain {
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            for view in self.image_views.drain(..) {
                device.destroy_image_view(view, None);
            }
            device.destroy_swapchain_khr(self.swapchain, None);
        }
    }
}

pub fn create_logical_device(instance: &Instance, physical_device: vk::PhysicalDevice,
                             queue_families: QueueFamilies, surface: vk::SurfaceKHR,
                             window_size: (u32, u32)) -> Result<(Device, Swapchain)> {
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

    let features = vk::PhysicalDeviceFeatures::builder()
        .shader_int64(true)
        .multi_draw_indirect(true)
        .sampler_anisotropy(true);

    let mut dynamic_rendering = vk::PhysicalDeviceDynamicRenderingFeaturesKHR::builder()
        .dynamic_rendering(true);

    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features)
        .push_next(&mut dynamic_rendering);
    let device = unsafe { instance.create_device(physical_device, &info, None)? };

    let swapchain = create_swapchain(instance, physical_device, surface, window_size, queue_families, &device)?;

    Ok((device, swapchain))
}

fn create_shader_module(device: &Device, spirv: &[u8]) -> Result<vk::ShaderModule> {
    let bytecode = Bytecode::new(spirv)?;
    let info = vk::ShaderModuleCreateInfo::builder()
        .code(bytecode.code())
        .code_size(bytecode.code_size());

    Ok(unsafe { device.create_shader_module(&info, None)? })
}

fn create_pipeline_stage(stage: vk::ShaderStageFlags, module: vk::ShaderModule) -> vk::PipelineShaderStageCreateInfoBuilder<'static> {
    vk::PipelineShaderStageCreateInfo::builder()
        .stage(stage)
        .module(module)
        .name(b"main\0")
}

fn get_topology(topology: Topology) -> vk::PrimitiveTopology {
    match topology {
        Topology::PointList => vk::PrimitiveTopology::POINT_LIST,
        Topology::LineList => vk::PrimitiveTopology::LINE_LIST,
        Topology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        Topology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        Topology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
        Topology::TriangleFan => vk::PrimitiveTopology::TRIANGLE_FAN,
    }
}

fn get_cull_mode(cull: Cull) -> vk::CullModeFlags {
    match cull {
        Cull::None => vk::CullModeFlags::NONE,
        Cull::CCW => vk::CullModeFlags::BACK,
        Cull::CW => vk::CullModeFlags::BACK,
        Cull::All => vk::CullModeFlags::FRONT_AND_BACK,
    }
}

fn get_front_face(cull: Cull) -> vk::FrontFace {
    match cull {
        Cull::None => vk::FrontFace::CLOCKWISE,
        Cull::CCW => vk::FrontFace::CLOCKWISE,
        Cull::CW => vk::FrontFace::COUNTER_CLOCKWISE,
        Cull::All => vk::FrontFace::CLOCKWISE,
    }
}

fn get_sample_count_flag(count: u8) -> vk::SampleCountFlags {
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

fn create_blend_attachment_state(blend_desc: &BlendDesc) -> vk::PipelineColorBlendAttachmentStateBuilder {
    // TODO: the rest
    vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::from_bits(blend_desc.color_write_mask.0).unwrap())
        .blend_enable(true)
}

pub fn create_graphics_pipeline<'a>(gpu: &'a Gpu, vertex_spirv: &[u8], pixel_spirv: &[u8],
                                    raster_desc: RasterDesc<'_>) -> Result<Pipeline<'a>> {
    let vertex_module = create_shader_module(&gpu.device, vertex_spirv)?;
    let pixel_module = create_shader_module(&gpu.device, pixel_spirv)?;

    let vertex_stage = create_pipeline_stage(vk::ShaderStageFlags::VERTEX, vertex_module);
    let pixel_stage = create_pipeline_stage(vk::ShaderStageFlags::FRAGMENT, pixel_module);

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(get_topology(raster_desc.topology))
        .primitive_restart_enable(raster_desc.primitive_restart);

    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(gpu.swapchain.extent.width as f32)
        .height(gpu.swapchain.extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(gpu.swapchain.extent);

    let viewports = [viewport];
    let scissors = [scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&viewports)
        .scissors(&scissors);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(get_cull_mode(raster_desc.cull))
        .front_face(get_front_face(raster_desc.cull))
        .depth_bias_enable(false);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(true)
        .rasterization_samples(get_sample_count_flag(raster_desc.sample_count));

    let attachment = if let Some(blend_desc) = raster_desc.blend_state {
        create_blend_attachment_state(blend_desc)
    } else {
        vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(false)
    };
    let attachments = [attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(&attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    let layout_info = vk::PipelineLayoutCreateInfo::builder();

    let layout = unsafe { gpu.device.create_pipeline_layout(&layout_info, None)? };

    let color_attachment_formats = [gpu.swapchain.format];
    let mut rendering_info = vk::PipelineRenderingCreateInfoKHR::builder()
        .color_attachment_formats(&color_attachment_formats);

    let stages = [vertex_stage, pixel_stage];
    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .color_blend_state(&color_blend_state)
        .layout(layout)
        .push_next(&mut rendering_info);

    let pipeline = unsafe {
        gpu.device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?.0[0] };

    unsafe {
        gpu.device.destroy_shader_module(vertex_module, None);
        gpu.device.destroy_shader_module(pixel_module, None);
    }

    Ok(Pipeline {
        layout,
        pipeline,
        bind_point: vk::PipelineBindPoint::GRAPHICS,
        gpu,
    })
}
