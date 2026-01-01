use std::collections::HashSet;
use anyhow::{anyhow, Result};
use vulkanalia::{vk, Device, Instance};
use vulkanalia::vk::{Handle, HasBuilder, InstanceV1_0, KhrSurfaceExtensionInstanceCommands, KhrSwapchainExtensionDeviceCommands};
use crate::{need_portability_ext, VALIDATION_ENABLED, VALIDATION_LAYER};

const FEATURE_REQUIREMENTS: &[(fn(&vk::PhysicalDeviceFeatures) -> vk::Bool32, &str)] = &[
    (|f| f.shader_int64, "shader int64"),
    (|f| f.multi_draw_indirect, "multi draw indirect"),
    (|f| f.sampler_anisotropy, "sampler anisotropy"),
];

const EXTENSION_REQUIREMENTS: &[vk::ExtensionName] = &[
    vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION.name,
    vk::KHR_SWAPCHAIN_EXTENSION.name,
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

    Ok(Swapchain {
        swapchain,
        format: surface_format.format,
        extent,
        images: unsafe { device.get_swapchain_images_khr(swapchain)? },
    })
}

fn check_device(instance: &Instance, device: vk::PhysicalDevice, surface: vk::SurfaceKHR) -> Result<(QueueFamilies, SwapchainSupport)> {
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

    Ok((queue_families, swapchain_support))
}

pub fn find_suitable_device(instance: &Instance, surface: vk::SurfaceKHR) -> Result<(vk::PhysicalDevice, QueueFamilies, SwapchainSupport)> {
    let mut devices = unsafe { instance.enumerate_physical_devices()? };
    devices.sort_unstable_by_key(|&device| rank_device(instance, device));

    let mut first_error = None;
    for device in devices {
        match check_device(instance, device, surface) {
            Ok((queue_families, swapchain_support)) => return Ok((device, queue_families, swapchain_support)),
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
}

impl Swapchain {
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_swapchain_khr(self.swapchain, None);
        }
    }
}

pub fn create_logical_device(instance: &Instance, physical_device: vk::PhysicalDevice,
                             queue_families: QueueFamilies, swapchain_support: SwapchainSupport,
                             surface: vk::SurfaceKHR, window_size: (u32, u32)) -> Result<(Device, Swapchain)> {
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

    let layers = if VALIDATION_ENABLED {
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

    let features = vk::PhysicalDeviceFeatures::builder();

    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);
    let device = unsafe { instance.create_device(physical_device, &info, None)? };

    let swapchain = create_swapchain(instance, physical_device, surface, window_size, queue_families, &device)?;

    Ok((device, swapchain))
}
