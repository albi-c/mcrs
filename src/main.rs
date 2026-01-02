mod application;

use std::collections::HashSet;
use std::ffi::{c_void, CStr};
use anyhow::{anyhow, Result};
use vulkanalia::{vk, Entry, Instance};
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::vk::{EntryV1_0, ExtDebugUtilsExtensionInstanceCommands, HasBuilder, InstanceV1_0};
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::{Window, WindowBuilder};
use crate::application::Application;

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

fn create_instance(window: &Window, entry: &Entry) -> Result<(Instance, Option<vk::DebugUtilsMessengerEXT>)> {
    let app_info = vk::ApplicationInfo::builder()
        .application_name(b"MCRS")
        .application_version(vk::make_version(0, 0, 1))
        .engine_name(b"No Engine")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    let mut extensions = vulkanalia::window::get_required_instance_extensions(window)
        .iter()
        .map(|ext| ext.as_ptr())
        .collect::<Vec<_>>();

    let flags = if gpu::need_portability_ext(entry.version()?) {
        log::debug!("enabling compatibility extensions for macos");
        extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name.as_ptr());
        extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    if gpu::validation_enabled() {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    let available_layers = unsafe { entry.enumerate_instance_layer_properties() }?
        .iter()
        .map(|layer| layer.layer_name)
        .collect::<HashSet<_>>();

    if gpu::validation_enabled() && !available_layers.contains(&gpu::VALIDATION_LAYER) {
        return Err(anyhow!("validation layer not supported"));
    }

    let (layers, mut debug_info) = if gpu::validation_enabled() {
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE)
            .user_callback(Some(vulkan_debug_callback));
        (vec![gpu::VALIDATION_LAYER.as_ptr()], Some(debug_info))
    } else {
        (vec![], None)
    };

    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    if gpu::validation_enabled() {
        info = info.push_next(debug_info.as_mut().unwrap());
    }

    let instance = unsafe { entry.create_instance(&info, None)? };

    if gpu::validation_enabled() {
        let messenger = unsafe { instance.create_debug_utils_messenger_ext(&debug_info.unwrap(), None)? };
        Ok((instance, Some(messenger)))
    } else {
        Ok((instance, None))
    }
}

struct App {
    instance: Instance,
    messenger: Option<vk::DebugUtilsMessengerEXT>,
    gpu: Box<gpu::Gpu>,
}

impl App {
    fn create(window: &Window) -> Result<Self> {
        let loader = unsafe { LibloadingLoader::new(LIBRARY)? };
        let entry = unsafe { Entry::new(loader) }.map_err(|e| anyhow!("{}", e))?;
        let (instance, messenger) = create_instance(window, &entry)?;
        let surface = unsafe { vulkanalia::window::create_surface(&instance, &window, &window)? };
        let size = window.inner_size();
        let gpu = Box::new(gpu::Gpu::new(&instance, surface, (size.width, size.height))?);
        Ok(Self {
            instance,
            messenger,
            gpu,
        })
    }

    fn render(&self, application: &mut Application) -> Result<()> {
        application.render()
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.gpu.destroy(&self.instance);
            if gpu::validation_enabled() {
                if let Some(messenger) = self.messenger {
                    self.instance.destroy_debug_utils_messenger_ext(messenger, None);
                }
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn main() -> Result<()> {
    let mut event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("MCRS")
        .with_inner_size(LogicalSize::new(1280, 720))
        .build(&event_loop)?;

    let app = App::create(&window)?;
    let mut application = Application::new(&app.gpu)?;
    event_loop.run_on_demand(|event, target| {
        match event {
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::RedrawRequested => {
                    if !target.exiting() {
                        app.render(&mut application).unwrap();
                    }
                },
                WindowEvent::CloseRequested => target.exit(),
                _ => {},
            },
            _ => {},
        }
    })?;

    Ok(())
}
