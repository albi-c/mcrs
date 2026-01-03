use anyhow::Result;
use glam::Vec4;
use gpu::{MemoryAllocation, MemoryAllocator};

const FRAMES_IN_FLIGHT: u64 = 2;

pub struct Application<'a> {
    gpu: &'a gpu::Gpu,
    vertex_shader: gpu::Shader<'a>,
    pixel_shader: gpu::Shader<'a>,
    queue: gpu::Queue<'a>,
    frame_semaphore: gpu::Semaphore<'a>,
    frame_arenas: Box<[gpu::Arena<'a>]>,
    next_frame: u64,
}

impl<'a> Application<'a> {
    pub fn new(gpu: &'a gpu::Gpu) -> Result<Self> {
        let queue = gpu.create_queue(gpu::QueueType::Graphics)?;
        Ok(Self {
            gpu,
            vertex_shader: gpu.create_shader(include_bytes!("../shaders/vert.spv"), gpu::ShaderStage::Vertex)?,
            pixel_shader: gpu.create_shader(include_bytes!("../shaders/frag.spv"), gpu::ShaderStage::Pixel)?,
            queue,
            frame_semaphore: gpu.create_semaphore(0)?,
            frame_arenas: (0..FRAMES_IN_FLIGHT)
                .map(|_| gpu.create_arena(1024 * 1024))
                .collect::<Result<_>>()?,
            next_frame: 1,
        })
    }

    fn get_frame_arena(&self) -> &gpu::Arena<'_> {
        &self.frame_arenas[(self.next_frame % FRAMES_IN_FLIGHT) as usize]
    }

    pub fn render(&mut self, ctx: &dyn gpu::SwapchainContext) -> Result<()> {
        if self.next_frame > FRAMES_IN_FLIGHT {
            self.frame_semaphore.wait(self.next_frame - FRAMES_IN_FLIGHT)?;
        }

        let arena = self.get_frame_arena();
        arena.reset();

        let mut indices = arena.alloc(3)?;
        indices.host_mut()[0..3].copy_from_slice(&[0u32, 1, 2]);

        let mut pixel_data = arena.alloc(2)?;
        pixel_data.host_mut()[0..2].copy_from_slice(&[
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(1.0, 1.0, 0.5, 1.0),
        ]);

        let mut command_buffer = self.queue.create_buffer()?;

        command_buffer.begin_recording()?;

        let image_index = self.gpu.next_swapchain_image(ctx)?;
        let render_pass_desc = gpu::RenderPassDesc {
            ..Default::default()
        };
        command_buffer.begin_render_pass(&render_pass_desc, image_index);
        command_buffer.bind_shaders([&self.vertex_shader, &self.pixel_shader]);
        command_buffer.draw_indexed(
            gpu::DevicePointer::null(), pixel_data.device(),
            indices.device(), 3, gpu::IndexType::U32);
        command_buffer.end_render_pass(image_index);

        command_buffer.end_recording()?;

        self.queue.submit(command_buffer, &self.frame_semaphore, self.next_frame)?;
        self.gpu.swapchain_present(&self.frame_semaphore, self.next_frame)?;

        self.next_frame += 1;

        Ok(())
    }
}
