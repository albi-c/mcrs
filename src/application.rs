use anyhow::Result;

const FRAMES_IN_FLIGHT: u64 = 2;

pub struct Application<'a> {
    gpu: &'a gpu::Gpu,
    vertex_shader: gpu::Shader<'a>,
    pixel_shader: gpu::Shader<'a>,
    queue: gpu::Queue<'a>,
    frame_semaphore: gpu::Semaphore<'a>,
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
            next_frame: 1,
        })
    }

    pub fn render(&mut self) -> Result<()> {
        if self.next_frame > FRAMES_IN_FLIGHT {
            self.frame_semaphore.wait(self.next_frame - FRAMES_IN_FLIGHT)?;
        }

        let mut command_buffer = self.queue.create_buffer()?;

        command_buffer.begin_recording()?;

        let image_index = self.gpu.next_swapchain_image()?;
        let render_pass_desc = gpu::RenderPassDesc {
            ..Default::default()
        };
        command_buffer.begin_render_pass(&render_pass_desc, image_index);
        command_buffer.bind_shaders([&self.vertex_shader, &self.pixel_shader]);
        command_buffer.draw_instanced(3, 1, 0, 0);
        command_buffer.end_render_pass(image_index);

        command_buffer.end_recording()?;

        self.queue.submit(&command_buffer, &self.frame_semaphore, self.next_frame)?;
        self.gpu.swapchain_present(&self.frame_semaphore, self.next_frame)?;

        self.next_frame += 1;

        Ok(())
    }
}
