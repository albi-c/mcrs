use anyhow::Result;

pub struct Application<'a> {
    gpu: &'a gpu::Gpu,
    vertex_shader: gpu::Shader<'a>,
    pixel_shader: gpu::Shader<'a>,
    // pipeline: gpu::Pipeline<'a>,
    queue: gpu::Queue<'a>,
    command_buffer: gpu::CommandBuffer<'a>,
    present_queue: gpu::Queue<'a>,
    image_available_semaphore: gpu::Semaphore<'a>,
    render_finished_semaphore: gpu::Semaphore<'a>,
}

impl<'a> Application<'a> {
    pub fn new(gpu: &'a gpu::Gpu) -> Result<Self> {
        let queue = gpu.create_queue(gpu::QueueType::Graphics, 0)?;
        Ok(Self {
            gpu,
            vertex_shader: gpu.create_shader(include_bytes!("../shaders/vert.spv"), gpu::ShaderStage::Vertex)?,
            pixel_shader: gpu.create_shader(include_bytes!("../shaders/frag.spv"), gpu::ShaderStage::Pixel)?,
            // pipeline: gpu.create_graphics_pipeline(
            //     include_bytes!("../shaders/vert.spv"),
            //     include_bytes!("../shaders/frag.spv"),
            //     gpu::RasterDesc {
            //         ..Default::default()
            //     },
            // )?,
            command_buffer: queue.create_buffer()?,
            queue,
            present_queue: gpu.create_queue(gpu::QueueType::Present, 0)?,
            image_available_semaphore: gpu.create_semaphore()?,
            render_finished_semaphore: gpu.create_semaphore()?,
        })
    }

    pub fn render(&mut self) -> Result<()> {
        let command_buffer = &mut self.command_buffer;

        command_buffer.begin_recording()?;

        let image_index = self.gpu.next_swapchain_image(&self.image_available_semaphore)?;
        let render_pass_desc = gpu::RenderPassDesc {
            ..Default::default()
        };
        command_buffer.begin_render_pass(&render_pass_desc, image_index);
        // command_buffer.set_pipeline(&self.pipeline);
        command_buffer.bind_shaders([&self.vertex_shader, &self.pixel_shader]);
        command_buffer.draw_instanced(3, 1, 0, 0);
        command_buffer.end_render_pass(image_index);

        command_buffer.end_recording()?;

        self.queue.submit(&self.command_buffer, &self.image_available_semaphore, &self.render_finished_semaphore)?;

        self.gpu.swapchain_present(image_index, &self.present_queue, &self.render_finished_semaphore)?;

        Ok(())
    }
}
