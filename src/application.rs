use anyhow::Result;

pub struct Application<'a> {
    gpu: &'a gpu::Gpu,
    pipeline: gpu::Pipeline<'a>,
    render_pass: gpu::RenderPass<'a>,
    queue: gpu::Queue<'a>,
    command_buffer: gpu::CommandBuffer<'a>,
    present_queue: gpu::Queue<'a>,
    image_available_semaphore: gpu::Semaphore<'a>,
    render_finished_semaphore: gpu::Semaphore<'a>,
}

impl<'a> Application<'a> {
    pub fn new(gpu: &'a gpu::Gpu) -> Result<Self> {
        let render_pass = gpu.create_render_pass()?;
        let queue = gpu.create_queue(gpu::QueueType::Graphics, 0)?;
        Ok(Self {
            gpu,
            pipeline: gpu.create_graphics_pipeline(
                include_bytes!("../shaders/vert.spv"),
                include_bytes!("../shaders/frag.spv"),
                gpu::RasterDesc {
                    ..Default::default()
                },
                &render_pass,
            )?,
            render_pass,
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
        command_buffer.begin_render_pass(&self.render_pass, image_index);
        command_buffer.set_pipeline(&self.pipeline);
        command_buffer.draw_instanced(3, 1, 0, 0);
        command_buffer.end_render_pass();

        command_buffer.end_recording()?;

        self.queue.submit(&self.command_buffer, &self.image_available_semaphore, &self.render_finished_semaphore)?;

        self.gpu.swapchain_present(image_index, &self.present_queue, &self.render_finished_semaphore)?;

        Ok(())
    }
}
