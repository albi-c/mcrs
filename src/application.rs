use anyhow::Result;

pub struct Application<'a> {
    gpu: &'a gpu::Gpu,
    pipeline: gpu::Pipeline<'a>,
    render_pass: gpu::RenderPass<'a>,
}

impl<'a> Application<'a> {
    pub fn new(gpu: &'a gpu::Gpu) -> Result<Self> {
        let render_pass = gpu.create_render_pass()?;
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
        })
    }

    pub fn render(&mut self) -> Result<()> {
        let _ = self.gpu.get_queue(gpu::QueueType::Graphics, 0);

        Ok(())
    }
}
