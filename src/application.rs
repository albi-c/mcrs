use std::io::Cursor;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::{Mat2, Vec2, Vec3A, Vec4};
use image::{EncodableLayout, ImageReader};
use gpu::{DevicePointer, MemoryAllocation, MemoryAllocator};

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

        let img = ImageReader::new(Cursor::new(include_bytes!("../textures/wall.jpg")))
            .with_guessed_format()?.decode()?.into_rgba8();
        let img_alloc = gpu.allocator().alloc_data(img.as_bytes())?;

        let mut command_buffer = queue.create_buffer()?;

        command_buffer.begin_recording()?;

        let tex = gpu.create_texture(gpu::TextureDesc {
            ty: gpu::TextureType::Tex2D,
            dimensions: (img.width(), img.height(), 1),
            format: gpu::Format::RGBA8UNorm,
            usage: gpu::TextureUsageFlags::Sampled,
            ..Default::default()
        }, &mut command_buffer)?;
        command_buffer.copy_to_texture(img_alloc.device(), &tex);

        command_buffer.end_recording()?;

        queue.submit_no_signal(command_buffer)?.wait();

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

    pub fn render(&mut self, time: f64, ctx: &dyn gpu::SwapchainContext) -> Result<()> {
        if self.next_frame > FRAMES_IN_FLIGHT {
            self.frame_semaphore.wait(self.next_frame - FRAMES_IN_FLIGHT)?;
        }

        let arena = self.get_frame_arena();
        arena.reset();

        let indices = arena.alloc_data(&[0u32, 1, 2])?;
        let vertex_data_positions = arena.alloc_data(&[
            Vec2::new(0.0, -0.5),
            Vec2::new(0.5, 0.5),
            Vec2::new(-0.5, 0.5),
        ])?;
        let vertex_data_colors = arena.alloc_data(&[
            Vec3A::new(1.0, 0.0, 0.0),
            Vec3A::new(0.0, 1.0, 0.0),
            Vec3A::new(0.0, 0.0, 1.0),
        ])?;
        #[repr(C)]
        #[derive(Copy, Clone, Debug, Pod, Zeroable)]
        struct VertexData(DevicePointer, DevicePointer, Mat2);
        let vertex_data = arena.alloc_data(&[VertexData(
            vertex_data_positions.device(),
            vertex_data_colors.device(),
            Mat2::from_angle(time as f32),
        )])?;
        let pixel_data = arena.alloc_data(&[
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(1.0, 1.0, 0.5, 1.0),
        ])?;

        let mut command_buffer = self.queue.create_buffer()?;

        command_buffer.begin_recording()?;

        let image_index = self.gpu.next_swapchain_image(ctx)?;
        let render_pass_desc = gpu::RenderPassDesc {
            ..Default::default()
        };
        command_buffer.begin_render_pass(&render_pass_desc, image_index);
        command_buffer.bind_shaders([&self.vertex_shader, &self.pixel_shader]);
        command_buffer.draw_indexed(
            vertex_data.device(), pixel_data.device(),
            indices.device(), 3, gpu::IndexType::U32);
        command_buffer.end_render_pass(image_index);

        command_buffer.end_recording()?;

        self.queue.submit(command_buffer, &self.frame_semaphore, self.next_frame)?;
        self.gpu.swapchain_present(&self.frame_semaphore, self.next_frame)?;

        self.next_frame += 1;

        Ok(())
    }
}
