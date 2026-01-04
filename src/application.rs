use std::io::Cursor;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec3A};
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

    tex: gpu::Texture<'a>,
    tex_descriptors: gpu::DescriptorHeap<'a>,
    sampler_descriptors: gpu::DescriptorHeap<'a>,
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

        let mut tex_descriptors = gpu.alloc_texture_descriptor_heap()?;
        let mut sampler_descriptors = gpu.alloc_sampler_descriptor_heap()?;

        tex.view_descriptor(&mut tex_descriptors[0])?;
        gpu.sampler_descriptor(gpu::SamplerDesc {
            ..Default::default()
        }, &mut sampler_descriptors[0])?;

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

            tex,
            tex_descriptors,
            sampler_descriptors,
        })
    }

    fn get_frame_arena(&self) -> &gpu::Arena<'_> {
        &self.frame_arenas[(self.next_frame % FRAMES_IN_FLIGHT) as usize]
    }

    const fn pack_tex_vertex(u: f32, v: f32, tex: u16) -> u32 {
        ((tex as u32) << 16) | (((v * 8.0) as u32 & 0xff) << 8) | ((u * 8.0) as u32 & 0xff)
    }

    pub fn render(&mut self, time: f64, ctx: &dyn gpu::SwapchainContext) -> Result<()> {
        if self.next_frame > FRAMES_IN_FLIGHT {
            self.frame_semaphore.wait(self.next_frame - FRAMES_IN_FLIGHT)?;
        }

        let arena = self.get_frame_arena();
        arena.reset();

        let indices = arena.alloc_data(&[0u32, 1, 2, 0, 2, 3])?;

        let view_size = gpu::SwapchainContext::get_window_size(ctx);
        let mat_perspective = Mat4::perspective_infinite_rh(
            100.0f32.to_radians(),
            view_size.0 as f32 / view_size.1 as f32,
            0.001f32,
        );
        let mat_view = Mat4::look_at_rh(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.5, 0.0, -1.0).normalize(),
            Vec3::Y,
        );
        let mat_model = Mat4::from_translation(Vec3::new(0.0, 0.0, -1.0))
            * Mat4::from_rotation_y(time as f32 * 0.5);
        let mat_mvp = mat_perspective * mat_view * mat_model;
        #[repr(C)]
        #[derive(Copy, Clone, Debug, Pod, Zeroable)]
        struct Vertex(Vec3, u32);
        const { assert!(size_of::<Vertex>() == 16) };
        let vertex_data_vertices = arena.alloc_data(&[
            Vertex(Vec3::new(-0.5, -0.5, 0.0), Self::pack_tex_vertex(0.0, 0.0, 0)),
            Vertex(Vec3::new(0.5, -0.5, 0.0), Self::pack_tex_vertex(1.0, 0.0, 0)),
            Vertex(Vec3::new(0.5, 0.5, 0.0), Self::pack_tex_vertex(1.0, 1.0, 0)),
            Vertex(Vec3::new(-0.5, 0.5, 0.0), Self::pack_tex_vertex(0.0, 1.0, 0)),
        ])?;
        #[repr(C)]
        #[derive(Copy, Clone, Debug, Pod, Zeroable)]
        struct VertexData(Mat4, DevicePointer, u64);
        let vertex_data = arena.alloc_data(&[VertexData(
            mat_mvp,
            vertex_data_vertices.device(),
            0,
        )])?;

        #[repr(C)]
        #[derive(Copy, Clone, Debug, Pod, Zeroable)]
        struct PixelData(Vec3A);
        let pixel_data = arena.alloc_data(&[PixelData(
            Vec3A::new(1.0, 1.0, 1.0),
        )])?;

        let mut command_buffer = self.queue.create_buffer()?;

        command_buffer.begin_recording()?;

        let mut swapchain_target = self.gpu.next_swapchain_image(ctx, &mut command_buffer)?;
        swapchain_target.clear_value = gpu::ClearValue::Color([0.08, 0.0, 0.0, 1.0]);
        let render_pass_desc = gpu::RenderPassDesc {
            color_targets: &[swapchain_target],
            ..Default::default()
        };
        command_buffer.begin_render_pass(&render_pass_desc);
        command_buffer.bind_shaders([&self.vertex_shader, &self.pixel_shader]);
         command_buffer.set_texture_heap(
             Some(&self.tex_descriptors),
             None,
             Some(&self.sampler_descriptors),
         );
        command_buffer.draw_indexed(
            vertex_data.device(), pixel_data.device(),
            indices.device(), 6, gpu::IndexType::U32);
        command_buffer.end_render_pass();

        command_buffer.end_recording()?;

        self.queue.submit(command_buffer, &self.frame_semaphore, self.next_frame)?;
        self.gpu.swapchain_present(&self.frame_semaphore, self.next_frame)?;

        self.next_frame += 1;

        Ok(())
    }
}
