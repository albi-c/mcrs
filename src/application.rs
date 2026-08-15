use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::path::Path;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec3A, Vec4};
use half::f16;
use image::{EncodableLayout, ImageReader};
use winit::event::ElementState;
use winit::keyboard::{KeyCode, PhysicalKey};
use gpu::{MemoryAllocation, MemoryAllocator};
use crate::gltf;

const FRAMES_IN_FLIGHT: u64 = 2;

// TODO: use u16 instead of f16
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex(pub [f16; 3], pub u16, pub [f16; 2], pub u32);

impl Vertex {
    pub fn pack_normal(normal: Vec3A) -> u32 {
        let x = (((normal.x + 1.0) * 1024.0) as u32).clamp(0, 0x7ff);
        let y = (((normal.y + 1.0) * 512.0) as u32).clamp(0, 0x3ff);
        let z = (((normal.z + 1.0) * 1024.0) as u32).clamp(0, 0x7ff);
        x | (y << 11) | (z << 21)
    }
}

impl meshopt::DecodePosition for Vertex {
    fn decode_position(&self) -> [f32; 3] {
        self.0.map(|x| x.to_f32())
    }
}

impl Default for Vertex {
    fn default() -> Self {
        Self::zeroed()
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Material {
    pub tex_offsets: u16,
    pub tex_diffuse: u16,

    pub ambient_and_roughness: u32,
    pub diffuse_and_normal: u32,
    pub specular_and_exp: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Light {
    pub pos: Vec3,
    pub intensity: f32,
    pub color: Vec3A,
}

impl Material {
    pub fn pack_vec4(vec: Vec4) -> u32 {
        let vec = (vec * 255.0).clamp(Vec4::splat(0.0), Vec4::splat(255.0));
        u32::from_le_bytes([
            vec.x as u8,
            vec.y as u8,
            vec.z as u8,
            vec.w as u8,
        ])
    }

    pub fn pack_tex_offsets(normal: u16, metallic_roughness: u16) -> u16 {
        assert!(normal < 16, "normal texture offset out of range: {normal}");
        assert!(metallic_roughness < 16, "metallic_roughness texture offset out of range: {metallic_roughness}");
        (metallic_roughness << 4) | normal
    }
}

fn hash(value: impl Hash) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn load_image(path: impl AsRef<Path>, gpu: &gpu::Gpu) -> Result<(u32, u32, gpu::Allocation<'_, u8>)> {
    let cache_path = format!(
        "image_cache/{:016x}_{}", hash(path.as_ref()),
        path.as_ref().file_name()
            .expect("path has no file name").to_str()
            .expect("unable to convert path to string"));
    if fs::exists(&cache_path)? {
        let mut file = fs::File::open(cache_path)?;
        let mut header = [0u32; 2];
        file.read_exact(bytemuck::cast_slice_mut(&mut header))?;
        let width = header[0];
        let height = header[1];
        let mut decoder = lz4::Decoder::new(file)?;
        let length = width as usize * height as usize * 4;
        let mut alloc = gpu.alloc::<u8>(length)?;
        decoder.read_exact(bytemuck::cast_slice_mut(alloc.host_mut()))?;
        Ok((width, height, alloc))
    } else {
        let img = ImageReader::open(path)?
            .with_guessed_format()?.decode()?.into_rgba8();
        let width = img.width();
        let height = img.height();
        let alloc = gpu.allocator().alloc_data(img.as_bytes())?;
        let mut file = fs::File::create(cache_path)?;
        file.write_all(bytemuck::cast_slice(&[width, height]))?;
        let mut encoder = lz4::EncoderBuilder::new().level(9).build(file)?;
        encoder.write_all(img.as_bytes())?;
        Ok((img.width(), img.height(), alloc))
    }
}

pub fn create_texture<'a, 'b>(gpu: &'a gpu::Gpu, (width, height): (u32, u32), alloc: gpu::Allocation<'a, u8>,
                              format: gpu::Format, cmd_buf: &mut gpu::CommandBuffer<'b>) -> Result<gpu::Texture<'a>> where 'a: 'b {
    let mip_count = width.max(height).checked_ilog2().unwrap_or(0).max(0) + 1;
    let tex = gpu.create_texture(gpu::TextureDesc {
        ty: gpu::TextureType::Tex2D,
        dimensions: (width, height, 1),
        format,
        usage: gpu::TextureUsageFlags::Sampled,
        mip_count,
        ..Default::default()
    }, cmd_buf)?;
    cmd_buf.copy_to_texture_mipmapped(alloc.device(), &tex);
    cmd_buf.give_ownership(alloc);

    Ok(tex)
}

fn load_texture<'a, 'b>(path: impl AsRef<Path>, gpu: &'a gpu::Gpu,
                        cmd_buf: &mut gpu::CommandBuffer<'b>) -> Result<gpu::Texture<'a>> where 'a: 'b {
    let (width, height, alloc) = load_image(path, gpu)?;
    create_texture(gpu, (width, height), alloc, gpu::Format::RGBA8UNorm, cmd_buf)
}

fn load_gltf_cached<'a>(path: impl AsRef<Path>, cache_path: impl AsRef<Path>, gpu: &'a gpu::Gpu,
                        tex_descriptors: &mut gpu::DescriptorHeap<'a>, tex_offset: u16) -> Result<gltf::Model<'a>> {
    if fs::exists(&cache_path)? {
        let mut file = fs::File::open(cache_path)?;
        gltf::Model::deserialize(gpu, tex_descriptors, &mut file)
    } else {
        let gltf = gltf::Model::load(gpu, path, tex_descriptors, tex_offset)?;
        let mut file = fs::File::create(cache_path)?;
        gltf.serialize(&mut file)?;
        Ok(gltf)
    }
}

struct WindowSizedTextures<'a> {
    pub depth_buffer: gpu::Texture<'a>,
    pub post_color: gpu::Texture<'a>,
}

struct Shaders<'a> {
    pub vertex: gpu::Shader<'a>,
    pub pixel: gpu::Shader<'a>,

    pub mesh_post: gpu::Shader<'a>,
    pub pixel_post: gpu::Shader<'a>,
    pub compute_post: gpu::Shader<'a>,
}

struct RayTracing<'a> {
    bl: gpu::rt::BottomLevelAS<'a>,
    tl: gpu::rt::TopLevelAS<'a>,
}

pub struct Application<'a> {
    gpu: &'a gpu::Gpu,
    shaders: Shaders<'a>,
    queue: gpu::Queue<'a>,
    frame_semaphore: gpu::Semaphore<'a>,
    frame_arenas: Box<[gpu::Arena<'a>]>,
    next_frame: u64,
    last_time: Option<f64>,

    textures: WindowSizedTextures<'a>,
    tex_descriptors: gpu::DescriptorHeap<'a>,
    sampler_descriptors: gpu::DescriptorHeap<'a>,
    accel_struct_descriptors: gpu::DescriptorHeap<'a>,

    gltf: gltf::Model<'a>,
    lights: gpu::Allocation<'a, Light>,
    rt: RayTracing<'a>,

    keys: HashMap<PhysicalKey, bool>,
    camera_pos: Vec3,
    camera_front: Vec3,
    camera_look: Vec2,
}

fn create_bottom_level_as<'a>(gpu: &'a gpu::Gpu, queue: &gpu::Queue<'a>, scene: &gltf::Scene<'a>,
                              scale: Vec3) -> Result<gpu::rt::BottomLevelAS<'a>> {
    let allocator = gpu.allocator_mem(gpu::Memory::AccelerationStructureInput);
    let mut bl_as = gpu::rt::BottomLevelAS::builder();
    let mut vertices = allocator.alloc(scene.vertices.len())?;
    for (dst, src) in vertices.host_mut().into_iter().zip(scene.vertices.host()) {
        *dst = src.0.map(|x| x.to_f32());
    }
    let indices = allocator.alloc_data(scene.indices.host())?;
    let transform_data = [
        [scale.x, 0.0, 0.0, 0.0],
        [0.0, scale.y, 0.0, 0.0],
        [0.0, 0.0, scale.z, 0.0],
    ];

    let mut cmd_buf = queue.create_buffer()?;

    let transform = allocator.alloc_data(&[transform_data])?;
    bl_as.geometry(
        vertices, gpu::rt::VertexFormat::XYZ32Float, indices, transform, 0);
    let bl_as = bl_as.build(gpu, &mut cmd_buf)?;

    queue.submit_no_signal(cmd_buf)?.wait();

    Ok(bl_as)
}

fn create_top_level_as<'a>(gpu: &'a gpu::Gpu, queue: &gpu::Queue<'a>,
                           bl_as: &gpu::rt::BottomLevelAS<'a>) -> Result<gpu::rt::TopLevelAS<'a>> {
    let transform_data = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];

    let mut cmd_buf = queue.create_buffer()?;

    let mut tl_as = gpu::rt::TopLevelAS::builder();
    tl_as.add(&bl_as, &transform_data);
    let tl_as = tl_as.build(gpu, &mut cmd_buf)?;

    queue.submit_no_signal(cmd_buf)?.wait();

    Ok(tl_as)
}

impl<'a> Application<'a> {
    pub fn new(gpu: &'a gpu::Gpu, ctx: &dyn gpu::SwapchainContext) -> Result<Self> {
        let queue = gpu.create_queue(gpu::QueueType::Graphics)?;

        let mut command_buffer = queue.create_buffer()?;

        let textures = Self::create_window_sized(gpu, ctx, &mut command_buffer)?;

        let mut tex_descriptors = gpu.alloc_texture_descriptor_heap()?;
        let mut sampler_descriptors = gpu.alloc_sampler_descriptor_heap()?;
        let mut accel_struct_descriptors = gpu.alloc_accel_struct_descriptor_heap()?;

        gpu.sampler_descriptor(gpu::SamplerDesc {
            ..Default::default()
        }, &mut sampler_descriptors[0])?;

        queue.submit_no_signal(command_buffer)?.wait();

        let gltf = load_gltf_cached(
            "models/Sponza_gltf/glTF/Sponza.gltf", "models/sponza_gltf.cache",
            gpu, &mut tex_descriptors, 1)?;
        // let gltf = gltf::Model::load(
        //     gpu, "models/Sponza_gltf/glTF/Sponza.gltf",
        //     &mut tex_descriptors, 1)?;

        let bl_as = create_bottom_level_as(
            gpu, &queue, &gltf.scenes[0], Vec3::new(1.0, -1.0, 1.0))?;
        let tl_as = create_top_level_as(gpu, &queue, &bl_as)?;
        tl_as.descriptor(&mut accel_struct_descriptors[0]);

        // TODO: compute queue, compute, texture.view_rw
        // TODO: conditional rendering
        // TODO: particles using compute, fluctuating light
        // TODO: fsr
        // TODO: ray query for shadows
        // TODO: point shadows using multi view
        // TODO: model loading abstraction
        // TODO: synchronization 2 everywhere
        // TODO: shader auto build

        let intensity = 1.5;
        let color = Vec3A::new(0.85, 0.65, 0.05);
        let lights = gpu.allocator().alloc_data(&[
            Light {
                pos: Vec3::new(-5.0, 1.3, 1.1),
                intensity,
                color,
            },
            Light {
                pos: Vec3::new(-5.0, 1.3, -1.7),
                intensity,
                color,
            },
            Light {
                pos: Vec3::new(3.9, 1.3, 1.1),
                intensity,
                color,
            },
            Light {
                pos: Vec3::new(3.9, 1.3, -1.7),
                intensity,
                color,
            },
        ])?;

        Ok(Self {
            gpu,
            shaders: Shaders {
                vertex: gpu.create_shader(&fs::read("shaders/vert.spv")?,
                                          gpu::ShaderStage::Vertex)?,
                pixel: gpu.create_shader(&fs::read("shaders/frag.spv")?,
                                         gpu::ShaderStage::Pixel)?,

                mesh_post: gpu.create_shader(&fs::read("shaders/mesh_post.spv")?,
                                             gpu::ShaderStage::Mesh)?,
                pixel_post: gpu.create_shader(&fs::read("shaders/frag_post.spv")?,
                                              gpu::ShaderStage::Pixel)?,
                compute_post: gpu.create_shader(&fs::read("shaders/comp_post.spv")?,
                                                gpu::ShaderStage::Compute)?,
            },
            queue,
            frame_semaphore: gpu.create_semaphore(0)?,
            frame_arenas: (0..FRAMES_IN_FLIGHT)
                .map(|_| gpu.create_arena(1024 * 1024))
                .collect::<Result<_>>()?,
            next_frame: 1,
            last_time: None,

            textures,
            tex_descriptors,
            sampler_descriptors,
            accel_struct_descriptors,

            gltf,
            lights,
            rt: RayTracing {
                bl: bl_as,
                tl: tl_as,
            },

            keys: HashMap::new(),
            camera_pos: Vec3::new(0.0, 0.0, 0.0),
            camera_front: Vec3::new(0.0, 0.0, 1.0),
            camera_look: Vec2::new(0.0, 0.0),
        })
    }

    fn create_tex<'b>(gpu: &'b gpu::Gpu, ctx: &dyn gpu::SwapchainContext,
                      cmd_buf: &mut gpu::CommandBuffer<'_>, format: gpu::Format,
                      usage: gpu::TextureUsageFlags) -> Result<gpu::Texture<'b>> {
        let dims = ctx.get_window_size();
        gpu.create_texture(gpu::TextureDesc {
            ty: gpu::TextureType::Tex2D,
            dimensions: (dims.0, dims.1, 1),
            format,
            usage,
            ..Default::default()
        }, cmd_buf)
    }

    fn create_window_sized<'b>(gpu: &'b gpu::Gpu, ctx: &dyn gpu::SwapchainContext,
                               cmd_buf: &mut gpu::CommandBuffer<'_>) -> Result<WindowSizedTextures<'b>> {
        Ok(WindowSizedTextures {
            depth_buffer: Self::create_tex(
                gpu, ctx, cmd_buf, gpu::Format::Depth32Float,
                gpu::TextureUsageFlags::DepthStencilAttachment,
            )?,
            post_color: Self::create_tex(
                gpu, ctx, cmd_buf, gpu::Format::RGBA8UNorm,
                gpu::TextureUsageFlags::Sampled | gpu::TextureUsageFlags::ColorAttachment,
            )?,
        })
    }

    fn get_frame_arena(&self) -> &gpu::Arena<'_> {
        &self.frame_arenas[(self.next_frame % FRAMES_IN_FLIGHT) as usize]
    }

    fn get_key(&self, code: KeyCode) -> bool {
        *self.keys.get(&PhysicalKey::Code(code)).unwrap_or(&false)
    }

    fn update(&mut self, dt: f32) {
        let front = self.camera_front;
        let up = Vec3::Y;
        let right = front.cross(up);

        let move_front = Vec3::new(front.x, 0.0, front.z).normalize();
        let move_right = Vec3::new(right.x, 0.0, right.z).normalize();

        let mut vel = Vec3::new(0.0, 0.0, 0.0);
        if self.get_key(KeyCode::KeyW) {
            vel += move_front;
        }
        if self.get_key(KeyCode::KeyS) {
            vel -= move_front;
        }
        if self.get_key(KeyCode::KeyD) {
            vel += move_right;
        }
        if self.get_key(KeyCode::KeyA) {
            vel -= move_right;
        }
        if self.get_key(KeyCode::Space) {
            vel += up;
        }
        if self.get_key(KeyCode::ShiftLeft) {
            vel -= up;
        }

        self.camera_pos += dt * vel * if self.get_key(KeyCode::ControlLeft) {
            10.0
        } else {
            2.5
        };
    }

    fn get_view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(
            self.camera_pos,
            self.camera_pos + self.camera_front,
            Vec3::Y,
        )
    }

    pub fn render(&mut self, time: f64, ctx: &dyn gpu::SwapchainContext) -> Result<()> {
        let last_time = self.last_time.replace(time).unwrap_or(time);
        let dt = (time - last_time) as f32;
        self.update(dt);

        if self.next_frame > FRAMES_IN_FLIGHT {
            self.frame_semaphore.wait(self.next_frame - FRAMES_IN_FLIGHT)?;
        }

        self.textures.post_color.view_descriptor(&mut self.tex_descriptors[0])?;

        let arena = self.get_frame_arena();
        arena.reset();

        let view_size = gpu::SwapchainContext::get_window_size(ctx);
        let mat_perspective = Mat4::perspective_infinite_rh(
            100.0f32.to_radians(),
            view_size.0 as f32 / view_size.1 as f32,
            0.001f32,
        );
        let mat_flip = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0));
        let mat_view = self.get_view_matrix();
        let mat_model = Mat4::IDENTITY;
        let mat_mvp = mat_perspective * mat_flip * mat_view * mat_model;

        const { assert!(size_of::<Vertex>() == 16) };

        #[repr(C)]
        #[derive(Copy, Clone, Debug, Pod, Zeroable)]
        struct VertexData(Mat4, Mat4, gpu::DevicePointer, gpu::DevicePointer);
        let vertex_data = arena.alloc_data(&[VertexData(
            mat_mvp,
            mat_model,
            self.gltf.scenes[0].vertices.device(),
            self.gltf.scenes[0].materials.device(),
        )])?;

        #[repr(C)]
        #[derive(Copy, Clone, Debug, Pod, Zeroable)]
        struct PixelData(Vec3, u32, gpu::DevicePointer);
        let pixel_data = arena.alloc_data(&[PixelData(
            self.camera_pos,
            self.lights.len() as u32,
            self.lights.device(),
        )])?;

        let mut command_buffer = self.queue.create_buffer()?;

        let mut swapchain_target = self.gpu.next_swapchain_image(ctx, &mut command_buffer)?;
        swapchain_target.clear_value = gpu::ClearValue::Color([0.08, 0.0, 0.0, 1.0]);

        let color_target = gpu::Target {
            view: self.textures.post_color.view()?,
            load_op: gpu::Load::Clear,
            store_op: gpu::Store::Store,
            clear_value: gpu::ClearValue::Color([0.08, 0.0, 0.0, 1.0]),
        };
        let depth_target = gpu::Target {
            view: self.textures.depth_buffer.view()?,
            load_op: gpu::Load::Clear,
            store_op: gpu::Store::Store,
            clear_value: gpu::ClearValue::DepthStencil(1.0, 0),
        };
        let depth_test_desc = gpu::DepthTestDesc {
            ..Default::default()
        };
        let render_pass_desc = gpu::RenderPassDesc {
            cull: gpu::Cull::CW,
            depth_test_state: Some(&depth_test_desc),
            color_targets: &[color_target],
            depth_target: Some(&depth_target),
            ..Default::default()
        };
        command_buffer.begin_render_pass(&render_pass_desc);
        command_buffer.bind_shaders([&self.shaders.vertex, &self.shaders.pixel]);
        command_buffer.set_descriptor_heap(
            Some(&self.tex_descriptors),
            None,
            Some(&self.sampler_descriptors),
            Some(&self.accel_struct_descriptors),
        );
        let indices = &self.gltf.scenes[0].indices;
        command_buffer.draw_indexed(
            vertex_data.device(), pixel_data.device(),
            indices.device(), indices.len() as u32, gpu::IndexType::U32);
        command_buffer.end_render_pass();

        #[repr(C)]
        #[derive(Copy, Clone, Debug, Pod, Zeroable)]
        struct PostPixelData(u32);
        let post_pixel_data = arena.alloc_data(&[PostPixelData(
            0,
        )])?;
        let post_pass_desc = gpu::RenderPassDesc {
            cull: gpu::Cull::None,
            color_targets: &[swapchain_target],
            ..Default::default()
        };
        command_buffer.begin_render_pass(&post_pass_desc);
        command_buffer.bind_shaders([&self.shaders.mesh_post, &self.shaders.pixel_post]);
        command_buffer.set_descriptor_heap(
            Some(&self.tex_descriptors),
            None,
            Some(&self.sampler_descriptors),
            Some(&self.accel_struct_descriptors),
        );
        command_buffer.draw_meshlets(
            gpu::DevicePointer::null(), post_pixel_data.device(),
            (1, 1, 1));
        command_buffer.end_render_pass();

        self.queue.submit(command_buffer, &self.frame_semaphore, self.next_frame)?;
        self.gpu.swapchain_present(&self.frame_semaphore, self.next_frame)?;

        self.next_frame += 1;

        Ok(())
    }

    pub fn resize(&mut self, ctx: &dyn gpu::SwapchainContext) -> Result<()> {
        let mut command_buffer = self.queue.create_buffer()?;

        let textures = Self::create_window_sized(self.gpu, ctx, &mut command_buffer)?;

        self.queue.submit_no_signal(command_buffer)?.wait();

        self.textures = textures;

        Ok(())
    }

    pub fn key(&mut self, key: PhysicalKey, state: ElementState) {
        self.keys.insert(key, state == ElementState::Pressed);
        if key == PhysicalKey::Code(KeyCode::KeyP) && state == ElementState::Pressed {
            println!("{:?}", self.camera_pos);
        }
    }

    pub fn mouse_move(&mut self, delta: (f64, f64)) {
        const SENSITIVITY: f32 = 0.5;
        let motion = Vec2::new(delta.0 as f32, delta.1 as f32) * SENSITIVITY;
        self.camera_look.x = (self.camera_look.x + motion.x).rem_euclid(360.0);
        self.camera_look.y = (self.camera_look.y - motion.y).clamp(-89.99, 89.99);

        let yaw = self.camera_look.x.to_radians();
        let pitch = self.camera_look.y.to_radians();

        self.camera_front = Vec3::new(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        ).normalize();
    }
}
