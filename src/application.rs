use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec3A, Vec3Swizzles, Vec4};
use image::{EncodableLayout, ImageReader};
use winit::event::ElementState;
use winit::keyboard::{KeyCode, PhysicalKey};
use gpu::{CommandBuffer, MemoryAllocation, MemoryAllocator};
use crate::obj;

const FRAMES_IN_FLIGHT: u64 = 2;

fn load_obj<C>(path: impl AsRef<Path> + Debug,
               get_material: impl Fn(&str) -> Option<u32>,
               push_vertex: impl Fn(&mut C, usize, Vec3, Vec2, Vec3, u32),
               make_vertex_container: impl FnOnce(usize) -> Result<C>) -> Result<C> {
    let model = obj::Model::read(
        path, |mat| get_material(mat).ok_or_else(|| anyhow!("invalid material id: {}", mat)))?;

    let mut length = 0;
    for obj in &model.objects {
        length += 3 * obj.faces.len();
    }

    let mut vertices = make_vertex_container(length)?;
    let mut index = 0;
    for obj in model.objects {
        for face in obj.faces {
            for vertex in face {
                push_vertex(
                    &mut vertices,
                    index,
                    model.vertices[vertex.vertex as usize],
                    model.tex_coords[vertex.tex_coord as usize],
                    model.normals[vertex.normal as usize],
                    vertex.material,
                );
                index += 1;
            }
        }
    }

    assert_eq!(index, length);

    Ok(vertices)

    // let (models, _) = tobj::load_obj(path, &tobj::LoadOptions {
    //     triangulate: true,
    //     ..Default::default()
    // })?;
    //
    // let mut length = 0;
    // for model in &models {
    //     length += model.mesh.indices.len();
    //     assert_eq!(model.mesh.indices.len(), model.mesh.texcoord_indices.len());
    //     assert_eq!(model.mesh.indices.len(), model.mesh.normal_indices.len());
    // }
    //
    // let mut vertices = make_vertex_container(length)?;
    // let mut index = 0;
    // for model in models {
    //     let mesh = model.mesh;
    //     let material = u32::try_from(mesh.material_id.unwrap_or(0))?;
    //     for (i_p, (i_t, i_n)) in mesh.indices.into_iter()
    //         .zip(mesh.texcoord_indices.into_iter().zip(mesh.normal_indices.into_iter())) {
    //         let b_p = i_p as usize * 3;
    //         let b_t = i_t as usize * 2;
    //         let b_n = i_n as usize * 3;
    //         push_vertex(
    //             &mut vertices,
    //             index,
    //             Vec3::new(
    //                 mesh.positions[b_p + 0],
    //                 mesh.positions[b_p + 1],
    //                 mesh.positions[b_p + 2],
    //             ),
    //             Vec2::new(
    //                 mesh.texcoords[b_t + 0],
    //                 mesh.texcoords[b_t + 1],
    //             ),
    //             Vec3::new(
    //                 mesh.normals[b_n + 0],
    //                 mesh.normals[b_n + 1],
    //                 mesh.normals[b_n + 2],
    //             ),
    //             material,
    //         );
    //         index += 1;
    //         assert!(index <= length);
    //     }
    // }
    //
    // Ok(vertices)
}

fn load_obj_cached<V: Pod, C>(path: &str, cache_path: impl AsRef<Path>,
                              get_material: impl Fn(&str) -> Option<u32>,
                              make_vertex: impl Fn(Vec3, Vec2, Vec3, u32) -> V,
                              make_vertex_container: impl FnOnce(usize) -> Result<C>,
                              vertex_container_slice: impl Fn(&mut C) -> &mut [V]) -> Result<C> {
    if fs::exists(&cache_path)? {
        let mut file = fs::File::open(cache_path)?;
        let length_bytes = file.seek(SeekFrom::End(0))? as usize;
        file.seek(SeekFrom::Start(0))?;
        let count = length_bytes / size_of::<V>();
        let mut container = make_vertex_container(count)?;
        let slice = vertex_container_slice(&mut container);
        assert_eq!(slice.len(), count);
        file.read_exact(bytemuck::cast_slice_mut(slice))?;
        Ok(container)
    } else {
        let mut container = load_obj(
            path,
            get_material,
            |container, index, pos, uv, nor, mat| {
                vertex_container_slice(container)[index] = make_vertex(pos, uv, nor, mat);
            },
            make_vertex_container,
        )?;
        let slice = vertex_container_slice(&mut container);
        fs::write(cache_path, bytemuck::cast_slice(slice))?;
        Ok(container)
    }
}

fn load_materials(path: impl AsRef<Path> + Debug, gpu: &gpu::Gpu,
                  texture_offset: u16) -> Result<(gpu::Allocation<'_, Material>, Vec<String>,
                                                  impl Fn(&str) -> Option<u32>)> {
    let (materials, by_name) = tobj::load_mtl(path)?;

    let mut allocation = gpu.alloc::<Material>(materials.len())?;
    let mem = allocation.host_mut();
    let mut texture_paths = vec![];

    for (i, material) in materials.into_iter().enumerate() {
        assert_eq!(material.ambient_texture, material.diffuse_texture);
        let mat = &mut mem[i];
        if let Some(disp) = material.normal_texture {
            mat.tex_disp = texture_offset + u16::try_from(texture_paths.len())?;
            texture_paths.push(disp);
        }
        if let Some(diff) = material.diffuse_texture {
            mat.tex_diffuse = texture_offset + u16::try_from(texture_paths.len())?;
            texture_paths.push(diff);
        }
        let [r, g, b] = material.ambient.unwrap_or_default();
        mat.ambient = Material::pack_vec4(Vec4::new(r, g, b, 0.0));
        let [r, g, b] = material.diffuse.unwrap_or_default();
        mat.diffuse_and_dissolve = Material::pack_vec4(
            Vec4::new(r, g, b, material.dissolve.unwrap_or_default()));
        let [r, g, b] = material.specular.unwrap_or_default();
        mat.specular_and_exp = Material::pack_vec4
            (Vec4::new(r, g, b, material.shininess.unwrap_or_default()));
    }

    Ok((allocation, texture_paths, move |name: &str| {
        by_name.get(name).map(|&index| index as u32)
    }))
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex(Vec3, u32, Vec2, Vec2);

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Material {
    pub tex_disp: u16,
    pub tex_diffuse: u16,

    pub ambient: u32,
    pub diffuse_and_dissolve: u32,
    pub specular_and_exp: u32,
}

impl Material {
    fn pack_vec4(vec: Vec4) -> u32 {
        let vec = (vec * 255.0).clamp(Vec4::splat(0.0), Vec4::splat(255.0));
        u32::from_le_bytes([
            vec.x as u8,
            vec.y as u8,
            vec.z as u8,
            vec.w as u8,
        ])
    }
}

fn load_texture<'a>(path: impl AsRef<Path>, gpu: &'a gpu::Gpu,
                    cmd_buf: &mut CommandBuffer<'_>) -> Result<(gpu::Texture<'a>, gpu::Allocation<'a, u8>)> {
    let img = ImageReader::open(path)?
        .with_guessed_format()?.decode()?.into_rgba8();
    let img_alloc = gpu.allocator().alloc_data(img.as_bytes())?;

    let tex = gpu.create_texture(gpu::TextureDesc {
        ty: gpu::TextureType::Tex2D,
        dimensions: (img.width(), img.height(), 1),
        format: gpu::Format::RGBA8UNorm,
        usage: gpu::TextureUsageFlags::Sampled,
        ..Default::default()
    }, cmd_buf)?;
    cmd_buf.copy_to_texture(img_alloc.device(), &tex);

    Ok((tex, img_alloc))
}

pub struct Application<'a> {
    gpu: &'a gpu::Gpu,
    vertex_shader: gpu::Shader<'a>,
    pixel_shader: gpu::Shader<'a>,
    queue: gpu::Queue<'a>,
    frame_semaphore: gpu::Semaphore<'a>,
    frame_arenas: Box<[gpu::Arena<'a>]>,
    next_frame: u64,
    last_time: Option<f64>,

    tex: gpu::Texture<'a>,
    depth_buffer: gpu::Texture<'a>,
    tex_descriptors: gpu::DescriptorHeap<'a>,
    sampler_descriptors: gpu::DescriptorHeap<'a>,
    model: gpu::Allocation<'a, Vertex>,
    materials: gpu::Allocation<'a, Material>,
    material_textures: Vec<gpu::Texture<'a>>,

    keys: HashMap<PhysicalKey, bool>,
    camera_pos: Vec3,
    camera_front: Vec3,
    camera_look: Vec2,
}

impl<'a> Application<'a> {
    pub fn new(gpu: &'a gpu::Gpu, ctx: &dyn gpu::SwapchainContext) -> Result<Self> {
        let queue = gpu.create_queue(gpu::QueueType::Graphics)?;

        let mut command_buffer = queue.create_buffer()?;

        command_buffer.begin_recording()?;

        let (tex, tex_alloc) = load_texture("models/viking_room/viking_room.png", gpu, &mut command_buffer)?;

        let depth_buffer = Self::create_depth_buffer(gpu, ctx, &mut command_buffer)?;

        let mut tex_descriptors = gpu.alloc_texture_descriptor_heap()?;
        let mut sampler_descriptors = gpu.alloc_sampler_descriptor_heap()?;

        tex.view_descriptor(&mut tex_descriptors[0])?;
        gpu.sampler_descriptor(gpu::SamplerDesc {
            ..Default::default()
        }, &mut sampler_descriptors[0])?;

        let (materials, texture_paths, mat_by_name) = load_materials(
            "models/Sponza/sponza.mtl", gpu, 1)?;
        let model = load_obj_cached(
            "models/Sponza/sponza.obj", "models/sponza.cache",
            mat_by_name,
            |pos, uv, normal, mat| Vertex(pos, mat, uv, normal.normalize().xy()),
            |n| gpu.alloc::<Vertex>(n),
            |container| container.host_mut())?;

        let mut material_textures = Vec::with_capacity(texture_paths.len());
        let mut material_tex_allocations = Vec::with_capacity(texture_paths.len());
        for (i, texture_path) in texture_paths.into_iter().enumerate() {
            let (tex, tex_alloc) = load_texture("models/Sponza/".to_owned() + &texture_path, gpu, &mut command_buffer)?;
            // texture index offset
            tex.view_descriptor(&mut tex_descriptors[i + 1])?;
            material_textures.push(tex);
            material_tex_allocations.push(tex_alloc);
        }

        command_buffer.end_recording()?;

        queue.submit_no_signal(command_buffer)?.wait();

        drop(tex_alloc);
        drop(material_tex_allocations);

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
            last_time: None,

            tex,
            depth_buffer,
            tex_descriptors,
            sampler_descriptors,
            model,
            materials,
            material_textures,

            keys: HashMap::new(),
            camera_pos: Vec3::new(0.0, 0.0, 0.0),
            camera_front: Vec3::new(0.0, 0.0, 1.0),
            camera_look: Vec2::new(0.0, 0.0),
        })
    }

    fn create_depth_buffer<'b>(gpu: &'b gpu::Gpu, ctx: &dyn gpu::SwapchainContext,
                               cmd_buf: &mut gpu::CommandBuffer<'_>) -> Result<gpu::Texture<'b>> {
        let dims = ctx.get_window_size();
        gpu.create_texture(gpu::TextureDesc {
            ty: gpu::TextureType::Tex2D,
            dimensions: (dims.0, dims.1, 1),
            format: gpu::Format::Depth32Float,
            usage: gpu::TextureUsageFlags::DepthStencilAttachment,
            layout: gpu::TextureLayout::DepthStencilAttachmentOptimal,
            ..Default::default()
        }, cmd_buf)
    }

    fn get_frame_arena(&self) -> &gpu::Arena<'_> {
        &self.frame_arenas[(self.next_frame % FRAMES_IN_FLIGHT) as usize]
    }

    const fn pack_tex_vertex(u: f32, v: f32, tex: u16) -> u32 {
        ((tex as u32) << 20) | (((v * 512.0) as u32 & ((1 << 10) - 1)) << 10) | ((u * 512.0) as u32 & ((1 << 10) - 1))
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

        vel *= 2.5;
        self.camera_pos += dt * vel;
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
        let mat_model = Mat4::from_translation(Vec3::new(0.0, 0.0, -1.0))
             * Mat4::from_scale(Vec3::splat(0.01));
        let mat_mvp = mat_perspective * mat_flip * mat_view * mat_model;

        const { assert!(size_of::<Vertex>() == 32) };

        #[repr(C)]
        #[derive(Copy, Clone, Debug, Pod, Zeroable)]
        struct VertexData(Mat4, gpu::DevicePointer, gpu::DevicePointer);
        let vertex_data = arena.alloc_data(&[VertexData(
            mat_mvp,
            self.model.device(),
            self.materials.device(),
        )])?;

        #[repr(C)]
        #[derive(Copy, Clone, Debug, Pod, Zeroable)]
        struct PixelData(Vec3A, Vec3A);
        let pixel_data = arena.alloc_data(&[PixelData(
            Vec3A::new(0.0, -1.0, 1.0).normalize(),
            self.camera_front.to_vec3a(),
        )])?;

        let mut command_buffer = self.queue.create_buffer()?;

        command_buffer.begin_recording()?;

        let mut swapchain_target = self.gpu.next_swapchain_image(ctx, &mut command_buffer)?;
        swapchain_target.clear_value = gpu::ClearValue::Color([0.08, 0.0, 0.0, 1.0]);
        let depth_target = gpu::Target {
            view: self.depth_buffer.view()?,
            load_op: gpu::Load::Clear,
            store_op: gpu::Store::Store,
            clear_value: gpu::ClearValue::DepthStencil(1.0, 0),
        };
        let depth_test_desc = gpu::DepthTestDesc {
            ..Default::default()
        };
        let render_pass_desc = gpu::RenderPassDesc {
            depth_test_state: Some(&depth_test_desc),
            color_targets: &[swapchain_target],
            depth_target: Some(&depth_target),
            ..Default::default()
        };
        command_buffer.begin_render_pass(&render_pass_desc);
        command_buffer.bind_shaders([&self.vertex_shader, &self.pixel_shader]);
         command_buffer.set_texture_heap(
             Some(&self.tex_descriptors),
             None,
             Some(&self.sampler_descriptors),
         );
        command_buffer.draw_instanced(
            vertex_data.device(), pixel_data.device(),
            self.model.len() as u32, 1, 0, 0);
        command_buffer.end_render_pass();

        command_buffer.end_recording()?;

        self.queue.submit(command_buffer, &self.frame_semaphore, self.next_frame)?;
        self.gpu.swapchain_present(&self.frame_semaphore, self.next_frame)?;

        self.next_frame += 1;

        Ok(())
    }

    pub fn resize(&mut self, ctx: &dyn gpu::SwapchainContext) -> Result<()> {
        let mut command_buffer = self.queue.create_buffer()?;

        command_buffer.begin_recording()?;

        let depth_buffer = Self::create_depth_buffer(self.gpu, ctx, &mut command_buffer)?;

        command_buffer.end_recording()?;

        self.queue.submit_no_signal(command_buffer)?.wait();

        self.depth_buffer = depth_buffer;

        Ok(())
    }

    pub fn key(&mut self, key: PhysicalKey, state: ElementState) {
        self.keys.insert(key, state == ElementState::Pressed);
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
