use std::time::Instant;
use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec3A, Vec4};
use half::f16;
use gpu::{MemoryAllocation, MemoryAllocator};
use macros::multi_allocation;
use crate::application::load_shader;
use crate::gltf::Scene;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Frustum([Vec4; 5]);

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct AABB {
    center: Vec3,
    extent: Vec3,
}

impl AABB {
    fn from_min_max(min: Vec3, max: Vec3) -> Self {
        Self {
            center: min.midpoint(max),
            extent: (max - min) * 0.5,
        }
    }

    fn to_min_max(self) -> (Vec3, Vec3) {
        (self.center - self.extent, self.center + self.extent)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct Vertex {
    pos: [f16; 3],
    mat: u16,
    uv: [f16; 2],
    // 11:10:11
    // z / 1024 - 1, y / 512 - 1, x / 1024 - 1
    normal: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Meshlet {
    vertices: [Vertex; 64],
    // packed: each uint: 0..8 - idx 1, 8..16 - idx 2, 16..24 - idx 3
    triangles: [u32; 128],
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct MeshletInfoFlags(u8);
bitflags! {
    impl MeshletInfoFlags : u8 {
        const NoRender = 0x01;
        const NoCulling = 0x02;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct MeshletInfo {
    aabb: AABB,
    transform: gpu::DevicePointer,  // Mat4
    vertex_count: u8,
    triangle_count: u8,
    flags: MeshletInfoFlags,
    _padding: [u8; 13],
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ModelFlags(u32);
bitflags! {
    impl ModelFlags : u32 {
        const NoRender = 0x01;
        // ignores meshlet transforms, only uses model transform when adjusting AABB
        const EnableCulling = 0x02;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Model {
    meshlets: gpu::DevicePointer,  // [Meshlet]
    meshlet_infos: gpu::DevicePointer,  // [MeshletInfo]
    transform: gpu::DevicePointer,  // Mat4
    meshlet_count: u32,
    flags: ModelFlags,
    aabb: AABB,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct MeshData {
    view_proj: Mat4,
    frustum: Frustum,
    model_parts: gpu::DevicePointer,  // [ModelPart]
    materials: gpu::DevicePointer,  // [UVec4]
    camera_pos: Vec3A,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ModelPart {
    model: gpu::DevicePointer,  // Model
    transform: gpu::DevicePointer,  // Mat4
    aabb: AABB,
    meshlet_offset: u32,
    meshlet_count: u32,
}

#[multi_allocation]
struct MultiAlloc {
    meshlet_transforms: Mat4,
    meshlet_infos: MeshletInfo,
    meshlets: Meshlet,
    model_transforms: Mat4,
    models: Model,
    model_parts: ModelPart,
}

pub struct MeshModels<'a> {
    shader_task: gpu::Shader<'a>,
    shader_mesh: gpu::Shader<'a>,
    shader_task2: gpu::Shader<'a>,
    shader_mesh2: gpu::Shader<'a>,
    shader_frag: gpu::Shader<'a>,

    alloc: MultiAlloc<'a>,
}

fn get_frustum(view_proj: &Mat4) -> Frustum {
    let x = view_proj.row(0);
    let y = view_proj.row(1);
    let z = view_proj.row(2);
    let w = view_proj.row(3);
    Frustum([
        w + x,
        w - x,
        w + y,
        w - y,
        z,
    ])
}

const MODEL_PART_SIZE: usize = 64;

impl<'a> MeshModels<'a> {
    pub fn from_gltf_scene(gpu: &'a gpu::Gpu, scene: &Scene, repeat: usize, offset: Vec2) -> anyhow::Result<Self> {
        // TODO: !! levels of detail with dynamic selection in task shader
        // TODO: cone culling

        let mut offsets = vec![];
        for y in 0..repeat {
            for x in 0..repeat {
                offsets.push((x + repeat * y, offset * Vec2::new(x as f32, y as f32)));
            }
        }

        let start_time = Instant::now();

        let vertex_positions = scene.vertices.host().iter()
            .map(|v| Vec3::from_array(v.0.map(f16::to_f32))).collect::<Vec<_>>();
        let vertex_adapter = meshopt::VertexDataAdapter::new(
            bytemuck::cast_slice(&vertex_positions), 12, 0)?;
        let meshlets = meshopt::build_meshlets(
            scene.indices.host(), &vertex_adapter, 64, 96, 0.0);

        let model_parts = meshlets.len().div_ceil(MODEL_PART_SIZE);
        let mut alloc = MultiAlloc::new(gpu, MultiAllocCounts {
            meshlet_transforms: 1,
            meshlet_infos: meshlets.len(),
            meshlets: meshlets.len(),
            model_transforms: repeat * repeat,
            models: 1,
            model_parts: model_parts * repeat * repeat,
        })?;

        alloc.meshlet_transforms.host_mut()[0] = Mat4::IDENTITY;

        let mut model_min_coord = Vec3::INFINITY;
        let mut model_max_coord = Vec3::NEG_INFINITY;
        for (i, meshlet) in meshlets.meshlets.iter().enumerate() {
            let mut vertices = [bytemuck::zeroed(); 64];
            let mut triangles = [0; 128];
            let mut min_coord = Vec3::INFINITY;
            let mut max_coord = Vec3::NEG_INFINITY;
            for j in 0..meshlet.vertex_count as usize {
                let idx = meshlets.vertices[meshlet.vertex_offset as usize + j] as usize;
                min_coord = min_coord.min(vertex_positions[idx]);
                max_coord = max_coord.max(vertex_positions[idx]);
                let vert = scene.vertices.host()[idx];
                vertices[j] = Vertex {
                    pos: vert.0,
                    mat: vert.1,
                    uv: vert.2,
                    normal: vert.3,
                };
            }

            model_min_coord = model_min_coord.min(min_coord);
            model_max_coord = model_max_coord.max(max_coord);

            for j in 0..meshlet.triangle_count as usize {
                let tri = &meshlets.triangles[meshlet.triangle_offset as usize + 3 * j..][..3];
                triangles[j] = u32::from_le_bytes([tri[0], tri[1], tri[2], 0]);
            }

            alloc.meshlet_infos.host_mut()[i] = MeshletInfo {
                aabb: AABB::from_min_max(min_coord, max_coord),
                transform: alloc.meshlet_transforms.device().add_typed::<Mat4>(0),
                vertex_count: meshlet.vertex_count as u8,
                triangle_count: meshlet.triangle_count as u8,
                flags: MeshletInfoFlags::empty(),
                _padding: [0u8; 13],
            };
            alloc.meshlets.host_mut()[i] = Meshlet {
                vertices,
                triangles,
            };
        }

        for &(i, offset) in &offsets {
            alloc.model_transforms.host_mut()[i] = Mat4::from_translation(Vec3::new(offset.x, 0.0, offset.y));
        }
        alloc.model_transforms.host_mut().sort_unstable_by_key(|m| (m.z_axis.truncate().length() * 16384.0) as isize);

        alloc.models.host_mut()[0] = Model {
            meshlets: alloc.meshlets.device(),
            meshlet_infos: alloc.meshlet_infos.device(),
            transform: alloc.model_transforms.device().add_typed::<Mat4>(0),
            meshlet_count: meshlets.len() as u32,
            flags: ModelFlags::EnableCulling,
            aabb: AABB::from_min_max(model_min_coord, model_max_coord),
            _padding: [0u32, 2],
        };
        for &(j, _offset) in &offsets {
            for i in 0..model_parts {
                let offset = (i * MODEL_PART_SIZE) as u32;
                let count = if i == model_parts - 1 {
                    meshlets.len() - (model_parts - 1) * MODEL_PART_SIZE
                } else {
                    MODEL_PART_SIZE
                } as u32;

                let mut min_bound = Vec3::INFINITY;
                let mut max_bound = Vec3::NEG_INFINITY;
                for i in offset..offset + count {
                    let (min, max) = alloc.meshlet_infos.host()[i as usize].aabb.to_min_max();
                    min_bound = min_bound.min(min);
                    max_bound = max_bound.max(max);
                }

                alloc.model_parts.host_mut()[i * repeat * repeat + j] = ModelPart {
                    model: alloc.models.device().add_typed::<Model>(0),
                    transform: alloc.model_transforms.device().add_typed::<Mat4>(j),
                    aabb: AABB::from_min_max(min_bound, max_bound),
                    meshlet_offset: offset,
                    meshlet_count: count,
                };
            }
        }

        println!("Clusterized scene in {:.03} seconds", start_time.elapsed().as_secs_f32());

        Ok(Self {
            shader_task: load_shader("shaders/task_model.glsl", gpu::ShaderStage::Task, gpu)?,
            shader_mesh: load_shader("shaders/mesh_model.glsl", gpu::ShaderStage::MeshWithTask, gpu)?,
            shader_task2: load_shader("shaders/task_model2.glsl", gpu::ShaderStage::Task, gpu)?,
            shader_mesh2: load_shader("shaders/mesh_model2.glsl", gpu::ShaderStage::MeshWithTask, gpu)?,
            shader_frag: load_shader("shaders/frag.glsl", gpu::ShaderStage::Pixel, gpu)?,

            alloc,
        })
    }

    pub fn render(&self, arena: &gpu::Arena<'a>, cmd_buf: &mut gpu::CommandBuffer<'a>,
                  pixel_data: gpu::DevicePointer, materials: gpu::DevicePointer,
                  view_proj: &Mat4, old_shaders: bool, camera_pos: Vec3A) -> anyhow::Result<()> {
        let model_pointers = arena.alloc_data(&[
            self.alloc.models.device().add_typed::<Model>(0),
        ])?;
        let mesh_data = arena.alloc_data(&[MeshData {
            view_proj: *view_proj,
            frustum: get_frustum(view_proj),
            model_parts: if old_shaders { model_pointers.device() } else { self.alloc.model_parts.device() },
            materials,
            camera_pos,
        }])?;

        if old_shaders {
            cmd_buf.bind_shaders([&self.shader_task, &self.shader_mesh, &self.shader_frag]);
            cmd_buf.draw_meshlets(
                mesh_data.device(), pixel_data, (self.alloc.models.len() as u32, 1, 1));
        } else {
            cmd_buf.bind_shaders([&self.shader_task2, &self.shader_mesh2, &self.shader_frag]);
            cmd_buf.draw_meshlets(
                mesh_data.device(), pixel_data, (self.alloc.model_parts.len() as u32, 1, 1));
        }

        Ok(())
    }
}
