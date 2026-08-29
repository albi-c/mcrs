use std::time::Instant;
use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec3A, Vec4};
use half::f16;
use itertools::Itertools;
use smallvec::SmallVec;
use gpu::{MemoryAllocation, MemoryAllocator};
use macros::multi_allocation;
use crate::application::{load_shader, load_shader_spirv};
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
    vertex_count: u8,
    triangle_count: u8,
    flags: MeshletInfoFlags,
    _padding: [u8; 5],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct LODChunk {
    aabb: AABB,
    meshlet_offset: u32,
    meshlet_count: u16,
    _padding: [u16; 1],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct LOD {
    meshlets: gpu::DevicePointer,  // [Meshlet]
    meshlet_infos: gpu::DevicePointer,  // [MeshletInfo]
    chunks: gpu::DevicePointer,  // [LODChunk]
    chunk_count: u32,
    _padding: [u32; 1],
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ModelFlags(u32);
bitflags! {
    impl ModelFlags : u32 {
        const NoRender = 0x01;
        // ignores meshlet transforms, only uses model transform when adjusting AABB
        const EnableCulling = 0x02;
        const EnableLODs = 0x04;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Model {
    lods: gpu::DevicePointer,  // [LOD]
    lod_count: u32,
    flags: ModelFlags,
    aabb: AABB,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CompData {
    frustum: Frustum,
    model_instances: gpu::DevicePointer,  // [ModelInstanceFiltered]
    model_parts: gpu::DevicePointer,  // [ModelPart]
    part_count: gpu::DevicePointer,  // gpu::DrawMeshTasksIndirectCommandA - only X count is written
    max_model_part_count: u32,
    _padding: [u32; 1],
    camera_pos_and_viewport: Vec4,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CompFilterData {
    frustum: Frustum,
    model_instances: gpu::DevicePointer,  // [ModelInstance]
    model_instances_filtered: gpu::DevicePointer,  // [ModelInstanceFiltered]
    filtered_instance_count: gpu::DevicePointer,  // gpu::DrawMeshTasksIndirectCommandA - only X count is written
    instance_count: u32,
    padding: [u32; 1],
    camera_pos_and_viewport: Vec4,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CompResetData {
    counts: [gpu::DevicePointer; 2],  // gpu::DrawMeshTasksIndirectCommandA or compute equivalent
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
    meshlets: gpu::DevicePointer,  // [Meshlet]
    meshlet_infos: gpu::DevicePointer,  // [MeshletInfo]
    transform: gpu::DevicePointer,  // Mat4
    meshlet_offset: u32,
    meshlet_count: u16,
    // used for communication between compute and task shader, meanings do not matter here
    flags: u16,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ModelInstanceFiltered {
    lods: gpu::DevicePointer,  // [LOD]
    transform: gpu::DevicePointer,  // Mat4
    // used for communication between compute and task shader, meanings do not matter here
    flags: u32,
    lod: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ModelInstance {
    model: gpu::DevicePointer,  // Model
    transform: gpu::DevicePointer,  // Mat4
}

#[multi_allocation]
struct MultiAlloc {
    meshlet_infos: MeshletInfo,
    meshlets: Meshlet,
    lod_chunks: LODChunk,
    lods: LOD,
    models: Model,
    instance_transforms: Mat4,
    model_instances: ModelInstance,
}

#[multi_allocation]
struct MultiAllocGpu {
    model_parts: ModelPart,
    model_instances_filtered: ModelInstanceFiltered,
    part_count: gpu::DrawMeshTasksIndirectCommandA,
    filtered_instance_count: gpu::DispatchIndirectCommandA,
}

pub struct MeshModels<'a> {
    shader_comp2: gpu::Shader<'a>,
    shader_reset_comp2: gpu::Shader<'a>,
    shader_filter_comp2: gpu::Shader<'a>,
    shader_task2: gpu::Shader<'a>,
    shader_mesh2: gpu::Shader<'a>,
    shader_frag: gpu::Shader<'a>,

    alloc: MultiAlloc<'a>,
    alloc_gpu: MultiAllocGpu<'a>,
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
        // TODO: level of detail selection
        // TODO: incorrect output for fullyInside - always returns true
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

        let mut levels = vec![scene.indices.host().to_owned()];
        for i in 0..12 {
            let prev_indices = levels.last().unwrap().as_slice();
            let new_indices = meshopt::simplify(
                &prev_indices, &vertex_adapter, prev_indices.len() / 2,
                0.02, meshopt::SimplifyOptions::None, None);
            let ratio = new_indices.len() as f64 / prev_indices.len() as f64;
            if ratio > 0.8 {
                break;
            }
            println!("simplified [{}]: {}, prev {}", i + 1, new_indices.len(), prev_indices.len());
            levels.push(new_indices);
            if ratio > 0.6 {
                break;
            }
        }

        let level_meshlets = levels.iter()
            .map(|indices| meshopt::build_meshlets(
                &indices, &vertex_adapter, 64, 96, 0.0))
            .collect::<Vec<_>>();
        let total_meshlets = level_meshlets.iter().map(|m| m.len()).sum::<usize>();
        let level_model_parts = level_meshlets.iter().map(|m| m.len().div_ceil(MODEL_PART_SIZE)).collect::<Vec<_>>();

        let mut alloc = MultiAlloc::new(gpu, MultiAllocCounts {
            meshlet_infos: total_meshlets,
            meshlets: total_meshlets,
            lod_chunks: level_model_parts.iter().sum(),
            lods: level_meshlets.len(),
            models: 1,
            instance_transforms: repeat * repeat,
            model_instances: repeat * repeat,
        })?;
        let alloc_gpu = MultiAllocGpu::new_mem(gpu, MultiAllocGpuCounts {
            model_parts: repeat * repeat * level_model_parts[0],
            model_instances_filtered: repeat * repeat,
            part_count: 1,
            filtered_instance_count: 1,
        }, gpu::Memory::Gpu)?;

        let mut model_min_coord = Vec3::INFINITY;
        let mut model_max_coord = Vec3::NEG_INFINITY;
        let mut meshlet_offset = 0;
        let mut chunk_offset = 0;
        for (lod, meshlets) in level_meshlets.iter().enumerate() {
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

                alloc.meshlet_infos.host_mut()[meshlet_offset + i] = MeshletInfo {
                    aabb: AABB::from_min_max(min_coord, max_coord),
                    vertex_count: meshlet.vertex_count as u8,
                    triangle_count: meshlet.triangle_count as u8,
                    flags: MeshletInfoFlags::empty(),
                    _padding: [0u8; 5],
                };
                alloc.meshlets.host_mut()[meshlet_offset + i] = Meshlet {
                    vertices,
                    triangles,
                };
            }

            let part_count = level_model_parts[lod];

            for i in 0..part_count {
                let offset = (i * MODEL_PART_SIZE) as u32;
                let count = if i == part_count - 1 {
                    level_meshlets[lod].len() - (part_count - 1) * MODEL_PART_SIZE
                } else {
                    MODEL_PART_SIZE
                } as u32;

                let mut min_bound = Vec3::INFINITY;
                let mut max_bound = Vec3::NEG_INFINITY;
                for j in offset..offset + count {
                    let (min, max) = alloc.meshlet_infos.host()[meshlet_offset + j as usize].aabb.to_min_max();
                    min_bound = min_bound.min(min);
                    max_bound = max_bound.max(max);
                }

                alloc.lod_chunks.host_mut()[chunk_offset + i] = LODChunk {
                    aabb: AABB::from_min_max(min_bound, max_bound),
                    meshlet_offset: offset,
                    meshlet_count: count.try_into().expect("meshlet count does not fit into u16"),
                    _padding: [0u16; 1],
                };
            }

            alloc.lods.host_mut()[lod] = LOD {
                meshlets: alloc.meshlets.device().add_typed::<Meshlet>(meshlet_offset),
                meshlet_infos: alloc.meshlet_infos.device().add_typed::<MeshletInfo>(meshlet_offset),
                chunks: alloc.lod_chunks.device().add_typed::<LODChunk>(chunk_offset),
                chunk_count: part_count as u32,
                _padding: [0u32; 1],
            };

            meshlet_offset += meshlets.len();
            chunk_offset += part_count;
        }

        alloc.models.host_mut()[0] = Model {
            lods: alloc.lods.device(),
            lod_count: alloc.lods.len() as u32,
            flags: ModelFlags::EnableCulling | ModelFlags::EnableLODs,
            aabb: AABB::from_min_max(model_min_coord, model_max_coord),
            _padding: [0u32; 2],
        };

        for &(i, offset) in &offsets {
            alloc.instance_transforms.host_mut()[i] = Mat4::from_translation(Vec3::new(offset.x, 0.0, offset.y));
            alloc.model_instances.host_mut()[i] = ModelInstance {
                model: alloc.models.device().add_typed::<Model>(0),
                transform: alloc.instance_transforms.device().add_typed::<Mat4>(i),
            };
        }
        alloc.instance_transforms.host_mut().sort_unstable_by_key(|m| (m.z_axis.truncate().length() * 16384.0) as isize);

        println!("Clusterized scene in {:.03} seconds", start_time.elapsed().as_secs_f32());

        const SHADER_FILES: &'static [(&'static str, gpu::ShaderStage)] = &[
            ("shaders/task_model2.glsl", gpu::ShaderStage::Task),
            ("shaders/mesh_model2.glsl", gpu::ShaderStage::MeshWithTask),
            ("shaders/frag.glsl", gpu::ShaderStage::Pixel),
        ];
        let mut spirv = SmallVec::<[_; 3]>::new();
        for &(f, s) in SHADER_FILES {
            spirv.push((load_shader_spirv(f, s)?, s));
        }
        let [shader_task2, shader_mesh2, shader_frag] = gpu.create_shaders_linked_arr(
            spirv.iter().map(|(c, s)| (c.as_slice(), *s)))?;

        Ok(Self {
            shader_comp2: load_shader("shaders/comp_model2.glsl", gpu::ShaderStage::Compute, gpu)?,
            shader_reset_comp2: load_shader("shaders/comp_model2_reset.glsl", gpu::ShaderStage::Compute, gpu)?,
            shader_filter_comp2: load_shader("shaders/comp_model2_filter.glsl", gpu::ShaderStage::Compute, gpu)?,
            shader_task2,
            shader_mesh2,
            shader_frag,

            alloc,
            alloc_gpu,
        })
    }

    pub fn prepare_render(&self, arena: &gpu::Arena<'a>, cmd_buf: &mut gpu::CommandBuffer<'a>,
                          view_proj: &Mat4, camera_pos: Vec3A, viewport: f32) -> anyhow::Result<()> {
        // TODO: extreme frame time fluctuations

        let frustum = get_frustum(view_proj);

        let comp_reset_data = arena.alloc_data(&[CompResetData {
            counts: [
                self.alloc_gpu.part_count.device(),
                self.alloc_gpu.filtered_instance_count.device(),
            ]
        }])?;

        cmd_buf.bind_compute_shader(&self.shader_reset_comp2);
        cmd_buf.dispatch(comp_reset_data.device(), (1, 1, 1));

        cmd_buf.barrier(
            gpu::Stage::Compute,
            gpu::Stage::Compute,
            gpu::HazardFlags::ShaderMemory);

        let comp_filter_data = arena.alloc_data(&[CompFilterData {
            frustum,
            model_instances: self.alloc.model_instances.device(),
            model_instances_filtered: self.alloc_gpu.model_instances_filtered.device(),
            filtered_instance_count: self.alloc_gpu.filtered_instance_count.device(),
            instance_count: self.alloc.model_instances.len() as u32,
            padding: [0u32; 1],
            camera_pos_and_viewport: camera_pos.extend(viewport),
        }])?;

        cmd_buf.bind_compute_shader(&self.shader_filter_comp2);
        cmd_buf.dispatch(
            comp_filter_data.device(),
            (self.alloc.model_instances.len().div_ceil(64) as u32, 1, 1));

        cmd_buf.barrier(
            gpu::Stage::Compute,
            gpu::Stage::Compute,
            gpu::HazardFlags::ShaderMemory);
        cmd_buf.barrier(
            gpu::Stage::Compute,
            gpu::Stage::DrawIndirect,
            gpu::HazardFlags::IndirectDrawArguments);

        let comp_data = arena.alloc_data(&[CompData {
            frustum,
            model_instances: self.alloc_gpu.model_instances_filtered.device(),
            model_parts: self.alloc_gpu.model_parts.device(),
            part_count: self.alloc_gpu.part_count.device(),
            max_model_part_count: self.alloc_gpu.model_parts.len() as u32,
            _padding: [0u32; 1],
            camera_pos_and_viewport: camera_pos.extend(viewport),
        }])?;

        cmd_buf.bind_compute_shader(&self.shader_comp2);
        cmd_buf.dispatch_indirect(
            comp_data.device(),
            self.alloc_gpu.filtered_instance_count.device());

        cmd_buf.barrier(
            gpu::Stage::Compute,
            gpu::Stage::TaskShader,
            gpu::HazardFlags::ShaderMemory);
        cmd_buf.barrier(
            gpu::Stage::Compute,
            gpu::Stage::DrawIndirect,
            gpu::HazardFlags::IndirectDrawArguments);

        Ok(())
    }

    pub fn render(&self, arena: &gpu::Arena<'a>, cmd_buf: &mut gpu::CommandBuffer<'a>,
                  pixel_data: gpu::DevicePointer, materials: gpu::DevicePointer,
                  view_proj: &Mat4, camera_pos: Vec3A) -> anyhow::Result<()> {
        let mesh_data = arena.alloc_data(&[MeshData {
            view_proj: *view_proj,
            frustum: get_frustum(view_proj),
            model_parts: self.alloc_gpu.model_parts.device(),
            materials,
            camera_pos,
        }])?;

        cmd_buf.bind_shaders([&self.shader_task2, &self.shader_mesh2, &self.shader_frag]);
        cmd_buf.draw_meshlets_indirect(
            mesh_data.device(), pixel_data,
            self.alloc_gpu.part_count.device(),
            1, 16);

        Ok(())
    }
}
