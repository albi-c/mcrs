use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use half::f16;
use gpu::{MemoryAllocation, MemoryAllocator};
use crate::application::load_shader;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Frustum([Vec4; 5]);

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct AABB {
    center: Vec3,
    extent: Vec3,
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

fn pack_normal(normal: Vec3) -> u32 {
    let x = (((normal.x + 1.0) * 1024.0) as u32).clamp(0, 0x7ff);
    let y = (((normal.y + 1.0) * 512.0) as u32).clamp(0, 0x3ff);
    let z = (((normal.z + 1.0) * 1024.0) as u32).clamp(0, 0x7ff);
    x | (y << 11) | (z << 21)
}

fn pack_triangle(indices: [usize; 3]) -> u32 {
    u32::from_le_bytes([indices[0] as u8, indices[1] as u8, indices[2] as u8, 0u8])
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Meshlet {
    vertices: [Vertex; 64],
    // packed: each uint: 0..8 - idx 1, 8..16 - idx 2, 16..24 - idx 3
    triangles: [u32; 126],
    _padding: [u32; 2],
}

unsafe impl Pod for Meshlet {}
unsafe impl Zeroable for Meshlet {}

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
    models_pointers: gpu::DevicePointer,  // [*Model]
    materials: gpu::DevicePointer,  // [UVec4]
}

pub struct MeshModels<'a> {
    shader_task: gpu::Shader<'a>,
    shader_mesh: gpu::Shader<'a>,
    shader_frag: gpu::Shader<'a>,

    meshlet_transforms: gpu::Allocation<'a, Mat4>,
    meshlet_infos: gpu::Allocation<'a, MeshletInfo>,
    meshlets: gpu::Allocation<'a, Meshlet>,
    models: gpu::Allocation<'a, Model>,
    model_transforms: gpu::Allocation<'a, Mat4>,
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

impl<'a> MeshModels<'a> {
    pub fn new(gpu: &'a gpu::Gpu) -> anyhow::Result<Self> {
        let aabb = AABB {
            center: Vec3::new(2.0, 2.0, 2.0),
            extent: Vec3::new(2.0, 0.0, 2.0),
        };

        let mut vertices = [Default::default(); 64];
        for z in 0..5 {
            for x in 0..5 {
                vertices[x + 5 * z] = Vertex {
                    pos: [
                        f16::from_f32(x as f32),
                        f16::from_f32(2.0),
                        f16::from_f32(z as f32),
                    ],
                    mat: 0,
                    uv: [
                        f16::from_f32(0.5),
                        f16::from_f32(0.5),
                    ],
                    normal: pack_normal(Vec3::new(0.0, 1.0, 0.0)),
                }
            }
        }

        let mut triangles = [0u32; 126];
        for z in 0..4 {
            for x in 0..4 {
                let base = x + 5 * z;
                let next_row = x + 5 * (z + 1);
                triangles[2 * (x + 4 * z) + 0] = pack_triangle([base, base + 1, next_row]);
                triangles[2 * (x + 4 * z) + 1] = pack_triangle([base, next_row, next_row + 1]);
            }
        }

        let meshlet_transforms = gpu.allocator().alloc_data(&[
            Mat4::IDENTITY,
        ])?;
        let meshlet_infos = gpu.allocator().alloc_data(&[
            MeshletInfo {
                // TODO: somehow modify model AABB when using meshlet transform to fix top level culling
                // probably just disable top level culling when meshlet matrix can change
                aabb,
                transform: meshlet_transforms.device().add_typed::<Mat4>(0),
                vertex_count: 25,
                triangle_count: 32,
                flags: MeshletInfoFlags::empty(),
                _padding: [0u8; 13],
            },
        ])?;
        let meshlets = gpu.allocator().alloc_data(&[
            Meshlet {
                vertices,
                triangles,
                _padding: [0u32; 2],
            }
        ])?;
        let model_transforms = gpu.allocator().alloc_data(&[
            Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0)),
        ])?;
        let models = gpu.allocator().alloc_data(&[
            Model {
                meshlets: meshlets.device(),
                meshlet_infos: meshlet_infos.device(),
                transform: model_transforms.device().add_typed::<Mat4>(0),
                meshlet_count: 1,
                flags: ModelFlags::EnableCulling,
                aabb,
                _padding: [0u32; 2],
            },
        ])?;

        // TODO: instead of many small allocations, put data for one model into one large allocation

        Ok(Self {
            shader_task: load_shader("shaders/task_model.glsl", gpu::ShaderStage::Task, gpu)?,
            shader_mesh: load_shader("shaders/mesh_model.glsl", gpu::ShaderStage::MeshWithTask, gpu)?,
            // TODO: use normal shader once everything else works
            // shader_frag: load_shader("shaders/frag.glsl", gpu::ShaderStage::Pixel, gpu)?,
            shader_frag: load_shader("shaders/frag_dummy.glsl", gpu::ShaderStage::Pixel, gpu)?,

            meshlet_transforms,
            meshlet_infos,
            meshlets,
            model_transforms,
            models,
        })
    }

    pub fn render(&self, arena: &gpu::Arena<'a>, cmd_buf: &mut gpu::CommandBuffer<'a>,
                  pixel_data: gpu::DevicePointer, materials: gpu::DevicePointer,
                  view_proj: &Mat4) -> anyhow::Result<()> {
        let model_pointers = arena.alloc_data(&[
            self.models.device().add_typed::<Model>(0),
        ])?;

        let mesh_data = arena.alloc_data(&[MeshData {
            view_proj: *view_proj,
            frustum: get_frustum(view_proj),
            models_pointers: model_pointers.device(),
            materials,
        }])?;

        cmd_buf.bind_shaders([&self.shader_task, &self.shader_mesh, &self.shader_frag]);
        cmd_buf.draw_meshlets(
            mesh_data.device(), pixel_data, (self.models.len() as u32, 1, 1));

        Ok(())
    }
}
