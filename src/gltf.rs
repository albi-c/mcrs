use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use anyhow::{anyhow, Result};
use easy_gltf::model::Mode;
use glam::{Vec2, Vec3, Vec3Swizzles, Vec4};
use gpu::MemoryAllocation;
use crate::application::{Material, Vertex};

pub struct Scene<'a> {
    indices: gpu::Allocation<'a, u32>,
    vertices: gpu::Allocation<'a, Vertex>,
    materials: gpu::Allocation<'a, Material>,
}

pub struct Model<'a> {
    scenes: Vec<Scene<'a>>,
}

trait ConvertVector : Copy + 'static {
    type Output: Copy + 'static;
    fn conv(self) -> Self::Output;
}

macro_rules! convert_vec {
    (2 $vec:expr) => {
        {
            let vec = $vec;
            Vec2::new(vec.x, vec.y)
        }
    };
    (3 $vec:expr) => {
        {
            let vec = $vec;
            Vec3::new(vec.x, vec.y, vec.z)
        }
    };
    (4 $vec:expr) => {
        {
            let vec = $vec;
            Vec4::new(vec.x, vec.y, vec.z, vec.w)
        }
    };
}

fn load_scene(scene: easy_gltf::Scene, gpu: &gpu::Gpu) -> Result<Scene<'_>> {
    let mut index_count = 0;
    let mut vertex_count = 0;
    let mut material_count = 0;
    let mut material_indices = HashMap::new();
    for model in &scene.models {
        match model.mode() {
            Mode::Triangles => {},
            _ => return Err(anyhow!("gltf model must be simple triangles")),
        }
        index_count += model.indices().ok_or_else(|| anyhow!("gltf model must have indices"))?.len();
        vertex_count += model.vertices().len();
        let material = model.material();
        if material_indices.try_insert(Arc::as_ptr(&material), (material, material_count)).is_err() {
            material_count += 1;
        }
    }

    let mut materials = gpu.alloc::<Material>(material_count as usize)?;
    for (material, index) in material_indices.values() {
        let mat = Material {
            tex_disp: 0,
            tex_diffuse: 0,

            ambient: 0,
            diffuse_and_dissolve: 0,
            specular_and_exp: 0,
        };
        materials.host_mut()[*index as usize] = mat;
    }

    let mut indices = gpu.alloc::<u32>(index_count)?;
    let mut vertices = gpu.alloc::<Vertex>(vertex_count)?;

    let mut index_offset = 0;
    let mut vertex_offset = 0;

    for model in scene.models {
        let material = material_indices.get(&Arc::as_ptr(&model.material())).unwrap().1;

        let mod_indices = model.indices().unwrap();
        let mod_vertices = model.vertices();

        // not a bug, should use vertex_offset
        let add_to_index = u32::try_from(vertex_offset)?;
        for (dst, src) in indices.host_mut()[index_offset..index_offset + mod_indices.len()]
            .iter_mut().zip(mod_indices.iter().copied()) {
            *dst = src + add_to_index;
        }
        index_offset += mod_indices.len();

        for (dst, src) in vertices.host_mut()[vertex_offset..vertex_offset + mod_vertices.len()]
            .iter_mut().zip(mod_vertices.iter()) {
            let normal = convert_vec!(3 src.normal).normalize();
            *dst = Vertex(
                convert_vec!(3 src.position),
                material,
                convert_vec!(2 src.tex_coords),
                normal.xy(),
            );
        }
        vertex_offset += mod_vertices.len();
    }

    Ok(Scene {
        indices,
        vertices,
        materials,
    })
}

pub fn load_gltf(path: impl AsRef<Path>, gpu: &gpu::Gpu) -> Result<Model<'_>> {
    let scenes = easy_gltf::load(path)
        .map_err(|e| anyhow!("failed to load gltf: {}", e))?;

    Ok(Model {
        scenes: scenes.into_iter().map(|scene| load_scene(scene, gpu)).collect::<Result<_>>()?,
    })
}
