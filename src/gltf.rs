use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use anyhow::{anyhow, Result};
use easy_gltf::model::Mode;
use glam::{Vec2, Vec3, Vec3Swizzles, Vec4};
use image::EncodableLayout;
use itertools::Itertools;
use gpu::{MemoryAllocation, MemoryAllocator};
use crate::application::{create_texture, Material, Vertex};

pub struct Scene<'a> {
    pub indices: gpu::Allocation<'a, u32>,
    pub vertices: gpu::Allocation<'a, Vertex>,
    pub materials: gpu::Allocation<'a, Material>,
    pub textures: Vec<gpu::Texture<'a>>,
}

pub struct Model<'a> {
    pub scenes: Vec<Scene<'a>>,
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

fn load_materials<'a, 'b>(material_iter: impl IntoIterator<Item = (&'b easy_gltf::Material, u32)>, count: usize, gpu: &'a gpu::Gpu,
                          tex_descriptors: &mut gpu::DescriptorHeap<'a>, tex_offset: &mut u16) -> Result<(Vec<gpu::Texture<'a>>, gpu::Allocation<'a, Material>)> {
    let mut texture_allocations = vec![];
    let mut textures = vec![];
    let queue = gpu.create_queue(gpu::QueueType::Graphics)?;
    let mut cmd_buf = queue.create_buffer()?;
    cmd_buf.begin_recording()?;  // TODO: move into create and submit methods

    let mut materials = gpu.alloc::<Material>(count)?;
    for (material, index) in material_iter.into_iter().sorted_unstable_by_key(|(_, i)| *i) {
        // TODO: emissive
        // TODO: pbr textures
        let pbr = &material.pbr;
        let color_tex = pbr.base_color_texture.as_ref().ok_or_else(|| anyhow!("material missing color texture"))?;
        let alloc = gpu.allocator().alloc_data(color_tex.as_bytes())?;
        let (tex, alloc) = create_texture(
            gpu, (color_tex.width(), color_tex.height()),
            alloc, gpu::Format::RGBA8UNorm, &mut cmd_buf)?;
        let tex_diffuse = *tex_offset;
        tex.view_descriptor(&mut tex_descriptors[tex_diffuse as usize])?;
        *tex_offset += 1;
        textures.push(tex);
        texture_allocations.push(alloc);

        let mut ambient_and_intensity = convert_vec!(4 pbr.base_color_factor);
        ambient_and_intensity.w = 0.0;

        let mat = Material {
            tex_disp: 0,
            tex_diffuse,

            ambient_and_intensity: Material::pack_vec4(ambient_and_intensity),
            diffuse_and_dissolve: Material::pack_vec4(Vec4::new(1.0, 1.0, 1.0, 0.0)),
            specular_and_exp: Material::pack_vec4(Vec4::new(0.0, 0.0, 0.0, 0.0)),
        };
        materials.host_mut()[index as usize] = mat;
    }

    cmd_buf.end_recording()?;
    queue.submit_no_signal(cmd_buf)?.wait();

    drop(texture_allocations);

    Ok((textures, materials))
}

fn load_scene<'a>(scene: easy_gltf::Scene, gpu: &'a gpu::Gpu,
                  tex_descriptors: &mut gpu::DescriptorHeap<'a>, tex_offset: &mut u16) -> Result<Scene<'a>> {
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
        if material_indices.try_insert(Arc::as_ptr(&material), (material, material_count)).is_ok() {
            material_count += 1;
        }
    }

    let (textures, materials) = load_materials(
        material_indices
            .values()
            .map(|(mat, idx)| (mat.as_ref(), *idx)),
        material_count as usize,
        gpu,
        tex_descriptors,
        tex_offset
    )?;

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
        textures,
    })
}

pub fn load_gltf<'a>(path: impl AsRef<Path>, gpu: &'a gpu::Gpu,
                     tex_descriptors: &mut gpu::DescriptorHeap<'a>, mut tex_offset: u16) -> Result<Model<'a>> {
    let scenes = easy_gltf::load(path)
        .map_err(|e| anyhow!("failed to load gltf: {}", e))?;

    Ok(Model {
        scenes: scenes.into_iter()
            .map(|scene| load_scene(scene, gpu, tex_descriptors, &mut tex_offset))
            .collect::<Result<_>>()?,
    })
}
