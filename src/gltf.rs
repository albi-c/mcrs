use std::path::Path;
use anyhow::{anyhow, Result};
use glam::{Vec2, Vec3};
use half::f16;
use itertools::izip;
use gpu::MemoryAllocation;
use crate::application::{Material, Vertex};

pub struct Scene<'a> {
    pub vertices: gpu::Allocation<'a, Vertex>,
    pub indices: gpu::Allocation<'a, u32>,
    pub materials: gpu::Allocation<'a, Material>,
    pub textures: Vec<gpu::Texture<'a>>,
}

pub struct Model<'a> {
    pub scenes: Vec<Scene<'a>>,
}

fn _visit_nodes_impl<'a>(nodes: impl IntoIterator<Item = gltf::Node<'a>>,
                         visitor: &mut impl FnMut(gltf::Node<'a>) -> Result<()>) -> Result<()> {
    for node in nodes {
        visitor(node.clone())?;
        _visit_nodes_impl(node.children(), visitor)?;
    }
    Ok(())
}

fn visit_nodes<'a>(nodes: impl IntoIterator<Item = gltf::Node<'a>>,
                   mut visitor: impl FnMut(gltf::Node<'a>) -> Result<()>) -> Result<()> {
    _visit_nodes_impl(nodes, &mut visitor)
}

fn visit_meshes<'a>(nodes: impl IntoIterator<Item = gltf::Node<'a>>,
                    mut visitor: impl FnMut(gltf::Node<'a>, gltf::Mesh<'a>) -> Result<()>) -> Result<()> {
    visit_nodes(nodes, |node| {
        if let Some(mesh) = node.mesh() {
            visitor(node, mesh)?;
        }
        Ok(())
    })
}

fn load_scene<'a>(gpu: &'a gpu::Gpu, scene: gltf::Scene<'_>, buffers: &[gltf::buffer::Data]) -> Result<Scene<'a>> {
    let mut vertex_count = 0;
    let mut index_count = 0;
    visit_meshes(scene.nodes(), |_node, mesh| {
        for prim in mesh.primitives() {
            // TODO: materials
            if prim.mode() != gltf::mesh::Mode::Triangles {
                return Err(anyhow!("gltf model must be simple triangles"));
            }

            let indices = prim.indices().ok_or_else(|| anyhow!("gltf model must have indices"))?;
            index_count += indices.count();

            let mut prev_count = None;
            for (name, sem) in [
                ("positions", gltf::Semantic::Positions),
                ("texture coordinates", gltf::Semantic::TexCoords(0)),
                ("normals", gltf::Semantic::Normals),
            ] {
                let acc = prim.get(&sem).ok_or_else(|| anyhow!("gltf model must have {}", name))?;
                if acc.count() != *prev_count.get_or_insert(acc.count()) {
                    return Err(anyhow!("gltf model has incorrect {} count", name));
                }
            }
            vertex_count += prev_count.unwrap();
        }
        Ok(())
    })?;

    let mut vertices_alloc = gpu.alloc(vertex_count)?;
    let vertices = vertices_alloc.host_mut();
    let mut vertex_index = 0;

    let mut indices_alloc = gpu.alloc(index_count)?;
    let indices = indices_alloc.host_mut();
    let mut index_index = 0;

    let mut materials_alloc = gpu.alloc(1)?;

    visit_meshes(scene.nodes(), |_node, mesh| {
        // TODO: use node.transform(), also use parent transform
        for prim in mesh.primitives() {
            assert_eq!(prim.mode(), gltf::mesh::Mode::Triangles);
            let reader = prim.reader(
                |buf| buffers.get(buf.index()).map(|b| b.0.as_slice()));

            let indices_src = reader.read_indices().unwrap().into_u32();
            let dst_index = index_index;
            let index_offset = u32::try_from(vertex_index).expect("too many indices");
            index_index += indices_src.len();
            for (dst, src) in indices[dst_index..].iter_mut().zip(indices_src) {
                *dst = src + index_offset;
            }

            for (pos, uv, nor) in izip!(
                reader.read_positions().unwrap().into_iter(),
                reader.read_tex_coords(0).unwrap().into_f32().into_iter(),
                reader.read_normals().unwrap().into_iter(),
            ) {
                let pos = Vec3::from_array(pos);
                let uv = Vec2::from_array(uv);
                let nor = Vec3::from_array(nor);
                let vert = Vertex(
                    pos,
                    0,  // TODO: material
                    uv,
                    [f16::from_f32(nor.x), f16::from_f32(nor.y), f16::from_f32(nor.z), f16::default()],
                );
                vertices[vertex_index] = vert;
                vertex_index += 1;
            }
        }
        Ok(())
    })?;

    Ok(Scene {
        vertices: vertices_alloc,
        indices: indices_alloc,
        materials: materials_alloc,
        textures: vec![],
    })
}

pub fn load_gltf<'a>(gpu: &'a gpu::Gpu, path: impl AsRef<Path>) -> Result<Model<'a>> {
    let start = std::time::Instant::now();
    let (document, buffers, _images) = gltf::import(path)?;
    let mut scenes = vec![];
    for scene in document.scenes() {
        scenes.push(load_scene(gpu, scene, &buffers)?);
    }
    println!("Loaded model in {} seconds", start.elapsed().as_secs_f64());
    Ok(Model {
        scenes,
    })
}

// use std::collections::HashMap;
// use std::path::Path;
// use std::sync::Arc;
// use anyhow::{anyhow, Result};
// use easy_gltf::model::Mode;
// use glam::{Vec2, Vec3, Vec4};
// use half::f16;
// use image::EncodableLayout;
// use itertools::Itertools;
// use gpu::{CommandBuffer, MemoryAllocation, MemoryAllocator};
// use crate::application::{create_texture, Material, Vertex};
//
// pub struct Scene<'a> {
//     pub indices: gpu::Allocation<'a, u32>,
//     pub vertices: gpu::Allocation<'a, Vertex>,
//     pub materials: gpu::Allocation<'a, Material>,
//     pub textures: Vec<gpu::Texture<'a>>,
// }
//
// pub struct Model<'a> {
//     pub scenes: Vec<Scene<'a>>,
// }
//
// macro_rules! convert_vec {
//     (2 $vec:expr) => {
//         {
//             let vec = $vec;
//             Vec2::new(vec.x, vec.y)
//         }
//     };
//     (3 $vec:expr) => {
//         {
//             let vec = $vec;
//             Vec3::new(vec.x, vec.y, vec.z)
//         }
//     };
//     (4 $vec:expr) => {
//         {
//             let vec = $vec;
//             Vec4::new(vec.x, vec.y, vec.z, vec.w)
//         }
//     };
// }
//
// fn add_texture<'a, 'b>(gpu: &'a gpu::Gpu, format: gpu::Format, width: u32, height: u32, data: &[u8],
//                        cmd_buf: &mut CommandBuffer<'b>, tex_descriptors: &mut gpu::DescriptorHeap<'a>,
//                        tex_offset: &mut u16) -> Result<(u16, gpu::Texture<'a>)> where 'a: 'b {
//     let alloc = gpu.allocator().alloc_data(data)?;
//     let tex = create_texture(gpu, (width, height), alloc, format, cmd_buf)?;
//     let tex_index = *tex_offset;
//     tex.view_descriptor(&mut tex_descriptors[tex_index as usize])?;
//     *tex_offset += 1;
//     Ok((tex_index, tex))
// }
//
// fn load_materials<'a, 'b>(material_iter: impl IntoIterator<Item = (&'b easy_gltf::Material, u32)>, count: usize, gpu: &'a gpu::Gpu,
//                           tex_descriptors: &mut gpu::DescriptorHeap<'a>, tex_offset: &mut u16) -> Result<(Vec<gpu::Texture<'a>>, gpu::Allocation<'a, Material>)> {
//     let mut textures = vec![];
//     let queue = gpu.create_queue(gpu::QueueType::Graphics)?;
//     let mut cmd_buf = queue.create_buffer()?;
//
//     let mut materials = gpu.alloc::<Material>(count)?;
//     for (material, index) in material_iter.into_iter().sorted_unstable_by_key(|(_, i)| *i) {
//         // TODO: emissive
//         let pbr = &material.pbr;
//
//         let (tex_diffuse, prev_tex) = if let Some(tex) = &pbr.base_color_texture {
//             let (tex_diffuse, tex) = add_texture(
//                 gpu, gpu::Format::RGBA8UNorm, tex.width(), tex.height(),
//                 tex.as_bytes(), &mut cmd_buf, tex_descriptors, tex_offset)?;
//             textures.push(tex);
//             (tex_diffuse, tex_diffuse)
//         } else {
//             ((1 << 15) | *tex_offset - 1, *tex_offset - 1)
//         };
//
//         let (tex_normal, normal_factor) = if let Some(normals) = &material.normal {
//             let tex = &normals.texture;
//             let src_data = tex.as_bytes();
//             let pixel_count = tex.width() as usize * tex.height() as usize;
//             let mut data = vec![0u8; pixel_count * 4];
//             let mut si = 0;
//             let mut di = 0;
//             for _ in 0..pixel_count {
//                 data[di + 0] = src_data[si + 0];
//                 data[di + 1] = src_data[si + 1];
//                 data[di + 2] = src_data[si + 2];
//                 data[di + 3] = 0;
//                 si += 3;
//                 di += 4;
//             }
//             let (tex_normal, tex) = add_texture(
//                 gpu, gpu::Format::RGBA8UNorm, tex.width(), tex.height(),
//                 &data, &mut cmd_buf, tex_descriptors, tex_offset)?;
//             textures.push(tex);
//             (tex_normal - prev_tex, normals.factor)
//         } else {
//             (0, 0.0)
//         };
//
//         let tex_metallic = if let Some(tex) = &pbr.metallic_texture {
//             let (tex_metallic, tex) = add_texture(
//                 gpu, gpu::Format::R8UNorm, tex.width(), tex.height(),
//                 tex.as_bytes(), &mut cmd_buf, tex_descriptors, tex_offset)?;
//             textures.push(tex);
//             tex_metallic - prev_tex
//         } else {
//             0
//         };
//
//         let tex_roughness = if let Some(tex) = &pbr.roughness_texture {
//             let (tex_roughness, tex) = add_texture(
//                 gpu, gpu::Format::R8UNorm, tex.width(), tex.height(),
//                 tex.as_bytes(), &mut cmd_buf, tex_descriptors, tex_offset)?;
//             textures.push(tex);
//             tex_roughness - prev_tex
//         } else {
//             0
//         };
//
//         let mut diffuse_and_normal = convert_vec!(4 pbr.base_color_factor);
//         diffuse_and_normal.w = normal_factor;
//
//         let mat = Material {
//             tex_offsets: Material::pack_tex_offsets(tex_normal, tex_metallic, tex_roughness),
//             tex_diffuse,
//
//             // ambient is unused
//             ambient_and_roughness: Material::pack_vec4(Vec4::new(0.0, 0.0, 0.0, pbr.roughness_factor)),
//             diffuse_and_normal: Material::pack_vec4(diffuse_and_normal),
//             // specular is unused
//             specular_and_exp: Material::pack_vec4(Vec4::new(0.0, 0.0, 0.0, pbr.metallic_factor)),
//         };
//         materials.host_mut()[index as usize] = mat;
//     }
//
//     queue.submit_no_signal(cmd_buf)?.wait();
//
//     Ok((textures, materials))
// }
//
// fn load_scene<'a>(scene: easy_gltf::Scene, gpu: &'a gpu::Gpu,
//                   tex_descriptors: &mut gpu::DescriptorHeap<'a>, tex_offset: &mut u16) -> Result<Scene<'a>> {
//     let mut index_count = 0;
//     let mut vertex_count = 0;
//     let mut material_count = 0;
//     let mut material_indices = HashMap::new();
//     for model in &scene.models {
//         match model.mode() {
//             Mode::Triangles => {},
//             _ => return Err(anyhow!("gltf model must be simple triangles")),
//         }
//         index_count += model.indices().ok_or_else(|| anyhow!("gltf model must have indices"))?.len();
//         vertex_count += model.vertices().len();
//         let material = model.material();
//         if material_indices.try_insert(Arc::as_ptr(&material), (material, material_count)).is_ok() {
//             material_count += 1;
//         }
//     }
//
//     let (textures, materials) = load_materials(
//         material_indices
//             .values()
//             .map(|(mat, idx)| (mat.as_ref(), *idx)),
//         material_count as usize,
//         gpu,
//         tex_descriptors,
//         tex_offset
//     )?;
//
//     let mut indices = gpu.alloc::<u32>(index_count)?;
//     let mut vertices = gpu.alloc::<Vertex>(vertex_count)?;
//
//     let mut index_offset = 0;
//     let mut vertex_offset = 0;
//
//     for model in scene.models {
//         let material = material_indices.get(&Arc::as_ptr(&model.material())).unwrap().1;
//
//         let mod_indices = model.indices().unwrap();
//         let mod_vertices = model.vertices();
//
//         // not a bug, should use vertex_offset
//         let add_to_index = u32::try_from(vertex_offset)?;
//         for (dst, src) in indices.host_mut()[index_offset..index_offset + mod_indices.len()]
//             .iter_mut().zip(mod_indices.iter().copied()) {
//             *dst = src + add_to_index;
//         }
//         index_offset += mod_indices.len();
//
//         for (dst, src) in vertices.host_mut()[vertex_offset..vertex_offset + mod_vertices.len()]
//             .iter_mut().zip(mod_vertices.iter()) {
//             let normal = convert_vec!(3 src.normal).normalize();
//             *dst = Vertex(
//                 convert_vec!(3 src.position),
//                 material,
//                 convert_vec!(2 src.tex_coords),
//                 [f16::from_f32(normal.x), f16::from_f32(normal.y), f16::from_f32(normal.z), f16::default()],
//             );
//         }
//         vertex_offset += mod_vertices.len();
//     }
//
//     Ok(Scene {
//         indices,
//         vertices,
//         materials,
//         textures,
//     })
// }
//
// pub fn load_gltf<'a>(path: impl AsRef<Path>, gpu: &'a gpu::Gpu,
//                      tex_descriptors: &mut gpu::DescriptorHeap<'a>, mut tex_offset: u16) -> Result<Model<'a>> {
//     let scenes = easy_gltf::load(path)
//         .map_err(|e| anyhow!("failed to load gltf: {}", e))?;
//
//     Ok(Model {
//         scenes: scenes.into_iter()
//             .map(|scene| load_scene(scene, gpu, tex_descriptors, &mut tex_offset))
//             .collect::<Result<_>>()?,
//     })
// }
