use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;
use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Quat, Vec2, Vec3, Vec3A, Vec4, Vec4Swizzles};
use half::f16;
use itertools::izip;
use gpu::{MemoryAllocation, MemoryAllocator};
use crate::application::{create_texture, Material, Vertex};

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct TextureInfo {
    width: u32,
    height: u32,
    offset: u32,
    components: u32,
}

impl TextureInfo {
    fn new(width: u32, height: u32, offset: u32, components: u32) -> Self {
        Self {
            width,
            height,
            offset,
            components,
        }
    }
}

pub struct Scene<'a> {
    pub vertices: gpu::Allocation<'a, Vertex>,
    pub indices: gpu::Allocation<'a, u32>,
    pub materials: gpu::Allocation<'a, Material>,
    pub textures: Vec<gpu::Texture<'a>>,
    texture_offset: u16,
    texture_info: Vec<TextureInfo>,
    texture_data: Vec<u8>,
}

pub struct Model<'a> {
    pub scenes: Vec<Scene<'a>>,
}

fn _visit_nodes_impl<'a>(nodes: impl IntoIterator<Item = gltf::Node<'a>>,
                         visitor: &mut impl FnMut(gltf::Node<'a>, &Mat4, &Mat3) -> Result<()>,
                         transform: &Mat4, scale_transform: &Mat3) -> Result<()> {
    for node in nodes {
        let mat = match node.transform() {
            gltf::scene::Transform::Matrix { matrix } => {
                Mat4::from_cols_array_2d(&matrix)
            },
            gltf::scene::Transform::Decomposed { translation, rotation, scale } => {
                Mat4::from_scale_rotation_translation(
                    Vec3::from_array(scale),
                    Quat::from_array(rotation).normalize(),
                    Vec3::from_array(translation),
                )
            },
        };
        let tr = transform * mat;
        let s_mat = Mat3::from_cols(
            mat.x_axis.xyz(),
            mat.y_axis.xyz(),
            mat.z_axis.xyz(),
        );
        let s_tr = scale_transform * s_mat;
        visitor(node.clone(), &tr, &s_tr)?;
        _visit_nodes_impl(node.children(), visitor, &tr, &s_tr)?;
    }
    Ok(())
}

fn visit_nodes<'a>(nodes: impl IntoIterator<Item = gltf::Node<'a>>,
                   mut visitor: impl FnMut(gltf::Node<'a>, &Mat4, &Mat3) -> Result<()>) -> Result<()> {
    _visit_nodes_impl(nodes, &mut visitor, &Mat4::default(), &Mat3::default())
}

fn visit_meshes<'a>(nodes: impl IntoIterator<Item = gltf::Node<'a>>,
                    mut visitor: impl FnMut(gltf::Node<'a>, &Mat4, &Mat3, gltf::Mesh<'a>) -> Result<()>) -> Result<()> {
    visit_nodes(nodes, |node, transform, scale_transform| {
        if let Some(mesh) = node.mesh() {
            visitor(node, transform, scale_transform, mesh)?;
        }
        Ok(())
    })
}

fn add_texture<'a, 'b>(gpu: &'a gpu::Gpu, format: gpu::Format, width: u32, height: u32, data: &[u8],
                       cmd_buf: &mut gpu::CommandBuffer<'b>, tex_descriptors: &mut gpu::DescriptorHeap<'a>,
                       tex_offset: &mut u16) -> Result<(u16, gpu::Texture<'a>)> where 'a: 'b {
    let alloc = gpu.allocator().alloc_data(data)?;
    let tex = create_texture(gpu, (width, height), alloc, format, cmd_buf)?;
    let tex_index = *tex_offset;
    tex.view_descriptor(&mut tex_descriptors[tex_index as usize])?;
    *tex_offset += 1;
    Ok((tex_index, tex))
}

fn expand_rgb_to_rgba(src: &[u8], expand: u8) -> Vec<u8> {
    let count = src.len() / 3;
    let mut dst = Vec::with_capacity(count * 4);
    let mut i = 0;
    for _ in 0..count {
        dst.push(src[i + 0]);
        dst.push(src[i + 1]);
        dst.push(src[i + 2]);
        dst.push(expand);
        i += 3;
    }
    dst
}

fn shrink_to_gb(pixel_size: usize, src: &[u8]) -> Vec<u8> {
    let count = src.len() / pixel_size;
    let mut dst = Vec::with_capacity(count * 2);
    let mut i = 0;
    for _ in 0..count {
        dst.push(src[i + 1]);
        dst.push(src[i + 2]);
        i += pixel_size;
    }
    dst
}

fn load_material<'a, 'b>(gpu: &'a gpu::Gpu, mat: &gltf::Material<'_>, cmd_buf: &mut gpu::CommandBuffer<'b>,
                         tex_descriptors: &mut gpu::DescriptorHeap<'a>, tex_offset: &mut u16,
                         images: &[gltf::image::Data], textures: &mut Vec<gpu::Texture<'a>>,
                         tex_info: &mut Vec<TextureInfo>, tex_data: &mut Vec<u8>) -> Result<Material> where 'a: 'b {
    let pbr = mat.pbr_metallic_roughness();

    let (tex_diffuse, base_tex) = if let Some(info) = pbr.base_color_texture() {
        if info.tex_coord() != 0 {
            return Err(anyhow!("diffuse uv index is {}, must be 0", info.tex_coord()));
        }
        let tex = &images[info.texture().source().index()];
        let data = if tex.format == gltf::image::Format::R8G8B8 {
            Cow::Owned(expand_rgb_to_rgba(&tex.pixels, 0xff))
        } else if tex.format == gltf::image::Format::R8G8B8A8 {
            Cow::Borrowed(&tex.pixels)
        } else {
            return Err(anyhow!("invalid diffuse texture format: {:?}", tex.format));
        };
        tex_info.push(TextureInfo::new(tex.width, tex.height, u32::try_from(tex_data.len())
            .expect("too much texture data"), 4));
        tex_data.extend_from_slice(&data);
        let (tex_diffuse, tex) = add_texture(
            gpu, gpu::Format::RGBA8UNorm, tex.width, tex.height, &data,
            cmd_buf, tex_descriptors, tex_offset)?;
        textures.push(tex);
        (tex_diffuse, tex_diffuse)
    } else {
        (0x8000, *tex_offset)
    };

    let tex_normal = if let Some(info) = mat.normal_texture() {
        if info.tex_coord() != 0 {
            return Err(anyhow!("normal uv index is {}, must be 0", info.tex_coord()));
        }
        let tex = &images[info.texture().source().index()];
        let data = if tex.format == gltf::image::Format::R8G8B8 {
            Cow::Owned(expand_rgb_to_rgba(&tex.pixels, 0x0))
        } else if tex.format == gltf::image::Format::R8G8B8A8 {
            Cow::Borrowed(&tex.pixels)
        } else {
            return Err(anyhow!("invalid normal texture format: {:?}", tex.format));
        };
        tex_info.push(TextureInfo::new(tex.width, tex.height, u32::try_from(tex_data.len())
            .expect("too much texture data"), 4));
        tex_data.extend_from_slice(&data);
        let (tex_normal, tex) = add_texture(
            gpu, gpu::Format::RGBA8UNorm, tex.width, tex.height, &data,
            cmd_buf, tex_descriptors, tex_offset)?;
        textures.push(tex);
        tex_normal - base_tex
    } else {
        0
    };

    let tex_metallic_roughness = if let Some(info) = pbr.metallic_roughness_texture() {
        if info.tex_coord() != 0 {
            return Err(anyhow!("metallic_roughness index is {}, must be 0", info.tex_coord()));
        }
        let tex = &images[info.texture().source().index()];
        let data = if tex.format == gltf::image::Format::R8G8B8 {
            Cow::Owned(shrink_to_gb(3, &tex.pixels))
        } else if tex.format == gltf::image::Format::R8G8B8A8 {
            Cow::Owned(shrink_to_gb(4, &tex.pixels))
        } else if tex.format == gltf::image::Format::R8G8 {
            Cow::Borrowed(&tex.pixels)
        } else {
            return Err(anyhow!("invalid metallic_roughness texture format: {:?}", tex.format));
        };
        tex_info.push(TextureInfo::new(tex.width, tex.height, u32::try_from(tex_data.len())
            .expect("too much texture data"), 2));
        tex_data.extend_from_slice(&data);
        let (tex_metallic_roughness, tex) = add_texture(
            gpu, gpu::Format::RG8UNorm, tex.width, tex.height, &data,
            cmd_buf, tex_descriptors, tex_offset)?;
        textures.push(tex);
        tex_metallic_roughness - base_tex
    } else {
        0
    };

    if mat.occlusion_texture().is_some() {
        return Err(anyhow!("occlusion textures are not allowed"));
    }

    let mut diffuse_and_normal = Vec4::from_array(pbr.base_color_factor());
    diffuse_and_normal.w = 1.0;

    Ok(Material {
        tex_offsets: Material::pack_tex_offsets(tex_normal, tex_metallic_roughness),
        tex_diffuse,

        ambient_and_roughness: Material::pack_vec4(Vec4::new(0.0, 0.0, 0.0, pbr.roughness_factor())),
        diffuse_and_normal: Material::pack_vec4(diffuse_and_normal),
        specular_and_exp: Material::pack_vec4(Vec4::new(0.0, 0.0, 0.0, pbr.metallic_factor())),
    })
}

fn load_scene<'a>(gpu: &'a gpu::Gpu, scene: gltf::Scene<'_>, buffers: &[gltf::buffer::Data],
                  images: &[gltf::image::Data], tex_descriptors: &mut gpu::DescriptorHeap<'a>,
                  tex_offset: &mut u16) -> Result<Scene<'a>> {
    let start_tex_offset = *tex_offset;
    let mut tex_info = vec![];
    let mut tex_data = vec![];

    let mut vertex_count = 0;
    let mut index_count = 0;
    let mut material_indices = HashMap::new();
    visit_meshes(scene.nodes(), |_node, _transform, _scale_transform, mesh| {
        for prim in mesh.primitives() {
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

            let mat = prim.material();
            let mat_index = mat.index().ok_or_else(|| anyhow!("gltf model must have material index"))?;
            let _ = material_indices.try_insert(mat_index, (material_indices.len(), mat));
        }
        Ok(())
    })?;

    let mut vertices_alloc = gpu.alloc(vertex_count)?;
    let vertices = vertices_alloc.host_mut();
    let mut vertex_index = 0;

    let mut indices_alloc = gpu.alloc(index_count)?;
    let indices = indices_alloc.host_mut();
    let mut index_index = 0;

    let mut materials_alloc = gpu.alloc(material_indices.len())?;
    let materials = materials_alloc.host_mut();
    let mut textures = vec![];

    let queue = gpu.create_queue(gpu::QueueType::Graphics)?;
    let mut cmd_buf = queue.create_buffer()?;
    for (index, src) in material_indices.values() {
        materials[*index] = load_material(
            gpu, src, &mut cmd_buf, tex_descriptors, tex_offset, images,
            &mut textures, &mut tex_info, &mut tex_data)?;
    }
    let queue_submit = queue.submit_no_signal(cmd_buf)?;

    visit_meshes(scene.nodes(), |_node, transform, scale_transform, mesh| {
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

            let (mat, _) = material_indices.get(&prim.material().index().unwrap())
                .expect("material index not in map");
            let mat = u16::try_from(*mat).expect("too many materials");

            for (pos, uv, nor) in izip!(
                reader.read_positions().unwrap().into_iter(),
                reader.read_tex_coords(0).unwrap().into_f32().into_iter(),
                reader.read_normals().unwrap().into_iter(),
            ) {
                let pos = transform.transform_point3a(Vec3A::from_array(pos));
                let uv = Vec2::from_array(uv);
                let nor = (scale_transform * Vec3A::from_array(nor)).normalize();
                let vert = Vertex(
                    [f16::from_f32(pos.x), f16::from_f32(pos.y), f16::from_f32(pos.z)],
                    mat,
                    [f16::from_f32(uv.x), f16::from_f32(uv.y)],
                    Vertex::pack_normal(nor),
                );
                vertices[vertex_index] = vert;
                vertex_index += 1;
            }
        }
        Ok(())
    })?;

    meshopt::optimize::optimize_vertex_cache_in_place(indices, vertices.len());
    meshopt::optimize::optimize_overdraw_in_place_decoder(indices, vertices, 1.05);
    meshopt::optimize::optimize_vertex_fetch_in_place(indices, vertices);

    queue_submit.wait();

    Ok(Scene {
        vertices: vertices_alloc,
        indices: indices_alloc,
        materials: materials_alloc,
        textures,
        texture_offset: start_tex_offset,
        texture_info: tex_info,
        texture_data: tex_data,
    })
}

impl<'a> Model<'a> {
    pub fn load(gpu: &'a gpu::Gpu, path: impl AsRef<Path>,
                tex_descriptors: &mut gpu::DescriptorHeap<'a>,
                mut tex_offset: u16) -> Result<Self> {
        let start = std::time::Instant::now();
        let (document, buffers, images) = gltf::import(path)?;
        let mut scenes = vec![];
        for scene in document.scenes() {
            scenes.push(load_scene(
                gpu, scene, &buffers, &images, tex_descriptors, &mut tex_offset)?);
        }
        println!("Loaded {} scenes in {:.3} seconds", scenes.len(), start.elapsed().as_secs_f64());
        Ok(Self {
            scenes,
        })
    }

    pub fn serialize(&self, dst: &mut impl Write) -> Result<()> {
        dst.write(&self.scenes.len().to_le_bytes())?;
        for scene in &self.scenes {
            scene.serialize(dst)?;
        }
        Ok(())
    }

    pub fn deserialize(gpu: &'a gpu::Gpu, tex_descriptors: &mut gpu::DescriptorHeap<'a>,
                       src: &mut impl Read) -> Result<Self> {
        let start = std::time::Instant::now();
        let mut count = [0usize];
        src.read_exact(bytemuck::cast_slice_mut(&mut count))?;
        let mut scenes = vec![];
        for _ in 0..count[0] {
            scenes.push(Scene::deserialize(gpu, tex_descriptors, src)?);
        }
        println!("Loaded {} scenes in {:.3} seconds", scenes.len(), start.elapsed().as_secs_f64());
        Ok(Self {
            scenes,
        })
    }
}

fn write_raw<T: Pod>(dst: &mut impl Write, data: &[T]) -> Result<()> {
    dst.write(bytemuck::cast_slice(data))?;
    Ok(())
}

fn read_raw<T: Pod>(src: &mut impl Read, data: &mut [T]) -> Result<()> {
    src.read_exact(bytemuck::cast_slice_mut(data))?;
    Ok(())
}

impl<'a> Scene<'a> {
    pub fn serialize(&self, dst: &mut impl Write) -> Result<()> {
        assert_eq!(self.textures.len(), self.texture_info.len());

        dst.write(bytemuck::cast_slice(&[
            u32::try_from(self.vertices.len())?,
            u32::try_from(self.indices.len())?,
            u32::try_from(self.materials.len())?,
        ]))?;
        dst.write(bytemuck::cast_slice(&[
            u16::try_from(self.textures.len())?,
            self.texture_offset,
        ]))?;
        dst.write(bytemuck::cast_slice(&[self.texture_data.len()]))?;

        write_raw(dst, &self.vertices.host())?;
        write_raw(dst, &self.indices.host())?;
        write_raw(dst, &self.materials.host())?;
        write_raw(dst, &self.texture_info)?;
        write_raw(dst, &self.texture_data)?;

        Ok(())
    }

    pub fn deserialize(gpu: &'a gpu::Gpu, tex_descriptors: &mut gpu::DescriptorHeap<'a>,
                       src: &mut impl Read) -> Result<Self> {
        let mut lengths = [0u32; 3];
        src.read_exact(bytemuck::cast_slice_mut(&mut lengths))?;
        let mut tex_lengths = [0u16; 2];
        src.read_exact(bytemuck::cast_slice_mut(&mut tex_lengths))?;
        let mut tex_info_len = [0usize];
        src.read_exact(bytemuck::cast_slice_mut(&mut tex_info_len))?;
        let [len_vertices, len_indices, len_materials] = lengths;
        let [len_textures, texture_offset] = tex_lengths;
        let [len_tex_data] = tex_info_len;

        let mut vertices = gpu.alloc(len_vertices as usize)?;
        let mut indices = gpu.alloc(len_indices as usize)?;
        let mut materials = gpu.alloc(len_materials as usize)?;
        let mut textures = Vec::with_capacity(len_textures as usize);
        let mut texture_info = vec![TextureInfo::zeroed(); len_textures as usize];
        let mut texture_data = vec![0u8; len_tex_data];

        read_raw(src, vertices.host_mut())?;
        read_raw(src, indices.host_mut())?;
        read_raw(src, materials.host_mut())?;
        read_raw(src, &mut texture_info)?;
        read_raw(src, &mut texture_data)?;

        let queue = gpu.create_queue(gpu::QueueType::Graphics)?;
        let mut cmd_buf = queue.create_buffer()?;

        let mut tex_offset = texture_offset;
        for &info in &texture_info {
            let format = match info.components {
                2 => gpu::Format::RG8UNorm,
                4 => gpu::Format::RGBA8UNorm,
                _ => return Err(anyhow!("invalid texture component count: {}", info.components)),
            };
            let offset = info.offset as usize;
            let length = info.width as usize * info.height as usize * info.components as usize;
            let data = &texture_data[offset..offset + length];
            let (_, tex) = add_texture(
                gpu, format, info.width, info.height, data,
                &mut cmd_buf, tex_descriptors, &mut tex_offset)?;
            textures.push(tex);
        }

        queue.submit_no_signal(cmd_buf)?.wait();

        Ok(Self {
            vertices,
            indices,
            materials,
            textures,
            texture_offset,
            texture_info,
            texture_data,
        })
    }
}
