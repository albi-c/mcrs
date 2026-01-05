use std::fs;
use std::io::BufRead;
use std::path::Path;
use anyhow::{anyhow, Result};
use glam::{Vec2, Vec3};

#[derive(Debug, Copy, Clone, Default)]
pub struct Vertex {
    pub vertex: u32,
    pub tex_coord: u32,
    pub normal: u32,
    pub material: u32,
}

#[derive(Debug)]
pub struct Object {
    pub name: String,
    pub faces: Vec<[Vertex; 3]>,
}

#[derive(Debug)]
pub struct Model {
    pub material_library: Option<String>,
    pub objects: Vec<Object>,
    pub vertices: Vec<Vec3>,
    pub tex_coords: Vec<Vec2>,
    pub normals: Vec<Vec3>,
}

fn parse_to_array<'a, T: Copy + Default, const N: usize>(parts: impl IntoIterator<Item = &'a str>,
                                                         parse: impl Fn(&'a str) -> Result<T>,
                                                         line: &str) -> Result<[T; N]> {

    let mut parts = parts.into_iter();
    let mut result = [Default::default(); N];
    for res in result.iter_mut() {
        let part = parts.next().ok_or_else(|| anyhow!("unexpected end of line: {line}"))?;
        *res = parse(part)?;
    }
    // if parts.next().is_some() {
    //     return Err(anyhow!("expected end of line: {line}"));
    // }
    Ok(result)
}

fn parse_numbers<const N: usize>(line: &str) -> Result<[f32; N]> {
    parse_to_array(line.trim().split(" "), |num| Ok(num.parse()?), line)
}

impl Model {
    pub fn read(path: impl AsRef<Path>, get_material_id: impl Fn(&str) -> Result<u32>) -> Result<Self> {
        let mut file = fs::File::open_buffered(path)?;

        let mut material_library = None;
        let mut objects = vec![];

        let mut line = String::new();
        let mut object: Option<Object> = None;
        let mut smooth = true;
        let mut material = 0;

        let mut vertices = vec![];
        let mut tex_coords = vec![];
        let mut normals = vec![];
        loop {
            line.clear();
            if file.read_line(&mut line)? == 0 {
                break;
            }
            let line = line.trim();
            if line.is_empty() || line.as_bytes()[0] == b'#' {
                continue;
            }

            if let Some(rest) = line.strip_prefix("v ") {
                let vertex = Vec3::from_array(parse_numbers(rest)?);
                vertices.push(vertex);
            } else if let Some(rest) = line.strip_prefix("vt ") {
                let tex_coord = Vec2::from_array(parse_numbers(rest)?);
                tex_coords.push(tex_coord);
            } else if let Some(rest) = line.strip_prefix("vn ") {
                let normal = Vec3::from_array(parse_numbers(rest)?);
                normals.push(normal);
            } else if let Some(rest) = line.strip_prefix("f ") {
                let face = parse_to_array(rest.trim().split(" "), |part| {
                    let [a, b, c]: [u32; 3] = parse_to_array(part.split("/"), |num| Ok(num.parse()?), part)?;
                    Ok(Vertex {
                        vertex: a - 1,
                        tex_coord: b - 1,
                        normal: c - 1,
                        material: material | ((smooth as u32) << 31),
                    })
                }, rest)?;
                object.get_or_insert_with(|| Object {
                    name: "".into(),
                    faces: vec![],
                }).faces.push(face);
            } else if let Some(rest) = line.strip_prefix("s ") {
                smooth = rest != "off";
            } else if let Some(rest) = line.strip_prefix("o ") {
                smooth = true;
                if let Some(obj) = object.replace(Object {
                    name: rest.into(),
                    faces: vec![],
                }) {
                    objects.push(obj);
                }
            } else if let Some(_rest) = line.strip_prefix("g ") {
                smooth = true;
            } else if let Some(rest) = line.strip_prefix("usemtl ") {
                material = get_material_id(rest)?;
            } else if let Some(rest) = line.strip_prefix("mtllib ") {
                material_library = Some(rest.to_string());
            } else if let Some(_) = line.strip_prefix("l ") {
                return Err(anyhow!("lines are not supported"));
            } else {
                return Err(anyhow!("invalid line: {}", line));
            }
        }
        if let Some(obj) = object {
            objects.push(obj);
        }

        Ok(Self {
            material_library,
            objects,
            vertices,
            tex_coords,
            normals,
        })
    }
}
