use std::path::Path;
use anyhow::Result;
use gltf::Gltf;

pub fn load_gltf(path: impl AsRef<Path>) -> Result<()> {
    let gltf = Gltf::open(path)?;

    todo!()
}
