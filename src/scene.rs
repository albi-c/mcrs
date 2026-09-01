use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use anyhow::Result;
use russimp_ng as ai;

#[repr(transparent)]
struct SceneFsFile(fs::File);

impl ai::fs::FileOperations for SceneFsFile {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, ()> {
        self.0.read(buf).map_err(|_| ())
    }

    fn write(&mut self, buf: &[u8]) -> Result<usize, ()> {
        self.0.write(buf).map_err(|_| ())
    }

    fn tell(&mut self) -> usize {
        self.0.stream_position().expect("failed to get stream position")
            .try_into().expect("failed to convert u64 to usize")
    }

    fn size(&mut self) -> usize {
        self.0.stream_len().expect("failed to get stream length")
            .try_into().expect("failed to convert u64 to usize")
    }

    fn seek(&mut self, seek_from: SeekFrom) -> Result<(), ()> {
        self.0.seek(seek_from).map(|_| ()).map_err(|_| ())
    }

    fn flush(&mut self) {
        let _ = self.0.flush();
    }

    fn close(&mut self) {}
}

fn make_boxed_file_operations(file: impl ai::fs::FileOperations + 'static) -> Box<dyn ai::fs::FileOperations> {
    Box::new(file)
}

struct SceneFs;

impl ai::fs::FileSystem for SceneFs {
    fn open(&self, path: &str, mode: &str) -> Option<Box<dyn ai::fs::FileOperations>> {
        let mut opt = fs::OpenOptions::new();
        for ch in mode.chars() {
            match ch {
                'r' => { opt.read(true); },
                'w' => { opt.write(true); },
                'b' => {},
                _ => panic!("unknown mode character: '{}'", ch),
            };
        }
        opt.open(path).ok().map(|f| make_boxed_file_operations(SceneFsFile(f)))
    }
}

pub struct Node {
    transform: glam::Mat4,
    children: Vec<Node>,
}

pub struct Scene {}

impl Scene {
    pub fn load(path: &str) -> Result<Self> {
        let mut properties = ai::property::PropertyStore::default();
        properties.set_integer(b"AI_CONFIG_PP_LBW_MAX_WEIGHTS\0", 4);
        properties.set_float(b"AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE\0", 80.0);
        let scene = ai::scene::Scene::from_file_system_with_props(path, vec![
            ai::scene::PostProcess::ValidateDataStructure,
            ai::scene::PostProcess::FixOrRemoveInvalidData,
            ai::scene::PostProcess::Triangulate,
            ai::scene::PostProcess::GenerateSmoothNormals,
            ai::scene::PostProcess::GenerateUVCoords,
            ai::scene::PostProcess::TransformUVCoords,
            ai::scene::PostProcess::CalculateTangentSpace,
            ai::scene::PostProcess::LimitBoneWeights,
            ai::scene::PostProcess::RemoveRedundantMaterials,
            ai::scene::PostProcess::EmbedTextures,
            ai::scene::PostProcess::JoinIdenticalVertices,
            ai::scene::PostProcess::SortByPrimitiveType,
            ai::scene::PostProcess::OptimizeGraph,
            ai::scene::PostProcess::OptimizeMeshes,
            ai::scene::PostProcess::GenerateBoundingBoxes,
            ai::scene::PostProcess::ImproveCacheLocality,
        ], &mut SceneFs, &properties)?;

        Ok(Self {})
    }
}
