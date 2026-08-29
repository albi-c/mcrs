use std::cell::RefCell;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use dear_imgui_rs::{BackendFlags, Context, DrawCmd, SnapshotTextureId, SynchronousRendererConsumer, TextureFormat, TextureId, TextureOp};
use glam::{Mat4, Vec2};
use gpu::{MemoryAllocation, MemoryAllocator, TextureUsageFlags};
use crate::application::load_shader_spirv;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    pos: Vec2,
    uv: Vec2,
    color: u32,
    // bit 32: use linear sampler, bit 31: no texture, bits 16..1: texture id
    texture: u32,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct VertData {
    proj: Mat4,
    vertices: gpu::DevicePointer,  // [Vertex]
    _padding: u64,
}

pub struct Imgui<'a> {
    gpu: &'a gpu::Gpu,
    context: RefCell<Context>,
    consumer: SynchronousRendererConsumer,
    shaders: [gpu::Shader<'a>; 2],
    textures: RefCell<Box<[Option<gpu::Texture<'a>>]>>,
    texture_offest: u16,
}

impl<'a> Imgui<'a> {
    pub fn new(gpu: &'a gpu::Gpu, texture_offest: u16) -> Result<Self> {
        assert_ne!(texture_offest, 0, "0 means a null texture");

        let mut context = Context::create();
        let consumer = context.create_synchronous_renderer_consumer()?;
        let flags = context.io().backend_flags()
            | BackendFlags::RENDERER_HAS_TEXTURES | BackendFlags::RENDERER_HAS_VTX_OFFSET;
        context.io_mut().set_backend_flags(flags);
        context.style_mut().scale_all_sizes(1.5);
        context.style_mut().set_font_scale_dpi(1.5);

        let spirv_vert = load_shader_spirv(
            "shaders/vert_imgui.glsl", gpu::ShaderStage::Vertex)?;
        let spirv_frag = load_shader_spirv(
            "shaders/frag_imgui.glsl", gpu::ShaderStage::Pixel)?;
        let shaders = gpu.create_shaders_linked_arr([
            (spirv_vert.as_slice(), gpu::ShaderStage::Vertex),
            (spirv_frag.as_slice(), gpu::ShaderStage::Pixel),
        ])?;

        Ok(Self {
            gpu,
            context: RefCell::new(context),
            consumer,
            shaders,
            textures: RefCell::new(Vec::from_fn(1, |_| None).into_boxed_slice()),
            texture_offest,
        })
    }

    pub fn io<R>(&self, func: impl FnOnce(&mut dear_imgui_rs::Io) -> R) -> R {
        func(self.context.borrow_mut().io_mut())
    }

    pub fn frame<R>(&self, func: impl FnOnce(&dear_imgui_rs::Ui) -> R) -> R {
        func(self.context.borrow_mut().frame())
    }

    pub fn start_frame(&self, screen_size: (u32, u32), dt: f32) {
        self.io(|io| {
            io.set_display_size([screen_size.0 as f32, screen_size.1 as f32]);
            io.set_delta_time(dt);
        });
    }

    pub fn render<'b>(&self, cmd_buf: &mut gpu::CommandBuffer<'b>, arena: &gpu::Arena<'_>,
                      screen_size: (u32, u32), color_target: gpu::Target<'_>,
                      tex_descriptors: &mut gpu::DescriptorHeap<'_>) -> Result<()> where 'a: 'b {
        let mut context = self.context.borrow_mut();

        let pending = context.render(&self.consumer);
        let mut feedback = vec![];
        for request in pending.texture_requests() {
            // TODO: do not unwrap_or(0) when subtraction underflows, this only works because texture count is limited to 1
            let req_tex: usize = match request.texture() {
                SnapshotTextureId::User(id) => id.context_id(),
                SnapshotTextureId::FontAtlas { context, .. } => context,
            }.get().get().checked_sub(self.texture_offest as u64).unwrap_or(0)
                .try_into().expect("invalid texture id");

            feedback.push(match request.operation() {
                TextureOp::Create { format, width, height, row_pitch, pixels } => {
                    let mut idx = 0;
                    for (i, tex) in self.textures.borrow_mut().iter_mut().enumerate() {
                        if tex.is_some() {
                            continue;
                        }
                        idx = self.texture_offest as u64 + i as u64;

                        let texture = self.gpu.create_texture(gpu::TextureDesc {
                            ty: gpu::TextureType::Tex2D,
                            dimensions: (*width, *height, 1),
                            format: match *format {
                                TextureFormat::RGBA32 => gpu::Format::RGBA8UNorm,
                                TextureFormat::Alpha8 => gpu::Format::A8UNorm,
                            },
                            usage: TextureUsageFlags::Sampled,
                            ..Default::default()
                        }, cmd_buf)?;
                        let components = match *format {
                            TextureFormat::RGBA32 => 4,
                            TextureFormat::Alpha8 => 1,
                        };
                        let data = arena.alloc_data(pixels)?;
                        cmd_buf.copy_to_texture_rl(
                            data.device(), &texture, *row_pitch as u32 / components);
                        cmd_buf.barrier(
                            gpu::Stage::AllTransfer, gpu::Stage::PixelShader | gpu::Stage::AllTransfer,
                            gpu::HazardFlags::empty());
                        texture.view_descriptor(&mut tex_descriptors[idx as usize])?;

                        tex.replace(texture);

                        break;
                    }
                    if idx == 0 {
                        panic!("imgui out of textures")
                    } else {
                        request.uploaded(TextureId::new(idx))?
                    }
                },
                TextureOp::Update { format, width, height, rects } => {
                    let width = *width;
                    let height = *height;

                    let mut textures = self.textures.borrow_mut();
                    let tex = textures[req_tex].as_ref()
                        .expect("imgui texture update refers to non existing texture");

                    let gpu_format = match *format {
                        TextureFormat::RGBA32 => gpu::Format::RGBA8UNorm,
                        TextureFormat::Alpha8 => gpu::Format::A8UNorm,
                    };
                    assert_eq!(gpu_format, tex.format(), "imgui texture cannot change format");

                    let (tex, barrier) = if (width, height) != tex.dimensions2() {
                        let (old_width, old_height) = tex.dimensions2();
                        let new_tex = self.gpu.create_texture(gpu::TextureDesc {
                            ty: gpu::TextureType::Tex2D,
                            dimensions: (width, height, 1),
                            format: gpu_format,
                            usage: TextureUsageFlags::Sampled,
                            ..Default::default()
                        }, cmd_buf)?;
                        cmd_buf.copy_texture_to_texture(
                            tex, &new_tex,
                            [(
                                (0, 0, 0), (0, 0, 0),
                                (width.min(old_width), height.min(old_height), 1),
                            )]);

                        cmd_buf.barrier(
                            gpu::Stage::AllTransfer, gpu::Stage::AllTransfer, gpu::HazardFlags::empty());

                        let old_tex = textures[req_tex].replace(new_tex);
                        // TODO: proper retirement
                        cmd_buf.give_ownership(old_tex);

                        let new_tex = textures[req_tex].as_ref().unwrap();

                        (new_tex, true)
                    } else {
                        (tex, false)
                    };

                    let components = match *format {
                        TextureFormat::RGBA32 => 4,
                        TextureFormat::Alpha8 => 1,
                    };
                    for rect in rects {
                        let data = arena.alloc_data(&rect.data)?;
                        cmd_buf.copy_to_texture_rl_at(
                            data.device(), tex, rect.row_pitch as u32 / components,
                            (rect.rect.x as i32, rect.rect.y as i32, 0),
                            (rect.rect.w as u32, rect.rect.h as u32, 1));
                    }
                    if barrier || !rects.is_empty() {
                        cmd_buf.barrier(
                            gpu::Stage::AllTransfer, gpu::Stage::PixelShader, gpu::HazardFlags::empty());
                    }

                    request.uploaded(TextureId::new(req_tex as u64 + self.texture_offest as u64))?
                },
                TextureOp::Destroy => {
                    // TODO: proper retirement
                    cmd_buf.give_ownership(self.textures.borrow_mut()[req_tex].take());

                    request.destroyed()?
                },
            });
        }
        let frame = pending.reconcile_texture_feedback(feedback)?;
        let draw_data = frame.draw_data();

        assert!(!draw_data.requirements().requires_raw_callback_support(),
                "imgui raw callbacks are not supported");

        let mut vertices = arena.alloc::<Vertex>(draw_data.total_idx_count())?;
        let mut vertex_count = 0;
        let mut sampler_nearest = false;

        for draw_list in draw_data.draw_lists() {
            let index_buffer = draw_list.idx_buffer();
            let vertex_buffer = draw_list.vtx_buffer();

            for cmd in draw_list.commands() {
                match cmd {
                    DrawCmd::Elements { cmd_params, count } => {
                        let texture = if cmd_params.texture_id.is_null() {
                            1 << 30
                        } else {
                            let texture: u16 = cmd_params.texture_id.id()
                                .try_into()
                                .expect("imgui texture id does not fit into 16 bits");
                            let mut texture = texture as u32;
                            if sampler_nearest {
                                texture |= 1 << 31;
                            }
                            texture
                        };

                        for i in cmd_params.idx_offset..cmd_params.idx_offset + count {
                            let idx = index_buffer[i] as usize + cmd_params.vtx_offset;
                            let vert = vertex_buffer[idx];
                            let vert = Vertex {
                                pos: Vec2::from_array(vert.pos),
                                uv: Vec2::from_array(vert.uv),
                                color: vert.col,
                                texture,
                                _padding: [0u32; 2],
                            };
                            vertices.host_mut()[vertex_count] = vert;
                            vertex_count += 1;
                        }
                    },
                    DrawCmd::RawCallback(_callback) => {
                        unimplemented!("imgui raw callbacks are not supported");
                    },
                    DrawCmd::ResetRenderState => {
                        sampler_nearest = true;
                    },
                    DrawCmd::SetSamplerNearest => {
                        sampler_nearest = true;
                    },
                    DrawCmd::SetSamplerLinear => {
                        sampler_nearest = false;
                    },
                }
            }
        }

        let screen_size_i = 1.0 / Vec2::new(screen_size.0 as f32, screen_size.1 as f32);
        let proj = Mat4::from_cols_array(&[
            screen_size_i.x * 2.0, 0.0, 0.0, 0.0,
            0.0, screen_size_i.y * 2.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            -1.0, -1.0, 0.0, 1.0,
        ]);
        let vert_data = arena.alloc_data(&[VertData {
            proj,
            vertices: vertices.device(),
            _padding: 0,
        }])?;

        let render_pass = gpu::RenderPassDesc {
            cull: gpu::Cull::None,
            color_targets: &[color_target],
            ..Default::default()
        };
        cmd_buf.begin_render_pass(&render_pass);

        cmd_buf.bind_shaders([&self.shaders[0], &self.shaders[1]]);
        cmd_buf.draw_instanced(
            vert_data.device(), gpu::DevicePointer::null(),
            vertex_count.try_into().expect("imgui vertex count too large"), 1, 0, 0);

        cmd_buf.end_render_pass();

        Ok(())
    }
}

impl<'a> Drop for Imgui<'a> {
    fn drop(&mut self) {
        self.context.borrow_mut().prepare_renderer_texture_reset(&self.consumer).unwrap().commit();
    }
}
