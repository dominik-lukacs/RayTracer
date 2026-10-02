use std::sync::Arc;

use wgpu::CurrentSurfaceTexture::{Lost, Occluded, Outdated, Suboptimal, Success, Timeout, Validation};
use wgpu::ExperimentalFeatures;
use winit::event_loop::OwnedDisplayHandle;
use winit::window::{Window, WindowId};

use crate::scene::{Material, GpuMaterial};
use crate::{fps_counter::FpsCounter, scene::Scene};
use crate::gui_app::GuiApp;
use crate::gpu_buffer::StorageBuffer;

pub struct Renderer {
    window: Arc<Window>,
    window_id: Option<WindowId>,
    start_time: std::time::Instant,
    last_time: std::time::Instant,

    instance: wgpu::Instance,
    //adapter: wgpu::Adapter,
    device: wgpu::Device,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    storage_format: wgpu::TextureFormat,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,

    color_buffer: wgpu::Texture,
    color_buffer_view: wgpu::TextureView,
    sampler: wgpu::Sampler,

    ray_tracing_pipeline: wgpu::ComputePipeline,
    ray_tracing_bind_group: wgpu::BindGroup,
    screen_pipeline: wgpu::RenderPipeline,
    screen_bind_group: wgpu::BindGroup,

    //scene stuff
    scene_bind_group: wgpu::BindGroup,
    scene_bind_group_layout: wgpu::BindGroupLayout,

    //egui stuff
    fps_counter: FpsCounter,
    gui_app: GuiApp,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

impl Renderer {
    pub async fn new(display: OwnedDisplayHandle, window: Arc<Window>, scene: Scene) -> Self {
        // Create the instance, adapter, device, and queue, and setup the surface
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(Box::new(display)));

        let surface = instance.create_surface(&window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: Some("Device"),
                    experimental_features: ExperimentalFeatures::disabled(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                },
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::Srgb,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let storage_format = wgpu::TextureFormat::Rgba8Unorm;

        // Create the color buffer and sampler
        let color_buffer = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Color Buffer"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: storage_format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[storage_format],
        });

        let color_buffer_view = color_buffer.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Color Buffer Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        // Create pipelines
        let ray_tracing_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Ray Tracing Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: storage_format,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        let ray_tracing_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ray Tracing Bind Group"),
            layout: &ray_tracing_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&color_buffer_view),
            }],
        });


        // scene stuff (buffers and bind groups)
        let (scene_bind_group_layout, scene_bind_group) = {
            let sphere_buffer = StorageBuffer::new_from_bytes(
                &device,
                bytemuck::cast_slice(scene.spheres.as_slice()),
                0_u32,
                Some("scene buffer"),
            );

            let mut global_texture_data: Vec<[f32; 3]> = Vec::new();
            let mut material_data: Vec<GpuMaterial> = Vec::with_capacity(scene.materials.len());

            for material in scene.materials.iter() {
                let gpu_material = match material {
                    Material::Lambertian { albedo } => {
                        GpuMaterial::lambertian(albedo, &mut global_texture_data)
                    }
                    Material::Metal { albedo, fuzz } => {
                        GpuMaterial::metal(albedo, *fuzz, &mut global_texture_data)
                    }
                    Material::Dielectric { refraction_index } => {
                        GpuMaterial::dielectric(*refraction_index)
                    }
                    Material::Checkerboard { odd, even } => {
                        GpuMaterial::checkerboard(odd, even, &mut global_texture_data)
                    }
                    Material::Emissive { emit } => {
                        GpuMaterial::emissive(emit, &mut global_texture_data)
                    }
                };

                material_data.push(gpu_material);
            }

            let material_buffer = StorageBuffer::new_from_bytes(
                &device,
                bytemuck::cast_slice(material_data.as_slice()),
                1_u32,
                Some("materials buffer"),
            );

            let texture_buffer = StorageBuffer::new_from_bytes(
                &device,
                bytemuck::cast_slice(global_texture_data.as_slice()),
                2_u32,
                Some("textures buffer"),
            );

            let light_indices: Vec<u32> = scene
                .spheres
                .iter()
                .enumerate()
                .filter(|(_, s)| {
                    matches!(
                        scene.materials[s.material_idx as usize],
                        Material::Emissive { .. }
                    )
                })
                .map(|(idx, _)| idx as u32)
                .collect();

            let light_buffer = StorageBuffer::new_from_bytes(
                &device,
                bytemuck::cast_slice(light_indices.as_slice()),
                3_u32,
                Some("lights buffer"),
            );

            let scene_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        sphere_buffer.layout(wgpu::ShaderStages::COMPUTE, true),
                        material_buffer.layout(wgpu::ShaderStages::COMPUTE, true),
                        texture_buffer.layout(wgpu::ShaderStages::COMPUTE, true),
                        light_buffer.layout(wgpu::ShaderStages::COMPUTE, true),
                    ],
                    label: Some("scene layout"),
                });
            let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &scene_bind_group_layout,
                entries: &[
                    sphere_buffer.binding(),
                    material_buffer.binding(),
                    texture_buffer.binding(),
                    light_buffer.binding(),
                ],
                label: Some("scene bind group"),
            });

            (scene_bind_group_layout, scene_bind_group)
        };


        let ray_tracing_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ray Tracing Pipeline Layout"),
            bind_group_layouts: &[Some(&ray_tracing_bind_group_layout), Some(&scene_bind_group_layout)],
            immediate_size: 0,
        });

        let ray_tracing_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ray Tracing Pipeline"),
            layout: Some(&ray_tracing_pipeline_layout),
            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Ray Tracing Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("ray_tracing_kernel.wgsl").into()),
            }),
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let screen_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Screen Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        let screen_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Screen Bind Group"),
            layout: &screen_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&color_buffer_view),
            }],
        });

        let screen_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Screen Pipeline Layout"),
            bind_group_layouts: &[Some(&screen_bind_group_layout)],
            immediate_size: 0,
        });

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Screen Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("screen_shader.wgsl").into()),
        });

        let screen_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Screen Pipeline"),
            layout: Some(&screen_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vert_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("frag_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // egui stuff
        let fps_counter = FpsCounter::new();
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            surface_format,
            egui_wgpu::RendererOptions { msaa_samples: 1, depth_stencil_format: None, dithering: false, predictable_texture_filtering: false },
        );
        let gui_app = GuiApp::new();

        let window_id = Some(window.id());
        let start_time = std::time::Instant::now();
        let last_time = start_time.clone();

        Renderer {
            window,
            instance,
            surface_format,
            storage_format,
            surface,
            device,
            queue,
            config,
            size,
            color_buffer,
            color_buffer_view,
            sampler,
            ray_tracing_bind_group,
            ray_tracing_pipeline,
            screen_bind_group,
            screen_pipeline,
            scene_bind_group,
            scene_bind_group_layout,
            fps_counter,
            gui_app,
            egui_renderer,
            egui_ctx,
            egui_state,
            window_id,
            start_time,
            last_time,
        }
    }

    pub fn render(&mut self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut ray_trace_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ray Tracing Pass"),
                timestamp_writes: None,
            });
            ray_trace_pass.set_pipeline(&self.ray_tracing_pipeline);
            ray_trace_pass.set_bind_group(0, &self.ray_tracing_bind_group, &[]);
            ray_trace_pass.set_bind_group(1, &self.scene_bind_group, &[]);
            ray_trace_pass.dispatch_workgroups(self.size.width, self.size.height, 1);
        }

        let output = match self.surface.get_current_texture() {
            Success(texture) => texture,
            Occluded | Timeout => return,
            Suboptimal(texture) => {
                drop(texture);
                self.configure_surface();
                return;
            },
            Outdated => {
                self.configure_surface();
                return;
            },
            Validation => {
                // TODO not sure what to do here
                return;
            },
            Lost => {
                self.surface = self.instance.create_surface(self.window.clone()).unwrap();
                self.configure_surface();
                return;
            },
            _ => return,
        };
        let texture_view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Screen Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color{
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            render_pass.set_pipeline(&self.screen_pipeline);
            render_pass.set_bind_group(0, &self.screen_bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }
        // egui render pass
        let raw_input = egui::RawInput::default();
        let full_output = self.egui_ctx.run_ui(raw_input, |ui| {
            self.gui_app.ui(&self.egui_ctx, self.fps_counter.average_fps(), self.fps_counter.average_frame_time());
        });
        self.egui_state.handle_platform_output(&self.window, full_output.platform_output);
        let clipped_primitives = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        let screen_descriptor = &egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.size.width, self.size.height],
            pixels_per_point: (self.window.scale_factor() as f32)
        };
        let egui_command_buffers = self.egui_renderer.update_buffers(&self.device, &self.queue, &mut encoder, &clipped_primitives, screen_descriptor);
        // self.egui_renderer.update_texture(&self.device, &self.queue, );
        self.egui_renderer.render(render_pass, paint_jobs, screen_descriptor);


        self.queue.submit(std::iter::once(encoder.finish()));
        // output.present();
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;

        self.configure_surface();
    }

    pub fn configure_surface(&mut self) {
        self.config.width = self.size.width;
        self.config.height = self.size.height;
        self.surface.configure(&self.device, &self.config);

        // Create a new color buffer with the new size
        self.color_buffer = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Color Buffer"),
            size: wgpu::Extent3d {
                width: self.size.width,
                height: self.size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.storage_format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[self.storage_format],
        });

        self.color_buffer_view = self.color_buffer.create_view(&wgpu::TextureViewDescriptor::default());

        // Create pipelines

        let ray_tracing_bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Ray Tracing Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: self.storage_format,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        self.ray_tracing_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ray Tracing Bind Group"),
            layout: &ray_tracing_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.color_buffer_view),
            }],
        });

        let ray_tracing_pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ray Tracing Pipeline Layout"),
            bind_group_layouts: &[Some(&ray_tracing_bind_group_layout), Some(&self.scene_bind_group_layout)],
            immediate_size: 0,
        });

        self.ray_tracing_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ray Tracing Pipeline"),
            layout: Some(&ray_tracing_pipeline_layout),
            module: &self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Ray Tracing Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("ray_tracing_kernel.wgsl").into()),
            }),
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let screen_bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Screen Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        self.screen_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Screen Bind Group"),
            layout: &screen_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(&self.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&self.color_buffer_view),
            }],
        });

        let screen_pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Screen Pipeline Layout"),
            bind_group_layouts: &[Some(&screen_bind_group_layout)],
            immediate_size: 0,
        });

        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Screen Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("screen_shader.wgsl").into()),
        });

        self.screen_pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Screen Pipeline"),
            layout: Some(&screen_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vert_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("frag_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
    }

    pub fn update(&mut self, _delta_time: f32) {
        self.fps_counter.update(_delta_time);
        //println!("FPS: {}", self.fps_counter.average_fps());
    }
}
