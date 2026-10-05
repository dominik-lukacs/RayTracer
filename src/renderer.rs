use std::sync::Arc;

use wgpu::CurrentSurfaceTexture::{Lost, Occluded, Outdated, Suboptimal, Success, Timeout, Validation};
use wgpu::ExperimentalFeatures;
use winit::event_loop::OwnedDisplayHandle;
use winit::window::{Window, WindowId};

use crate::budget::{Budget, MAX_STEPS_PER_PIXEL};
use crate::gpu_timer::GpuTimer;
use crate::gpu_buffer::{StorageBuffer, UniformBuffer};
use crate::gui_app::{GuiApp, Settings, Stats};
use crate::scene::{Material, GpuMaterial};
use crate::{fps_counter::FpsCounter, scene::Scene};

// Must match `@workgroup_size` in ray_tracing_kernel.wgsl
const WORKGROUP_SIZE: u32 = 8;
const PATH_STATE_BYTES: u64 = 64;
const ACCUM_BYTES: u64 = 16;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    max_bounces: u32,
    steps: u32,
    active_count: u32,
    group_start: u32,
    epoch: u32,
    _pad: u32,
}

// One frame's worth of ray tracing work
struct Dispatch {
    steps: u32,
    active_groups: u32,
    total_groups: u32,
    work: f32,
}

pub struct Renderer {
    window: Arc<Window>,
    pub window_id: Option<WindowId>,

    instance: wgpu::Instance,
    device: wgpu::Device,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,

    // Resolution-dependent GPU state
    path_buffer: wgpu::Buffer,
    accum_buffer: wgpu::Buffer,
    ray_tracing_bind_group: wgpu::BindGroup,
    screen_bind_group: wgpu::BindGroup,

    // Resolution-independent GPU state
    params_buffer: UniformBuffer,
    ray_tracing_bind_group_layout: wgpu::BindGroupLayout,
    screen_bind_group_layout: wgpu::BindGroupLayout,
    ray_tracing_pipeline: wgpu::ComputePipeline,
    screen_pipeline: wgpu::RenderPipeline,
    scene_bind_group: wgpu::BindGroup,
    scene_bind_group_layout: wgpu::BindGroupLayout,

    // Progressive rendering state
    settings: Settings,
    budget: Budget,
    gpu_timer: Option<GpuTimer>,
    last_gpu_ms: Option<f32>,
    epoch: u32,
    group_start: u32,
    total_steps: f64,
    needs_clear: bool,

    // egui stuff
    fps_counter: FpsCounter,
    gui_app: GuiApp,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

impl Renderer {
    pub async fn new(display: OwnedDisplayHandle, window: Arc<Window>, scene: Arc<Scene>) -> Self {
        // Create the instance, adapter, device, and queue, and setup the surface
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(Box::new(display)));

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .unwrap();

        let adapter_limits = adapter.limits();
        let required_limits = if cfg!(target_arch = "wasm32") {
            wgpu::Limits::downlevel_webgl2_defaults()
        } else {
            wgpu::Limits {
                max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                max_buffer_size: adapter_limits.max_buffer_size,
                ..wgpu::Limits::defaults()
            }
        };

        let timestamp_support = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: timestamp_support,
                    required_limits,
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
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // Pipelines and layouts
        let params_buffer = UniformBuffer::new(
            &device,
            std::mem::size_of::<Params>() as wgpu::BufferAddress,
            0_u32,
            Some("params buffer"),
        );

        // Create pipelines
        let ray_tracing_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Ray Tracing Bind Group Layout"),
            entries: &[
                params_buffer.layout(wgpu::ShaderStages::COMPUTE),
                Self::storage_layout_entry(1, wgpu::ShaderStages::COMPUTE, false),// path states
                Self::storage_layout_entry(2, wgpu::ShaderStages::COMPUTE, false),// accumulation
            ],
        });

        let screen_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Screen Bind Group Layout"),
            entries: &[
                params_buffer.layout(wgpu::ShaderStages::FRAGMENT),
                Self::storage_layout_entry(1, wgpu::ShaderStages::FRAGMENT, true),
            ],
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

        // Resolution-dependent buffers
        let (path_buffer, accum_buffer, ray_tracing_bind_group, screen_bind_group) = Self::create_targets(
            &device,
            config.width,
            config.height,
            &params_buffer,
            &ray_tracing_bind_group_layout,
            &screen_bind_group_layout,
        );

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

        // Frame budget: aim at monitor's refresh rate
        let target_hz = window
            .current_monitor()
            .and_then(|m| m.refresh_rate_millihertz())
            .map(|mhz| mhz as f32 / 1000.0)
            .unwrap_or(144.0);
        let settings = Settings { max_bounces: 100, gpu_fraction: 0.6, target_hz };
        let mut budget = Budget::new(target_hz);
        budget.gpu_fraction = settings.gpu_fraction;

        let gpu_timer = if timestamp_support.is_empty() {
            None
        } else {
            Some(GpuTimer::new(&device, &queue))
        };

        let window_id = Some(window.id());

        Renderer {
            window,
            instance,
            surface_format,
            surface,
            device,
            queue,
            config,
            size,
            path_buffer,
            accum_buffer,
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
            params_buffer,
            ray_tracing_bind_group_layout,
            screen_bind_group_layout,
            settings,
            budget,
            gpu_timer,
            last_gpu_ms: None,
            epoch: 0,
            group_start: 0,
            total_steps: 0.0,
            needs_clear: false,
        }
    }

    fn storage_layout_entry(binding: u32, visibility: wgpu::ShaderStages, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn create_targets(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        params_buffer: &UniformBuffer,
        ray_tracing_layout: &wgpu::BindGroupLayout,
        screen_layout: &wgpu::BindGroupLayout,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup, wgpu::BindGroup) {
        let pixels = width as u64 * height as u64;

        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let path_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Path state buffer"),
            size: pixels * PATH_STATE_BYTES,
            usage,
            mapped_at_creation: false,
        });
        let accum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Accumulation buffer"),
            size: pixels * ACCUM_BYTES,
            usage,
            mapped_at_creation: false,
        });

        let ray_tracing_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ray Tracing Bind Group"),
            layout: ray_tracing_layout,
            entries: &[
                params_buffer.binding(),
                wgpu::BindGroupEntry { binding: 1, resource: path_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: accum_buffer.as_entire_binding() },
            ],
        });

        let screen_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Screen Bind Group"),
            layout: screen_layout,
            entries: &[
                params_buffer.binding(),
                wgpu::BindGroupEntry { binding: 1, resource: accum_buffer.as_entire_binding() },
            ],
        });

        (path_buffer, accum_buffer, ray_tracing_bind_group, screen_bind_group)
    }

    pub fn reset_accumulation(&mut self) {
        self.needs_clear = true;
    }

    fn total_groups(&self) -> u32 {
        self.size.width.div_ceil(WORKGROUP_SIZE) * self.size.height.div_ceil(WORKGROUP_SIZE)
    }

    fn plan_dispatch(&self) -> Dispatch {
        let total_groups = self.total_groups().max(1);
        let s = self.budget.steps_per_pixel();
        let steps = (s.ceil() as u32).clamp(1, MAX_STEPS_PER_PIXEL as u32);
        let fraction = (s / steps as f32).clamp(0.0, 1.0);
        let active_groups = ((fraction * total_groups as f32).round() as u32).clamp(1, total_groups);
        let work = steps as f32 * active_groups as f32 / total_groups as f32;
        Dispatch { steps, active_groups, total_groups, work }
    }

    pub fn render(&mut self) {
        if self.size.width == 0 || self.size.height == 0 {
            return;
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
        };
        let texture_view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let _ = self.device.poll(wgpu::PollType::Poll);
        if let Some(timer) = self.gpu_timer.as_mut() {
            for (work, ms) in timer.collect() {
                self.budget.on_gpu_sample(work, ms);
                self.last_gpu_ms = Some(ms);
            }
        }
        self.budget.target_hz = self.settings.target_hz;
        self.budget.gpu_fraction = self.settings.gpu_fraction;

        let dispatch = self.plan_dispatch();
        let params = Params {
            width: self.size.width,
            height: self.size.height,
            max_bounces: self.settings.max_bounces.max(1),
            steps: dispatch.steps,
            active_count: dispatch.active_groups,
            group_start: self.group_start,
            epoch: self.epoch,
            _pad: 0,
        };
        self.queue.write_buffer(self.params_buffer.handle(), 0, bytemuck::bytes_of(&params));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        if self.needs_clear {
            encoder.clear_buffer(&self.path_buffer, 0, None);
            encoder.clear_buffer(&self.accum_buffer, 0, None);
            self.needs_clear = false;
            self.total_steps = 0.0;
        }

        // Ray tracing
        let timing_slot = self.gpu_timer.as_mut().and_then(|t| t.acquire_slot(dispatch.work));
        {
            let timestamp_writes = match (&self.gpu_timer, timing_slot) {
                (Some(timer), Some(_)) => Some(timer.pass_writes()),
                _ => None,
            };
            let mut ray_trace_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ray Tracing Pass"),
                timestamp_writes,
            });
            ray_trace_pass.set_pipeline(&self.ray_tracing_pipeline);
            ray_trace_pass.set_bind_group(0, &self.ray_tracing_bind_group, &[]);
            ray_trace_pass.set_bind_group(1, &self.scene_bind_group, &[]);
            ray_trace_pass.dispatch_workgroups(
                self.size.width.div_ceil(WORKGROUP_SIZE),
                self.size.height.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }
        if let (Some(timer), Some(slot)) = (&self.gpu_timer, timing_slot) {
            timer.resolve(&mut encoder, slot);
        }

        self.group_start = (self.group_start + dispatch.active_groups) % dispatch.total_groups;
        self.total_steps += dispatch.work as f64;

        // UI
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let stats = Stats {
            fps: self.fps_counter.average_fps(),
            frame_time: self.fps_counter.average_frame_time(),
            gpu_ms: self.last_gpu_ms,
            steps_per_pixel: self.budget.steps_per_pixel(),
            total_steps: self.total_steps,
        };
        let mut restart = false;
        let full_output = self.egui_ctx.run_ui(raw_input, |_ui| {
            restart = self.gui_app.ui(&self.egui_ctx, &stats, &mut self.settings);
        });
        if restart {
            self.reset_accumulation();
        }
        self.egui_state.handle_platform_output(&self.window, full_output.platform_output);
        let clipped_primitives = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        let screen_descriptor = &egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, &delta[0]);
        }
        let egui_command_buffers = self.egui_renderer.update_buffers(&self.device, &self.queue, &mut encoder, &clipped_primitives, screen_descriptor);
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Screen Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
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
            })
            .forget_lifetime();
            render_pass.set_pipeline(&self.screen_pipeline);
            render_pass.set_bind_group(0, &self.screen_bind_group, &[]);
            render_pass.draw(0..3, 0..1);

            self.egui_renderer.render(&mut render_pass, &clipped_primitives, &screen_descriptor);
        }
        self.queue.submit(egui_command_buffers.into_iter().chain(std::iter::once(encoder.finish())));
        if let (Some(timer), Some(slot)) = (self.gpu_timer.as_mut(), timing_slot) {
            timer.after_submit(slot);
        }
        self.window.pre_present_notify();
        self.queue.present(output);

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;

        self.configure_surface();
    }

    pub fn configure_surface(&mut self) {
        // Minimized windows report a zero size, configuring that is a validation error
        if self.size.width == 0 || self.size.height == 0 {
            return;
        }
        self.config.width = self.size.width;
        self.config.height = self.size.height;
        self.surface.configure(&self.device, &self.config);

        let (path_buffer, accum_buffer, ray_tracing_bind_group, screen_bind_group) = Self::create_targets(
            &self.device,
            self.size.width,
            self.size.height,
            &self.params_buffer,
            &self.ray_tracing_bind_group_layout,
            &self.screen_bind_group_layout,
        );
        self.path_buffer = path_buffer;
        self.accum_buffer = accum_buffer;
        self.ray_tracing_bind_group = ray_tracing_bind_group;
        self.screen_bind_group = screen_bind_group;

        self.group_start = 0;
        self.total_steps = 0.0;
        self.needs_clear = false;
        self.budget.invalidate();
    }
    
    pub fn update(&mut self, delta_time: f32) {
        self.fps_counter.update(delta_time);
        // if GPU timestamps aren't available fall back to CPU frame time
        if self.gpu_timer.is_none() {
            self.budget.on_frame_time(delta_time);
        }
    }

    pub fn handle_egui_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        let response = self.egui_state.on_window_event(&self.window, event);
        if response.repaint {
            self.window.request_redraw();
        }
        response.consumed
    }
}
