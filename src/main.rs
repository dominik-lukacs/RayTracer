use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub extern crate nalgebra_glm as glm;

mod renderer;
mod fps_counter;
mod gui_app;
mod sphere;
mod gpu_buffer;
mod scene;
use renderer::Renderer;

use scene::{Material, Scene, Texture};
use sphere::Sphere;

use wgpu;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("GPU Ray Tracer")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
        .build(&event_loop)
        .unwrap();

    let scene = setup_scene();

    let mut renderer = pollster::block_on(Renderer::new(window, scene));

    let start_time = std::time::Instant::now();
    let mut last_time = start_time.clone();

    event_loop.run(move |event, _, control_flow| {
        renderer.platform.handle_event(&event);

        match event {
            Event::WindowEvent {
                ref event,
                window_id
            } if window_id == renderer.window.id() => {
                match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::ExitWithCode(0),
                    WindowEvent::Resized(physical_size) => {
                        renderer.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        renderer.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(_) => {
                renderer.platform.update_time(start_time.elapsed().as_secs_f64());//TODO: maybe this can be moved to renderer.update()?
                match renderer.render() {
                    Ok(_) => {}
                    // Recreate the swap_chain if lost
                    Err(wgpu::SurfaceError::Lost) => {renderer.resize(renderer.size)}
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => {*control_flow = ControlFlow::Exit}
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            Event::MainEventsCleared => {
                {
                    let delta_time = last_time.elapsed().as_secs_f32();
                    last_time = std::time::Instant::now();
                    renderer.update(delta_time);
                }
                renderer.window.request_redraw();
            }
            _ => {}
        }
    });
}


fn setup_scene() -> scene::Scene {
    let materials = vec![
        Material::Checkerboard {
            even: Texture::new_from_color(glm::vec3(0.5_f32, 0.7_f32, 0.8_f32)),
            odd: Texture::new_from_color(glm::vec3(0.9_f32, 0.9_f32, 0.9_f32)),
        },
        Material::Lambertian {
            albedo: Texture::new_from_image("assets/moon.jpeg")
                .expect("Hardcoded path should be valid"),
        },
        Material::Metal {
            albedo: Texture::new_from_color(glm::vec3(1_f32, 0.85_f32, 0.57_f32)),
            fuzz: 0.3_f32,
        },
        Material::Metal {
            albedo: Texture::new_from_color(glm::vec3(0.5_f32, 0.85_f32, 1_f32)),
            fuzz: 0.0_f32,
        },
        Material::Dielectric {
            refraction_index: 1.5_f32,
        },
        Material::Lambertian {
            albedo: Texture::new_from_image("assets/earthmap.jpeg")
                .expect("Hardcoded path should be valid"),
        },
        Material::Emissive {
            emit: Texture::new_from_scaled_image("assets/sun.jpeg", 50.0)
                .expect("Hardcoded path should be valid"),
        },
        Material::Lambertian {
            albedo: Texture::new_from_color(glm::vec3(0.3_f32, 0.9_f32, 0.9_f32)),
        },
        Material::Emissive {
            emit: Texture::new_from_color(glm::vec3(50.0_f32, 0.0_f32, 0.0_f32)),
        },
        Material::Emissive {
            emit: Texture::new_from_color(glm::vec3(0.0_f32, 50.0_f32, 0.0_f32)),
        },
        Material::Emissive {
            emit: Texture::new_from_color(glm::vec3(0.0, 0.0, 50.0)),
        },
    ];

    let spheres = vec![
        Sphere::new(glm::vec3(0.0, -510.0, -1.0), 500.0, 10_u32),
        // left row
        Sphere::new(glm::vec3(-2.0, 0.0, -3.0), 1.0, 2_u32),
        //Sphere::new(glm::vec3(0.0, 0.0, -3.0), 1.0, 1_u32),
        Sphere::new(glm::vec3(2.0, 0.0, -3.0), 1.0, 3_u32),
        // middle row
        //Sphere::new(glm::vec3(-5.0, 1.0, 0.0), 1.0, 2_u32),
        //Sphere::new(glm::vec3(0.0, 1.0, 0.0), 1.0, 3_u32),
        //Sphere::new(glm::vec3(5.0, 1.0, 0.0), 1.0, 6_u32),
        // right row
        //Sphere::new(glm::vec3(-5.0, 0.8, 4.0), 0.8, 1_u32),
        //Sphere::new(glm::vec3(0.0, 1.2, 4.0), 1.2, 4_u32),
        //Sphere::new(glm::vec3(5.0, 2.0, 4.0), 2.0, 5_u32),
    ];

    Scene { spheres, materials }
}
