use std::sync::Arc;

use winit::event_loop::{ControlFlow, EventLoop};

mod app;
mod renderer;
mod scene;
mod fps_counter;
mod gui_app;
mod gpu_buffer;
mod sphere;

use app::App;
use scene::{Scene, Material, Texture};
use nalgebra_glm as glm;
use sphere::Sphere;

fn main() {
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);

    let start_time = std::time::Instant::now();
    let last_time = start_time.clone();
    
    let mut app = App {
        renderer: None,
        scene: Arc::new(setup_scene()),
        window: None,
        window_id: None,
        start_time,
        last_time,
    };
    let _ = event_loop.run_app(&mut app);
}

fn setup_scene() -> Scene {
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
        Sphere::new(glm::vec3(0.0, 0.0, -3.0), 1.0, 1_u32),
        Sphere::new(glm::vec3(2.0, 0.0, -3.0), 1.0, 3_u32),
        // middle row
        Sphere::new(glm::vec3(-5.0, 1.0, 0.0), 1.0, 2_u32),
        Sphere::new(glm::vec3(0.0, 1.0, 1.0), 1.0, 3_u32),
        Sphere::new(glm::vec3(5.0, 1.0, 0.0), 1.0, 6_u32),
        // right row
        Sphere::new(glm::vec3(-5.0, 0.8, 4.0), 0.8, 1_u32),
        Sphere::new(glm::vec3(0.0, 1.2, 4.0), 1.2, 4_u32),
        Sphere::new(glm::vec3(5.0, 2.0, 4.0), 2.0, 5_u32),
    ];

    Scene { spheres, materials }
}
