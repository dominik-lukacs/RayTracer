use crate::sphere::Sphere;

pub struct Scene {
    pub spheres: Vec<Sphere>,
    pub materials: Vec<Material>,
}

pub enum Material {
    Lambertian { albedo: Texture },
    Metal { albedo: Texture, fuzz: f32 },
    Dielectric { refraction_index: f32 },
    Checkerboard { even: Texture, odd: Texture },
    Emissive { emit: Texture },
}

use image::RgbaImage;
use thiserror::Error;
use nalgebra_glm as glm;

pub struct Texture {
    dimensions: (u32, u32),
    data: Vec<[f32; 3]>,
}

impl Texture {
    pub fn new_from_image(path: &str) -> Result<Self, TextureError> {
        Self::new_from_scaled_image(path, 1_f32)
    }

    pub fn new_from_scaled_image(path: &str, scale: f32) -> Result<Self, TextureError> {
        use std::fs::*;
        use std::io::BufReader;

        let file = File::open(path)?;
        let pixels: RgbaImage =
            image::load(BufReader::new(file), image::ImageFormat::Jpeg)?.into_rgba8();
        let tex_scale = scale / 255_f32;
        let dimensions = pixels.dimensions();
        let data = pixels
            .pixels()
            .map(|p| -> [f32; 3] {
                [
                    tex_scale * (p[0] as f32),
                    tex_scale * (p[1] as f32),
                    tex_scale * (p[2] as f32),
                ]
            })
            .collect();

        Ok(Self { dimensions, data })
    }

    pub fn new_from_color(color: glm::Vec3) -> Self {
        let data = vec![[color.x, color.y, color.z]];
        let dimensions = (1_u32, 1_u32);

        Self { dimensions, data }
    }

    pub fn as_slice(&self) -> &[[f32; 3]] {
        self.data.as_slice()
    }

    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }
}

#[derive(Error, Debug)]
pub enum TextureError {
    #[error(transparent)]
    FileIoError(#[from] std::io::Error),
    #[error(transparent)]
    ImageLoadError(#[from] image::ImageError),
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TextureDescriptor {
    width: u32,
    height: u32,
    offset: u32,
}

impl TextureDescriptor {
    pub fn empty() -> Self {
        Self {
            width: 0_u32,
            height: 0_u32,
            offset: 0xffffffff,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterial {
    id: u32,
    desc1: TextureDescriptor,
    desc2: TextureDescriptor,
    x: f32,
}

impl GpuMaterial {
    pub fn lambertian(albedo: &Texture, global_texture_data: &mut Vec<[f32; 3]>) -> Self {
        Self {
            id: 0_u32,
            desc1: Self::append_to_global_texture_data(albedo, global_texture_data),
            desc2: TextureDescriptor::empty(),
            x: 0_f32,
        }
    }

    pub fn metal(albedo: &Texture, fuzz: f32, global_texture_data: &mut Vec<[f32; 3]>) -> Self {
        Self {
            id: 1_u32,
            desc1: Self::append_to_global_texture_data(albedo, global_texture_data),
            desc2: TextureDescriptor::empty(),
            x: fuzz,
        }
    }

    pub fn dielectric(refraction_index: f32) -> Self {
        Self {
            id: 2_u32,
            desc1: TextureDescriptor::empty(),
            desc2: TextureDescriptor::empty(),
            x: refraction_index,
        }
    }

    pub fn checkerboard(
        even: &Texture,
        odd: &Texture,
        global_texture_data: &mut Vec<[f32; 3]>,
    ) -> Self {
        Self {
            id: 3_u32,
            desc1: Self::append_to_global_texture_data(even, global_texture_data),
            desc2: Self::append_to_global_texture_data(odd, global_texture_data),
            x: 0_f32,
        }
    }

    pub fn emissive(emit: &Texture, global_texture_data: &mut Vec<[f32; 3]>) -> Self {
        Self {
            id: 4_u32,
            desc1: Self::append_to_global_texture_data(emit, global_texture_data),
            desc2: TextureDescriptor::empty(),
            x: 0_f32,
        }
    }

    fn append_to_global_texture_data(
        texture: &Texture,
        global_texture_data: &mut Vec<[f32; 3]>,
    ) -> TextureDescriptor {
        let dimensions = texture.dimensions();
        let offset = global_texture_data.len() as u32;
        global_texture_data.extend_from_slice(texture.as_slice());
        TextureDescriptor {
            width: dimensions.0,
            height: dimensions.1,
            offset,
        }
    }
}
