use nalgebra_glm as glm;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sphere {
    pub center: glm::Vec4,
    pub radius: f32,
    pub material_idx: u32,
    _padding: [u32; 2],
}

impl Sphere {
    pub fn new(center: glm::Vec3, radius: f32, material_idx: u32) -> Self {
        Self {
            center: glm::vec3_to_vec4(&center),
            radius,
            material_idx,
            _padding: [0; 2],
        }
    }
}
