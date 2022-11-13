use bytemuck::{Pod, Zeroable};
use nalgebra_glm::{identity, TMat4};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

vulkano::impl_vertex!(Vertex, position, normal, color);

#[derive(Debug, Clone)]
pub struct MVP {
    pub model: TMat4<f32>,
    pub view: TMat4<f32>,
    pub projection: TMat4<f32>,
}

impl MVP {
    pub fn new() -> MVP {
        MVP {
            model: identity(),
            view: identity(),
            projection: identity(),
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct AmbientLight {
    pub color: [f32; 3],
    pub intensity: f32,
}

#[derive(Default, Debug, Clone)]
pub struct DirectionalLight {
    pub position: [f32; 4],
    pub color: [f32; 3],
}
