use bytemuck::{Pod, Zeroable};
use nalgebra_glm::{identity, TMat4};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

vulkano::impl_vertex!(Vertex, position, color);

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
