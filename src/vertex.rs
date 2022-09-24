
use bytemuck::{Zeroable, Pod};


#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Vertex {
    pub position: [f32; 3],
}


vulkano::impl_vertex!(Vertex, position);
