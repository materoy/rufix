use std::{sync::Arc, time::Instant};

use nalgebra_glm::{
    identity, look_at, perspective, pi, rotate_normalized_axis, translate, vec3, TMat4,
};
use vertex::Vertex;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess},
    command_buffer::AutoCommandBufferBuilder,
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    swapchain::{self, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError},
    sync::{self, FlushError, GpuFuture},
    Version,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::{
    shaders::{fs, vs},
    vertex::{AmbientLight, DirectionalLight, MVP},
};

extern crate vulkano;
extern crate vulkano_win;
extern crate winit;

mod shaders;
mod vertex;

fn main() {
    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(InstanceCreateInfo {
            enabled_extensions: extensions,
            max_api_version: Some(Version::V1_1),
            ..Default::default()
        })
        .unwrap()
    };

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            vulkano::device::physical::PhysicalDeviceType::DiscreteGpu => 0,
            vulkano::device::physical::PhysicalDeviceType::IntegratedGpu => 1,
            vulkano::device::physical::PhysicalDeviceType::VirtualGpu => 2,
            vulkano::device::physical::PhysicalDeviceType::Cpu => 3,
            vulkano::device::physical::PhysicalDeviceType::Other => 4,
        })
        .unwrap();
    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );
    println!(
        "Our physical device supports Vulkan: {:?}",
        physical_device.properties().api_version
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: physical_device
                .required_extensions()
                .union(&device_extensions),
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let image_format = Some(
            physical_device
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count,
                image_format,
                image_extent: surface.window().inner_size().into(),
                image_usage: usage,
                composite_alpha: alpha,
                ..Default::default()
            },
        )
        .unwrap()
    };

    let vs = shaders::vs::load(device.clone()).unwrap();
    let fs = shaders::fs::load(device.clone()).unwrap();

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16_UNORM,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    )
    .unwrap();

    // pipeline
    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    let vertex_buffer = get_vertex_buffer(device.clone());

    let uniform_buffer = CpuBufferPool::<vs::ty::MVP_Data>::uniform_buffer(device.clone());

    let ambient_buffer = CpuBufferPool::<fs::ty::Ambient_Data>::uniform_buffer(device.clone());

    let directional_buffer =
        CpuBufferPool::<fs::ty::Directional_Light_Data>::uniform_buffer(device.clone());

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let mut framebuffers =
        window_size_dependent_setup(device.clone(), &images, render_pass.clone(), &mut viewport);

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

    let rotation_start = Instant::now();

    let mut mvp = MVP::new();
    mvp.model = translate(&identity(), &vec3(0.0, 0.0, -2.5));
    let ambient_light = AmbientLight {
        color: [1.0, 1.0, 1.0],
        intensity: 0.2,
    };
    let directional_light = DirectionalLight {
        position: [-4.0, -4.0, 0.0, 1.0],
        color: [1.0, 1.0, 1.0],
    };

    event_loop.run(move |event, _, control_flow| match event {
        winit::event::Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        winit::event::Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain = true;
        }
        winit::event::Event::RedrawEventsCleared => {
            // Render operations here

            previous_frame_end
                .as_mut()
                .take()
                .unwrap()
                .cleanup_finished();

            if recreate_swapchain {
                let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                    image_extent: surface.window().inner_size().into(),
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                    Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                };

                swapchain = new_swapchain;
                framebuffers = window_size_dependent_setup(
                    device.clone(),
                    &new_images,
                    render_pass.clone(),
                    &mut viewport,
                );
                recreate_swapchain = false;
            }

            let uniform_buffer_subbuffer = {
                let dimensions: [u32; 2] = surface.window().inner_size().into();
                mvp.projection = perspective(
                    dimensions[0] as f32 / dimensions[1] as f32,
                    180.0,
                    0.01,
                    100.0,
                );
                mvp.view = look_at(
                    &vec3(0.0, 0.0, 0.01),
                    &vec3(0.0, 0.0, 0.0),
                    &vec3(0.0, -1.0, 0.0),
                );

                // Rotation animation
                let elapsed = rotation_start.elapsed().as_secs() as f64
                    + rotation_start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0;
                let elapsed_as_radians = elapsed * pi::<f64>() / 180.0;
                let mut model: TMat4<f32> = rotate_normalized_axis(
                    &identity(),
                    elapsed_as_radians as f32 * 50.0,
                    &vec3(0.0, 0.0, 1.0),
                );
                model = rotate_normalized_axis(
                    &model,
                    elapsed_as_radians as f32 * 30.0,
                    &vec3(0.0, 1.0, 0.0),
                );

                model = rotate_normalized_axis(
                    &model,
                    elapsed_as_radians as f32 * 20.0,
                    &vec3(1.0, 0.0, 0.0),
                );
                model = mvp.model * model;

                let uniform_data = vs::ty::MVP_Data {
                    world: model.into(),
                    view: mvp.view.into(),
                    projection: mvp.projection.into(),
                };

                uniform_buffer.next(uniform_data).unwrap()
            };

            let ambient_uniform_subbufer = {
                let uniform_data = fs::ty::Ambient_Data {
                    color: ambient_light.color.into(),
                    intensity: ambient_light.intensity.into(),
                };

                ambient_buffer.next(uniform_data).unwrap()
            };

            let directional_uniform_subbuffer = {
                let uniform_data = fs::ty::Directional_Light_Data {
                    position: directional_light.position.into(),
                    color: directional_light.color.into(),
                };

                directional_buffer.next(uniform_data).unwrap()
            };

            let layout = pipeline.layout().set_layouts().get(0).unwrap();
            let set = PersistentDescriptorSet::new(
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer),
                    WriteDescriptorSet::buffer(1, ambient_uniform_subbufer),
                    WriteDescriptorSet::buffer(2, directional_uniform_subbuffer),
                ],
            )
            .unwrap();

            let (image_num, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into(), 1f32.into()];

            let mut cmd_buffer_builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            cmd_buffer_builder
                .begin_render_pass(
                    framebuffers[image_num].clone(),
                    vulkano::command_buffer::SubpassContents::Inline,
                    clear_values,
                )
                .unwrap()
                .set_viewport(0, [viewport.clone()])
                .bind_pipeline_graphics(pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline.layout().clone(),
                    0,
                    set.clone(),
                )
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap();

            let command_buffer = cmd_buffer_builder.build().unwrap();

            let future = previous_frame_end
                .take()
                .unwrap()
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => previous_frame_end = Some(Box::new(future) as Box<_>),
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>)
                }
                Err(e) => {
                    eprintln!("Failed to flush future: {:?}", e);
                    previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>)
                }
            }
        }
        _ => {}
    });
}

fn window_size_dependent_setup(
    device: Arc<Device>,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(device.clone(), dimensions, Format::D16_UNORM).unwrap(),
    )
    .unwrap();

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                vulkano::render_pass::FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

fn get_vertex_buffer(device: Arc<Device>) -> Arc<CpuAccessibleBuffer<[vertex::Vertex]>> {
    CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        [
            // front face
            Vertex {
                position: [-1.000000, -1.000000, 1.000000],
                normal: [0.0000, 0.0000, 1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, 1.000000, 1.000000],
                normal: [0.0000, 0.0000, 1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, 1.000000, 1.000000],
                normal: [0.0000, 0.0000, 1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, -1.000000, 1.000000],
                normal: [0.0000, 0.0000, 1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, 1.000000, 1.000000],
                normal: [0.0000, 0.0000, 1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, -1.000000, 1.000000],
                normal: [0.0000, 0.0000, 1.0000],
                color: [1.0, 0.35, 0.137],
            },
            // back face
            Vertex {
                position: [1.000000, -1.000000, -1.000000],
                normal: [0.0000, 0.0000, -1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, 1.000000, -1.000000],
                normal: [0.0000, 0.0000, -1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, 1.000000, -1.000000],
                normal: [0.0000, 0.0000, -1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, -1.000000, -1.000000],
                normal: [0.0000, 0.0000, -1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, 1.000000, -1.000000],
                normal: [0.0000, 0.0000, -1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, -1.000000, -1.000000],
                normal: [0.0000, 0.0000, -1.0000],
                color: [1.0, 0.35, 0.137],
            },
            // top face
            Vertex {
                position: [-1.000000, -1.000000, 1.000000],
                normal: [0.0000, -1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, -1.000000, 1.000000],
                normal: [0.0000, -1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, -1.000000, -1.000000],
                normal: [0.0000, -1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, -1.000000, 1.000000],
                normal: [0.0000, -1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, -1.000000, -1.000000],
                normal: [0.0000, -1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, -1.000000, -1.000000],
                normal: [0.0000, -1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            // bottom face
            Vertex {
                position: [1.000000, 1.000000, 1.000000],
                normal: [0.0000, 1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, 1.000000, 1.000000],
                normal: [0.0000, 1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, 1.000000, -1.000000],
                normal: [0.0000, 1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, 1.000000, 1.000000],
                normal: [0.0000, 1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, 1.000000, -1.000000],
                normal: [0.0000, 1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, 1.000000, -1.000000],
                normal: [0.0000, 1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            // left face
            Vertex {
                position: [-1.000000, -1.000000, -1.000000],
                normal: [-1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, 1.000000, -1.000000],
                normal: [-1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, 1.000000, 1.000000],
                normal: [-1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, -1.000000, -1.000000],
                normal: [-1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, 1.000000, 1.000000],
                normal: [-1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, -1.000000, 1.000000],
                normal: [-1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            // right face
            Vertex {
                position: [1.000000, -1.000000, 1.000000],
                normal: [1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, 1.000000, 1.000000],
                normal: [1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, 1.000000, -1.000000],
                normal: [1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, -1.000000, 1.000000],
                normal: [1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, 1.000000, -1.000000],
                normal: [1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, -1.000000, -1.000000],
                normal: [1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
        ]
        .iter()
        .cloned(),
    )
    .unwrap()
}
