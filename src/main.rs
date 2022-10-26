extern crate vulkano;
extern crate vulkano_win;
extern crate winit;

use std::{sync::Arc, time::Instant};

use nalgebra_glm::{look_at, perspective, pi, rotate_normalized_axis, translate, vec3, TMat4};
use shaders::vs;
use vertex::MVP;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
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
        GraphicsPipeline, Pipeline,
    },
    render_pass::{self, Framebuffer, RenderPass, Subpass},
    swapchain::{
        self, AcquireError, PresentInfo, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync::{FlushError, GpuFuture},
    Version, VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod shaders;
mod vertex;

fn main() {
    let instance = {
        let library = VulkanLibrary::new().unwrap();
        let extensions = vulkano_win::required_extensions(&library);
        Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: extensions,
                max_api_version: Some(Version::V1_1),
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| {
            panic!("Error creating an instance: {:?}", e);
        })
    };

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_ext))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    // q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false)
                    q.queue_flags.graphics && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .expect("No suitable device found");

    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            enabled_extensions: device_ext,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
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

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let mut framebuffers =
        window_size_dependent_setup(&images, render_pass.clone(), &mut viewport, device.clone());

    let mut recreate_swapchain = false;
    let mut previous_frame_end =
        Some(Box::new(vulkano::sync::now(device.clone())) as Box<dyn GpuFuture>);

    // let rotation_start = Instant::now();

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }

        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain = true;
        }

        Event::RedrawEventsCleared => {
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
                    Err(e) => panic!("Failed to recreate swapchain {:?}", e),
                };

                swapchain = new_swapchain;
                framebuffers = window_size_dependent_setup(
                    &new_images,
                    render_pass.clone(),
                    &mut viewport,
                    device.clone(),
                );
                recreate_swapchain = false
            }

            let uniform_buffer = CpuBufferPool::<vs::ty::MVP_Data>::uniform_buffer(device.clone());

            let uniform_buffer_subbufer = {
                let mut mvp = MVP::new();
                mvp.view = look_at(
                    &vec3(0.0, 0.0, 0.01),
                    &vec3(0.0, 0.0, 0.0),
                    &vec3(0.0, -1.0, 0.0),
                );

                let dimensions: [u32; 2] = surface.window().inner_size().into();
                mvp.projection = perspective(
                    dimensions[0] as f32 / dimensions[1] as f32,
                    180.0,
                    0.01,
                    100.0,
                );

                // mvp.model = translate(&nalgebra_glm::identity(), &vec3(0.0, 0.0, -0.5));

                // let elapsed = rotation_start.elapsed().as_secs() as f64
                //     + rotation_start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0;
                // let elapsed_as_radians = elapsed * pi::<f64>() / 180.0 * 30.0;
                // let model = rotate_normalized_axis(
                //     &mvp.model,
                //     elapsed_as_radians as f32,
                //     &vec3(0.0, 0.0, 1.0),
                // );

                let uniform_data = vs::ty::MVP_Data {
                    world: mvp.model.into(),
                    view: mvp.view.into(),
                    projection: mvp.projection.into(),
                };

                uniform_buffer.from_data(uniform_data).unwrap()
            };

            let vs = shaders::vs::load(device.clone()).unwrap();
            let fs = shaders::fs::load(device.clone()).unwrap();

            let pipeline = GraphicsPipeline::start()
                .vertex_input_state(BuffersDefinition::new().vertex::<vertex::Vertex>())
                .vertex_shader(vs.entry_point("main").unwrap(), ())
                .input_assembly_state(InputAssemblyState::new())
                .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
                .fragment_shader(fs.entry_point("main").unwrap(), ())
                .depth_stencil_state(DepthStencilState::simple_depth_test())
                .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(device.clone())
                .unwrap();

            let layout = pipeline.layout().set_layouts().get(0).unwrap();
            let set = PersistentDescriptorSet::new(
                layout.clone(),
                [WriteDescriptorSet::buffer(0, uniform_buffer_subbufer)],
            )
            .unwrap();

            let (image_num, suboptimal, acquired_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

            if suboptimal {
                recreate_swapchain = true
            }

            let clear_values = vec![Some([0.0, 0.68, 1.0, 1.0].into()), Some(1f32.into())];

            let vertex_buffer = CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::empty(),
                false,
                [
                    vertex::Vertex {
                        position: [-0.5, 0.5, -0.5],
                        color: [0.0, 0.0, 0.0],
                    },
                    vertex::Vertex {
                        position: [0.5, 0.5, -0.5],
                        color: [0.0, 0.0, 0.0],
                    },
                    vertex::Vertex {
                        position: [0.0, -0.5, -0.5],
                        color: [0.0, 0.0, 0.0],
                    },
                    // 2
                    vertex::Vertex {
                        position: [-0.5, -0.5, -0.6],
                        color: [1.0, 1.0, 1.0],
                    },
                    vertex::Vertex {
                        position: [0.5, -0.5, -0.6],
                        color: [1.0, 1.0, 1.0],
                    },
                    vertex::Vertex {
                        position: [0.0, 0.5, -0.6],
                        color: [1.0, 1.0, 1.0],
                    },
                ]
                .iter()
                .cloned(),
            )
            .unwrap();

            let mut cmd_buffer_builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue_family_index,
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            cmd_buffer_builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values,
                        ..RenderPassBeginInfo::framebuffer(framebuffers[image_num].clone())
                    },
                    SubpassContents::Inline,
                )
                .unwrap()
                .set_viewport(0, [viewport.clone()])
                .bind_pipeline_graphics(pipeline.clone())
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Graphics,
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
                .join(acquired_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(
                    queue.clone(),
                    PresentInfo {
                        index: image_num,
                        ..PresentInfo::swapchain(swapchain.clone())
                    },
                )
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame_end = Some(Box::new(future) as Box<_>);
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end =
                        Some(Box::new(vulkano::sync::now(device.clone())) as Box<_>)
                }
                Err(e) => {
                    eprintln!("Failed to flush future: {:?}", e);
                    previous_frame_end =
                        Some(Box::new(vulkano::sync::now(device.clone())) as Box<_>)
                }
            }
        }
        _ => {}
    });
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
    device: Arc<Device>,
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
                render_pass::FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
