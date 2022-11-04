#![allow(dead_code)]
use std::iter;
use wgpu::{util::DeviceExt, RenderPipeline};
use winit::{
    event::*,
    window::Window,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use rand::{
    distributions::{Distribution, Uniform},
    SeedableRng,
};

struct State {
    // transform configuration
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    particle_buffers: Vec<wgpu::Buffer>,
}

const NUM_PARTICLES: u32 = 5000;


impl State {
    async fn new(window: &Window) ->Self {
        
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter:false,
            }
        ).await.unwrap();
        
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None, // Trace path
        ).await.unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_supported_formats(&adapter)[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
        };
        surface.configure(&device, &config);
        
        let sprite_shader = device.create_shader_module(
            wgpu::ShaderModuleDescriptor{
            label: Some("sprite shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader_sprite.wgsl").into()),
            }
        );

        let compute_shader = device.create_shader_module(
            wgpu::ShaderModuleDescriptor{
            label: Some("compute shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader_compute.wgsl").into()),
            }
        );

        // what kind of buffer we need?
        // 1. vertex buffer
        let vertex_data = [-0.01f32, -0.02, 0.01, -0.02, 0.00, 0.02];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("vertex buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX,   
        });

        // 2. param buffer
        let param_data = [0.04f32, 0.1, 0.025, 0.025, 0.02, 0.05, 0.005];
        let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("param buffer"),
            contents: bytemuck::cast_slice(&param_data),
            usage: wgpu::BufferUsages::VERTEX,   
        });

        // 3. particle buffer for read&write before&after particles.
        let mut particle_data = vec![0.0f32; (4 * NUM_PARTICLES) as usize];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let unif = Uniform::new_inclusive(-1.0, 1.0);
        for particle_instance_chunk in particle_data.chunks_mut(4){
            particle_instance_chunk[0] = unif.sample(&mut rng);         // posX
            particle_instance_chunk[1] = unif.sample(&mut rng);         // posY
            particle_instance_chunk[2] = unif.sample(&mut rng) * 0.1;   // velX
            particle_instance_chunk[3] = unif.sample(&mut rng) * 0.1;   // velY
        }

        let mut particle_buffers = Vec::<wgpu::Buffer>::new();

        for i in 0..2 {
            particle_buffers.push(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
                    label: Some(&format!("Particle Buffer {}", i)),
                    contents: bytemuck::cast_slice(&particle_data),
                    usage: wgpu::BufferUsages::VERTEX |
                            wgpu::BufferUsages::STORAGE |
                            wgpu::BufferUsages::COPY_DST,   
                }),
            );
        }
        // next step would be we need to feed vertex&particle buffer's value to vertex shader
        // since vertex shader need this two to render objects
        // we also need to pass particles buffer to compute shader (using bind group)

        // first create render pipeline
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("render"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor{
            label: None,
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState{
                module: &sprite_shader,
                entry_point: "vs_main",
                buffers: &[
                    wgpu::VertexBufferLayout{
                        array_stride: 4 * 4,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 =>Float32x2],
                    },
                    wgpu::VertexBufferLayout{
                        array_stride: 2 * 4,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![2 => Float32x2],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &sprite_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            surface: surface,
            device: device,
            queue: queue,
            config: config,
            size: size,
            render_pipeline: render_pipeline,
            vertex_buffer: vertex_buffer,
            particle_buffers: particle_buffers
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    #[allow(unused_variables)]
    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {}

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.particle_buffers[0].slice(..)); 
            render_pass.set_vertex_buffer(1, self.vertex_buffer.slice(..));
            render_pass.draw(0..3, 0..NUM_PARTICLES);
        }
        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub async fn run() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    let mut state = State::new(&window).await;
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                if !state.input(event) {
                    // UPDATED!
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            // new_inner_size is &&mut so w have to dereference it twice
                            state.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
            }
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    
                    Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                }
            }
            Event::RedrawEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                window.request_redraw();
            }
            _ => {}
        }
    });
}
