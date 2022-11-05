#![allow(dead_code)]
use std::iter;
use wgpu::{util::DeviceExt, RenderPipeline, ComputePipeline};
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
    particle_bind_groups: Vec<wgpu::BindGroup>,
    compute_pipeline: ComputePipeline,
    frame: usize,
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
        let param_data = [0.04f32, 0.1, 0.025, 0.025, 0.02, 0.05, 0.005].to_vec();
        let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("param buffer"),
            contents: bytemuck::cast_slice(&param_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 3. particle buffer for read&write before&after particles.
        let mut particle_data = vec![0.0f32; (4 * NUM_PARTICLES) as usize];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let unif = Uniform::new_inclusive(-1.0, 1.0); 
        for particle_chunk in particle_data.chunks_mut(4){
            // particle_chunk[0] = 2.0 * (unif.sample(&mut rng) - 0.5);         // posX
            // particle_chunk[1] = 2.0 * (unif.sample(&mut rng) - 0.5);         // posY
            // particle_chunk[2] = 2.0 * (unif.sample(&mut rng) - 0.5) * 0.1;   // velX
            // particle_chunk[3] = 2.0 * (unif.sample(&mut rng) - 0.5) * 0.1;   // velY
            
            particle_chunk[0] = unif.sample(&mut rng);         // posX
            particle_chunk[1] = unif.sample(&mut rng);         // posY
            particle_chunk[2] = unif.sample(&mut rng);   // velX
            particle_chunk[3] = unif.sample(&mut rng);   // velY

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
            bind_group_layouts: &[], // The bind group layout pre-defines the types, 
                                     // purposes and uses of these GPU entities, 
                                     // which allows the GPU figure out 
                                     // how to run a pipeline most efficiently ahead of time. 
                                     // this is probably why it's seperated from the actual bind group
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
                        step_mode: wgpu::VertexStepMode::Instance, // important!!!!!!!
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
        
        // when we done with render pipeline, let's consider something about compute pipeline ^ ^
        // 1. bind_group_layout    
        // 2. bind_group           
        // 3. pipeline layout      
        // 4. pipeline     
        // we actually dont need bind_group for the initializing pipeline along its layout.
        // put it in the second just for concise   
        
        // for compute shader, we need
        // 1. parameter 
        // 2. particle A (read)
        // 3. particle B (write)
        let compute_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor{
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Uniform, 
                            has_dynamic_offset: false, 
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: true }, 
                            has_dynamic_offset: false, 
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: false }, 
                            has_dynamic_offset: false, 
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ]
            }
        );

        // remember when we have two particle buffer, one for read and one for write
        // if we only have one bind group, and particle[0] for reading, particles[1] for writing
        // then for compute shader, every time we feed the same bind group
        // then that would be incorret, cause in the next frame, we need to use the written buffer for reading
        
        // therefore we need to have two bind group. for example
        // In frame 0, pass in bind_group[0], read particles[0] write particles[1]
        // In frame 1, pass in bind_group[1], read particles[1] write particles[0]
        // so for bind_group[0], particles[0] is for reading and vice versa
        let mut particle_bind_groups = Vec::<wgpu::BindGroup>::new();
        for i in 0..2 {
            particle_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particle_buffers[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: particle_buffers[(i + 1) % 2].as_entire_binding(), // bind to opposite buffer
                    },
                ],
                label: None,
            }));
        }

        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("compute"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("Compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "cs_main",
        });

        let frame = 0;

        Self {
            surface: surface,
            device: device,
            queue: queue,
            config: config,
            size: size,
            render_pipeline: render_pipeline,
            vertex_buffer: vertex_buffer,
            particle_buffers: particle_buffers,
            compute_pipeline: compute_pipeline,
            particle_bind_groups: particle_bind_groups,
            frame: frame,
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


        // compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: None });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.particle_bind_groups[self.frame % 2], &[]);
            compute_pass.dispatch_workgroups(((NUM_PARTICLES as f32) / 64.0).ceil() as u32, 1, 1);
        }
        // render pass
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
            render_pass.set_vertex_buffer(0, self.particle_buffers[(self.frame + 1) % 2].slice(..)); 
            render_pass.set_vertex_buffer(1, self.vertex_buffer.slice(..));
            render_pass.draw(0..3, 0..NUM_PARTICLES);
        }
        self.frame += 1;
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
