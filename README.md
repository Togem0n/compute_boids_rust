A GPU based boid simulation written in rust based on [austin-Eng/webgpu-sample](https://austin-eng.com/webgpu-samples/samples/computeBoids), compute shader is used here to optimise the particles' computation.

The original example is written in Javascript. For better understanding webgpu pipeline in rust, I manage to migrate the project to rust and the final result behaves the same as the original one.

To view the simulation of birds, simply run

```
Cargo run
```

![Boid](https://github.com/Togem0n/compute_boids_rust/blob/main/computeBoid.gif)

