struct Input {
    @location(0) a_particle_pos : vec2<f32>,
    @location(1) a_particle_vel : vec2<f32>,
    @location(2) a_pos : vec2<f32>,
};

struct Output {
    @builtin(position) position : vec4<f32>,
    @location(0) v_vel:vec2<f32>,
};

@vertex
fn vs_main(input: Input) -> Output {
    var output: Output;
    var angle : f32 = -atan2(input.a_particle_vel.x, input.a_particle_vel.y);
    var pos : vec2<f32> = vec2<f32>(
        (input.a_pos.x * cos(angle)) - (input.a_pos.y * sin(angle)),
        (input.a_pos.x * sin(angle)) + (input.a_pos.y * cos(angle)));
    output.position = vec4<f32>(pos + input.a_particle_pos, 0.0, 1.0);
    output.v_vel = input.a_particle_vel;
    return output;
}

@fragment
fn fs_main(output: Output) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}