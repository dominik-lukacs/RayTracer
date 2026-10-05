struct Params {
    width: u32,
    height: u32,
    max_bounces: u32,
    steps: u32,
    active_count: u32,
    epoch: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> accum: array<vec4<f32>>;

@vertex
fn vert_main(@builtin(vertex_index) VertexIndex : u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((VertexIndex << 1u) & 2u), f32(VertexIndex & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn frag_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = min(u32(pos.x), params.width - 1u);
    let y = min(u32(pos.y), params.height - 1u);
    let a = accum[y * params.width + x];

    // Pixel hasn't finished a sample yet.
    if a.w < 0.5 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    return vec4<f32>(min(a.rgb / a.w, vec3<f32>(1.0)), 1.0);
}
