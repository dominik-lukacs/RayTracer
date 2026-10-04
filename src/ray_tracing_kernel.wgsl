const EPSILON = 0.001f;

const MIN_T: f32 = 0.001f;
const MAX_T: f32 = 1000.0f;

const PI = 3.1415927f;
const FRAC_1_PI = 0.31830987f;
const FRAC_PI_2 = 1.5707964f;


struct Params {
    width: u32,
    height: u32,
    max_bounces: u32,
    steps: u32,
    active_count: u32,
    group_start: u32,
    epoch: u32,
    _pad: u32,
}

struct PathState {
    origin: vec3<f32>,
    bounce: u32,
    direction: vec3<f32>,
    rng: u32,
    throughput: vec3<f32>,
    flags: u32, // 0 = uninitialized
    radiance: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> paths: array<PathState>;
@group(0) @binding(2) var<storage, read_write> accum: array<vec4<f32>>;

@group(1) @binding(0) var<storage, read> spheres: array<Sphere>;
@group(1) @binding(1) var<storage, read> materials: array<Material>;
@group(1) @binding(2) var<storage, read> textures: array<array<f32, 3>>;
@group(1) @binding(3) var<storage, read> lights: array<u32>;

fn length_squared(v: vec3<f32>) -> f32 {
    return v.x*v.x + v.y*v.y + v.z*v.z;
}

fn cameraRay(pixel: vec2<u32>, rng: ptr<function, u32>) -> Ray {
    let size = vec2<f32>(f32(params.width), f32(params.height));
    let aspect_ratio = size.x / size.y;
    let viewport_height = size.y / 500.0;
    let viewport_width = aspect_ratio * viewport_height;
    let focal_length = 1.0;

    let origin = vec3<f32>(0.0, 0.0, 0.0);
    let horizontal = vec3<f32>(viewport_width, 0.0, 0.0);
    let vertical = vec3<f32>(0.0, viewport_height, 0.0);
    let upper_left_corner = origin - horizontal/2.0 + vertical/2.0 - vec3<f32>(0.0, 0.0, focal_length);

    // Sub-pixel jitter
    let u = (f32(pixel.x) + rngNextFloat(rng)) / size.x;
    let v = (f32(pixel.y) + rngNextFloat(rng)) / size.y;

    var ray: Ray;
    ray.origin = origin;
    ray.direction = normalize(upper_left_corner + u*horizontal - v*vertical - origin);
    return ray;
}

fn newPath(pixel: vec2<u32>, rng: ptr<function, u32>) -> PathState {
    let ray = cameraRay(pixel, rng);
    return PathState(ray.origin, 0u, ray.direction, 0u, vec3(1f), 1u, vec3(0f), 0f);
}

fn isFiniteColor(c: vec3<f32>) -> bool {
    // Written so that NaN compares false and gets rejected
    return all(abs(c) < vec3<f32>(1e20));
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let group_total = nwg.x * nwg.y;
    let group_idx = wid.y * nwg.x + wid.x;
    if (group_idx + group_total - params.group_start) % group_total >= params.active_count {
        return;
    }
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }

    let idx = gid.y * params.width + gid.x;
    var st = paths[idx];
    var acc = accum[idx];
    var rng = st.rng;

    if st.flags == 0u {
        rng = initRng(gid.xy, vec2<u32>(params.width, params.height), params.epoch);
        st = newPath(gid.xy, &rng);
    }

    for (var i = 0u; i < params.steps; i += 1u) {
        if advancePath(&st, &rng) {
            if isFiniteColor(st.radiance) {
                acc += vec4<f32>(st.radiance, 1.0);
            }
            st = newPath(gid.xy, &rng);
        }
    }

    st.rng = rng;
    paths[idx] = st;
    accum[idx] = acc;
}

fn ray_color(ray: Ray) -> vec3<f32> {
    var pixel_color: vec3<f32>;


    var didHit: bool = false;
    for (var i = 0u; i < arrayLength(&spheres); i = i + 1u) {
        let t = hit(ray, spheres[i]);
        if (t > 0.0) {
            let N: vec3<f32> = normalize((ray.origin + normalize(ray.direction) * t) - vec3<f32>(0.0, 0.0, -1.0));
            pixel_color = 0.5 * vec3<f32>(N.x+1.0, N.y+1.0, N.z+1.0);
            didHit = true;
            break;
        }
    }
    if (!didHit) {
        let t = 0.5 * (ray.direction.y + 1.0);
        pixel_color = (1.0 - t) * vec3<f32>(1.0, 1.0, 1.0) + t * vec3<f32>(0.5, 0.7, 1.0);
    }

    return pixel_color;
}

fn hit(ray: Ray, sphere: Sphere) -> f32 {
    let oc = ray.origin - sphere.center.xyz;
    let a: f32 = length_squared(ray.direction);
    let half_b: f32 = dot(ray.direction, oc);
    let c: f32 = length_squared(oc) - sphere.radius * sphere.radius;
    let discriminant: f32 = half_b*half_b - a * c;

    if discriminant < 0.0 {
        return -1.0;
    } else {
        return (-half_b - sqrt(discriminant) ) / a;
    }
}

// models
struct Sphere {
    center: vec4<f32>,
    radius: f32,
    material_idx: u32,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct Scatter {
    ray: Ray,
    throughput: vec3<f32>,
}

struct Intersection {
    p: vec3<f32>,
    n: vec3<f32>,
    u: f32,
    v: f32,
    t: f32,
    material_idx: u32,
    sphere_idx: u32,
}

struct Material {
    id: u32,
    desc1: TextureDescriptor,
    desc2: TextureDescriptor,
    x: f32,
}

struct TextureDescriptor {
    width: u32,
    height: u32,
    offset: u32,
}


fn sphereIntersection(ray: Ray, sphere: Sphere, sphere_idx: u32, t: f32) -> Intersection {
    let p = rayPointAtParameter(ray, t);
    let n = (1f / sphere.radius) * (p - sphere.center.xyz);
    let theta = acos(-n.y);
    let phi = atan2(-n.z, n.x) + PI;
    let u = 0.5 * FRAC_1_PI * phi;
    let v = FRAC_1_PI * theta;

    // TODO: passing sphereIdx in here just to pass it to Intersection
    return Intersection(p, n, u, v, t, sphere.material_idx, sphere_idx);
}

fn rayPointAtParameter(ray: Ray, t: f32) -> vec3<f32> {
    return ray.origin + t * ray.direction;
}

fn rayIntersectSphere(ray: Ray, sphereIdx: u32, tmin: f32, tmax: f32, hit: ptr<function, Intersection>) -> bool {
    let sphere = spheres[sphereIdx];
    let oc = ray.origin - sphere.center.xyz;
    let a = dot(ray.direction, ray.direction);
    let b = dot(oc, ray.direction);
    let c = dot(oc, oc) - sphere.radius * sphere.radius;
    let discriminant = b * b - a * c;

    if discriminant >= 0f {
        var t = (-b - sqrt(discriminant)) / a;
        if t < tmax && t > tmin {
            *hit = sphereIntersection(ray, sphere, sphereIdx, t);
            return true;
        }

        t = (-b + sqrt(discriminant)) / a;
        if t < tmax && t > tmin {
            *hit = sphereIntersection(ray, sphere, sphereIdx, t);
            return true;
        }
    }

    return false;
}

fn intersect(ray: Ray, intersection: ptr<function, Intersection>) -> bool {
    var closestT = MAX_T;
    var closestIntersection = Intersection();

    for (var idx = 0u; idx < arrayLength(&spheres); idx = idx + 1u) {
        var testIntersect = Intersection();
        if rayIntersectSphere(ray, idx, MIN_T, closestT, &testIntersect) {
            closestT = testIntersect.t;
            closestIntersection = testIntersect;
        }
    }

    if closestT < MAX_T {
        *intersection = closestIntersection;
        return true;
    }

    return false;
}

fn advancePath(st: ptr<function, PathState>, rng: ptr<function, u32>) -> bool {
    let ray = Ray((*st).origin, (*st).direction);
    var intersection = Intersection();

    if !intersect(ray, &intersection) {
        // Ray missed
        let t = 0.5 * (normalize(ray.direction).y + 1.0);
        let sky_color = (1.0 - t) * vec3<f32>(1.0, 1.0, 1.0) + t * vec3<f32>(0.5, 0.7, 1.0);
        (*st).radiance += (*st).throughput * sky_color;
        return true;
    }

    let material = materials[intersection.material_idx];


    if material.id == 4u {
        let emissionColor = textureLookup(material.desc1, intersection.u, intersection.v);
        (*st).radiance += (*st).throughput * emissionColor;
        return true;
    }

    if (*st).bounce + 1u >= params.max_bounces {
        return true;
    }

    let scatter = scatterRay(ray, intersection, material, rng);
    (*st).origin = scatter.ray.origin;
    (*st).direction = scatter.ray.direction;
    (*st).throughput = scatter.throughput;
    (*st).bounce += 1u;

    // Russian roulette: low contribution paths killed early
    if (*st).bounce >= 3u  {
        let tp = (*st).throughput;
        let p = clamp(max(tp.x, max(tp.y, tp.z)), 0.05, 1.0);
        if rngNextFloat(rng) > p {
            return true;
        }
        (*st).throughput = tp / p;
    }

    return false;
}

fn scatterRay(wo: Ray, hit: Intersection, material: Material, rngState: ptr<function, u32>) -> Scatter {
    switch material.id {
        case 0u: {
            let texture = material.desc1;
            return scatterMixtureDensity(hit, texture, rngState);
        }

        case 1u: {
            let texture = material.desc1;
            let fuzz = material.x;
            return scatterMetal(wo, hit, texture, fuzz, rngState);
        }

        default: {
            return scatterMissingMaterial(hit, rngState);
        }
    }
}

fn scatterMissingMaterial(hit: Intersection, rngState: ptr<function, u32>) -> Scatter {
    let scatterDirection = hit.n + rngNextVec3InUnitSphere(rngState);
    // An aggressive pink color to indicate an error
    let albedo = vec3(0.5f, 0.7f, 0.9f);
    return Scatter(Ray(hit.p, scatterDirection), albedo);
}

fn textureLookup(desc: TextureDescriptor, u: f32, v: f32) -> vec3<f32> {
    let u_ = clamp(u, 0f, 1f);
    let v_ = 1f - clamp(v, 0f, 1f);

    let j = u32(u_ * f32(desc.width));
    let i = u32(v_ * f32(desc.height));
    let idx = i * desc.width + j;

    let elem = textures[desc.offset + idx];
    return vec3(elem[0u], elem[1u], elem[2u]);
}

fn scatterMixtureDensity(hit: Intersection, albedo: TextureDescriptor, rngState: ptr<function, u32>) -> Scatter {
    let scatterDirection = sampleMixtureDensity(hit, rngState);
    let materialValue = evalLambertian(hit, albedo, scatterDirection);
    let materialPdf = pdfLambertian(hit, scatterDirection);
    let lightPdf = pdfLight(hit, scatterDirection);
    let throughput = materialValue / max(EPSILON, (0.5f * materialPdf + 0.5f * lightPdf));
    return Scatter(Ray(hit.p, scatterDirection), throughput);
}

fn sampleMixtureDensity(hit: Intersection, rngState: ptr<function, u32>) -> vec3<f32> {
    if rngNextFloat(rngState) < 0.5f {
        return sampleLambertian(hit, rngState);
    } else {
        return sampleLight(hit, rngState);
    }
}

fn evalLambertian(hit: Intersection, texture: TextureDescriptor, wi: vec3<f32>) -> vec3<f32> {
    return textureLookup(texture, hit.u, hit.v) * FRAC_1_PI * max(EPSILON, dot(hit.n, wi));
}

fn sampleLambertian(hit: Intersection, rngState: ptr<function, u32>) -> vec3<f32> {
    let v = rngNextInCosineWeightedHemisphere(rngState);
    let onb = pixarOnb(hit.n);
    return onb * v;
}

fn pdfLambertian(hit: Intersection, wi: vec3<f32>) -> f32 {
    return max(EPSILON, dot(hit.n, wi) * FRAC_1_PI);
}

fn sampleLight(hit: Intersection, rngState: ptr<function, u32>) -> vec3<f32> {
    // Select a random light using a uniform distribution.
    let numLights = arrayLength(&lights);   // TODO: what about when there are no lights?
    let lightIdx = rngNextUintInRange(rngState, 0u, numLights - 1u);
    let sphereIdx = lights[lightIdx];
    let sphere = spheres[sphereIdx];

    return sampleHemisphere(hit, sphere, rngState);
}

fn sampleHemisphere(hit: Intersection, sphere: Sphere, rngState: ptr<function, u32>) -> vec3<f32> {
    let v = rngNextInUnitHemisphere(rngState);

    // Sample the hemisphere facing the intersection point.
    let dir = normalize(hit.p - sphere.center.xyz);
    let onb = pixarOnb(dir);

    let pointOnSphere = sphere.center.xyz + onb * sphere.radius * v;
    let toPointOnSphere = pointOnSphere - hit.p;

    return normalize(toPointOnSphere);
}

fn pdfLight(hit: Intersection, wi: vec3<f32>) -> f32 {
    let ray = Ray(hit.p, wi);
    var lightHit = Intersection();
    var pdf = 0f;

    if intersect(ray, &lightHit) {
        let sphereIdx = lightHit.sphere_idx;
        let sphere = spheres[sphereIdx];
        let numSpheres = arrayLength(&spheres);
        let toLight = lightHit.p - hit.p;
        let lengthSqr = dot(toLight, toLight);
        let cosine = abs(dot(wi, lightHit.n));
        let areaHalfSphere = 2f * PI * sphere.radius * sphere.radius;

        // lengthSqr / cosine is the inverse of the geometric factor, as defined in
        // "MULTIPLE IMPORTANCE SAMPLING 101".
        pdf = lengthSqr / max(EPSILON, cosine * areaHalfSphere * f32(numSpheres));
    }

    return pdf;
}

fn pixarOnb(n: vec3<f32>) -> mat3x3<f32> {
    // https://www.jcgt.org/published/0006/01/01/paper-lowres.pdf
    let s = select(-1f, 1f, n.z >= 0f);
    let a = -1f / (s + n.z);
    let b = n.x * n.y * a;
    let u = vec3<f32>(1f + s * n.x * n.x * a, s * b, -s * n.x);
    let v = vec3<f32>(b, s + n.y * n.y * a, -n.y);

    return mat3x3<f32>(u, v, n);
}

fn scatterMetal(wo: Ray, hit: Intersection, texture: TextureDescriptor, fuzz: f32, rngState: ptr<function, u32>) -> Scatter {
    let scatterDirection = reflect(wo.direction, hit.n) + fuzz * rngNextVec3InUnitSphere(rngState);
    let albedo = textureLookup(texture, hit.u, hit.v);
    return Scatter(Ray(hit.p, scatterDirection), albedo);
}


// random number generation

fn rngNextInCosineWeightedHemisphere(state: ptr<function, u32>) -> vec3<f32> {
    let r1 = rngNextFloat(state);
    let r2 = rngNextFloat(state);
    let sqrt_r2 = sqrt(r2);

    let z = sqrt(1f - r2);
    let phi = 2f * PI * r1;
    let x = cos(phi) * sqrt_r2;
    let y = sin(phi) * sqrt_r2;

    return vec3<f32>(x, y, z);
}

fn rngNextInUnitHemisphere(state: ptr<function, u32>) -> vec3<f32> {
    let r1 = rngNextFloat(state);
    let r2 = rngNextFloat(state);

    let phi = 2f * PI * r1;
    let sinTheta = sqrt(1f - r2 * r2);

    let x = cos(phi) * sinTheta;
    let y = sin(phi) * sinTheta;
    let z = r2;

    return vec3(x, y, z);
}

fn rngNextVec3InUnitDisk(state: ptr<function, u32>) -> vec3<f32> {
    // Generate numbers uniformly in a disk:
    // https://stats.stackexchange.com/a/481559

    // r^2 is distributed as U(0, 1).
    let r = sqrt(rngNextFloat(state));
    let alpha = 2f * PI * rngNextFloat(state);

    let x = r * cos(alpha);
    let y = r * sin(alpha);

    return vec3(x, y, 0f);
}

fn rngNextVec3InUnitSphere(state: ptr<function, u32>) -> vec3<f32> {
    // probability density is uniformly distributed over r^3
    let r = pow(rngNextFloat(state), 0.33333f);
    let theta = PI * rngNextFloat(state);
    let phi = 2f * PI * rngNextFloat(state);

    let x = r * sin(theta) * cos(phi);
    let y = r * sin(theta) * sin(phi);
    let z = r * cos(theta);

    return vec3(x, y, z);
}

fn rngNextUintInRange(state: ptr<function, u32>, min: u32, max: u32) -> u32 {
    return min + rngNextInt(state) % (max - min);
}

fn rngNextFloat(state: ptr<function, u32>) -> f32 {
    // rngNextInt(state);
    // return f32(*state) / f32(0xffffffffu);
    // TODO AI generated, check if above is better
    return f32(rngNextInt(state) >> 8u) * (1.0 / 16777216.0);
}

fn initRng(pixel: vec2<u32>, resolution: vec2<u32>, frame: u32) -> u32 {
    // Adapted from https://github.com/boksajak/referencePT
    let seed = dot(pixel, vec2<u32>(1u, resolution.x)) ^ jenkinsHash(frame);
    return jenkinsHash(seed);
}

fn rngNextInt(state: ptr<function, u32>) -> u32 {
    // PCG random number generator
    // Based on https://www.shadertoy.com/view/XlGcRh

    let oldState = *state;
    *state = oldState + 747796405u + 2891336453u;
    let word = ((oldState >> ((oldState >> 28u) + 4u)) ^ oldState) * 277803737u;
    return (word >> 22u) ^ word;
}

fn jenkinsHash(input: u32) -> u32 {
    var x = input;
    x += x << 10u;
    x ^= x >> 6u;
    x += x << 3u;
    x ^= x >> 11u;
    x += x << 15u;
    return x;
}

