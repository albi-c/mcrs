#version 450

#include "common.glsl"

layout(local_size_x = 128) in;

layout(set = 0, binding = 0) uniform texture2D textures[];
layout(set = 1, binding = 0) uniform writeonly image2D textures_rw[];
layout(set = 2, binding = 0) uniform sampler samplers[];

struct Particle {
    float16_t x;
    float16_t y;
    float16_t z;
    uint16_t group;
    float lifetime;
    uint velSpeed;
};

struct ParticleGroup {
    float16_t x;
    float16_t y;
    float16_t z;
    uint16_t tex;
    float16_t scale_x;
    float16_t scale_y;
    float16_t rot_speed;
    // 0..8 rotation speed variability (/256), 8..16 scale variability (/256)
    uint16_t rot_speed_scale_var;
};

layout(std430, buffer_reference, buffer_reference_align = 8) restrict buffer CompDataParticles {
    Particle data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer CompDataParticleGroups {
    ParticleGroup data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer CompData {
    CompDataParticles particles;
    CompDataParticleGroups groups;
    float dt;
    float maxLifetime;
};

layout(std430, push_constant) uniform Data {
    CompData comp;
} data;

vec4 readPacked(uint packed) {
    vec4 color = vec4(
        (vec3(
            float(packed & 0xff),
            float((packed >> 8) & 0xff),
            float((packed >> 16) & 0xff)
        ) / 255.0 - 0.5) * 2.0,
        float(packed >> 24) / 255.0
    );
    return color;
}

uvec2 murmurHash21(uint src) {
    const uint M = 0x5bd1e995u;
    uvec2 h = uvec2(1190494759u, 2147483647u);
    src *= M; src ^= src>>24u; src *= M;
    h *= M; h ^= src;
    h ^= h>>13u; h *= M; h ^= h>>15u;
    return h;
}

vec2 hash21(uint src) {
    uvec2 h = murmurHash21(src);
    return uintBitsToFloat(h & 0x007fffffu | 0x3f800000u) - 1.0;
}

void main() {
    CompData d = data.comp;
    uint idx = gl_GlobalInvocationID.x;
    Particle p = d.particles.data[idx];
    vec4 velSpeed = readPacked(p.velSpeed);
    vec3 vel = normalize(velSpeed.xyz) * velSpeed.w * 16.0;

    // TODO: move to particle generation
    vel.y = abs(vel.y);
    vel *= vec3(0.3, 1.5, 0.3);

    float lt = p.lifetime + d.dt;
    if (lt > d.maxLifetime) {
        lt = 0.0;
    }
    ParticleGroup g = d.groups.data[p.group];
    vec2 offset = (hash21(idx) - 0.5) * 0.25;
    vec3 pos = vec3(float(g.x), float(g.y), float(g.z)) + vec3(offset.x, 0.0, offset.y);
    pos += vel.xyz * log(lt + 1.0);
    d.particles.data[idx] = Particle(float16_t(pos.x), float16_t(pos.y), float16_t(pos.z), p.group, lt, p.velSpeed);
}
