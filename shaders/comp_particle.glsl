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
        float(packed & 0xff),
        float((packed >> 8) & 0xff),
        float((packed >> 16) & 0xff),
        float(packed >> 24)
    );
    return color / 255.0;
}

void main() {
    CompData d = data.comp;
    uint idx = gl_GlobalInvocationID.x;
    Particle p = d.particles.data[idx];
    vec4 velSpeed = readPacked(p.velSpeed);
    vec3 vel = velSpeed.xyz * velSpeed.w * 16.0;
    vec3 pos = vec3(float(p.x), float(p.y), float(p.z));
    pos += vel * d.dt;
    float lt = p.lifetime + d.dt;
    if (lt > d.maxLifetime) {
        lt = 0.0;
        ParticleGroup g = d.groups.data[p.group];
        pos = vec3(float(g.x), float(g.y), float(g.z));
    }
//    d.particles.data[idx] = Particle(float16_t(pos.x), float16_t(pos.y), float16_t(pos.z), p.group, lt, p.velSpeed);
}
