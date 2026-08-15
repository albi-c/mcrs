#version 450

#include "common.glsl"

layout(location = 0) out vec2 outUv;
layout(location = 1) flat out uint16_t outTex;

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
    uint _padding;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertDataParticles {
    Particle data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertDataParticleGroups {
    ParticleGroup data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertData {
    mat4 mvp;
    mat4 rotation;
    VertDataParticles particles;
    VertDataParticleGroups groups;
};

layout(std430, push_constant) uniform Data {
    VertData vert;
    Pointer frag;
} data;

const vec2 OFFSETS[6] = vec2[6](
    vec2(-0.5, -0.5),
    vec2(0.5, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

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

void main() {
    // TODO: rotate particles towards camera, gravity

    VertData d = data.vert;

    uint i = gl_VertexIndex / 6;
    uint j = gl_VertexIndex % 6;

    Particle particle = d.particles.data[i];
    ParticleGroup group = d.groups.data[particle.group];
    outTex = group.tex;

    vec4 velSpeed = readPacked(particle.velSpeed);
    vec3 vel = velSpeed.xyz;

    vec3 worldPos = vec3(float(particle.x), float(particle.y), float(particle.z));
    vec2 scale = vec2(float(group.scale_x), float(group.scale_y));
    vec2 offset = OFFSETS[j];
    outUv = offset + 0.5;
    vec3 pos = worldPos + (d.rotation * vec4(offset * scale, 0.0, 0.0)).xyz;
    gl_Position = d.mvp * vec4(pos, 1.0);
}
