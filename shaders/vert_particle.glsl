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
    float16_t rot_speed;
    // 0..8 rotation speed variability (/256), 8..16 scale variability (/256)
    uint16_t rot_speed_scale_var;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertDataParticles {
    Particle data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertDataParticleGroups {
    ParticleGroup data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertData {
    mat4 mvp;
    vec4 camera_right;
    vec4 camera_up;
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

float variabilityModifier(float variability, float inp) {
    // input between 0 and 1, output around -1 or 1
    float halfVar = 0.5 * variability;
    inp -= 0.5;
    inp = inp / 0.5 * halfVar;
    inp += sign(inp) - 0.5 * halfVar;
    return inp;
}

float variabilityModifierPos(float variability, float inp) {
    // input between 0 and 1, output around 1
    inp *= variability;
    inp += 1.0 - 0.5 * variability;
    return inp;
}

void main() {
    // TODO: rotate particles towards camera, gravity

    // TODO!!!: remove compute for particles, calculate everything in vertex shader - cur_lifetime = time % lifetime

    VertData d = data.vert;

    uint i = gl_VertexIndex / 6;
    uint j = gl_VertexIndex % 6;

    vec2 hashed = hash21(i);

    Particle particle = d.particles.data[i];
    ParticleGroup group = d.groups.data[particle.group];
    outTex = group.tex;

    float rotSpeedVar = float(group.rot_speed_scale_var & 0xff) / 256.0;
    float scaleVar = float(group.rot_speed_scale_var >> 8) / 256.0;

    float rotation = hashed.x + particle.lifetime * float(group.rot_speed) * variabilityModifier(rotSpeedVar, hashed.y);
    vec2 basePointOffset = OFFSETS[j];
    vec2 pointOffset = vec2(0.0);
    pointOffset.x = basePointOffset.x * cos(rotation) + basePointOffset.y * sin(rotation);
    pointOffset.y = - basePointOffset.x * sin(rotation) + basePointOffset.y * cos(rotation);

    vec4 velSpeed = readPacked(particle.velSpeed);
    vec3 vel = velSpeed.xyz;

    vec3 worldPos = vec3(float(particle.x), float(particle.y), float(particle.z));
    // TODO: add lifetime to group - particles get smaller over time
    vec2 scale = vec2(float(group.scale_x), float(group.scale_y)) * variabilityModifierPos(scaleVar, hashed.x); // * (particle.lifetime / group.lifetime);
    outUv = OFFSETS[j] + 0.5;
    // first is spherical (broken when looking from the top), second is cylindrical
//    vec3 pos = worldPos + vec3(-1.0, 1.0, 1.0) * d.camera_right.xyz * pointOffset.x * scale.x + d.camera_up.xyz * pointOffset.y * scale.y;
    vec3 pos = worldPos + vec3(-1.0, 1.0, 1.0) * d.camera_right.xyz * pointOffset.x * scale.x + vec3(0.0, pointOffset.y * scale.y, 0.0);
    gl_Position = d.mvp * vec4(pos, 1.0);
}
