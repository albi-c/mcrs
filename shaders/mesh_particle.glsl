#version 450

#include "common.glsl"
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require

const uint PARTICLE_COUNT = 16;
const uint VERTEX_COUNT = PARTICLE_COUNT * 4;
const uint PRIMITIVE_COUNT = PARTICLE_COUNT * 2;

layout(local_size_x = 4, local_size_y = PARTICLE_COUNT, local_size_z = 1) in;
layout(triangles, max_vertices = VERTEX_COUNT, max_primitives = PRIMITIVE_COUNT) out;

layout(location = 0) out vec2 outUvs[];
layout(location = 1) flat out uint16_t outTextures[];

struct Particle {
    float16_t origin[3];
    float16_t velocity[3];
    float16_t acceleration[3];
    float16_t spiralRadius;
    float16_t spiralOffset;
    float16_t spiralSpeed;
    float16_t spiralVelocityInfluence;
    float16_t timeOffset;
    float16_t lifetime;
    uint16_t tex;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataParticles {
    Particle data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshData {
    mat4 mvp;
    vec4 cameraRight;
    vec4 cameraUp;
    MeshDataParticles particles;
    float time;
    uint _padding;
};

layout(std430, push_constant) uniform Data {
    MeshData mesh;
    Pointer frag;
} data;

const vec2 OFFSETS[4] = vec2[4](
    vec2(-0.5, -0.5),
    vec2(0.5, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

vec3 readVec16(in float16_t vec[3]) {
    return vec3(float(vec[0]), float(vec[1]), float(vec[2]));
}

vec3 getParticlePosition(in Particle p, float time) {
    float t = mod(time + float(p.timeOffset), float(p.lifetime));

    vec3 pOrigin = readVec16(p.origin);
    vec3 pVelocity = readVec16(p.velocity);
    vec3 pAcceleration = readVec16(p.acceleration);

    vec3 velocity = pVelocity + t * pAcceleration;
    vec3 position = pOrigin + t * pVelocity + t * t * 0.5 * pAcceleration;

    if (length(velocity) < 0.00001) {
        return position;
    }

    vec3 axis = normalize(velocity);

    vec3 reference = mix(
        vec3(1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        step(0.9, abs(axis.x))
    );

    vec3 u = normalize(cross(axis, reference));
    vec3 v = cross(axis, u);

    float speed0 = length(pVelocity);
    float speed1 = length(velocity);
    float angle = float(p.spiralOffset) + t * float(p.spiralSpeed) + t * 0.5 * float(p.spiralVelocityInfluence) * (speed0 + speed1);
    vec2 cs = vec2(cos(angle), sin(angle));

    return position + p.spiralRadius * cs.x * u + p.spiralRadius * cs.y * v;
}

void main() {
    MeshData d = data.mesh;
    if (gl_LocalInvocationIndex == 0) {
        SetMeshOutputsEXT(VERTEX_COUNT, PRIMITIVE_COUNT);
    }
    uvec2 localId = gl_LocalInvocationID.xy;
    if (localId.x == 0) {
        gl_PrimitiveTriangleIndicesEXT[2 * localId.y + 0] = uvec3(0, 1, 2) + gl_LocalInvocationIndex;
        gl_PrimitiveTriangleIndicesEXT[2 * localId.y + 1] = uvec3(0, 2, 3) + gl_LocalInvocationIndex;
    }

    Particle p = d.particles.data[PARTICLE_COUNT * gl_WorkGroupID.x + localId.y];

    // TODO: time calculations in f64
    // TODO: add texture rotation in screen space
    // TODO: fix spherical billboarding, make spherical/cylindrical selectable with parameter
    // TODO: add x/y scale parameters for non square textures and changing size
    // TODO: add shrinkage over time
    // TODO: add option to despawn or freeze once lifetime is reached instead of looping, would require f64 timeOffset
    // TODO: add tint color parameter
    // TODO: add acceleration over time

    vec3 basePos = getParticlePosition(p, d.time);
    float rotation = 0.0;
    vec2 basePointOffset = OFFSETS[localId.x];
    vec2 scale = vec2(0.1);
    vec2 pointOffset = vec2(
        basePointOffset.x * cos(rotation) + basePointOffset.y * sin(rotation),
        -basePointOffset.x * sin(rotation) + basePointOffset.y * cos(rotation)
    ) * scale;

    // spherical (broken) / cylindrical billboarding
    vec3 pos = basePos + vec3(-1.0, 1.0, 1.0) * d.cameraRight.xyz * pointOffset.x + d.cameraUp.xyz * pointOffset.y;
//    vec3 pos = basePos + vec3(-1.0, 1.0, 1.0) * d.cameraRight.xyz * pointOffset.x + vec3(0.0, pointOffset.y, 0.0);

    gl_MeshVerticesEXT[gl_LocalInvocationIndex].gl_Position = d.mvp * vec4(pos, 1.0);
    outUvs[gl_LocalInvocationIndex] = basePointOffset + 0.5;
    outTextures[gl_LocalInvocationIndex] = p.tex;
}
