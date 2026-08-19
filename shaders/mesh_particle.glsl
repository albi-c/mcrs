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
layout(location = 2) out vec3 outColors[];

struct Particle {
    double timeOffset;
    float16_t origin[3];
    float16_t velocity[3];
    float16_t acceleration[3];
    float16_t accelerationChange[3];
    float16_t scale[2];
    float16_t scaleChange;
    float16_t rotation;
    float16_t rotation_change;
    float16_t spiralRadius;
    float16_t spiralOffset;
    float16_t spiralSpeed;
    float16_t spiralVelocityInfluence;
    float16_t lifetime;
    uint16_t tex;
    uint8_t color[3];
    // bit 0: enable cylindrical billboarding, bit 1: stop on end, bit 2: hide on end, bit 3: no render
    uint8_t flags;
    uint16_t _padding[3];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataParticles {
    Particle data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshData {
    mat4 mvp;
    vec4 cameraRight;
    vec4 cameraUp;
    MeshDataParticles particles;
    double time;
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

vec2 readVec16_2(in float16_t vec[2]) {
    return vec2(float(vec[0]), float(vec[1]));
}

vec3 readVec16(in float16_t vec[3]) {
    return vec3(float(vec[0]), float(vec[1]), float(vec[2]));
}

vec3 getParticlePosition(in Particle p, float t) {
    vec3 pOrigin = readVec16(p.origin);
    vec3 pVelocity = readVec16(p.velocity);
    vec3 pAcceleration = readVec16(p.acceleration) + t * readVec16(p.accelerationChange);

    vec3 velocity = pVelocity + t * pAcceleration;
    vec3 position = pOrigin + t * pVelocity + t * t * 0.5 * pAcceleration;

    if (length(velocity) < 0.00001) {
        return position;
    }

    vec3 axis = normalize(velocity);

    vec3 reference = abs(axis.x) > 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);

    vec3 u = normalize(cross(axis, reference));
    vec3 v = cross(axis, u);

    float speed0 = length(pVelocity);
    float speed1 = length(velocity);
    float angle = float(p.spiralOffset) + t * float(p.spiralSpeed) + t * 0.5 * float(p.spiralVelocityInfluence) * (speed0 + speed1);
    vec2 cs = vec2(cos(angle), sin(angle));

    return position + p.spiralRadius * cs.x * u + p.spiralRadius * cs.y * v;
}

shared bool willAnyRender;

void main() {
    MeshData d = data.mesh;
    uvec2 localId = gl_LocalInvocationID.xy;

    Particle p = d.particles.data[PARTICLE_COUNT * gl_WorkGroupID.x + localId.y];
    uint flags = uint(p.flags);

    double timeWithOffset = d.time + double(p.timeOffset);
    double lifetime = double(p.lifetime);

    willAnyRender = false;

    memoryBarrierShared();

    bool noRender = (flags & 0x08u) != 0 || ((flags & 0x04u) != 0 && timeWithOffset >= lifetime);
    if (!noRender) {
        willAnyRender = true;
    }

    memoryBarrierShared();

    if (!willAnyRender) {
        SetMeshOutputsEXT(0, 0);
        return;
    }

    if (gl_LocalInvocationIndex == 0) {
        SetMeshOutputsEXT(VERTEX_COUNT, PRIMITIVE_COUNT);
    }
    if (localId.x == 0) {
        gl_PrimitiveTriangleIndicesEXT[2 * localId.y + 0] = uvec3(0, 1, 2) + gl_LocalInvocationIndex;
        gl_PrimitiveTriangleIndicesEXT[2 * localId.y + 1] = uvec3(0, 2, 3) + gl_LocalInvocationIndex;
    }

    if (noRender) {
        gl_MeshVerticesEXT[gl_LocalInvocationIndex].gl_Position = vec4(-1.0, -1.0, -1.0, 1.0);
        outTextures[gl_LocalInvocationIndex] = uint16_t(0);
        return;
    }

    float t = float((flags & 0x02u) != 0 ? clamp(timeWithOffset, 0.0, lifetime) : mod(timeWithOffset, lifetime));
    vec3 basePos = getParticlePosition(p, t);
    float rotation = float(p.rotation) + t * float(p.rotation_change);
    vec2 basePointOffset = OFFSETS[localId.x];
    vec2 scale = readVec16_2(p.scale) * max(1.0 + t * float(p.scaleChange), 0.0);
    vec2 pointOffset = vec2(
        basePointOffset.x * cos(rotation) + basePointOffset.y * sin(rotation),
        -basePointOffset.x * sin(rotation) + basePointOffset.y * cos(rotation)
    ) * scale;

    vec3 pos = basePos + d.cameraRight.xyz * pointOffset.x + ((flags & 0x01u) != 0 ? vec3(0.0, 1.0, 0.0) : d.cameraUp.xyz) * pointOffset.y;

    gl_MeshVerticesEXT[gl_LocalInvocationIndex].gl_Position = d.mvp * vec4(pos, 1.0);
    outUvs[gl_LocalInvocationIndex] = basePointOffset + 0.5;
    outTextures[gl_LocalInvocationIndex] = p.tex;
    outColors[gl_LocalInvocationIndex] = vec3(float(p.color[0]), float(p.color[1]), float(p.color[2])) / 255.0;
}
