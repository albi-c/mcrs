#version 450

#include "common.glsl"

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec2 outUv;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertDataPositions {
    vec2 positions[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertDataColors {
    vec4 colors[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertDataUvs {
    vec2 uvs[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertData {
    VertDataPositions positions;
    VertDataColors colors;
    VertDataUvs uvs;
    uint64_t _padding;
    mat2 matrix;
};

layout(std430, push_constant) uniform Data {
    VertData vert;
    Pointer frag;
} data;

void main() {
    gl_Position = vec4(data.vert.matrix *  data.vert.positions.positions[gl_VertexIndex], 0.0, 1.0);
    outColor = data.vert.colors.colors[gl_VertexIndex].xyz;
    outUv = data.vert.uvs.uvs[gl_VertexIndex];
}
