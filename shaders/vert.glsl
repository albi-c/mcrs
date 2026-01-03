#version 450

#include "common.glsl"

layout(location = 0) out vec3 outColor;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertDataPositions {
    vec2 positions[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertDataColors {
    vec4 colors[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertData {
    VertDataPositions positions;
    VertDataColors colors;
    mat2 matrix;
};

layout(std430, push_constant) uniform Data {
    VertData vert;
    Pointer frag;
} data;

void main() {
    gl_Position = vec4(data.vert.matrix *  data.vert.positions.positions[gl_VertexIndex], 0.0, 1.0);
    outColor = data.vert.colors.colors[gl_VertexIndex].xyz;
}
