#version 450

#include "common.glsl"

layout(location = 0) out vec2 outUv;
layout(location = 1) flat out uint outTex;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertDataPositions {
    vec4 positions[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertDataTex {
    uint tex[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertData {
    mat4 mvp;
    VertDataPositions positions;
    VertDataTex tex;
};

layout(std430, push_constant) uniform Data {
    VertData vert;
    Pointer frag;
} data;

void main() {
    VertData d = data.vert;
    gl_Position = d.mvp * vec4(d.positions.positions[gl_VertexIndex].xyz, 1.0);
    uint tex = d.tex.tex[gl_VertexIndex];
    float u = float(tex & 0xff);
    float v = float((tex >> 8) & 0xff);
    outUv = round(vec2(u, v)) / 8.0;
    outTex = tex >> 16;
}
