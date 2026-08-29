#version 450

#include "common.glsl"

layout(location = 0) out vec2 outUv;
layout(location = 1) flat out uint outColor;
layout(location = 2) flat out uint outTex;

struct Vertex {
    vec4 posUv;
    uint color;
    uint tex;
    uint _padding[2];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertDataVertices {
    Vertex data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertData {
    mat4 proj;
    VertDataVertices vertices;
};

layout(std430, push_constant) uniform Data {
    VertData vert;
    Pointer frag;
} data;

void main() {
    VertData d = data.vert;

    Vertex v = d.vertices.data[gl_VertexIndex];
    gl_Position = d.proj * vec4(v.posUv.xy, 0.0, 1.0);
    outUv = v.posUv.zw;
    outColor = v.color;
    outTex = v.tex;
}
