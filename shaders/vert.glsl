#version 450

#include "common.glsl"

layout(location = 0) out vec2 outUv;
layout(location = 1) flat out uint outTex;

struct Vertex {
    float x;
    float y;
    float z;
    uint tex;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertDataVertices {
    Vertex data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertData {
    mat4 mvp;
    VertDataVertices vertices;
    uint64_t _padding;
};

layout(std430, push_constant) uniform Data {
    VertData vert;
    Pointer frag;
} data;

void main() {
    VertData d = data.vert;
    Vertex vertex = d.vertices.data[gl_VertexIndex];
    gl_Position = d.mvp * vec4(vertex.x, vertex.y, vertex.z, 1.0);
    uint tex = vertex.tex;
    float u = float(tex & 0xff);
    float v = float((tex >> 8) & 0xff);
    outUv = round(vec2(u, v)) / 8.0;
    outTex = tex >> 16;
}
