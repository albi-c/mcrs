#version 450

#include "common.glsl"

layout(location = 0) out vec2 outUv;

vec2 positions[] = vec2[](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0)
);
vec2 uvs[] = vec2[](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0)
);

layout(std430, push_constant) uniform Data {
    Pointer vert;
    Pointer frag;
} data;

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0) * 4.0 - 1.0;
    outUv = uvs[gl_VertexIndex] * 2.0;
}
