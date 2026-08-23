#version 460

#include "common.glsl"

layout(location = 0) in vec2 inUv;
layout(location = 1) in vec3 inNormal;
layout(location = 2) flat in uvec4 inMat;
layout(location = 3) in vec3 inWorldPos;

layout(location = 0) out vec4 outColor;

layout(std430, push_constant) uniform Data {
    Pointer vert;
    Pointer frag;
} data;

void main() {
    outColor = vec4(inUv, 0.0, 1.0);
}
