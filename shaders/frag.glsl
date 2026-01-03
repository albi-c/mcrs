#version 450

#include "common.glsl"

layout (location = 0) in vec3 inColor;

layout (location = 0) out vec4 outColor;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer FragData {
    vec4 colorAdd;
    vec4 colorMul;
};

layout(std430, push_constant) uniform Data {
    Pointer vert;
    FragData frag;
} data;

void main() {
    outColor = (vec4(inColor, 1.0) + data.frag.colorAdd) * data.frag.colorMul;
}
