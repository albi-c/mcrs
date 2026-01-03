#version 450

#include "common.glsl"

layout(location = 0) in vec3 inColor;
layout(location = 1) in vec2 inUv;

layout(location = 0) out vec4 outColor;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer FragDataColors {
    vec4 colorAdd;
    vec4 colorMul;
};

layout(set = 0, binding = 0) uniform texture2D textures[];
layout(set = 1, binding = 0) uniform writeonly image2D textures_rw[];
layout(set = 2, binding = 0) uniform sampler samplers[];

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer FragData {
    FragDataColors colors;
};

layout(std430, push_constant) uniform Data {
    Pointer vert;
    FragData frag;
} data;

void main() {
    outColor = vec4(texture(sampler2D(textures[0], samplers[0]), inUv).rgb, 1.0);
//    outColor = (vec4(inColor, 1.0) + data.frag.colors.colorAdd) * data.frag.colors.colorMul;
}
