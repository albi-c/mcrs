#version 450

#include "common.glsl"

layout(location = 0) in vec2 inUv;
layout(location = 1) flat in uint inTex;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform texture2D textures[];
layout(set = 1, binding = 0) uniform writeonly image2D textures_rw[];
layout(set = 2, binding = 0) uniform sampler samplers[];

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer FragData {
    vec4 tint;
};

layout(std430, push_constant) uniform Data {
    Pointer vert;
    FragData frag;
} data;

void main() {
    outColor = vec4(texture(sampler2D(textures[nonuniformEXT(inTex)], samplers[0]), inUv).rgb * data.frag.tint.rgb, 1.0);
}
