#version 450

#include "common.glsl"

layout(location = 0) in vec2 inUv;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform texture2D textures[];
layout(set = 1, binding = 0) uniform writeonly image2D textures_rw[];
layout(set = 2, binding = 0) uniform sampler samplers[];

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer FragDataTexture {
    uint baseTexture;
    uint overlayTexture;
};

layout(std430, push_constant) uniform Data {
    Pointer vert;
    FragDataTexture frag;
} data;

void main() {
    vec4 color = texture(sampler2D(textures[nonuniformEXT(data.frag.baseTexture)], samplers[1]), inUv);
    vec4 overlay = texture(sampler2D(textures[nonuniformEXT(data.frag.overlayTexture)], samplers[0]), inUv);
    outColor = mix(vec4(color.rgb, 1.0), vec4(overlay.rgb, 1.0), overlay.a);
}
