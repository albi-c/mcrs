#version 450

#include "common.glsl"

layout(location = 0) in vec2 inUv;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform texture2D textures[];
layout(set = 1, binding = 0) uniform writeonly image2D textures_rw[];
layout(set = 2, binding = 0) uniform sampler samplers[];

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer FragDataTexture {
    uint texture;
};

layout(std430, push_constant) uniform Data {
    Pointer vert;
    FragDataTexture frag;
} data;

void main() {
    vec4 color = texture(sampler2D(textures[nonuniformEXT(data.frag.texture)], samplers[0]), inUv);
    outColor = vec4(color.rgb, 1.0);
}
