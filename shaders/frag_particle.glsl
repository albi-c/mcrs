#version 450

#include "common.glsl"

layout(location = 0) in vec2 inUv;
layout(location = 1) flat in uint16_t inTexture;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform texture2D textures[];
layout(set = 1, binding = 0) uniform writeonly image2D textures_rw[];
layout(set = 2, binding = 0) uniform sampler samplers[];

layout(std430, push_constant) uniform Data {
    Pointer vert;
    Pointer frag;
} data;

void main() {
    vec4 texColor = texture(sampler2D(textures[nonuniformEXT(inTexture)], samplers[0]), inUv);
    if (texColor.a < 0.1) {
        discard;
    }
    outColor = vec4(texColor.rgb, 1.0);
}
