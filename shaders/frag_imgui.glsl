#version 450

#include "common.glsl"

layout(location = 0) in vec2 inUv;
layout(location = 1) flat in uint inColor;
layout(location = 2) flat in uint inTex;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform texture2D textures[];
layout(set = 1, binding = 0) uniform writeonly image2D textures_rw[];
layout(set = 2, binding = 0) uniform sampler samplers[];

layout(std430, push_constant) uniform Data {
    Pointer vert;
    Pointer frag;
} data;

vec4 readPacked(uint packed) {
    vec4 color = vec4(
        float(packed & 0xffu),
        float((packed >> 8) & 0xffu),
        float((packed >> 16) & 0xffu),
        float(packed >> 24)
    );
    return color / 255.0;
}

void main() {
    vec4 color = readPacked(inColor);
    if (color.a < 0.1) {
        discard;
    }
    if ((inTex & (1u << 30)) == 0) {
        vec4 tex = texture(sampler2D(textures[nonuniformEXT(inTex & 0xffffu)], samplers[nonuniformEXT((inTex & (1u << 31)) == 0 ? 0 : 1)]), inUv);
        if (tex.a < 0.1) {
            discard;
        }
        outColor = color * tex;
    } else {
        outColor = color;
    }
}
