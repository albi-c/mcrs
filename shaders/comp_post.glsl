#version 450

#include "common.glsl"

layout(local_size_x = 128) in;

layout(set = 0, binding = 0) uniform texture2D textures[];
layout(set = 1, binding = 0) uniform writeonly image2D textures_rw[];
layout(set = 2, binding = 0) uniform sampler samplers[];

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer CompData {
    uint texture;
};

layout(std430, push_constant) uniform Data {
    Pointer comp;
} data;

void main() {
}
