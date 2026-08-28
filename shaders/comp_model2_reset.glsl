#version 450

#include "common.glsl"

const uint COUNTS = 2;

layout(local_size_x = COUNTS) in;

struct PartCount {
    uint x;
    uint y;
    uint z;
};

layout(std430, buffer_reference, buffer_reference_align = 8) restrict buffer CompDataPartCountPointer {
    PartCount count;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer CompData {
    CompDataPartCountPointer counts[COUNTS];
};

layout(std430, push_constant) uniform Data {
    CompData comp;
} data;

void main() {
    data.comp.counts[gl_LocalInvocationIndex].count = PartCount(0, 1, 1);
}
