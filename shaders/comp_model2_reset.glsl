#version 450

#include "common.glsl"

layout(local_size_x = 1) in;

layout(std430, buffer_reference, buffer_reference_align = 8) restrict buffer CompDataPartCount {
    uint x;
    uint y;
    uint z;
};

layout(std430, push_constant) uniform Data {
    CompDataPartCount partCount;
} data;

void main() {
    data.partCount.x = 0;
    data.partCount.y = 1;
    data.partCount.z = 1;
}
