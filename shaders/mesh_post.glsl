#version 450

#include "common.glsl"
#extension GL_EXT_mesh_shader : require

layout(local_size_x = 3, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 3, max_primitives = 1) out;

layout(location = 0) out vec2 outUvs[];

vec2 positions[] = vec2[](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0)
);
vec2 uvs[] = vec2[](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0)
);

layout(std430, push_constant) uniform Data {
    Pointer mesh;
    Pointer frag;
} data;

void main() {
    uint id = gl_LocalInvocationIndex;
    if (id == 0) {
        SetMeshOutputsEXT(3, 1);
        gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
    }

    gl_MeshVerticesEXT[id].gl_Position = vec4(positions[id] * 4.0 - 1.0, 0.0, 1.0);
    outUvs[id] = uvs[id] * 2.0;
}
