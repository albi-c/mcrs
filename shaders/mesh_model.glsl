#version 450

#include "common.glsl"
#include "common_model.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 64, max_primitives = 126) out;

layout(location = 0) out vec2 outUvs[];
layout(location = 1) out vec3 outNormals[];
layout(location = 2) flat out uvec4 outMaterials[];
layout(location = 3) out vec3 outWorldPositions[];
layout(location = 4) flat out uint outDebugColors[];

taskPayloadSharedEXT Task IN;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataMaterials {
    // Material described in vert.glsl
    uvec4 data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshData {
    mat4 viewProj;
    Frustum frustum;
    Pointer models;
    MeshDataMaterials materials;
};

layout(std430, push_constant) uniform Data {
    MeshData mesh;
    Pointer frag;
} data;

vec4 getVertexPosition(in Vertex vertex) {
    return vec4(float(vertex.pos[0]), float(vertex.pos[1]), float(vertex.pos[2]), 1.0);
}
vec2 getVertexUv(in Vertex vertex) {
    return vec2(vertex.uv[0], vertex.uv[1]);
}
vec3 getVertexNormal(in Vertex vertex) {
    uint n = vertex.normal;
    float nx = float(n & 0x7ffu) / 1024.0 - 1.0;
    float ny = float((n >> 11) & 0x3ffu) / 512.0 - 1.0;
    float nz = float(n >> 21) / 1024.0 - 1.0;
    // TODO: multiply by inverse of model matrix if doing non uniform transforms
    vec3 normal = normalize(vec3(nx, ny, nz));
    return normal;
}

uvec3 readIndices(uint packed) {
    return uvec3(packed & 0xff, (packed >> 8) & 0xff, (packed >> 16) & 0xff);
}

uint murmurHash11(uint src) {
    const uint M = 0x5bd1e995u;
    uint h = 1190494759u;
    src *= M; src ^= src>>24u; src *= M;
    h *= M; h ^= src;
    h ^= h>>13u; h *= M; h ^= h>>15u;
    return h;
}

void main() {
    // TODO: cone culling
    // TODO: calculate tangent and bitangent for normal mapping
    // TODO: move culling into task shader

    MeshData d = data.mesh;
    MeshletInfo mi = IN.meshletInfos.data[gl_WorkGroupID.x];
    if ((mi.flags & 0x01) != 0) {
        if (gl_LocalInvocationIndex == 0) {
            SetMeshOutputsEXT(0, 0);
        }
        return;
    }

    if (gl_LocalInvocationIndex == 0) {
        g_isInFrustum = true;
        g_modelTransform = IN.model * mi.transform.transform;
    }

    memoryBarrierShared();

    if ((mi.flags & 0x02) == 0) {
        checkFrustum(d.frustum, mi.aabb, g_isInFrustum, g_modelTransform);

        memoryBarrierShared();
    }

    if (!g_isInFrustum) {
        if (gl_LocalInvocationIndex == 0) {
            SetMeshOutputsEXT(0, 0);
        }
        return;
    }

    if (gl_LocalInvocationIndex == 0) {
        SetMeshOutputsEXT(mi.vertexCount, mi.triangleCount);
    }

    if (gl_LocalInvocationIndex < mi.vertexCount) {
        Vertex v = IN.meshlets.data[gl_WorkGroupID.x].vertices[gl_LocalInvocationIndex];

        vec4 worldPos = g_modelTransform * getVertexPosition(v);
        gl_MeshVerticesEXT[gl_LocalInvocationIndex].gl_Position = d.viewProj * worldPos;
        outUvs[gl_LocalInvocationIndex] = getVertexUv(v);
        outNormals[gl_LocalInvocationIndex] = getVertexNormal(v);
        outMaterials[gl_LocalInvocationIndex] = d.materials.data[v.mat];
        outWorldPositions[gl_LocalInvocationIndex] = worldPos.xyz;
        outDebugColors[gl_LocalInvocationIndex] = murmurHash11(gl_WorkGroupID.x);
    }

    if (gl_LocalInvocationIndex < mi.triangleCount) {
        gl_PrimitiveTriangleIndicesEXT[gl_LocalInvocationIndex] = readIndices(IN.meshlets.data[gl_WorkGroupID.x].triangles[gl_LocalInvocationIndex]);
        if (gl_LocalInvocationIndex + 64 < mi.triangleCount) {
            gl_PrimitiveTriangleIndicesEXT[gl_LocalInvocationIndex + 64] = readIndices(IN.meshlets.data[gl_WorkGroupID.x].triangles[gl_LocalInvocationIndex + 64]);
        }
    }
}
