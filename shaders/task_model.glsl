#version 450

#include "common.glsl"
#include "common_model.glsl"

// TODO: make workgroup size a multiple of 64 and replace shared variables with arrays (indexed with Y index)
// e.g. local_size_x = 6, local_size_y = 32
// shared bool isInFrustum[32];
layout(local_size_x = 6, local_size_y = 1, local_size_z = 1) in;

taskPayloadSharedEXT Task OUT;

// --- 64 bytes
struct Model {
    MeshDataModelMeshlets meshlets;
    MeshDataModelMeshletInfos meshletInfos;
    MeshDataTransform transform;
    uint meshletCount;
    // bit 0: disable rendering, bit 1: disable frustum culling
    uint flags;
    AABB aabb;
    uint _padding[2];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataModel {
    Model model;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataModelPointers {
    MeshDataModel data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshData {
    mat4 viewProj;
    Frustum frustum;
    MeshDataModelPointers modelPointers;
    Pointer materials;
};

layout(std430, push_constant) uniform Data {
    MeshData mesh;
    Pointer frag;
} data;

void main() {
    MeshData d = data.mesh;
    Model m = d.modelPointers.data[gl_WorkGroupID.x].model;

    if ((m.flags & 0x01) != 0) {
        EmitMeshTasksEXT(0, 0, 0);
        return;
    }

    if (gl_LocalInvocationIndex == 0) {
        isInFrustum = true;
        modelTransform = m.transform.transform;
    }

    memoryBarrierShared();

    if ((m.flags & 0x02) != 0) {
        checkFrustum(d.frustum, m.aabb);

        memoryBarrierShared();
    }

    if (!isInFrustum) {
        EmitMeshTasksEXT(0, 0, 0);
        return;
    }

    if (gl_LocalInvocationIndex == 0) {
        OUT.model = modelTransform;
        OUT.meshlets = m.meshlets;
        OUT.meshletInfos = m.meshletInfos;
    }

    EmitMeshTasksEXT(m.meshletCount, 1, 1);
}
