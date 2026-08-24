#version 450

#include "common.glsl"
#include "common_model.glsl"

// TODO: make workgroup size a multiple of 64 and replace shared variables with arrays (indexed with Y index)
// e.g. local_size_x = 6, local_size_y = 32
// shared bool isInFrustum[32];
layout(local_size_x = 6, local_size_y = 1, local_size_z = 1) in;

taskPayloadSharedEXT Task OUT;

struct Model {
    MeshDataModelMeshlets meshlets;
    MeshDataModelMeshletInfos meshletInfos;
    uint meshletCount;
    // bit 0: disable rendering, bit 1: disable frustum culling
    uint flags;
    AABB aabb;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataModels {
    Model data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataMaterials {
    // Material described in vert.glsl
    uvec4 data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataModelTransforms {
    mat4 data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataModelIndices {
    uint data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshData {
    mat4 viewProj;
    Frustum frustum;
    MeshDataModels models;
    // TODO: use array of pointers to make dynamically adding and removing models easier
    MeshDataModelIndices modelIndices;
    MeshDataModelTransforms modelTransforms;
    Pointer meshletTransforms;
    Pointer materials;
    uint modelIndexOffset;
    uint modelTransformOffset;
    uint meshletTransformOffset;
    bool useModelIndexArray;
};

layout(std430, push_constant) uniform Data {
    MeshData mesh;
    Pointer frag;
} data;

void main() {
    MeshData d = data.mesh;
    uint modelIndex;
    if (subgroupElect()) {
        if (d.useModelIndexArray) {
            modelIndex = d.modelIndices.data[gl_WorkGroupID.x] + d.modelIndexOffset;
        } else {
            modelIndex = gl_WorkGroupID.x + d.modelIndexOffset;
        }
    }
    Model m = d.models.data[subgroupBroadcastFirst(modelIndex)];

    if ((m.flags & 0x01) != 0) {
        EmitMeshTasksEXT(0, 0, 0);
        return;
    }

    if (gl_LocalInvocationIndex == 0) {
        isInFrustum = true;
        modelTransform = d.modelTransforms.data[gl_WorkGroupID.x + d.modelTransformOffset];
    }

    memoryBarrierShared();

    if ((m.flags & 0x02) == 0) {
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
