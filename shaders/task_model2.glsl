#version 450

#include "common.glsl"
#include "common_model2.glsl"

layout(local_size_x = MODEL_PART_SIZE, local_size_y = 1, local_size_z = 1) in;

taskPayloadSharedEXT Task OUT;

// --- 64 bytes
struct Model {
    MeshDataModelMeshlets meshlets;
    MeshDataModelMeshletInfos meshletInfos;
    MeshDataTransform transform;
    uint meshletCount;
    // bit 0: disable rendering, bit 1: enable frustum culling
    uint flags;
    AABB aabb;
    uint _padding[2];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataModel {
    Model model;
};

struct ModelPart {
    MeshDataModel model;
    uint meshletOffset;
    uint meshletCount;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataModelParts {
    ModelPart data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshData {
    mat4 viewProj;
    Frustum frustum;
    MeshDataModelParts modelParts;
    Pointer materials;
};

layout(std430, push_constant) uniform Data {
    MeshData mesh;
    Pointer frag;
} data;

shared uint meshletOutputIndex;

void main() {
    bool alive = true;

    MeshData d = data.mesh;
    ModelPart mp = d.modelParts.data[gl_WorkGroupID.x];
    if (gl_LocalInvocationIndex >= mp.meshletCount) {
        return;
    }
    Model m = mp.model.model;

    if ((m.flags & 0x01) != 0) {
        return;
    }
    mat4 modelTransform = m.transform.transform;
    if ((m.flags & 0x02) != 0) {
        bool inFrustum;
        if (subgroupElect()) {
            inFrustum = checkFrustum(d.frustum, m.aabb, modelTransform);
        }
        if (!subgroupBroadcastFirst(inFrustum)) {
            return;
        }
    }

    MeshletInfo mi = m.meshletInfos.data[mp.meshletOffset + gl_LocalInvocationIndex];

    mat4 meshletTransform;
    if ((mi.flags & 0x01) != 0) {
        return;
    } else {
        meshletTransform = modelTransform * mi.transform.transform;
        if ((mi.flags & 0x02) == 0) {
            // TODO: cone culling
            if (!checkFrustum(d.frustum, mi.aabb, meshletTransform)) {
                return;
            }
        }
    }

    meshletOutputIndex = 0;
    memoryBarrierShared();

    uvec4 ballot = subgroupBallot(true);
    uint subgroupIndex = subgroupBallotExclusiveBitCount(ballot);
    uint subgroupOffset;
    if (subgroupElect()) {
        uint subgroupCount = subgroupBallotBitCount(ballot);
        subgroupOffset = atomicAdd(meshletOutputIndex, subgroupCount);
    }
    uint index = subgroupBroadcastFirst(subgroupOffset) + subgroupIndex;

    memoryBarrierShared();

    OUT.meshletOffsets[index] = uint8_t(gl_LocalInvocationIndex);

    if (subgroupElect()) {
        OUT.model = modelTransform;
        OUT.meshlets = m.meshlets;
        OUT.meshletInfos = m.meshletInfos;
        OUT.meshletBase = mp.meshletOffset;
        EmitMeshTasksEXT(meshletOutputIndex, 1, 1);
    }
}
