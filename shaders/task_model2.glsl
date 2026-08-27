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
    MeshDataTransform transform;
    AABB aabb;
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
    vec4 cameraPos;
};

layout(std430, push_constant) uniform Data {
    MeshData mesh;
    Pointer frag;
} data;

shared uint g_meshletOutputIndex;
shared bool g_inFrustum;
shared bool g_skipPerMeshletFrustum;

void main() {
    MeshData d = data.mesh;
    ModelPart mp = d.modelParts.data[gl_WorkGroupID.x];
    if (gl_LocalInvocationIndex >= mp.meshletCount) {
        return;
    }
    Model m = mp.model.model;

    if ((m.flags & 0x01) != 0) {
        return;
    }
    mat4 modelTransform = mp.transform.transform;
    if ((m.flags & 0x02) != 0) {
        if (gl_LocalInvocationIndex == 0) {
            bool completelyInFrustum;
            bool inFrustum = checkFrustumAndFullyInside(d.frustum, mp.aabb, modelTransform, completelyInFrustum);
            g_inFrustum = inFrustum;
            g_skipPerMeshletFrustum = completelyInFrustum;
        }
        memoryBarrierShared();
        if (!g_inFrustum) {
            return;
        }
    }

    MeshletInfo mi = m.meshletInfos.data[mp.meshletOffset + gl_LocalInvocationIndex];

    bool alive = true;
    if ((mi.flags & 0x01) != 0) {
        alive = false;
    } else if (!g_skipPerMeshletFrustum && (mi.flags & 0x02) == 0) {
        if (!checkFrustum(d.frustum, mi.aabb, modelTransform * mi.transform.transform)) {
            alive = false;
        }
    }

    g_meshletOutputIndex = 0;
    memoryBarrierShared();

    uvec4 ballot = subgroupBallot(alive);
    uint subgroupIndex = subgroupBallotExclusiveBitCount(ballot);
    uint subgroupOffset;
    if (subgroupElect()) {
        uint subgroupCount = subgroupBallotBitCount(ballot);
        subgroupOffset = atomicAdd(g_meshletOutputIndex, subgroupCount);
    }
    uint index = subgroupBroadcastFirst(subgroupOffset) + subgroupIndex;

    memoryBarrierShared();

    if (alive) {
        OUT.meshletOffsets[index] = uint8_t(gl_LocalInvocationIndex);
    }

    if (gl_LocalInvocationIndex == 0) {
        OUT.model = modelTransform;
        OUT.meshlets = m.meshlets;
        OUT.meshletInfos = m.meshletInfos;
        OUT.meshletBase = mp.meshletOffset;
        EmitMeshTasksEXT(g_meshletOutputIndex, 1, 1);
    }
}
