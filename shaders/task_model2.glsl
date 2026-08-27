#version 450

#include "common.glsl"
#include "common_model2_tm.glsl"
#include "common_model2_ct.glsl"

layout(local_size_x = MODEL_PART_SIZE, local_size_y = 1, local_size_z = 1) in;

taskPayloadSharedEXT Task OUT;

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

void main() {
    MeshData d = data.mesh;
    ModelPart mp = d.modelParts.data[gl_WorkGroupID.x];
    if (gl_LocalInvocationIndex >= mp.meshletCount) {
        return;
    }
    MeshletInfo mi = mp.meshletInfos.data[mp.meshletOffset + gl_LocalInvocationIndex];
    mat4 modelTransform = mp.transform.transform;

    // TODO: cone culling, DO NOT SKIP if mp.flags & 0x01 is set, that only applies to frustum culling
    bool alive = true;
    if ((mi.flags & 0x01) != 0) {
        alive = false;
    } else if ((mp.flags & 0x01) == 0 && (mi.flags & 0x02) == 0) {
        if (!checkFrustum(d.frustum, mi.aabb, modelTransform)) {
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
        OUT.meshlets = mp.meshlets;
        OUT.meshletInfos = mp.meshletInfos;
        OUT.meshletBase = mp.meshletOffset;
        EmitMeshTasksEXT(g_meshletOutputIndex, 1, 1);
    }
}
