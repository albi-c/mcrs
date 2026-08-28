#version 450

#include "common.glsl"
#include "common_model2_ct.glsl"

const uint SIZE_X = 64;

layout(local_size_x = SIZE_X) in;

// --- 16 bytes
struct ModelInstance {
    MeshDataModel model;
    MeshDataTransform transform;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer CompDataModelInstances {
    ModelInstanceFiltered data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) restrict buffer CompDataModelParts {
    ModelPart data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) restrict buffer CompDataPartCount {
    uint x;
    uint _y;
    uint _z;
    uint _padding;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer CompData {
    Frustum frustum;
    CompDataModelInstances modelInstances;
    CompDataModelParts modelParts;
    CompDataPartCount partCount;
    uint maxModelPartCount;
    uint _padding;
    vec4 cameraPosAndViewport;
};

layout(std430, push_constant) uniform Data {
    CompData comp;
} data;

void main() {
    CompData d = data.comp;
    ModelInstanceFiltered mif = d.modelInstances.data[gl_WorkGroupID.x];
    LOD lod = mif.lods.data[mif.lod];
    mat4 transform = mif.transform.transform;
    uint flags = mif.flags;

    for (uint baseChunk = 0; baseChunk < lod.chunkCount; baseChunk += SIZE_X) {
        subgroupBarrier();

        uint chunkIndex = baseChunk + gl_LocalInvocationIndex;
        if (chunkIndex >= lod.chunkCount) {
            return;
        }
        LODChunk chunk = lod.chunks.data[chunkIndex];

        bool fullyInside = false;
        // TODO: incorrect output for fullyInside - always returns true
        if ((flags & 0x01) == 0) {
            if (!checkFrustumAndFullyInside(d.frustum, chunk.aabb, transform, fullyInside)) {
                continue;
            }
        }

        ModelPart part;
        part.meshlets = lod.meshlets;
        part.meshletInfos = lod.meshletInfos;
        part.transform = mif.transform;
        part.meshletOffset = chunk.meshletOffset;
        part.meshletCount = chunk.meshletCount;
        part.flags = uint16_t(fullyInside ? 0x1 : 0x0);

        uvec4 ballot = subgroupBallot(true);
        uint subgroupIndex = subgroupBallotExclusiveBitCount(ballot);
        uint subgroupOffset;
        if (subgroupElect()) {
            uint subgroupCount = subgroupBallotBitCount(ballot);
            subgroupOffset = atomicAdd(d.partCount.x, subgroupCount);
        }
        uint index = subgroupBroadcastFirst(subgroupOffset) + subgroupIndex;

        if (index > d.maxModelPartCount) {
            continue;
        }

        d.modelParts.data[index] = part;
    }
}
