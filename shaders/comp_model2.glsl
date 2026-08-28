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
    ModelInstance data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) restrict buffer CompDataModelParts {
    ModelPart data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) restrict buffer CompDataPartCount {
    uint x;
    uint _y;
    uint _z;
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

uint selectLOD(in Model model, in mat4 modelTransform, vec4 cameraPosAndViewport) {
    if ((model.flags & 0x04) == 0) {
        return 0;
    } else {
        vec3 cameraPos = cameraPosAndViewport.xyz;
        float k = cameraPosAndViewport.w;

        vec3 center;
        vec3 extent;
        transformAABB(model.aabb, modelTransform, center, extent);

        float dist = length(cameraPos - center);
        float radius = length(extent);

        return uint(clamp((dist - radius) / (radius * 0.5), 0.0, float(model.lodCount - 1)));
    }
}

shared LOD g_lod;
shared mat4 g_transform;
shared MeshDataTransform g_transformPtr;
shared bool g_discardModel;
shared bool g_skipLodPartAABB;
shared uint g_outputIndex;

void perWorkgroup(in CompData d) {
    ModelInstance mi = d.modelInstances.data[gl_WorkGroupID.x];
    Model m = mi.model.model;
    if ((m.flags & 0x01) != 0) {
        g_discardModel = true;
        return;
    }
    mat4 transform = mi.transform.transform;
    if ((m.flags & 0x02) != 0) {
        bool fullyInside;
        if (!checkFrustumAndFullyInside(d.frustum, m.aabb, transform, fullyInside)) {
            g_discardModel = true;
            return;
        }
        g_skipLodPartAABB = fullyInside;
    } else {
        g_skipLodPartAABB = true;
    }
    uint lodIndex = selectLOD(m, transform, d.cameraPosAndViewport);
    g_lod = m.lods.data[lodIndex];
    g_transform = transform;
    g_transformPtr = mi.transform;
    g_discardModel = false;
}

void processChunk(CompData d, in Frustum f, in LOD lod, in LODChunk chunk) {
    bool fullyInside;
    // TODO: incorrect output for fullyInside - always returns true
    if (false && g_skipLodPartAABB) {
        fullyInside = false;
    } else {
        if (!checkFrustumAndFullyInside(f, chunk.aabb, g_transform, fullyInside)) {
            return;
        }
    }

    ModelPart part;
    part.meshlets = lod.meshlets;
    part.meshletInfos = lod.meshletInfos;
    part.transform = g_transformPtr;
    part.meshletOffset = chunk.meshletOffset;
    part.meshletCount = chunk.meshletCount;
    part.flags = uint16_t(fullyInside ? 0x1 : 0x0);

    g_outputIndex = 0;
    memoryBarrierShared();

    uvec4 ballot = subgroupBallot(true);
    uint subgroupIndex = subgroupBallotExclusiveBitCount(ballot);
    uint subgroupOffset;
    if (subgroupElect()) {
        uint subgroupCount = subgroupBallotBitCount(ballot);
        subgroupOffset = atomicAdd(d.partCount.x, subgroupCount);
    }
    uint index = subgroupBroadcastFirst(subgroupOffset) + subgroupIndex;

    if (index > d.maxModelPartCount) {
        return;
    }

    d.modelParts.data[index] = part;
}

void main() {
    CompData d = data.comp;

    if (gl_LocalInvocationIndex == 0) {
        perWorkgroup(d);
    }
    memoryBarrierShared();

    if (g_discardModel) {
        return;
    }

    LOD lod = g_lod;
    for (uint baseChunk = 0; baseChunk < lod.chunkCount; baseChunk += SIZE_X) {
        uint chunkIndex = baseChunk + gl_LocalInvocationIndex;
        if (chunkIndex >= lod.chunkCount) {
            return;
        }
        LODChunk chunk = lod.chunks.data[chunkIndex];
        processChunk(d, d.frustum, lod, chunk);
    }
}
