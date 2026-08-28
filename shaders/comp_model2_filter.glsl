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

layout(std430, buffer_reference, buffer_reference_align = 8) restrict buffer CompDataModelInstancesFiltered {
    ModelInstanceFiltered data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) restrict buffer CompDataInstanceCount {
    uint x;
    uint _y;
    uint _z;
    uint _padding;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer CompData {
    Frustum frustum;
    CompDataModelInstances modelInstances;
    CompDataModelInstancesFiltered modelInstanesFiltered;
    CompDataInstanceCount filteredInstanceCount;
    uint instanceCount;
    uint _padding;
    vec4 cameraPosAndViewport;
};

layout(std430, push_constant) uniform Data {
    CompData comp;
} data;

void main() {
    CompData d = data.comp;

    if (gl_GlobalInvocationID.x >= d.instanceCount) {
        return;
    }
    ModelInstance mi = d.modelInstances.data[gl_GlobalInvocationID.x];
    Model m = mi.model.model;
    if ((m.flags & 0x01) != 0) {
        return;
    }
    mat4 transform = mi.transform.transform;
    bool fullyInside = true;
    if ((m.flags & 0x02) != 0) {
        if (!checkFrustumAndFullyInside(d.frustum, m.aabb, transform, fullyInside)) {
            return;
        }
    }

    uint lodIndex = 0;
    if ((m.flags & 0x04) != 0) {
        vec3 cameraPos = d.cameraPosAndViewport.xyz;

        vec3 center;
        vec3 extent;
        transformAABB(m.aabb, transform, center, extent);

        float dist = length(cameraPos - center);
        float radius = length(extent);

        lodIndex = uint(clamp((dist - radius) / (radius * 0.125), 0.0, float(m.lodCount - 1)));
    }

    ModelInstanceFiltered mif;
    mif.lods = m.lods;
    mif.transform = mi.transform;
    mif.flags = uint(fullyInside ? 0x1 : 0x0);
    mif.lod = lodIndex;

    uvec4 ballot = subgroupBallot(true);
    uint subgroupIndex = subgroupBallotExclusiveBitCount(ballot);
    uint subgroupOffset;
    if (subgroupElect()) {
        uint subgroupCount = subgroupBallotBitCount(ballot);
        subgroupOffset = atomicAdd(d.filteredInstanceCount.x, subgroupCount);
    }
    uint index = subgroupBroadcastFirst(subgroupOffset) + subgroupIndex;

    d.modelInstanesFiltered.data[index] = mif;
}
