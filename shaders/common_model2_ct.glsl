#include "common_model2.glsl"

// --- 32 bytes
struct LODChunk {
    AABB aabb;
    uint meshletOffset;
    uint16_t meshletCount;
    uint16_t _padding;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataLODChunks {
    LODChunk data[];
};

// --- 32 bytes
struct LOD {
    MeshDataModelMeshlets meshlets;
    MeshDataModelMeshletInfos meshletInfos;
    MeshDataLODChunks chunks;
    uint chunkCount;
    uint _padding;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataLODs {
    LOD data[];
};

// --- 48 bytes
struct Model {
    MeshDataLODs lods;
    uint lodCount;
    // bit 0: disable rendering, bit 1: enable frustum culling, bit 2: enable LODs
    uint flags;
    AABB aabb;
    uint _padding[2];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataModel {
    Model model;
};

// --- 32 bytes
struct ModelPart {
    MeshDataModelMeshlets meshlets;
    MeshDataModelMeshletInfos meshletInfos;
    MeshDataTransform transform;
    uint meshletOffset;
    uint16_t meshletCount;
    // bit 0: fully inside AABB, skip per meshlet checks
    uint16_t flags;
};

// --- 24 bytes
struct ModelInstanceFiltered {
    MeshDataLODs lods;
    MeshDataTransform transform;
    // bit 0: skip per chunk frustum culling
    uint flags;
    uint lod;
};
