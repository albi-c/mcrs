#ifndef _COMMON_MODEL_2
#define _COMMON_MODEL_2

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require

// max 256 due to 8 bit relative indices in Task struct
const uint MODEL_PART_SIZE = 64;

// --- 16 bytes, possible to extend to 24 to make Meshlet 2k in size
struct Vertex {
    float16_t pos[3];
    uint16_t mat;
    float16_t uv[2];
    // 11:10:11
    // z / 1024 - 1, y / 512 - 1, x / 1024 - 1
    uint normal;
};

// --- 24 bytes
struct AABB {
    // vec3 center, vec3 extent
    float data[6];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataTransform {
    mat4 transform;
};

// --- 32 bytes
struct MeshletInfo {
    AABB aabb;
    uint8_t vertexCount;
    uint8_t triangleCount;
    // bit 0: disable rendering, bit 1: disable frustum culling, bit 2: disable rendering in depth prepass
    uint8_t flags;
    uint8_t _padding[5];
};

// --- 1536 bytes
struct Meshlet {
    // --- 1024 bytes
    Vertex vertices[64];

    // --- 512 bytes
    // packed: each uint: 0..8 - idx 1, 8..16 - idx 3, 16..24 - idx 3
    uint triangles[128];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataModelMeshlets {
    Meshlet data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataModelMeshletInfos {
    MeshletInfo data[];
};

struct Frustum {
    // xyz - normal, w - distance
    vec4 planes[5];
};

void transformAABB(in AABB aabb, in mat4 modelTransform, out vec3 center, out vec3 extent) {
    center = (modelTransform * vec4(aabb.data[0], aabb.data[1], aabb.data[2], 1.0)).xyz;
    extent = mat3(
        abs(modelTransform[0].xyz),
        abs(modelTransform[1].xyz),
        abs(modelTransform[2].xyz)
    ) * vec3(aabb.data[3], aabb.data[4], aabb.data[5]);
}

bool isOutsidePlane(vec4 plane, vec3 center, vec3 extent) {
    float dist = dot(plane.xyz, center) + plane.w;
    float radius = dot(abs(plane.xyz), extent);
    return dist + radius < 0.0;
}

bool checkFrustum(in Frustum f, in AABB aabb, in mat4 modelTransform) {
    vec3 center;
    vec3 extent;
    transformAABB(aabb, modelTransform, center, extent);
    for (uint i = 0; i < 5; i++) {
        if (isOutsidePlane(f.planes[i], center, extent)) {
            return false;
        }
    }
    return true;
}

bool checkFrustumAndFullyInside(in Frustum f, in AABB aabb, in mat4 modelTransform, out bool fullyInside) {
    vec3 center;
    vec3 extent;
    transformAABB(aabb, modelTransform, center, extent);
    for (uint i = 0; i < 5; i++) {
        if (isOutsidePlane(f.planes[i], center, extent)) {
            return false;
        }
    }
    for (uint i = 0; i < 5; i++) {
        vec4 plane = f.planes[i];
        vec3 farPoint = mix(center - extent, center + extent, greaterThan(plane.xyz, vec3(0.0)));
        if (dot(plane.xyz, farPoint) + plane.w < 0.0) {
            fullyInside = false;
            return true;
        }
    }
    fullyInside = true;
    return true;
}

#endif
