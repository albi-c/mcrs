#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require

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

// --- 1536 bytes
struct Meshlet {
    // --- 1024 bytes
    Vertex vertices[64];

    // --- 512 bytes
    // packed: each uint: 0..8 - idx 1, 8..16 - idx 3, 16..24 - idx 3
    uint triangles[126];
    uint _padding[2];
};

// --- 96 bytes
struct MeshletInfo {
    mat4 transform;
    AABB aabb;
    uint8_t vertexCount;
    uint8_t triangleCount;
    // bit 0: disable rendering, bit 1: disable frustum culling
    uint8_t flags;
    uint8_t _padding[5];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataModelMeshlets {
    Meshlet data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataModelMeshletInfos {
    MeshletInfo data[];
};

struct Task {
    mat4 model;
    MeshDataModelMeshlets meshlets;
    MeshDataModelMeshletInfos meshletInfos;
};

struct Frustum {
    // xyz - normal, w - distance
    vec4 planes[5];
};

shared bool isInFrustum;
shared mat4 modelTransform;

// !! shader local size must be at least 6
// TODO: make workgroup size a multiple of 64 and replace shared variables with arrays (indexed with Y index)
// e.g. local_size_x = 5, local_size_y = 64
// shared bool isInFrustum[64];
void checkFrustum(in Frustum f, in AABB aabb) {
    if (gl_LocalInvocationIndex >= 5) {
        return;
    }

    vec3 center = (modelTransform * vec4(aabb.data[0], aabb.data[1], aabb.data[2], 1.0)).xyz;
    mat3 model3 = mat3(modelTransform);
    vec3 extent = mat3(
        abs(model3[0]),
        abs(model3[1]),
        abs(model3[2])
    ) * vec3(aabb.data[3], aabb.data[4], aabb.data[5]);
    vec4 plane = f.planes[gl_LocalInvocationIndex];
    float dist = dot(plane.xyz, center) + plane.w;
    float radius = dot(abs(plane.xyz), extent);
    if (dist + radius < 0.0) {
        isInFrustum = false;
    }
}
