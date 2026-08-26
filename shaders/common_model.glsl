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

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MeshDataTransform {
    mat4 transform;
};

// --- 48 bytes
struct MeshletInfo {
    AABB aabb;
    MeshDataTransform transform;
    uint8_t vertexCount;
    uint8_t triangleCount;
    // bit 0: disable rendering, bit 1: disable frustum culling
    uint8_t flags;
    uint8_t _padding[13];
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

struct Task {
    mat4 model;
    MeshDataModelMeshlets meshlets;
    MeshDataModelMeshletInfos meshletInfos;
};

struct Frustum {
    // xyz - normal, w - distance
    vec4 planes[5];
};

shared bool g_isInFrustum;
shared mat4 g_modelTransform;

void checkFrustum(in Frustum f, in AABB aabb) {
    if (gl_LocalInvocationIndex >= 5) {
        return;
    }

    mat4 modelTransform = g_modelTransform;
    vec3 center = (modelTransform * vec4(aabb.data[0], aabb.data[1], aabb.data[2], 1.0)).xyz;
    vec3 extent = mat3(
        abs(modelTransform[0].xyz),
        abs(modelTransform[1].xyz),
        abs(modelTransform[2].xyz)
    ) * vec3(aabb.data[3], aabb.data[4], aabb.data[5]);
    vec4 plane = f.planes[gl_LocalInvocationIndex];
    float dist = dot(plane.xyz, center) + plane.w;
    float radius = dot(abs(plane.xyz), extent);
    if (dist + radius < 0.0) {
        g_isInFrustum = false;
    }
}
