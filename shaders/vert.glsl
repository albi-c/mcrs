#version 450

#include "common.glsl"

layout(location = 0) out vec2 outUv;
layout(location = 1) out vec3 outNormal;
layout(location = 2) flat out uvec4 outMat;
layout(location = 3) out vec3 outWorldPos;

struct Vertex {
    float x;
    float y;
    float z;
    uint mat;
    vec2 uv;
    float16_t nx;
    float16_t ny;
    float16_t nz;
    float16_t _pad;
};

struct Material {
    // (diffuse << 16) | offsets
    // ambient == diffuse
    // alpha is in diffuse
    // 4 offsets from diffuse texture (? | roughness << 8 | metallic << 4 | normal)
    uint texDiffuseOffsets;

    // (? << 24) | (b << 16) | (g << 8) | r
    uint ambientAndIntensity;
    uint diffuseAndNormal;
    uint specularAndExp;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertDataVertices {
    Vertex data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertDataMaterials {
    Material data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertData {
    mat4 mvp;
    VertDataVertices vertices;
    VertDataMaterials materials;
};

layout(std430, push_constant) uniform Data {
    VertData vert;
    Pointer frag;
} data;

void main() {
    VertData d = data.vert;

    Vertex vertex = d.vertices.data[gl_VertexIndex];
    vec4 position = d.mvp * vec4(vertex.x, vertex.y, vertex.z, 1.0);
    gl_Position = position;
    outWorldPos = position.xyz;
    outUv = vertex.uv;
    // TODO: multiply by inverse of model matrix if doing non uniform transforms
    vec3 normal = vec3(float(vertex.nx), float(vertex.ny), float(vertex.nz));
    outNormal = normal;

    Material material = d.materials.data[vertex.mat];
    outMat = uvec4(material.texDiffuseOffsets, material.ambientAndIntensity, material.diffuseAndNormal, material.specularAndExp);
}
