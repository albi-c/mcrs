#version 450

#include "common.glsl"

layout(location = 0) out vec2 outUv;
layout(location = 1) out vec3 outNormal;
layout(location = 2) flat out uvec4 outMat;
layout(location = 3) out vec3 outWorldPos;

struct Vertex {
    float16_t x;
    float16_t y;
    float16_t z;
    uint16_t mat;
    float16_t u;
    float16_t v;
    // 11:10:11
    // z / 1024 - 1, y / 512 - 1, x / 1024 - 1
    uint n;
};

struct Material {
    // diffuse << 16 | offsets
    // ambient == diffuse
    // alpha is in diffuse
    // 4 offsets from diffuse texture (? | metallic_roughness << 4 | normal)
    uint texDiffuseOffsets;

    // ? << 24 | b << 16 | g << 8 | r
    uint ambientAndRoughness;
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
    mat4 model;
    VertDataVertices vertices;
    VertDataMaterials materials;
};

layout(std430, push_constant) uniform Data {
    VertData vert;
    Pointer frag;
} data;

vec4 getVertexPosition(in Vertex vertex) {
    return vec4(float(vertex.x), float(vertex.y), float(vertex.z), 1.0);
}
vec2 getVertexUv(in Vertex vertex) {
    return vec2(vertex.u, vertex.v);
}
vec3 getVertexNormal(in Vertex vertex) {
    uint n = vertex.n;
    float nx = float(n & 0x7ff) / 1024.0 - 1.0;
    float ny = float((n >> 11) & 0x3ff) / 512.0 - 1.0;
    float nz = float(n >> 21) / 1024.0 - 1.0;
    // TODO: multiply by inverse of model matrix if doing non uniform transforms
    vec3 normal = normalize(vec3(nx, ny, nz));
    return normal;
}

void main() {
    VertData d = data.vert;

    Vertex vertex = d.vertices.data[gl_VertexIndex];
    vec4 vertexPos = getVertexPosition(vertex);
    vec4 position = d.mvp * vertexPos;
    gl_Position = position;
    outWorldPos = (d.model * vertexPos).xyz;
    outUv = getVertexUv(vertex);
    vec3 normal = getVertexNormal(vertex);
    outNormal = normal;

    Material material = d.materials.data[vertex.mat];
    outMat = uvec4(material.texDiffuseOffsets, material.ambientAndRoughness, material.diffuseAndNormal, material.specularAndExp);
}
