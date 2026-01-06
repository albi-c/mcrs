#version 450

#include "common.glsl"

layout(location = 0) out vec2 outUv;
layout(location = 1) out vec3 outNormal;
layout(location = 2) flat out uvec4 outMat;
layout(location = 3) flat out vec3 outFlatNormal;
layout(location = 4) flat out uint outUseFlatNormal;
layout(location = 5) out vec3 outWorldPos;

struct Vertex {
    float x;
    float y;
    float z;
    uint mat;
    vec2 uv;
    vec2 nxy;
};

struct Material {
    // (diffuse << 16) | disp
    // ambient == diffuse
    // alpha is in diffuse
    uint texDiffuseDisp;

    // (? << 24) | (b << 16) | (g << 8) | r
    uint ambientAndIntensity;
    uint diffuseAndDissolve;
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
//    outUv = vec2(vertex.uv.x, 1.0 - vertex.uv.y);
    outUv = vertex.uv;
    vec2 n = vertex.nxy;
    // TODO: multiply by inverse of model matrix if doing non uniform transforms
    vec3 normal = vec3(n, sqrt(max(1.0 - n.x*n.x - n.y*n.y, 0.0)));
    outNormal = normal;
    outFlatNormal = normal;

    Material material = d.materials.data[vertex.mat & ~(1 << 31)];
    outMat = uvec4(material.texDiffuseDisp, material.ambientAndIntensity, material.diffuseAndDissolve, material.specularAndExp);
    outUseFlatNormal = (vertex.mat & (1 << 31)) >> 31;

//    uint tex = vertex.tex;
//    float u = float(tex & ((1 << 10) - 1));
//    float v = float((tex >> 10) & ((1 << 10) - 1));
//    outUv = round(vec2(u, v)) / 512.0;
//    outTex = tex >> 20;
}
