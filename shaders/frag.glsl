#version 450

#include "common.glsl"

layout(location = 0) in vec2 inUv;
layout(location = 1) in vec3 inNormal;
layout(location = 2) flat in uvec4 inMat;
layout(location = 3) flat in vec3 inFlatNormal;
layout(location = 4) flat in uint inUseFlatNormal;
layout(location = 5) in vec3 inWorldPos;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform texture2D textures[];
layout(set = 1, binding = 0) uniform writeonly image2D textures_rw[];
layout(set = 2, binding = 0) uniform sampler samplers[];

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer FragData {
    vec4 sunPos;
    vec4 lookDirection;
};

layout(std430, push_constant) uniform Data {
    Pointer vert;
    FragData frag;
} data;

vec4 readPacked(uint packed) {
    vec4 color = vec4(
        float(packed & 0xff),
        float((packed >> 8) & 0xff),
        float((packed >> 16) & 0xff),
        float(packed >> 24)
    );
    return color / 255.0;
}

void main() {
    // ambient == diffuse
    // alpha is in diffuse
    uint texDiffuse = inMat.x >> 16;
    uint texDisp = inMat.x & 0xffff;

    vec4 sampleDiffuse = texture(sampler2D(textures[nonuniformEXT(texDiffuse)], samplers[0]), inUv);
    if (sampleDiffuse.a < 0.001) {
        discard;
    }
    vec4 sampleDisp = texture(sampler2D(textures[nonuniformEXT(texDisp)], samplers[0]), inUv);

    vec4 ambientAndIntensity = readPacked(inMat.y);
    vec3 ambient = ambientAndIntensity.rgb;
    float intensityAmbient = max(ambientAndIntensity.a, 0.2);

    vec4 diffuseAndDissolve = readPacked(inMat.z);
    vec3 diffuse = diffuseAndDissolve.rgb;
    float dissolve = diffuseAndDissolve.a;

    vec4 specularAndExp = readPacked(inMat.w);
    vec3 specular = specularAndExp.rgb;
    float specularExp = specularAndExp.a;

    vec3 normal = (inUseFlatNormal == 0) ? inNormal : inFlatNormal;
    vec3 sunDirection = normalize(data.frag.sunPos.xyz - inWorldPos);
    float intensityDiffuse = max(0.0, dot(normal, sunDirection)) * 0.6;

    outColor = vec4(sampleDiffuse.rgb * (ambient * intensityAmbient + diffuse * intensityDiffuse), 1.0);
//    outColor = vec4(normal, 1.0);
//    float fn = (inUseFlatNormal == 0) ? 0.3 : 0.0;
//    outColor = vec4(fn, 0.3 - fn, 0.0, 1.0);
}
