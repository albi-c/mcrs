#version 450

#include "common.glsl"

layout(location = 0) in vec2 inUv;
layout(location = 1) in vec3 inNormal;
layout(location = 2) flat in uvec4 inMat;
layout(location = 3) in vec3 inWorldPos;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform texture2D textures[];
layout(set = 1, binding = 0) uniform writeonly image2D textures_rw[];
layout(set = 2, binding = 0) uniform sampler samplers[];

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer FragData {
    vec4 sunPos;
    vec4 viewPos;
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
    uint texDiffuseRaw = inMat.x >> 16;
    uint texDiffuse = texDiffuseRaw & 0x7fff;
    uint texDisp = texDiffuse + (inMat.x & 0xf);
    uint texMetallic = texDiffuse + ((inMat.x >> 4) & 0xf);
    uint texRoughness = texDiffuse + ((inMat.x >> 8) & 0xf);
//    uint tex? = texDiffuse + ((inMat.x >> 12) & 0xf);

    vec4 sampleDiffuse = (texDiffuseRaw & 0x8000) != 0 ? vec4(1.0) : texture(sampler2D(textures[nonuniformEXT(texDiffuse)], samplers[0]), inUv);
    if (sampleDiffuse.a < 0.001) {
        discard;
    }
    vec3 sampleDisp = texDisp == texDiffuse ? vec3(0.0, 0.0, 1.0) : texture(sampler2D(textures[nonuniformEXT(texDisp)], samplers[0]), inUv).rgb;
    float sampleMetallic = texMetallic == texDiffuse ? 0.0 : texture(sampler2D(textures[nonuniformEXT(texDisp)], samplers[0]), inUv).r;
    float sampleRoughness = texRoughness == texDiffuse ? 0.0 : texture(sampler2D(textures[nonuniformEXT(texDisp)], samplers[0]), inUv).r;

    vec4 ambientAndRoughness = readPacked(inMat.y);
    // ambient is unused
//    float ambient = ambientAndRoughness.rgb;
    float roughness = ambientAndRoughness.a * sampleRoughness;

    float intensityAmbient = 0.2;

    vec4 diffuseAndNormal = readPacked(inMat.z);
    vec3 diffuse = diffuseAndNormal.rgb;
    float normalFactor = diffuseAndNormal.a;

    vec4 specularAndExp = readPacked(inMat.w);
    // specular is unused
//    vec3 specular = specularAndExp.rgb;
    float specularExp = specularAndExp.a * sampleMetallic;

    vec3 normal = inNormal;
    vec3 sunDirection = normalize(data.frag.sunPos.xyz - inWorldPos);
    float intensityDiffuse = max(0.0, dot(normal, sunDirection)) * 0.6;
    vec3 lookDirection = normalize(data.frag.viewPos.xyz - inWorldPos);
    vec3 halfwayDirection = normalize(sunDirection + lookDirection);
    float intensitySpecular = pow(max(dot(normal, halfwayDirection), 0.0), specularExp * 80.0) * 0.6;

    outColor = vec4(sampleDiffuse.rgb * diffuse * (intensityAmbient + intensityDiffuse) + intensitySpecular, 1.0);
//    outColor = vec4(reflectDirection, 1.0);
//    outColor = vec4(sampleMetallic, 0.0, 0.0, 1.0);
//    outColor = vec4(specularExp == roughness ? 0.0 : 1.0, 0.0, 0.0, 1.0);
//    outColor = vec4(specularExp, sampleMetallic, 0.0, 1.0);

//    outColor = vec4(normal, 1.0);
}
