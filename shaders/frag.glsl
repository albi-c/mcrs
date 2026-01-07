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

struct Light {
    vec4 posAndIntensity;
    vec4 color;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer FragDataLights {
    Light data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer FragData {
    vec4 viewPos;
    uint lightCount;
    uint _padding1[3];
    FragDataLights lights;
    uint _padding2[2];
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

vec3 getLight(in vec3 normal, in vec3 viewDir, float specExp, float metallic, in Light light, out vec3 specular) {
    vec3 lightDir = normalize(light.posAndIntensity.xyz - inWorldPos);
    float intDiff = max(0.0, dot(normal, lightDir));
    vec3 halfDir = normalize(lightDir + viewDir);
    float intSpec = pow(max(dot(normal, halfDir), 0.0), max(specExp, 1.0) * 12.0) * metallic;
    float distance = length(light.posAndIntensity.xyz - inWorldPos);
    specular = light.color.rgb * intSpec / max(pow(distance, 1.1), 1.0) * light.posAndIntensity.w * 0.7;
    return light.color.rgb * intDiff / max(pow(distance, 1.5), 1.0) * light.posAndIntensity.w;
}

void main() {
    uint texDiffuseRaw = inMat.x >> 16;
    uint texDiffuse = texDiffuseRaw & 0x7fff;
    uint texDisp = texDiffuse + (inMat.x & 0xf);
    uint texMetallicRoughness = texDiffuse + ((inMat.x >> 4) & 0xf);
//    uint tex? = texDiffuse + ((inMat.x >> 8) & 0xf);
//    uint tex? = texDiffuse + ((inMat.x >> 12) & 0xf);

    vec4 sampleDiffuse = (texDiffuseRaw & 0x8000) != 0 ? vec4(1.0) : texture(sampler2D(textures[nonuniformEXT(texDiffuse)], samplers[0]), inUv);
    if (sampleDiffuse.a < 0.001) {
        discard;
    }

    vec3 sampleDisp = texDisp == texDiffuse ? vec3(0.0, 0.0, 1.0) : texture(sampler2D(textures[nonuniformEXT(texDisp)], samplers[0]), inUv).rgb;
    vec2 sampleMetallicRoughness = texMetallicRoughness == texDiffuse ? vec2(1.0) : texture(sampler2D(textures[nonuniformEXT(texMetallicRoughness)], samplers[0]), inUv).rg;
    float sampleMetallic = sampleMetallicRoughness.g;
    float sampleRoughness = sampleMetallicRoughness.r;

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
    float specularExp = specularAndExp.a;

    vec3 diffuseBase = sampleDiffuse.rgb * diffuse;
    vec3 resultColor = diffuseBase * intensityAmbient;

    uint lightCount = data.frag.lightCount;
    FragDataLights lights = data.frag.lights;
    vec3 viewDir = normalize(data.frag.viewPos.xyz - inWorldPos);
    for (uint i = 0; i < lightCount; i++) {
        vec3 specular;
        resultColor += diffuseBase * getLight(inNormal, viewDir, specularExp, sampleMetallic, lights.data[i], specular);
        resultColor += specular;
    }

    outColor = vec4(resultColor, 1.0);
}
