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
    uint texDiffuse = inMat.x >> 16;
    uint texDisp = texDiffuse + (inMat.x & 0xf);
    uint texMetallic = texDiffuse + ((inMat.x >> 4) & 0xf);
    uint texRoughness = texDiffuse + ((inMat.x >> 8) & 0xf);
//    uint tex? = texDiffuse + ((inMat.x >> 12) & 0xf);

    vec4 sampleDiffuse = texture(sampler2D(textures[nonuniformEXT(texDiffuse)], samplers[0]), inUv);
    if (sampleDiffuse.a < 0.001) {
        discard;
    }
    vec3 sampleDisp = texDisp == texDiffuse ? vec3(0.0, 0.0, 1.0) : texture(sampler2D(textures[nonuniformEXT(texDisp)], samplers[0]), inUv).rgb;
    float sampleMetallic = texMetallic == texDiffuse ? 0.0 : texture(sampler2D(textures[nonuniformEXT(texDisp)], samplers[0]), inUv).r;

    vec4 ambientAndIntensity = readPacked(inMat.y);
    vec3 ambient = ambientAndIntensity.rgb;
    float intensityAmbient = max(ambientAndIntensity.a, 0.2);

    vec4 diffuseAndNormal = readPacked(inMat.z);
    vec3 diffuse = diffuseAndNormal.rgb;
    float normalFactor = diffuseAndNormal.a;

    vec4 specularAndExp = readPacked(inMat.w);
    vec3 specular = specularAndExp.rgb;
    // TODO: multiply by metallic sample
    float specularExp = specularAndExp.a;

    vec3 normal = inNormal;
    vec3 sunDirection = normalize(data.frag.sunPos.xyz - inWorldPos);
    float intensityDiffuse = max(0.0, dot(normal, sunDirection)) * 0.6;

//    outColor = vec4(sampleDiffuse.rgb * (ambient * intensityAmbient + diffuse * intensityDiffuse), 1.0);
//    outColor = sampleDiffuse;
    outColor = vec4(sampleDisp.xy, 1.0, 1.0);
//    outColor = vec4(sampleMetallic * specularExp, 0.0, 0.0, 1.0);

//    outColor = vec4(normal, 1.0);
}
