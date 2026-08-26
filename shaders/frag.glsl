#version 460

#include "common.glsl"
#extension GL_EXT_ray_query : require

layout(location = 0) in vec2 inUv;
layout(location = 1) in vec3 inNormal;
layout(location = 2) flat in uvec4 inMat;
layout(location = 3) in vec3 inWorldPos;
layout(location = 4) flat in uint inDebugColor;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform texture2D textures[];
layout(set = 1, binding = 0) uniform writeonly image2D textures_rw[];
layout(set = 2, binding = 0) uniform sampler samplers[];
layout(set = 3, binding = 0) uniform accelerationStructureEXT accelerationStructures[];

struct Light {
    vec4 posAndIntensity;
    vec4 color;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer FragDataLights {
    Light data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer FragData {
    float viewX;
    float viewY;
    float viewZ;
    uint lightCount;
    FragDataLights lights;
    bool useDebugColor;
};

layout(std430, push_constant) uniform Data {
    Pointer vert;
    FragData frag;
} data;

vec4 readPacked(uint packed) {
    vec4 color = vec4(
        float(packed & 0xffu),
        float((packed >> 8) & 0xffu),
        float((packed >> 16) & 0xffu),
        float(packed >> 24)
    );
    return color / 255.0;
}

vec3 getLight(in vec3 normal, in vec3 viewDir, float specExp, float metallic, float roughness, in Light light, out vec3 specular) {
    vec3 lightDiff = light.posAndIntensity.xyz - inWorldPos;
    float lightDist = length(lightDiff);
    vec3 lightDir = lightDiff / lightDist;

//    rayQueryEXT query;
//    rayQueryInitializeEXT(
//        query, accelerationStructures[0], gl_RayFlagsTerminateOnFirstHitEXT, 0xff, inWorldPos, 0.01,
//        lightDir, lightDist
//    );
//    rayQueryProceedEXT(query);
//    if (rayQueryGetIntersectionTypeEXT(query, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
//        return vec3(0.0);
//    }

    float intDiff = max(0.0, dot(normal, lightDir));
    vec3 halfDir = normalize(lightDir + viewDir);
    float intSpec = pow(max(dot(normal, halfDir), 0.0), max(specExp, 1.0) * 12.0) * metallic;
    float distance = length(light.posAndIntensity.xyz - inWorldPos);
    specular = light.color.rgb * intSpec / max(pow(distance, 0.95), 1.0) * light.posAndIntensity.w * 1.0;
    return light.color.rgb * intDiff / max(pow(distance, 1.4), 1.0) * light.posAndIntensity.w * 1.2;
}

mat3 getTBN(vec3 p, vec3 n) {
    // http://www.thetenthplanet.de/archives/1180
    vec3 dp1 = dFdx(p);
    vec3 dp2 = dFdy(p);
    vec2 duv1 = dFdx(inUv);
    vec2 duv2 = dFdy(inUv);
    vec3 dp2perp = cross(dp2, n);
    vec3 dp1perp = cross(n, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = inversesqrt(max(dot(T, T), dot(B, B)));
    return mat3(T * invmax, B * invmax, n);
}

vec3 getSampledNormal(uint tex, vec3 viewPos, float strength) {
    mat3 tbn = getTBN(inWorldPos - viewPos, normalize(inNormal));
    vec3 map = texture(sampler2D(textures[nonuniformEXT(tex)], samplers[0]), inUv).rgb * 2.0 - 1.0;
    return normalize(tbn * normalize(map * vec3(1.0, 1.0, 1.0 / max(strength * 3.0, 0.1))));
}

void main() {
    FragData d = data.frag;
    if (d.useDebugColor) {
        outColor = vec4(readPacked(inDebugColor).rgb, 1.0);
        return;
    }

    uint texDiffuseRaw = inMat.x >> 16;
    uint texDiffuse = texDiffuseRaw & 0x7fffu;
    uint texDisp = texDiffuse + (inMat.x & 0xfu);
    uint texMetallicRoughness = texDiffuse + ((inMat.x >> 4) & 0xfu);
//    uint tex? = texDiffuse + ((inMat.x >> 8) & 0xf);
//    uint tex? = texDiffuse + ((inMat.x >> 12) & 0xf);

    vec4 sampleDiffuse = (texDiffuseRaw & 0x8000u) != 0 ? vec4(1.0) : texture(sampler2D(textures[nonuniformEXT(texDiffuse)], samplers[0]), inUv);
    if (sampleDiffuse.a < 0.001) {
        discard;
    }

//    vec3 sampleDisp = texDisp == texDiffuse ? vec3(0.0, 0.0, 0.5) : texture(sampler2D(textures[nonuniformEXT(texDisp)], samplers[0]), inUv).rgb;
    vec2 sampleMetallicRoughness = texMetallicRoughness == texDiffuse ? vec2(1.0) : texture(sampler2D(textures[nonuniformEXT(texMetallicRoughness)], samplers[0]), inUv).rg;
    float sampleMetallic = sampleMetallicRoughness.g;
    float sampleRoughness = sampleMetallicRoughness.r;

    vec4 ambientAndRoughness = readPacked(inMat.y);
    // ambient is unused
//    float ambient = ambientAndRoughness.rgb;
    float roughness = ambientAndRoughness.a * sampleRoughness;

    float intensityAmbient = 0.1;

    vec4 diffuseAndNormal = readPacked(inMat.z);
    vec3 diffuse = diffuseAndNormal.rgb;
    float normalFactor = diffuseAndNormal.a;

    vec4 specularAndExp = readPacked(inMat.w);
    // specular is unused
//    vec3 specular = specularAndExp.rgb;
    float specularExp = specularAndExp.a;

    vec3 diffuseBase = sampleDiffuse.rgb * diffuse;
    vec3 resultColor = diffuseBase * intensityAmbient;

    vec3 viewPos = vec3(d.viewX, d.viewY, d.viewZ);

    vec3 normal = texDisp == texDiffuse ? inNormal : getSampledNormal(texDisp, viewPos, normalFactor);

    uint lightCount = d.lightCount;
    FragDataLights lights = d.lights;
    vec3 viewDir = normalize(viewPos - inWorldPos);
    for (uint i = 0; i < lightCount; i++) {
        vec3 specular;
        resultColor += diffuseBase * getLight(normal, viewDir, specularExp, sampleMetallic, roughness, lights.data[i], specular);
        resultColor += specular;
    }

    outColor = vec4(resultColor, 1.0);
}
