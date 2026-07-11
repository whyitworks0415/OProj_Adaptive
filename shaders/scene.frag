#version 450
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos; // .xyz는 카메라 위치, .w는 예비
    vec4 lightDir; // .xyz=활성 광원(태양/달) 방향, .w=1이면 밤
    mat4 lightVP; // 방향광 시점 뷰·투영 (섀도맵 샘플링용)
    vec4 skyParams; // x=시간(초), y=구름량, z=안개 밀도, w=timeOfDay
    vec4 envParams; // x=현실적 셰이더(1)/클래식(0), y=노출, zw=예비
} cam;

layout(set = 0, binding = 1) uniform sampler2D textures[64];

// 씬 조명 UBO(GLTF KHR_lights_punctual)
struct GpuLight {
    vec4 posRange; // Point/Spot: xyz=위치, w=범위(0=무제한) / Directional: xyz=방향, w=0
    vec4 dirType; // xyz=스팟 방향, w=타입(0=Point, 1=Directional, 2=Spot)
    vec4 colorEnab; // xyz=색상*강도, w=활성 여부(0 또는 1)
    vec4 coneParams; // x=cos(내부 콘각), y=cos(외부 콘각) — Spot 전용
};
layout(set = 0, binding = 2) uniform SceneLightUBO {
    int numLights;
    int useSceneLights; // 1=GLTF 조명, 0=셰이더 기본 조명
    int ambientOn;
    int emissiveOn;
    GpuLight lights[16]; // GLTF 조명 + 발광 재질 자동 조명 (최대 16)
} sl;

// 방향광 섀도맵 (set 0, binding 3)
layout(set = 0, binding = 3) uniform sampler2D shadowMap;

layout(push_constant) uniform PushConstants {
    mat4  model;
    vec4  baseColor;
    float shininess;
    float specularStrength;
    float reflectStrength;
    float textureIndex;
    vec4  emissive; // rgb=발광 색상(텍스처 있으면 곱셈 팩터), a=발광 강도
    vec4  texIndices2; // x=metallicRoughness 텍스처, y=emissive 텍스처 (-1=없음)
} pc;

#include "common_env.glsl"

// KHR_lights_punctual 거리 감쇄
float lightAtten(float dist, float range) {
    if (range > 0.0) {
        float ratio = clamp(dist / range, 0.0, 1.0);
        float a = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
        return a * a / (dist * dist + 1.0);
    }
    return 1.0 / (dist * dist + 1.0);
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(cam.cameraPos.xyz - fragPos);
    bool realistic = cam.envParams.x > 0.5;

    // 기본 albedo 계산
    // 주의: int(-1.0 + 0.5) 는 0으로 잘리므로(0쪽 절삭) floor 로 반올림해야
    // 텍스처 없음(-1)이 텍스처 0번으로 잘못 해석되지 않는다.
    int texIdx = int(floor(pc.textureIndex + 0.5));
    vec3 albedo;
    if (texIdx >= 0) {
        albedo = texture(textures[texIdx], fragUV).rgb * pc.baseColor.rgb;
    } else {
        albedo = fragColor * pc.baseColor.rgb;
    }

    // Blender metallicRoughness 텍스처: G=러프니스, B=메탈릭 (팩터와 곱)
    float roughness = envRoughness(pc.shininess);
    float metallic  = envMetallic(pc.specularStrength);
    int   mrIdx     = int(floor(pc.texIndices2.x + 0.5));
    if (mrIdx >= 0) {
        vec4 mr    = texture(textures[mrIdx], fragUV);
        roughness *= mr.g;
        metallic  *= mr.b;
    }

    // 발광: emissive 텍스처가 있으면 팩터와 곱한다.
    vec3 emis    = pc.emissive.rgb;
    int  emisIdx = int(floor(pc.texIndices2.y + 0.5));
    if (emisIdx >= 0) {
        emis *= texture(textures[emisIdx], fragUV).rgb;
    }

    vec3 result;

    if (realistic) {
        // ---- 현실적 경로: PBR + 반구 환경광 + 시간대 조명 + 안개 + ACES ----
        result = (sl.ambientOn != 0) ? hemiAmbient(N, albedo, cam.lightDir) : vec3(0.0);

        // 조명 배열(GLTF 조명 + 발광 재질 자동 조명)은 항상 적용한다.
        for (int i = 0; i < sl.numLights && i < 16; ++i) {
            if (sl.lights[i].colorEnab.w < 0.5) continue;
            int  ltype  = int(sl.lights[i].dirType.w + 0.5);
            vec3 lcolor = sl.lights[i].colorEnab.rgb;

            if (ltype == 1) {
                // 방향광 (소프트 섀도 + 시간대 감쇠·틴트)
                vec3  L   = normalize(-sl.lights[i].posRange.xyz);
                float vis = shadowVisPCF(shadowMap, cam.lightVP, fragPos, N, L);
                result += pbrLightRM(N, L, V, lcolor * envDaylightScale(cam.lightDir),
                                     albedo, roughness, metallic) * vis;
            } else {
                vec3  toLight = sl.lights[i].posRange.xyz - fragPos;
                float dist    = max(length(toLight), 0.001);
                // 범위 밖은 감쇄가 정확히 0 — BRDF 계산 전에 건너뛴다 (시각 결과 동일)
                float range   = sl.lights[i].posRange.w;
                if (range > 0.0 && dist > range) continue;
                vec3  L       = toLight / dist;
                float atten   = lightAtten(dist, range);
                if (ltype == 2) {
                    // Blender 스팟 콘 각도(inner/outer) 적용
                    vec3  spotDir = normalize(sl.lights[i].dirType.xyz);
                    float cosA    = dot(-L, spotDir);
                    float innerC  = sl.lights[i].coneParams.x;
                    float outerC  = sl.lights[i].coneParams.y;
                    atten *= clamp((cosA - outerC) / max(innerC - outerC, 1e-4), 0.0, 1.0);
                }
                result += pbrLightRM(N, L, V, lcolor * atten, albedo, roughness, metallic);
            }
        }

        if (sl.useSceneLights == 0) {
            // 진짜 GLTF 조명이 없는 씬: 기본 태양(낮)/달(밤) + 도시 램프를 더한다.
            vec3  sunL   = normalize(-cam.lightDir.xyz);
            float sunVis = shadowVisPCF(shadowMap, cam.lightVP, fragPos, N, sunL);
            result += pbrLightRM(N, sunL, V, envSunColor(cam.lightDir),
                                 albedo, roughness, metallic) * sunVis;

            const vec3  PT_POS[3] = vec3[](vec3( 4.0, 2.5,  4.0),
                                            vec3(-4.0, 2.5, -4.0),
                                            vec3( 0.0, 4.0,  0.0));
            const vec3  PT_COL[3] = vec3[](vec3(9.6, 7.2, 12.0),
                                            vec3(4.8, 9.6, 12.0),
                                            vec3(20.0,20.0,18.0));
            float lampScale = mix(0.45, 0.12, envDayFactor(cam.lightDir));
            for (int i = 0; i < 3; ++i) {
                vec3  toLight = PT_POS[i] - fragPos;
                float dist    = length(toLight);
                float atten   = 1.0 / (1.0 + 0.22 * dist + 0.20 * dist * dist);
                result += pbrLightRM(N, toLight / dist, V, PT_COL[i] * atten * lampScale,
                                     albedo, roughness, metallic);
            }
        } else if (cam.lightDir.w > 0.5) {
            // GLTF 조명 씬의 밤: 태양 대신 달빛을 추가한다 (GLTF 램프는 그대로 유지).
            vec3  moonL = normalize(-cam.lightDir.xyz);
            float mvis  = shadowVisPCF(shadowMap, cam.lightVP, fragPos, N, moonL);
            result += pbrLightRM(N, moonL, V, envSunColor(cam.lightDir),
                                 albedo, roughness, metallic) * mvis;
        }

        result += envSpecular(N, V, albedo, pc.shininess, pc.specularStrength,
                              pc.reflectStrength, cam.lightDir);
        if (sl.emissiveOn != 0) result += emis * 1.5;
        result = applyFog(result, length(cam.cameraPos.xyz - fragPos),
                          cam.lightDir, cam.skyParams.z);
        result = acesToneMap(result * cam.envParams.y);
    } else {
        // ---- 클래식 경로: 원본 Blinn-Phong + Reinhard (안개·시간대 감쇠 없음) ----
        result = (sl.ambientOn != 0) ? vec3(0.08) * albedo : vec3(0.0);

        // 조명 배열(GLTF + 자동 발광 조명)은 항상 적용한다.
        for (int i = 0; i < sl.numLights && i < 16; ++i) {
            if (sl.lights[i].colorEnab.w < 0.5) continue;
            int  ltype  = int(sl.lights[i].dirType.w + 0.5);
            vec3 lcolor = sl.lights[i].colorEnab.rgb;

            if (ltype == 1) {
                vec3  L   = normalize(-sl.lights[i].posRange.xyz);
                float vis = shadowVisPCF(shadowMap, cam.lightVP, fragPos, N, L);
                result += blinnPhongLegacy(N, L, V, lcolor, albedo,
                                           pc.shininess, pc.specularStrength) * vis;
            } else {
                vec3  toLight = sl.lights[i].posRange.xyz - fragPos;
                float dist    = max(length(toLight), 0.001);
                // 범위 밖은 감쇄가 정확히 0 — BRDF 계산 전에 건너뛴다 (시각 결과 동일)
                float range   = sl.lights[i].posRange.w;
                if (range > 0.0 && dist > range) continue;
                vec3  L       = toLight / dist;
                float atten   = lightAtten(dist, range);
                if (ltype == 2) {
                    vec3  spotDir  = normalize(sl.lights[i].dirType.xyz);
                    float cosAngle = dot(-L, spotDir);
                    atten *= clamp((cosAngle - 0.5) / 0.2, 0.0, 1.0);
                }
                result += blinnPhongLegacy(N, L, V, lcolor * atten, albedo,
                                           pc.shininess, pc.specularStrength);
            }
        }
        if (sl.useSceneLights == 0) {
            vec3  sunL   = -cam.lightDir.xyz;
            float sunVis = shadowVisPCF(shadowMap, cam.lightVP, fragPos, N, normalize(sunL));
            result += blinnPhongLegacy(N, sunL, V, vec3(0.9, 0.855, 0.765), albedo,
                                       pc.shininess, pc.specularStrength) * sunVis;

            const vec3  PT_POS[3] = vec3[](vec3( 4.0, 2.5,  4.0),
                                            vec3(-4.0, 2.5, -4.0),
                                            vec3( 0.0, 4.0,  0.0));
            const vec3  PT_COL[3] = vec3[](vec3(9.6, 7.2, 12.0),
                                            vec3(4.8, 9.6, 12.0),
                                            vec3(20.0,20.0,18.0));
            for (int i = 0; i < 3; ++i) {
                vec3  toLight = PT_POS[i] - fragPos;
                float dist    = length(toLight);
                float atten   = 1.0 / (1.0 + 0.22 * dist + 0.20 * dist * dist);
                result += blinnPhongLegacy(N, toLight / dist, V, PT_COL[i] * atten,
                                           albedo, pc.shininess, pc.specularStrength);
            }
        }

        result += legacyFresnel(N, V, pc.reflectStrength, cam.lightDir);
        if (sl.emissiveOn != 0) result += emis;
        result = result / (result + vec3(1.0)); // Reinhard
    }

    // 감마 보정
    result   = pow(result, vec3(1.0 / 2.2));
    outColor = vec4(result, pc.baseColor.a);
}
