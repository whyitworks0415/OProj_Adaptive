#version 450
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in  vec2 fragUV;
layout(location = 1) in  vec3 viewRay; // 버텍스에서 보간된 월드 시선 방향
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos; // .xyz는 카메라 위치, .w는 사용하지 않는다.
    vec4 lightDir; // .xyz=활성 광원(태양/달) 방향, .w=1이면 밤
    mat4 lightVP; // 방향광 시점 뷰·투영 (섀도맵 샘플링용)
    vec4 skyParams; // x=시간(초), y=구름량, z=안개 밀도, w=timeOfDay
    vec4 envParams; // x=현실적 셰이더(1)/클래식(0), y=노출, zw=예비
} cam;

// 디퍼드 조명 패스에서 읽는 G-Buffer 텍스처
layout(set = 1, binding = 0) uniform sampler2D gAlbedo; // RGB는 기본색, A는 스페큘러 강도
layout(set = 1, binding = 1) uniform sampler2D gNormal; // RGB는 0..1로 인코딩한 법선, A는 shininess/256
layout(set = 1, binding = 2) uniform sampler2D gPosition; // RGB는 월드 위치, A는 1.0+발광 강도(0이면 배경)
layout(set = 1, binding = 3) uniform sampler2D gMaterial; // R은 반사 강도

// 씬 조명 UBO(set 0, binding 2)
struct GpuLight {
    vec4 posRange; // Point/Spot: xyz=위치, w=범위 / Directional: xyz=방향, w=0
    vec4 dirType; // xyz=스팟 방향, w=타입(0=Point, 1=Directional, 2=Spot)
    vec4 colorEnab; // xyz=색상*강도, w=활성 여부
    vec4 coneParams; // x=cos(내부 콘각), y=cos(외부 콘각) — Spot 전용
};
layout(set = 0, binding = 2) uniform SceneLightUBO {
    int numLights;
    int useSceneLights;
    int ambientOn;
    int emissiveOn;
    GpuLight lights[16]; // GLTF 조명 + 발광 재질 자동 조명 (최대 16)
} sl;

// 방향광 섀도맵 (set 0, binding 3)
layout(set = 0, binding = 3) uniform sampler2D shadowMap;

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
    vec4 gbPos = texture(gPosition, fragUV);
    bool realistic = cam.envParams.x > 0.5;

    // gPosition.w < 0.5이면 지오메트리가 없는 배경 픽셀이다.
    if (gbPos.w < 0.5) {
        if (realistic) {
            // 절차적 하늘 (시선 레이는 버텍스에서 보간)
            vec3 dir = normalize(viewRay);
            vec3 sky = skyRadiance(dir, cam.lightDir, cam.cameraPos.xyz,
                                   cam.skyParams.x, cam.skyParams.y);
            sky      = acesToneMap(sky * cam.envParams.y);
            sky      = pow(sky, vec3(1.0 / 2.2));
            outColor = vec4(sky, 1.0);
        } else {
            outColor = vec4(0.53, 0.68, 0.85, 1.0); // 원본 하늘 배경색
        }
        return;
    }

    // G-Buffer에서 재질과 기하 정보를 복원한다.
    vec3  fragPos     = gbPos.rgb;
    float emissiveLum = max(0.0, gbPos.w - 1.0); // gPosition.w에는 1.0 + 발광 강도가 들어 있다.

    vec4  gbAlb     = texture(gAlbedo, fragUV);
    vec4  gbN       = texture(gNormal, fragUV);
    vec4  gbMat     = texture(gMaterial, fragUV);

    vec3  albedo    = gbAlb.rgb;
    float specStr   = gbAlb.a;
    vec3  N         = normalize(gbN.rgb * 2.0 - 1.0);
    float shininess = max(gbN.a * 256.0, 1.0);
    float reflectStrength = gbMat.r;

    vec3 V = normalize(cam.cameraPos.xyz - fragPos);
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
                vec3  L   = normalize(-sl.lights[i].posRange.xyz);
                float vis = shadowVisPCF(shadowMap, cam.lightVP, fragPos, N, L);
                result += pbrLight(N, L, V, lcolor * envDaylightScale(cam.lightDir),
                                   albedo, shininess, specStr) * vis;
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
                result += pbrLight(N, L, V, lcolor * atten, albedo, shininess, specStr);
            }
        }

        if (cam.lightDir.w > 0.5 && sl.useSceneLights != 0) {
            // GLTF 조명 씬의 밤: 태양 대신 달빛을 추가한다 (GLTF 램프는 그대로 유지).
            vec3  moonL = normalize(-cam.lightDir.xyz);
            float mvis  = shadowVisPCF(shadowMap, cam.lightVP, fragPos, N, moonL);
            result += pbrLight(N, moonL, V, envSunColor(cam.lightDir), albedo,
                               shininess, specStr) * mvis;
        }
        if (sl.useSceneLights == 0) {
            // 진짜 GLTF 조명이 없는 씬: 기본 태양(낮)/달(밤) + 도시 램프를 더한다.
            vec3  sunL   = normalize(-cam.lightDir.xyz);
            float sunVis = shadowVisPCF(shadowMap, cam.lightVP, fragPos, N, sunL);
            result += pbrLight(N, sunL, V, envSunColor(cam.lightDir), albedo,
                               shininess, specStr) * sunVis;

            const vec3 PT_POS[3] = vec3[](vec3( 4.0, 2.5,  4.0),
                                           vec3(-4.0, 2.5, -4.0),
                                           vec3( 0.0, 4.0,  0.0));
            const vec3 PT_COL[3] = vec3[](vec3(9.6, 7.2, 12.0),
                                           vec3(4.8, 9.6, 12.0),
                                           vec3(20.0,20.0,18.0));
            float lampScale = mix(0.45, 0.12, envDayFactor(cam.lightDir));
            for (int i = 0; i < 3; ++i) {
                vec3  toLight = PT_POS[i] - fragPos;
                float dist    = length(toLight);
                float atten   = 1.0 / (1.0 + 0.22 * dist + 0.20 * dist * dist);
                result += pbrLight(N, toLight / dist, V, PT_COL[i] * atten * lampScale,
                                   albedo, shininess, specStr);
            }
        }

        result += envSpecular(N, V, albedo, shininess, specStr,
                              reflectStrength, cam.lightDir);
        if (sl.emissiveOn != 0 && emissiveLum > 0.0) result += albedo * emissiveLum * 1.5;
        result = applyFog(result, length(cam.cameraPos.xyz - fragPos),
                          cam.lightDir, cam.skyParams.z);
        result = acesToneMap(result * cam.envParams.y);
    } else {
        // ---- 클래식 경로: 원본 Blinn-Phong + Reinhard ----
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
                                           shininess, specStr) * vis;
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
                                           shininess, specStr);
            }
        }
        if (sl.useSceneLights == 0) {
            vec3  sunL   = -cam.lightDir.xyz;
            float sunVis = shadowVisPCF(shadowMap, cam.lightVP, fragPos, N, normalize(sunL));
            result += blinnPhongLegacy(N, sunL, V, vec3(0.9, 0.855, 0.765), albedo,
                                       shininess, specStr) * sunVis;

            const vec3 PT_POS[3] = vec3[](vec3( 4.0, 2.5,  4.0),
                                           vec3(-4.0, 2.5, -4.0),
                                           vec3( 0.0, 4.0,  0.0));
            const vec3 PT_COL[3] = vec3[](vec3(9.6, 7.2, 12.0),
                                           vec3(4.8, 9.6, 12.0),
                                           vec3(20.0,20.0,18.0));
            for (int i = 0; i < 3; ++i) {
                vec3  toLight = PT_POS[i] - fragPos;
                float dist    = length(toLight);
                float atten   = 1.0 / (1.0 + 0.22 * dist + 0.20 * dist * dist);
                result += blinnPhongLegacy(N, toLight / dist, V, PT_COL[i] * atten,
                                           albedo, shininess, specStr);
            }
        }

        result += legacyFresnel(N, V, reflectStrength, cam.lightDir);
        if (sl.emissiveOn != 0 && emissiveLum > 0.0) result += albedo * emissiveLum;
        result = result / (result + vec3(1.0)); // Reinhard
    }

    // 감마 보정
    result   = pow(result, vec3(1.0 / 2.2));
    outColor = vec4(result, 1.0);
}
