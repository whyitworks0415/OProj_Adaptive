// common_env.glsl — PBR 조명·절차적 하늘·안개·톤매핑 공용 함수
// scene.frag / deferred_light.frag / sky.frag 에서 #include 로 사용한다.
//
// 규약:
// - lightDirW: cam.lightDir 그대로. xyz = 씬을 향하는 "활성 광원"(태양 또는 달) 방향,
//   w = 1.0 이면 밤(활성 광원이 달). 태양의 실제 위치는 밤에 지평선 아래에 있으므로
//   envToSun() 으로 복원한다.
// - 모든 radiance 는 HDR 선형 공간이며 마지막에 acesToneMap + 감마 보정을 적용한다.

const float PI = 3.14159265359;

// ---------------------------------------------------------------- 해시/노이즈
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 443.8975);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

float hash13(vec3 p) {
    p = fract(p * 443.8975);
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash12(i);
    float b = hash12(i + vec2(1, 0));
    float c = hash12(i + vec2(0, 1));
    float d = hash12(i + vec2(1, 1));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// 5옥타브 FBM (구름용)
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; ++i) {
        v += a * vnoise(p);
        p = p * 2.03 + vec2(17.3, 9.1);
        a *= 0.5;
    }
    return v;
}

// ---------------------------------------------------------------- 톤매핑
// ACES 필름 톤매핑 근사 (Narkowicz)
vec3 acesToneMap(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// ---------------------------------------------------------------- 태양/달 상태
// 실제 태양 방향(밤에는 지평선 아래)을 복원한다.
vec3 envToSun(vec4 lightDirW) {
    vec3 toL = -normalize(lightDirW.xyz);
    if (lightDirW.w > 0.5) toL.y = -toL.y; // 밤: 활성 광원(달)은 태양의 고도 반전 위치
    return toL;
}

// 낮 정도 (0=밤, 1=완전한 낮)
float envDayFactor(vec4 lightDirW) {
    return smoothstep(-0.08, 0.15, envToSun(lightDirW).y);
}

// 활성 광원(태양 또는 달)의 radiance. 시간대에 따라 색·강도가 변한다.
vec3 envSunColor(vec4 lightDirW) {
    vec3  toSun = envToSun(lightDirW);
    float e     = toSun.y;
    if (lightDirW.w > 0.5) {
        // 은은한 푸른 달빛
        return vec3(0.45, 0.55, 0.80) * (smoothstep(0.03, 0.30, -e) * 0.18);
    }
    vec3 horizonCol = vec3(1.0, 0.42, 0.15);      // 일출/일몰의 주황빛
    vec3 noonCol    = vec3(1.0, 0.96, 0.88);      // 정오의 백색광
    vec3 col        = mix(horizonCol, noonCol, smoothstep(0.03, 0.45, e));
    return col * (smoothstep(-0.03, 0.12, e) * 1.15);
}

// GLTF 방향광(태양)에 적용할 시간대 스케일·틴트.
// 정오 = 원본 강도(백색), 일출/일몰 = 주황 틴트, 밤 = 0 (달빛이 대신한다).
vec3 envDaylightScale(vec4 lightDirW) {
    float e    = envToSun(lightDirW).y;
    vec3  tint = mix(vec3(1.0, 0.55, 0.28), vec3(1.0), smoothstep(0.03, 0.45, e));
    return tint * smoothstep(-0.03, 0.12, e);
}

// ---------------------------------------------------------------- PBR (Cook-Torrance GGX)
float dGGX(float NdotH, float a) {
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float gSmith(float NdotV, float NdotL, float rough) {
    float k  = (rough + 1.0);
    k = k * k / 8.0;
    float gv = NdotV / (NdotV * (1.0 - k) + k);
    float gl = NdotL / (NdotL * (1.0 - k) + k);
    return gv * gl;
}

vec3 fresnelSchlick(float cosT, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosT, 0.0, 1.0), 5.0);
}

// 레거시 퐁 파라미터(shininess/specStr)를 러프니스/메탈릭으로 역매핑
// (GLTFLoader: shininess = mix(256,4,r^2), specStr = mix(0.05,0.95,metallic))
float envRoughness(float shininess) {
    return clamp(sqrt(clamp((256.0 - shininess) / 252.0, 0.0, 1.0)), 0.05, 1.0);
}
float envMetallic(float specStr) {
    return clamp((specStr - 0.05) / 0.90, 0.0, 1.0);
}

// 단일 광원에 대한 Cook-Torrance 조명 (러프니스/메탈릭 직접 전달).
// radiance = 광원 색 * 강도 * 감쇠.
vec3 pbrLightRM(vec3 N, vec3 L, vec3 V, vec3 radiance, vec3 albedo,
                float roughness, float metallic) {
    roughness = clamp(roughness, 0.05, 1.0);
    metallic  = clamp(metallic,  0.0,  1.0);
    vec3 F0   = mix(vec3(0.04), albedo, metallic);

    vec3  H     = normalize(L + V);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 1e-4);
    float NdotH = max(dot(N, H), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);

    float D    = dGGX(NdotH, roughness * roughness);
    float G    = gSmith(NdotV, NdotL, roughness);
    vec3  F    = fresnelSchlick(max(dot(H, V), 0.0), F0);
    // 확산항은 기존 조명 강도와의 호환을 위해 1/PI 를 생략하고,
    // 스페큘러항에 PI 를 곱해 상대 균형을 유지한다.
    vec3  spec = min(D * G * F / max(4.0 * NdotV * NdotL, 1e-4) * PI, vec3(16.0));
    vec3  kd   = (vec3(1.0) - F) * (1.0 - metallic);
    return (kd * albedo + spec) * radiance * NdotL;
}

// 레거시 퐁 파라미터(shininess/specStr) 버전 래퍼
vec3 pbrLight(vec3 N, vec3 L, vec3 V, vec3 radiance, vec3 albedo,
              float shininess, float specStr) {
    return pbrLightRM(N, L, V, radiance, albedo,
                      envRoughness(shininess), envMetallic(specStr));
}

// ---------------------------------------------------------------- 클래식(원본) 경로
// 현실적 셰이더 OFF 시 사용하는 원본 Blinn-Phong 조명·Fresnel 반사.
vec3 blinnPhongLegacy(vec3 N, vec3 L, vec3 V, vec3 lightColor, vec3 albedo,
                      float shininess, float specStr) {
    float diff = max(dot(N, L), 0.0);
    vec3  H    = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), shininess) * specStr;
    return lightColor * (diff * albedo + spec * vec3(1.0));
}

vec3 legacyFresnel(vec3 N, vec3 V, float reflectStrength, vec4 lightDirW) {
    float cosTheta = max(dot(N, V), 0.0);
    float f = pow(1.0 - cosTheta, 3.0);
    vec3 skyColor = mix(vec3(0.4, 0.55, 0.75), vec3(0.7, 0.8, 0.95),
                        clamp(0.5 + 0.5 * (-lightDirW.y), 0.0, 1.0));
    return f * skyColor * reflectStrength;
}

// 반구 환경광: 위쪽은 하늘색, 아래쪽은 지면 반사색. 시간대를 따라간다.
vec3 hemiAmbient(vec3 N, vec3 albedo, vec4 lightDirW) {
    float day  = envDayFactor(lightDirW);
    vec3  skyA = mix(vec3(0.010, 0.014, 0.030), vec3(0.36, 0.44, 0.58), day);
    vec3  gndA = mix(vec3(0.006, 0.006, 0.009), vec3(0.22, 0.20, 0.17), day);
    return mix(gndA, skyA, N.y * 0.5 + 0.5) * albedo * 0.22;
}

// 하늘색 환경 스페큘러 (러프니스가 낮을수록 선명한 반사)
vec3 envSpecular(vec3 N, vec3 V, vec3 albedo, float shininess, float specStr,
                 float reflectStrength, vec4 lightDirW) {
    float roughness = envRoughness(shininess);
    float metallic  = envMetallic(specStr);
    vec3  F0        = mix(vec3(0.04), albedo, metallic);
    float day       = envDayFactor(lightDirW);

    vec3  R      = reflect(-V, N);
    vec3  skyCol = mix(mix(vec3(0.012, 0.018, 0.045), vec3(0.62, 0.76, 0.92), day),
                       mix(vec3(0.004, 0.007, 0.018), vec3(0.13, 0.32, 0.72), day),
                       clamp(R.y, 0.0, 1.0));
    float NdotV  = max(dot(N, V), 1e-4);
    vec3  F      = F0 + (max(vec3(1.0 - roughness), F0) - F0)
                        * pow(1.0 - NdotV, 5.0);
    return F * skyCol * (1.0 - roughness) * (0.4 + 0.6 * reflectStrength) * 0.6;
}

// ---------------------------------------------------------------- 절차적 하늘
// dir: 정규화된 시선 방향. HDR 하늘 radiance 를 반환한다 (톤매핑 전).
vec3 skyRadiance(vec3 dir, vec4 lightDirW, vec3 camPos, float time, float cloudiness) {
    vec3  toSun    = envToSun(lightDirW);
    float sunE     = toSun.y;
    float day      = smoothstep(-0.08, 0.15, sunE);
    float night    = 1.0 - day;
    float twilight = exp(-abs(sunE) * 9.0); // 여명/황혼 정도

    // 대기 그라데이션 (지평선 -> 천정)
    float h       = clamp(dir.y, 0.0, 1.0);
    vec3  zenith  = mix(vec3(0.004, 0.007, 0.018), vec3(0.13, 0.32, 0.72), day);
    vec3  horizon = mix(vec3(0.012, 0.018, 0.045), vec3(0.62, 0.76, 0.92), day);
    vec3  sky     = mix(horizon, zenith, pow(h, 0.55));

    // 태양 방위 쪽 지평선에 황혼 산란광을 더한다.
    vec2  dirH      = normalize(dir.xz + vec2(1e-5));
    vec2  sunH      = normalize(toSun.xz + vec2(1e-5));
    float sunAmount = max(dot(dirH, sunH), 0.0) * 0.5 + 0.5;
    sky += vec3(0.95, 0.42, 0.14) * twilight * pow(1.0 - h, 3.0) * sunAmount * 1.4;

    // 별 (밤하늘, 구름보다 먼저 그려 구름이 가리게 한다)
    if (night > 0.15 && dir.y > -0.05) {
        vec3  cell    = floor(dir * 220.0);
        float star    = step(0.9985, hash13(cell));
        float twinkle = 0.55 + 0.45 * sin(time * 3.0 + hash13(cell + 1.0) * 40.0);
        sky += vec3(0.80, 0.85, 1.0) * star * twinkle
             * night * smoothstep(-0.02, 0.10, dir.y) * 1.6;
    }

    // 태양 원반 + 헤일로
    float cosSun = clamp(dot(dir, toSun), -1.0, 1.0);
    vec3  sunCol = mix(vec3(1.0, 0.45, 0.15), vec3(1.0, 0.98, 0.92),
                       smoothstep(0.0, 0.30, sunE));
    float disc = smoothstep(0.99930, 0.99975, cosSun);
    float halo = pow(max(cosSun, 0.0), 180.0) * 0.5
               + pow(max(cosSun, 0.0), 8.0) * 0.12;
    sky += sunCol * (disc * 24.0 + halo) * smoothstep(-0.05, 0.02, sunE);

    // 달 원반 (태양의 반대 고도)
    vec3  toMoon  = vec3(toSun.x, -toSun.y, toSun.z);
    float cosMoon = clamp(dot(dir, toMoon), -1.0, 1.0);
    float moonUp  = step(0.0, -sunE);
    sky += vec3(0.90, 0.95, 1.0)
         * (smoothstep(0.99965, 0.99985, cosMoon) * 1.4
            + pow(max(cosMoon, 0.0), 64.0) * 0.05)
         * night * moonUp;

    // FBM 구름층 (고도 400m 평면과의 교차점에서 샘플링)
    if (cloudiness > 0.01 && dir.y > 0.015) {
        float t    = 400.0 / dir.y;
        vec2  cuv  = (camPos.xz + dir.xz * t) * 0.0009
                   + vec2(time * 0.006, time * 0.0023);
        float f    = fbm(cuv);
        float cov  = smoothstep(1.0 - cloudiness, 1.0 - cloudiness + 0.35, f);
        vec3  lit  = mix(vec3(0.015, 0.018, 0.028), vec3(1.0, 0.98, 0.95), day);
        lit        = mix(lit, vec3(1.0, 0.55, 0.30), twilight * 0.7);
        vec3  cCol = mix(lit, lit * 0.55, smoothstep(0.40, 0.90, f)); // 두꺼운 곳은 어둡게
        float fade = smoothstep(0.015, 0.12, dir.y); // 지평선 근처 페이드
        sky = mix(sky, cCol, cov * fade * 0.92);
    }

    // 지평선 아래: 지면 안개색으로 채운다.
    if (dir.y < 0.0) {
        sky = mix(sky, horizon * 0.85, smoothstep(0.0, -0.08, dir.y));
    }
    return sky;
}

// ---------------------------------------------------------------- 안개
// 시간대를 따라가는 안개색 (하늘 지평선색과 일치시켜 이음새를 없앤다)
vec3 envFogColor(vec4 lightDirW) {
    vec3  toSun    = envToSun(lightDirW);
    float day      = smoothstep(-0.08, 0.15, toSun.y);
    float twilight = exp(-abs(toSun.y) * 9.0);
    vec3  c        = mix(vec3(0.012, 0.018, 0.045), vec3(0.62, 0.76, 0.92), day);
    return c + vec3(0.95, 0.42, 0.14) * twilight * 0.35; // 황혼엔 주황빛 안개
}

// 지수 거리 안개. density <= 0 이면 안개 없음.
vec3 applyFog(vec3 col, float dist, vec4 lightDirW, float density) {
    if (density <= 0.0) return col;
    float f = 1.0 - exp(-dist * density);
    return mix(col, envFogColor(lightDirW), clamp(f, 0.0, 1.0));
}

// ---------------------------------------------------------------- 소프트 섀도
// 회전 Poisson 디스크 16탭 PCF. 1.0=완전히 밝음, 0.0=완전히 그림자.
const vec2 POISSON16[16] = vec2[](
    vec2(-0.94201624, -0.39906216), vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870), vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432), vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845), vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554), vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023), vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507), vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367), vec2( 0.14383161, -0.14100790));

float shadowVisPCF(sampler2D shadowMap, mat4 lightVP, vec3 worldPos, vec3 N, vec3 L) {
    // 노멀 오프셋으로 셰도 acne 를 줄인다.
    vec4 lc   = lightVP * vec4(worldPos + N * 0.02, 1.0);
    vec3 proj = lc.xyz / lc.w;
    vec2 uv   = proj.xy * 0.5 + 0.5;

    // 라이트 절두체 밖이거나 최대 깊이를 넘으면 그림자 없음으로 본다.
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z > 1.0) {
        return 1.0;
    }

    float bias    = max(0.0018 * (1.0 - dot(N, L)), 0.0006);
    float current = proj.z - bias;

    // 픽셀마다 디스크를 무작위 회전시켜 밴딩을 노이즈로 바꾼다.
    float ang    = hash12(gl_FragCoord.xy) * 6.2831853;
    vec2  rot    = vec2(cos(ang), sin(ang));
    float radius = 2.0 / float(textureSize(shadowMap, 0).x);

    // 1단계: 4탭 프리패스 — 완전히 밝거나 완전히 그림자면 여기서 끝낸다.
    // 화면 대부분은 반그림자(penumbra)가 아니므로 고해상도에서 큰 절약이 된다.
    float lit = 0.0;
    for (int i = 0; i < 4; ++i) {
        vec2 o = POISSON16[i];
        o = vec2(o.x * rot.x - o.y * rot.y,
                 o.x * rot.y + o.y * rot.x) * radius;
        lit += (current <= texture(shadowMap, uv + o).r) ? 1.0 : 0.0;
    }
    if (lit == 0.0) return 0.0;
    if (lit == 4.0) return 1.0;

    // 2단계: 반그림자 픽셀만 나머지 12탭으로 부드럽게 만든다.
    for (int i = 4; i < 16; ++i) {
        vec2 o = POISSON16[i];
        o = vec2(o.x * rot.x - o.y * rot.y,
                 o.x * rot.y + o.y * rot.x) * radius;
        lit += (current <= texture(shadowMap, uv + o).r) ? 1.0 : 0.0;
    }
    return lit / 16.0;
}
