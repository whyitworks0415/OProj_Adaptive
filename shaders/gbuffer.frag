#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragUV;

// 지오메트리 패스가 기록하는 G-Buffer 출력
layout(location = 0) out vec4 gAlbedo; // RGB=기본색, A=스페큘러 강도
layout(location = 1) out vec4 gNormal; // RGB=0..1로 인코딩한 법선, A=shininess/256
layout(location = 2) out vec4 gPosition; // RGB=월드 위치, A=1.0+발광 강도(유효 픽셀 표시)
layout(location = 3) out vec4 gMaterial; // R=반사 강도

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos; // .xyz는 카메라 위치, .w는 예비
    vec4 lightDir;
} cam;

layout(set = 0, binding = 1) uniform sampler2D textures[64];

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

void main() {
    vec3 N = normalize(fragNormal);

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

    // Blender metallicRoughness 텍스처(G=러프니스, B=메탈릭)를 퍼픽셀로 반영해
    // 조명 패스가 읽는 shininess/specStr 인코딩을 갱신한다.
    float shininess = pc.shininess;
    float specStr   = pc.specularStrength;
    int   mrIdx     = int(floor(pc.texIndices2.x + 0.5));
    if (mrIdx >= 0) {
        vec4  mr    = texture(textures[mrIdx], fragUV);
        // 팩터 기반 러프니스/메탈릭을 복원해 텍스처와 곱한 뒤 재인코딩
        float rough = clamp(sqrt(clamp((256.0 - shininess) / 252.0, 0.0, 1.0)), 0.0, 1.0) * mr.g;
        float metal = clamp((specStr - 0.05) / 0.90, 0.0, 1.0) * mr.b;
        shininess = mix(256.0, 4.0, rough * rough);
        specStr   = mix(0.05, 0.95, metal);
    }

    // 발광 강도: emissive 텍스처가 있으면 팩터와 곱해 퍼픽셀 luminance 를 만든다.
    vec3 emis    = pc.emissive.rgb;
    int  emisIdx = int(floor(pc.texIndices2.y + 0.5));
    float emissiveLum;
    if (emisIdx >= 0) {
        emis *= texture(textures[emisIdx], fragUV).rgb;
        emissiveLum = max(max(emis.r, emis.g), emis.b);
    } else {
        emissiveLum = pc.emissive.a;
    }

    gAlbedo   = vec4(albedo, specStr);
    gNormal   = vec4(N * 0.5 + 0.5, shininess / 256.0);
    // gPosition.w가 1.0 이상이면 유효 픽셀이고, 0.0이면 배경이다.
    gPosition = vec4(fragPos, 1.0 + emissiveLum);
    gMaterial = vec4(pc.reflectStrength, 0.0, 0.0, 0.0);
}
