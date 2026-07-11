#version 450
#extension GL_GOOGLE_include_directive : require

// 절차적 하늘 전체화면 패스 (forward 경로).
// deferred_light.vert 의 전체화면 삼각형과 함께 사용하며,
// 깊이 테스트/쓰기 없이 렌더패스 맨 앞에서 그려 배경을 채운다.

layout(location = 0) in  vec2 fragUV;
layout(location = 1) in  vec3 viewRay; // 버텍스에서 보간된 월드 시선 방향
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos; // .xyz는 카메라 위치
    vec4 lightDir;  // .xyz=활성 광원 방향, .w=1이면 밤(달)
    mat4 lightVP;
    vec4 skyParams; // x=시간(초), y=구름량 0..1, z=안개 밀도, w=timeOfDay
} cam;

#include "common_env.glsl"

void main() {
    // 버텍스에서 보간된 시선 레이를 정규화 (역행렬 계산은 정점당 1회로 이동)
    vec3 dir = normalize(viewRay);

    vec3 sky = skyRadiance(dir, cam.lightDir, cam.cameraPos.xyz,
                           cam.skyParams.x, cam.skyParams.y);

    // 씬과 동일한 톤매핑·감마를 적용해 이음새를 없앤다.
    sky      = acesToneMap(sky);
    sky      = pow(sky, vec3(1.0 / 2.2));
    outColor = vec4(sky, 1.0);
}
