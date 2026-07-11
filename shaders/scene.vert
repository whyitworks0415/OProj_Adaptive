#version 450

// 버텍스 입력
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inUV;

// 프래그먼트 셰이더 출력
layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out vec2 fragUV;

// 공통 UBO
layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos; // .xyz는 카메라 위치, .w는 사용하지 않는다.
} cam;

// 드로우별 모델 행렬과 재질 값을 push constant에 묶어 전달한다.
layout(push_constant) uniform PushConstants {
    mat4  model;
    vec4  baseColor;
    float shininess;
    float specularStrength;
    float reflectStrength;
    float textureIndex;
    vec4  emissive; // rgb=발광 색상, a=발광 강도
} pc;

void main() {
    vec4 worldPos = pc.model * vec4(inPosition, 1.0);
    fragPos       = worldPos.xyz;
    fragNormal    = normalize(mat3(transpose(inverse(pc.model))) * inNormal);
    fragColor     = inColor * pc.baseColor.rgb;
    fragUV        = inUV;

    gl_Position   = cam.proj * cam.view * worldPos;
}
