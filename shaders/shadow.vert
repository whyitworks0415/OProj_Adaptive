#version 450

// 섀도맵 깊이 패스: 라이트 시점에서 깊이만 기록한다.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inUV;

// 라이트 뷰·투영 × 모델 행렬을 push constant로 전달한다.
layout(push_constant) uniform ShadowPush {
    mat4 lightMVP;
} pc;

void main() {
    gl_Position = pc.lightMVP * vec4(inPosition, 1.0);
}
