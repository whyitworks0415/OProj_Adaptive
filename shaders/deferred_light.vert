#version 450

layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec3 viewRay; // 월드 공간 시선 방향 (비정규화, 프래그먼트에서 normalize)

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 lightDir;
    mat4 lightVP;
    vec4 skyParams;
    vec4 envParams;
} cam;

// 버텍스 버퍼 없이 전체화면 삼각형을 생성한다.
// 세 정점만으로 전체 뷰포트를 덮는다.
// 시선 레이는 정점에서 한 번만 역행렬로 계산해 보간한다
// (프래그먼트마다 inverse(mat4) 2회를 돌리면 고해상도에서 매우 비싸다).
void main() {
    vec2 uv     = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    fragUV      = uv;
    vec2 ndc    = uv * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);

    vec4 vp = inverse(cam.proj) * vec4(ndc, 1.0, 1.0);
    viewRay = mat3(inverse(cam.view)) * (vp.xyz / vp.w);
}
