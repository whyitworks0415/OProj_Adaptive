#version 450
// 포스트프로세스 공용 풀스크린 삼각형 (UBO 불필요 버전)
layout(location = 0) out vec2 fragUV;
void main() {
    vec2 uv     = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    fragUV      = uv;
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
