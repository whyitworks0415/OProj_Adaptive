#version 450
// 블룸 1단계: 밝은 픽셀만 추출 (하프 해상도로 다운샘플하며)
layout(location = 0) in  vec2 fragUV;
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform sampler2D srcTex;
layout(push_constant) uniform Push {
    vec4 p0; // xy = 소스 텍셀 크기, zw = (블러 방향 — 이 패스에선 미사용)
    vec4 p1; // x = 밝기 임계값, y = 강도(합성 패스용)
} pc;
void main() {
    vec3  c = texture(srcTex, fragUV).rgb;
    float l = max(c.r, max(c.g, c.b));
    // 임계값 이하 픽셀은 0, 그 위는 부드럽게(soft knee) 통과시킨다.
    float w = smoothstep(pc.p1.x, 1.0, l);
    outColor = vec4(c * w, 1.0);
}
