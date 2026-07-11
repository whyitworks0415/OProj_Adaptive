#version 450
// 블룸 3단계: 블러 결과를 화면에 가산 합성 (파이프라인 블렌드 = ADD)
layout(location = 0) in  vec2 fragUV;
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform sampler2D bloomTex;
layout(push_constant) uniform Push {
    vec4 p0;
    vec4 p1; // y = 블룸 강도
} pc;
void main() {
    outColor = vec4(texture(bloomTex, fragUV).rgb * pc.p1.y, 1.0);
}
