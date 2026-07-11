#version 450
// 블룸 2단계: 분리형 가우시안 블러 (push 의 dir 로 가로/세로를 재사용)
layout(location = 0) in  vec2 fragUV;
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform sampler2D srcTex;
layout(push_constant) uniform Push {
    vec4 p0; // xy = 텍셀 크기, zw = 블러 방향 (1,0)=가로 (0,1)=세로
    vec4 p1;
} pc;
void main() {
    const float W[5] = float[](0.227027, 0.194594, 0.121621, 0.054054, 0.016216);
    vec2 step = pc.p0.zw * pc.p0.xy * 1.6; // 1.6px 간격으로 반경을 넓힌다
    vec3 acc = texture(srcTex, fragUV).rgb * W[0];
    for (int i = 1; i < 5; ++i) {
        acc += texture(srcTex, fragUV + step * float(i)).rgb * W[i];
        acc += texture(srcTex, fragUV - step * float(i)).rgb * W[i];
    }
    outColor = vec4(acc, 1.0);
}
