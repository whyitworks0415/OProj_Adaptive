#version 450

// 버텍스 입력
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inUV;
layout(location = 4) in mat4 inModel;

// 프래그먼트 셰이더 출력
layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out vec2 fragUV;

// 카메라 UBO(set 0, binding 0)
layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
} cam;

// Push constants keep the same layout as the non-instanced pipeline.
layout(push_constant) uniform PushConstants {
    mat4  model; // Unused here; the per-instance matrix is inModel.
    vec4  baseColor;
    float shininess;
    float specularStrength;
    float reflectStrength;
    float textureIndex;
    vec4  emissive; // rgb=발광 색상, a=발광 강도
} pc;

void main() {
    mat4 model    = inModel;
    vec4 worldPos = model * vec4(inPosition, 1.0);

    fragPos    = worldPos.xyz;
    fragNormal = normalize(mat3(transpose(inverse(model))) * inNormal);
    fragColor  = inColor * pc.baseColor.rgb;
    fragUV     = inUV;

    gl_Position = cam.proj * cam.view * worldPos;
}
