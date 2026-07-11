#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal; // 공통 Vertex 바인딩을 맞추기 위해 남겨둔다.
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
} cam;

void main() {
    gl_Position = cam.proj * cam.view * vec4(inPosition, 1.0);
    fragColor   = inColor;
}
