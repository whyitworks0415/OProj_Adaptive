// ============================================================================
// Camera.cpp — 1인칭 카메라 구현
// ============================================================================
// - yaw/pitch(도 단위)로부터 전방 벡터를 만들어 glm::lookAt 으로 뷰 행렬 생성
// - WASD/Space/Shift 이동: 눌린 키를 방향 벡터로 합산 후 정규화 (대각선 보정)
// - 시네마틱 모드: 입력은 target 값에 즉시 반영되고, 실제 위치/회전은
//   지수 보간(exponential smoothing)으로 타겟을 부드럽게 따라간다.
//   보간 계수 k = 1 - exp(-factor * dt) — 프레임레이트와 무관하게 같은 감속 곡선.
// ============================================================================
#include "Camera.h"
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>

// 피치를 제한해 짐벌락을 피한다.
static constexpr float kPitchLimit = 89.0f;

void Camera::clampPitch() {
    pitch = std::clamp(pitch, -kPitchLimit, kPitchLimit);
}

// 카메라 전방 벡터 계산
// yaw·pitch 로부터 단위 전방 벡터를 계산한다.
// 구면 좌표 변환:
// x = cos(pitch) * cos(yaw)
// y = sin(pitch)             ← 수직 성분
// z = cos(pitch) * sin(yaw)
glm::vec3 Camera::getCameraFront() const {
    float yawR   = glm::radians(yaw);
    float pitchR = glm::radians(pitch);
    return glm::normalize(glm::vec3{
        std::cos(pitchR) * std::cos(yawR),
        std::sin(pitchR),
        std::cos(pitchR) * std::sin(yawR)
    });
}

// 수평 전방 벡터 계산
// pitch 를 0으로 고정한 수평 전방 벡터.
// WASD 이동에 사용 -> 위아래를 바라보고 있어도 수평으로만 이동
glm::vec3 Camera::getHorizontalForward() const {
    float yawR = glm::radians(yaw);
    return glm::normalize(glm::vec3{std::cos(yawR), 0.0f, std::sin(yawR)});
}

// 수평 전방과 월드 업(Y) 의 외적으로 오른쪽 벡터를 구한다
glm::vec3 Camera::getRight() const {
    return glm::normalize(glm::cross(getHorizontalForward(), glm::vec3{0, 1, 0}));
}

// glm::lookAt 으로 뷰 행렬 반환 (position -> position+front, up=Y)
glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(position, position + getCameraFront(), glm::vec3{0, 1, 0});
}

// 외부에서 position/yaw/pitch 를 직접 수정한 직후에 호출.
// 시네마틱 모드의 누적 상태(타겟, 잔여 속도)를 현재 상태로 리셋해
// 토글하거나 텔레포트할 때 카메라가 옛 타겟으로 끌려가지 않게 한다.
void Camera::syncTarget() {
    targetYaw       = yaw;
    targetPitch     = pitch;
    currentVelocity = glm::vec3{0.0f};
}

// WASD/방향키 입력을 단위 방향 벡터로 묶고 속도를 곱해 desired velocity 를 만든다.
// 시네마틱 모드면 currentVelocity 를 desired 로 지수 보간하면서 position 을 적분.
// 비시네마틱 모드면 desired 를 그대로 적용 (= 기존 동작).
static void applyMovement(glm::vec3& position,
                          glm::vec3& currentVelocity,
                          const glm::vec3& desiredVelocity,
                          bool cinematic,
                          float positionSmoothFactor,
                          float dt) {
    if (cinematic) {
        // 1 - exp(-k*dt): 프레임레이트에 의존하지 않는 지수 보간 계수
        float k = 1.0f - std::exp(-positionSmoothFactor * dt);
        currentVelocity = glm::mix(currentVelocity, desiredVelocity, k);
        position += currentVelocity * dt;
    } else {
        currentVelocity = desiredVelocity;
        position += desiredVelocity * dt;
    }
}

// 시네마틱 모드에서 누적된 타겟 yaw/pitch 를 향해 yaw/pitch 를 지수 보간으로 끌어당김.
static void applyRotation(float& yaw, float& pitch,
                          float targetYaw, float targetPitch,
                          float rotationSmoothFactor, float dt) {
    float k = 1.0f - std::exp(-rotationSmoothFactor * dt);
    yaw   = glm::mix(yaw,   targetYaw,   k);
    pitch = glm::mix(pitch, targetPitch, k);
}

// WASD와 Space/Shift로 카메라를 이동한다.
// 왼쪽 Ctrl 키를 누르면 fastSpeed 로 전환 (탐색 시 빠르게 이동)
void Camera::processKeyboard(GLFWwindow* window, float dt) {
    bool  fast  = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS;
    float speed = fast ? fastSpeed : normalSpeed;

    glm::vec3 fwd   = getHorizontalForward();
    glm::vec3 right = getRight();

    // 눌린 키를 단위 방향 벡터로 합산한 뒤 정규화 -> 대각 이동이 √2 배 빨라지지 않음
    glm::vec3 dir{0.0f};
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) dir += fwd;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) dir -= fwd;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) dir -= right;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) dir += right;
    if (glfwGetKey(window, GLFW_KEY_SPACE)      == GLFW_PRESS) dir.y += 1.0f;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) dir.y -= 1.0f;

    glm::vec3 desiredVelocity{0.0f};
    if (glm::length(dir) > 1e-4f) desiredVelocity = glm::normalize(dir) * speed;

    applyMovement(position, currentVelocity, desiredVelocity,
                  cinematic, positionSmoothFactor, dt);

    if (cinematic)
        applyRotation(yaw, pitch, targetYaw, targetPitch, rotationSmoothFactor, dt);
}

// 방향키로 배치 카메라 위치를 조정한다.
// ghost 모드에서 원래 카메라 위치를 조정할 때 사용
void Camera::processArrowKeys(GLFWwindow* window, float dt) {
    glm::vec3 fwd   = getHorizontalForward();
    glm::vec3 right = getRight();

    glm::vec3 dir{0.0f};
    if (glfwGetKey(window, GLFW_KEY_UP)    == GLFW_PRESS) dir += fwd;
    if (glfwGetKey(window, GLFW_KEY_DOWN)  == GLFW_PRESS) dir -= fwd;
    if (glfwGetKey(window, GLFW_KEY_LEFT)  == GLFW_PRESS) dir -= right;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) dir += right;
    // Page Up / Page Down: 배치 카메라 수직 이동
    if (glfwGetKey(window, GLFW_KEY_PAGE_UP)   == GLFW_PRESS) dir.y += 1.0f;
    if (glfwGetKey(window, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS) dir.y -= 1.0f;

    glm::vec3 desiredVelocity{0.0f};
    if (glm::length(dir) > 1e-4f) desiredVelocity = glm::normalize(dir) * normalSpeed;

    applyMovement(position, currentVelocity, desiredVelocity,
                  cinematic, positionSmoothFactor, dt);

    if (cinematic)
        applyRotation(yaw, pitch, targetYaw, targetPitch, rotationSmoothFactor, dt);
}

// 마우스 이동량으로 yaw와 pitch를 갱신한다.
// dy 가 양수(마우스 아래로) -> pitch 감소(위를 봄) : 반전하여 직관적 조작
// 시네마틱 모드: 타겟에만 누적 -> 실제 yaw/pitch 는 processKeyboard 에서 보간으로 따라감
// 일반 모드: yaw/pitch 를 즉시 갱신하고 타겟도 동기화 (모드 전환 시 스냅 방지)
void Camera::processMouseDelta(float dx, float dy) {
    targetYaw   += dx * mouseSensitivity;
    targetPitch -= dy * mouseSensitivity; // dy 부호 반전: 마우스를 내리면 위를 봄
    targetPitch  = std::clamp(targetPitch, -kPitchLimit, kPitchLimit);

    if (!cinematic) {
        yaw   = targetYaw;
        pitch = targetPitch;
    }
}
