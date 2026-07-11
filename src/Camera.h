#pragma once
#include <glm/glm.hpp>

struct GLFWwindow;

// Camera: 1인칭 시점 카메라
// 좌표계: Y-up (위쪽이 +Y).
// yaw=0° -> +X 방향, yaw=90° -> +Z 방향 (기본값).
// pitch 범위: -89° ~ +89° (gimbal lock 방지).
class Camera {
public:
    glm::vec3 position        = {0.0f, 1.8f, -8.0f}; // 월드 공간 위치 (눈 높이 기준)
    float     yaw             =  90.0f; // 수평 회전각 (도). +90 -> +Z 방향을 바라봄
    float     pitch           =  -5.0f; // 수직 회전각 (도). 음수 = 약간 아래를 봄
    float     normalSpeed     =  5.0f; // 일반 이동 속도 (m/s)
    float     fastSpeed       = 20.0f; // 왼쪽 Ctrl 키 누를 때 빠른 이동 속도 (m/s)
    float     mouseSensitivity = 0.12f; // 마우스 픽셀당 회전각

    // 시네마틱 카메라 모드: 위치/회전을 지수 보간으로 부드럽게 따라가게 함.
    // C 키로 on/off. 입력은 "타겟" 값에 즉시 반영되고, 실제 카메라 상태는
    // 매 프레임 타겟을 향해 천천히 수렴한다.
    bool      cinematic              = false;
    float     positionSmoothFactor   =  6.0f; // 클수록 위치가 빠르게 따라감 (1/τ)
    float     rotationSmoothFactor   = 10.0f; // 클수록 회전이 빠르게 따라감

    // 매 프레임 호출 – WASD/Space/Shift 키로 카메라 이동
    void processKeyboard(GLFWwindow* window, float dt);

    // 배치된(placed) 카메라를 방향키로 이동 (ghost 모드에서 원래 카메라 제어)
    void processArrowKeys(GLFWwindow* window, float dt);

    // 마우스 이동 델타(픽셀)를 받아 yaw·pitch 업데이트
    void processMouseDelta(float dx, float dy);

    // 외부에서 position/yaw/pitch 를 직접 갈아끼운 직후에 호출해
    // 시네마틱 보간 타겟을 현재 값에 다시 맞춘다 (스냅 백 방지).
    void syncTarget();

    // 현재 yaw·pitch 에서 뷰 행렬 계산 (glm::lookAt)
    glm::mat4 getViewMatrix() const;

    // pitch 포함 완전한 3D 전방 벡터 (렌더링·조준에 사용)
    glm::vec3 getCameraFront() const;

    // pitch 를 무시한 수평 전방 벡터 (WASD 이동에 사용 – 경사면에서 뜨지 않음)
    glm::vec3 getHorizontalForward() const;

    // 오른쪽 벡터 (수평 전방 × 월드 업)
    glm::vec3 getRight() const;

private:
    // pitch 를 ±89° 이내로 클램프 (90° 이상이면 화면이 뒤집힘)
    void clampPitch();

    // 시네마틱 모드 내부 상태
    glm::vec3 currentVelocity = {0.0f, 0.0f, 0.0f}; // 이전 프레임까지 누적된 실제 이동 속도
    float     targetYaw       = 90.0f; // 누적되는 마우스 입력 (= 즉시 반영된 yaw)
    float     targetPitch     = -5.0f; // 누적되는 마우스 입력 (= 즉시 반영된 pitch)
};
