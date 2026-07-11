#pragma once
#include <cstdint>
#include <glm/glm.hpp>

// MeshRange: 글로벌 버퍼 안의 메시 범위
// Vulkan 은 단일 vertex+index 버퍼에 모든 메시를 이어붙인 후
// vkCmdDrawIndexed(indexCount, 1, indexStart, ...) 로 부분 드로우한다.
// MeshRange 는 그 '부분 범위' 를 기록하는 구조체다.
// (맵 로딩은 GLTFLoader 로 통일되었고, 이 구조체는 오클루전 쿼리용
//  절차적 박스 등 내부 지오메트리 범위 기록에 사용된다.)
struct MeshRange {
    uint32_t  indexStart; // indices 배열에서 이 메시가 시작하는 오프셋
    uint32_t  indexCount; // 이 메시의 인덱스 수 (삼각형 수 × 3)
    glm::vec3 bboxMin; // 축 정렬 경계 박스 최솟값 (원본 좌표계)
    glm::vec3 bboxMax; // 축 정렬 경계 박스 최댓값 (원본 좌표계)
};
