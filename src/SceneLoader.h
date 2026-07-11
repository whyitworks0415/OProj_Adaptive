#pragma once

#include <string>
#include <vector>

// SceneLoader: maps/ 디렉터리에서 렌더링 가능한 맵 파일을 탐색한다.
// 맵 형식은 GLB/GLTF 로 통일되어 있다 (Blender 내보내기 + Unity 비교 실험 호환).
// map/ 하위 폴더의 번호 맵을 목록 앞에 자연수 순으로 정렬한다.
std::vector<std::string> findMapFiles(const std::string& dir);
