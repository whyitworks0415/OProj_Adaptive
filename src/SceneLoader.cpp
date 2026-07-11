// ============================================================================
// SceneLoader.cpp — maps/ 디렉터리에서 맵 파일(GLB/GLTF) 탐색
// ============================================================================
// - 재귀 탐색으로 .glb/.gltf 만 수집 (.blend 원본 등은 무시)
// - 정렬 규칙: map/ 폴더 우선 → 자연수 순(natural sort: 2.glb < 10.glb)
//   TAB 키 순환과 HUD 목록이 이 순서를 따른다.
// ============================================================================
#include "SceneLoader.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <vector>

namespace {

std::string lowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string normalizedPath(std::filesystem::path path) {
    std::string p = path.generic_string();
    std::replace(p.begin(), p.end(), '\\', '/');
    return p;
}

bool isMapExtension(const std::string& ext) {
    // 맵 형식은 GLB/GLTF 로 통일 (.blend 원본 등은 무시)
    return ext == ".glb" || ext == ".gltf";
}

int mapDirectoryPriority(const std::string& path) {
    std::string p = lowerCopy(path);
    if (p.find("/map/") != std::string::npos) return 0;
    if (p.find("/maps/") != std::string::npos) return 1;
    return 2;
}

bool naturalLess(const std::string& a, const std::string& b) {
    int pa = mapDirectoryPriority(a);
    int pb = mapDirectoryPriority(b);
    if (pa != pb) return pa < pb;

    std::string la = lowerCopy(a);
    std::string lb = lowerCopy(b);
    size_t ia = 0, ib = 0;
    while (ia < la.size() && ib < lb.size()) {
        if (std::isdigit((unsigned char)la[ia]) && std::isdigit((unsigned char)lb[ib])) {
            size_t ea = ia;
            size_t eb = ib;
            while (ea < la.size() && std::isdigit((unsigned char)la[ea])) ++ea;
            while (eb < lb.size() && std::isdigit((unsigned char)lb[eb])) ++eb;

            std::string na = la.substr(ia, ea - ia);
            std::string nb = lb.substr(ib, eb - ib);
            na.erase(0, std::min(na.find_first_not_of('0'), na.size() - 1));
            nb.erase(0, std::min(nb.find_first_not_of('0'), nb.size() - 1));
            if (na.size() != nb.size()) return na.size() < nb.size();
            if (na != nb) return na < nb;
            ia = ea;
            ib = eb;
            continue;
        }
        if (la[ia] != lb[ib]) return la[ia] < lb[ib];
        ++ia;
        ++ib;
    }
    return la.size() < lb.size();
}

} // namespace

std::vector<std::string> findMapFiles(const std::string& dir) {
    std::vector<std::string> result;
    namespace fs = std::filesystem;

    std::error_code ec;
    for (auto& entry : fs::recursive_directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = lowerCopy(entry.path().extension().string());
        if (!isMapExtension(ext)) continue;
        result.push_back(normalizedPath(entry.path()));
    }

    std::sort(result.begin(), result.end(), naturalLess);
    return result;
}
