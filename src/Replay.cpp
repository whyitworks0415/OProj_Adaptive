// ============================================================================
// Replay.cpp — 카메라 경로 녹화/재생 + Unity 비교 실험용 CSV 내보내기
// ============================================================================
// 녹화: 매 프레임 (경과시간, 위치, yaw, pitch)를 ReplayFrame 으로 누적
//       → R 키 종료 시 replays/<맵이름>_NNN.replay(바이너리) + .csv(범용) 저장
// 재생: 시간 기반으로 인접 프레임을 선형 보간해 카메라에 적용 (VulkanApp 쪽)
// 파일이 맵별로 분리 저장되므로 "같은 맵 = 같은 movement" 벤치마크가 보장된다.
// CSV 는 Unity(BenchmarkApp.cs)가 그대로 읽어 동일 경로를 재생한다.
// ============================================================================
#include "Replay.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdio>

#ifdef _WIN32
#  include <windows.h>
#  include <direct.h>
#else
#  include <dirent.h>
#  include <sys/stat.h>
#endif

// saveReplay  –  녹화된 프레임 배열을 바이너리 파일로 저장
// 헤더: "VRPL" + 버전(1) + 프레임 수
// 본문: 각 프레임마다 [time, x, y, z, yaw, pitch] (모두 float32)
bool saveReplay(const std::string& path, const std::vector<ReplayFrame>& frames) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[Replay] Cannot write: " << path << "\n";
        return false;
    }

    // 파일 헤더 기록
    f.write("VRPL", 4);
    uint32_t ver = 1, count = (uint32_t)frames.size();
    f.write(reinterpret_cast<const char*>(&ver),   4);
    f.write(reinterpret_cast<const char*>(&count), 4);

    // 각 프레임의 카메라 상태를 순서대로 기록
    for (auto& fr : frames) {
        f.write(reinterpret_cast<const char*>(&fr.time),  4); // 경과 시간
        f.write(reinterpret_cast<const char*>(&fr.pos.x), 4); // X 위치
        f.write(reinterpret_cast<const char*>(&fr.pos.y), 4); // Y 위치
        f.write(reinterpret_cast<const char*>(&fr.pos.z), 4); // Z 위치
        f.write(reinterpret_cast<const char*>(&fr.yaw),   4); // 수평 회전
        f.write(reinterpret_cast<const char*>(&fr.pitch), 4); // 수직 회전
    }
    std::cout << "[Replay] Saved " << count << " frames to " << path << "\n";
    return true;
}

// loadReplay  –  바이너리 파일에서 프레임 배열을 읽어온다
// magic 검증 -> 버전/카운트 읽기 -> 프레임 배열 채우기
bool loadReplay(const std::string& path, std::vector<ReplayFrame>& frames) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[Replay] Cannot open: " << path << "\n";
        return false;
    }

    // magic 검증: 파일이 올바른 리플레이 포맷인지 확인
    char magic[4];
    f.read(magic, 4);
    if (magic[0]!='V'||magic[1]!='R'||magic[2]!='P'||magic[3]!='L') {
        std::cerr << "[Replay] Bad magic in " << path << "\n";
        return false;
    }

    // 버전·프레임 수 읽기 (버전은 현재 사용하지 않으나 하위 호환용으로 보존)
    uint32_t ver, count;
    f.read(reinterpret_cast<char*>(&ver),   4);
    f.read(reinterpret_cast<char*>(&count), 4);

    // 프레임 배열 크기 확보 후 데이터 채우기
    frames.resize(count);
    for (auto& fr : frames) {
        f.read(reinterpret_cast<char*>(&fr.time),  4);
        f.read(reinterpret_cast<char*>(&fr.pos.x), 4);
        f.read(reinterpret_cast<char*>(&fr.pos.y), 4);
        f.read(reinterpret_cast<char*>(&fr.pos.z), 4);
        f.read(reinterpret_cast<char*>(&fr.yaw),   4);
        f.read(reinterpret_cast<char*>(&fr.pitch), 4);
    }
    std::cout << "[Replay] Loaded " << count << " frames from " << path << "\n";
    return true;
}

// exportReplayCsv  –  프레임 배열을 범용 CSV 로 내보낸다 (Unity 비교 실험용)
bool exportReplayCsv(const std::string& csvPath, const std::vector<ReplayFrame>& frames) {
    std::ofstream f(csvPath);
    if (!f) {
        std::cerr << "[Replay] Cannot write CSV: " << csvPath << "\n";
        return false;
    }
    // 좌표계 규약을 파일 안에 명시해 다른 엔진에서 변환 실수를 막는다.
    // (주석은 인코딩 문제를 피하기 위해 영어로 쓴다)
    f << "# VulkanRenderer camera path export\n";
    f << "# coord: right-handed, Y-up, front = (cosP*cosY, sinP, cosP*sinY), angles in degrees\n";
    f << "# Unity (map imported via glTFast): use pos=(-x, y, z), flip x of front, then LookRotation\n";
    f << "time_s,pos_x,pos_y,pos_z,yaw_deg,pitch_deg\n";
    for (const auto& fr : frames) {
        f << fr.time  << ',' << fr.pos.x << ',' << fr.pos.y << ','
          << fr.pos.z << ',' << fr.yaw   << ',' << fr.pitch << '\n';
    }
    std::cout << "[Replay] Exported CSV: " << csvPath
              << " (" << frames.size() << " frames)\n";
    return true;
}

// exportMissingReplayCsvs  –  기존 *.replay 중 .csv 가 없는 것을 일괄 변환
void exportMissingReplayCsvs(const std::string& dir) {
    for (const std::string& path : findReplayFiles(dir)) {
        std::string csvPath = path.substr(0, path.size() - 7) + ".csv"; // ".replay" 제거
        std::ifstream exists(csvPath);
        if (exists.good()) continue;
        std::vector<ReplayFrame> frames;
        if (loadReplay(path, frames) && !frames.empty())
            exportReplayCsv(csvPath, frames);
    }
}

// findReplayFiles  –  dir 안의 *.replay 파일 목록을 이름순 정렬 후 반환
// Windows: Win32 FindFirstFile API 사용
// Linux/macOS: POSIX opendir/readdir 사용
std::vector<std::string> findReplayFiles(const std::string& dir) {
    std::vector<std::string> result;
#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    std::string pattern = dir + "\\*.replay";
    HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            // 디렉토리는 제외, 파일만 추가
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
                result.push_back(dir + "/" + fd.cFileName);
        } while (FindNextFileA(h, &fd));
        FindClose(h);
    }
#else
    DIR* d = opendir(dir.c_str());
    if (d) {
        struct dirent* e;
        while ((e = readdir(d))) {
            std::string n = e->d_name;
            if (n.size()>7 && n.substr(n.size()-7)==".replay")
                result.push_back(dir+"/"+n);
        }
        closedir(d);
    }
#endif
    std::sort(result.begin(), result.end()); // 이름순 정렬 -> 번호순과 동일
    return result;
}

// findReplayFilesFor  –  "<mapStem>_" 접두사가 붙은 리플레이만 필터링해 반환
std::vector<std::string> findReplayFilesFor(const std::string& dir,
                                            const std::string& mapStem) {
    std::vector<std::string> result;
    const std::string prefix = mapStem + "_";
    for (const std::string& path : findReplayFiles(dir)) {
        auto slash = path.find_last_of("/\\");
        std::string name = (slash != std::string::npos) ? path.substr(slash + 1) : path;
        if (name.rfind(prefix, 0) == 0)
            result.push_back(path);
    }
    return result;
}

// nextReplayPathFor  –  해당 맵의 기존 리플레이 수 +1 번호로 경로 생성
std::string nextReplayPathFor(const std::string& dir, const std::string& mapStem) {
#ifdef _WIN32
    _mkdir(dir.c_str());
#else
    mkdir(dir.c_str(), 0755);
#endif
    int next = (int)findReplayFilesFor(dir, mapStem).size() + 1;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "_%03d.replay", next);
    return dir + "/" + mapStem + buf;
}

// nextReplayPath  –  dir 안의 기존 파일 수에 +1 한 번호로 경로 생성
// 예: 기존 replay_001, _002 가 있으면 -> replay_003.replay 반환
// dir 가 없으면 mkdir 으로 생성
std::string nextReplayPath(const std::string& dir) {
    // 디렉토리 없으면 생성 (이미 있어도 무시됨)
#ifdef _WIN32
    _mkdir(dir.c_str());
#else
    mkdir(dir.c_str(), 0755);
#endif
    auto existing = findReplayFiles(dir);
    int  next = (int)existing.size() + 1; // 현재 파일 개수 + 1 = 다음 번호
    char buf[64];
    std::snprintf(buf, sizeof(buf), "/replay_%03d.replay", next);
    return dir + buf;
}
