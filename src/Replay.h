#pragma once
#include <vector>
#include <string>
#include <glm/glm.hpp>

// Replay: 카메라 경로 녹화와 재생
// 녹화: 매 프레임 카메라 위치·방향을 ReplayFrame 으로 저장
// 재생: 저장된 프레임을 시간순으로 보간하며 카메라에 적용

// 녹화된 카메라 스냅샷 한 장
struct ReplayFrame {
    float     time; // 녹화 시작으로부터의 경과 시간 (초)
    glm::vec3 pos; // 카메라 월드 위치
    float     yaw; // 수평 회전각 (도)
    float     pitch; // 수직 회전각 (도)
};

// 리플레이 바이너리 파일 포맷:
// magic   : char[4]    = "VRPL"          (파일 식별자)
// version : uint32     = 1               (버전)
// count   : uint32     = 프레임 수
// frames  : [time(f32), x,y,z(f32×3), yaw(f32), pitch(f32)] × count 형식으로 저장

// frames 를 path 에 바이너리로 저장. 실패 시 false 반환
bool saveReplay(const std::string& path, const std::vector<ReplayFrame>& frames);

// path 에서 바이너리를 읽어 frames 에 채워 넣음. 실패 시 false 반환
bool loadReplay(const std::string& path, std::vector<ReplayFrame>& frames);

// dir 안의 *.replay 파일 목록을 이름순으로 반환
std::vector<std::string> findReplayFiles(const std::string& dir);

// dir 안에서 특정 맵 전용 리플레이("<mapStem>_NNN.replay")만 이름순으로 반환
// (맵마다 같은 movement 로 벤치마크할 수 있도록 맵별로 경로를 분리 저장한다)
std::vector<std::string> findReplayFilesFor(const std::string& dir,
                                            const std::string& mapStem);

// dir/<mapStem>_001.replay, _002, ... 형식으로 해당 맵의 다음 번호 경로를 반환
std::string nextReplayPathFor(const std::string& dir, const std::string& mapStem);

// dir/replay_001.replay, _002, ... 형식으로 다음 번호의 경로를 반환
// (dir 가 없으면 생성)
std::string nextReplayPath(const std::string& dir);

// frames 를 다른 엔진(Unity 등)에서 읽을 수 있는 CSV 로 내보낸다.
// 형식: time_s,pos_x,pos_y,pos_z,yaw_deg,pitch_deg (헤더 1줄 + 프레임당 1줄)
// 좌표계는 이 렌더러 기준(오른손, Y-up, front = (cosP·cosY, sinP, cosP·sinY))이며
// 헤더 주석에 명시된다. 실패 시 false 반환
bool exportReplayCsv(const std::string& csvPath, const std::vector<ReplayFrame>& frames);

// dir 안의 모든 *.replay 에 대해 같은 이름의 .csv 가 없으면 만들어 준다.
// (기존에 녹화해 둔 리플레이도 Unity 비교 실험에 쓸 수 있게 함)
void exportMissingReplayCsvs(const std::string& dir);
