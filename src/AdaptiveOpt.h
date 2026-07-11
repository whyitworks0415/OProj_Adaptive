#pragma once

// AdaptiveOptimizer — 상황 적응형 최적화 컨트롤러 (` 키로 토글)
//
// 아이디어: 각 최적화 기법은 상황(씬·시점·오브젝트 분포)에 따라 이득일 수도,
// 오히려 탐색 오버헤드가 더 클 수도 있다. 이 컨트롤러는 실행 중에 기법 하나씩
// ON/OFF 를 번갈아 실측(A/B 테스트)해서, 실제 프레임 시간이 짧아지는 쪽만
// 유지한다 — "탐색 비용보다 프레임 이득이 큰 기법만 적용".
//
// 측정 설계:
//  - 한 기법당 A(현재 상태)/B(반전 상태) 서브윈도를 인터리브(A B A B)로 측정해
//    카메라 이동 같은 점진적 부하 변화가 한쪽 윈도에만 쏠리는 것을 상쇄한다.
//  - 서브윈도는 "최소 프레임 수 + 최소 시간"을 모두 채워야 끝난다 — 수천 FPS
//    상황에서 표본 수집이 몇 ms 만에 끝나 노이즈에 휘둘리는 것을 방지.
//  - 서브윈도 전환 직후 몇 프레임은 버린다 (파이프라인/오클루전 쿼리 지연 안정화).
//  - 프레임 시간은 가운데 50% 절사 평균으로 집계 — 스파이크에 강인.
//
// 판정 (노이즈 방지 3중 필터 — 셋 다 통과해야 상태를 바꾼다):
//  1. 상대 차이 >= HYST_PCT (%)
//  2. 절대 차이 >= MIN_DELTA_MS (ms) — 프레임타임이 1ms 미만인 고FPS 구간에서
//     %-기준만으로는 노이즈가 유의미해 보이는 문제를 차단
//  3. 라운드 간 부호 일관성 — 1라운드(A1 vs B1)와 2라운드(A2 vs B2)의 우열이
//     서로 반대면 판단 보류 (현 상태 유지)
//
// 탐색 비용 관리 (측정 자체가 프레임을 깎지 않도록):
//  - 반전(B) 상태가 명백히 나쁘면(평균 프레임타임 EARLY_RATIO 배 초과) 남은
//    측정을 건너뛰고 즉시 원상 판정 — 손해 상태로 머무는 시간에 상한을 둔다.
//  - 회전/이동이 크거나 히치(>100ms)가 끼면 그 테스트는 폐기하고 원상 복구.
//  - 카메라가 STILL_SEC 이상 조용할 때만 테스트를 시작한다 — 이동 중에 시작한
//    테스트는 어차피 폐기되므로 순수 낭비다.
//  - 전체 스윕이 끝나면 타이머 재측정을 하지 않는다. 시점이 크게 바뀌었을 때만
//    (RESWEEP_* 임계) 재스윕하고, 그 외에는 HEARTBEAT_SEC 간격으로 기법 하나씩만
//    저빈도 점검한다 (낮/밤 주기처럼 서서히 변하는 부하 대비).
//    → 같은 자리에 있으면 탐색 비용이 사실상 0 이 된다.
//
// 대상 기법: 화면 결과가 (거의) 달라지지 않는 기법만 자동 제어한다.
// 원거리 컬링(7)·소형 오브젝트 컬링(8)은 오브젝트가 눈에 띄게 사라지는
// 품질 트레이드오프 기법이라 대상에서 제외 — 수동 키로만 조작한다.
//
// 이 헤더는 OptFlags 정의 직후(VulkanApp.h 내부)에 include 되어야 한다.

#include <algorithm>
#include <cmath>
#include <vector>

#include <glm/glm.hpp>

class AdaptiveOptimizer {
public:
    static constexpr int   NUM_TECH        = 7;
    static constexpr int   WARM_FRAMES     = 6;     // 서브윈도 전환 후 버리는 프레임
    static constexpr int   WARM_FRAMES_OCC = 10;    // 오클루전 ON 전환은 쿼리 지연만큼 추가 예열
    static constexpr int   MEASURE_FRAMES  = 14;    // 서브윈도당 최소 측정 프레임 수
    static constexpr float WINDOW_MIN_MS   = 40.0f; // 서브윈도당 최소 측정 시간 (ms)
    static constexpr int   ROUNDS          = 2;     // A/B 인터리브 반복 (A B A B)
    static constexpr float HYST_PCT        = 2.0f;  // 판정 필터 1: 상대 차이 하한 (%)
    static constexpr float MIN_DELTA_MS    = 0.08f; // 판정 필터 2: 절대 차이 하한 (ms)
    static constexpr float EARLY_RATIO     = 1.30f; // 조기 원상 판정: B가 A보다 이 배율 이상 느림
    static constexpr int   EARLY_MIN_SAMPLES = 10;  // 조기 판정에 필요한 최소 B 표본
    static constexpr float EARLY_MIN_MS    = 15.0f; // 조기 판정에 필요한 최소 B 측정 시간 (ms)
    static constexpr float SWEEP_GAP_SEC   = 0.15f; // 스윕 중 테스트 간 간격 (초)
    static constexpr float HEARTBEAT_SEC   = 30.0f; // 안정 상태에서 기법 1개 저빈도 점검 간격
    static constexpr float RESWEEP_POS     = 8.0f;  // 재스윕 트리거: 스윕 시점 대비 이동 (m)
    static constexpr float RESWEEP_YAW     = 30.0f; // 재스윕 트리거: 회전 (도)
    static constexpr float RESWEEP_PITCH   = 15.0f;
    static constexpr float STILL_POS       = 0.8f;  // "정지" 판정 임계 (m / 도 / 초)
    static constexpr float STILL_ANG       = 5.0f;
    static constexpr float STILL_SEC       = 0.35f;
    static constexpr float MAX_YAW_DRIFT   = 25.0f; // 테스트 폐기 기준 회전량 (도)
    static constexpr float MAX_PITCH_DRIFT = 20.0f;
    static constexpr float MAX_POS_DRIFT   = 15.0f; // 테스트 폐기 기준 이동량 (m)
    static constexpr float HITCH_MS        = 100.0f;

    enum class Phase { Idle, Warmup, Measure };

    struct Tech {
        const char*      name;
        const char*      keyChip;    // 대응하는 수동 토글 키 (오버레이 표기용)
        bool OptFlags::* flag;
        bool  applicable = true;  // 씬에서 의미가 있는지 (LOD 메시/인스턴싱 그룹 존재 여부)
        bool  tested     = false; // 유효한 측정 결과 보유 여부
        float fpsGainPct = 0.f;   // ON이 OFF 대비 몇 % 빠른가 (음수 = ON이 손해)
        int   discards   = 0;     // 이동/히치로 폐기된 테스트 수
    };

    bool enabled = false;

    Tech techs[NUM_TECH] = {
        { "Frustum Culling",  "1", &OptFlags::frustumCulling   },
        { "LOD",              "2", &OptFlags::lod              },
        { "GPU Instancing",   "3", &OptFlags::instancing       },
        { "Backface Culling", "4", &OptFlags::backfaceCulling  },
        { "Depth Sort",       "5", &OptFlags::depthSort        },
        { "Occlusion Cull",   "6", &OptFlags::occlusionCulling },
        { "Deferred Shading", "9", &OptFlags::deferredShading  },
    };

    // 오버레이가 읽는 진행 상태
    Phase phase     = Phase::Idle;
    int   testTech  = -1;    // 측정 중인 기법 인덱스 (-1 = 대기)
    int   subWindow = 0;     // 0..2*ROUNDS-1 (짝수 = A 상태, 홀수 = B 상태)
    bool  paused    = false; // FPS 캡/벤치마크 중 → 측정 불가
    bool  sweepDone = false; // 적용 가능한 전 기법을 한 번씩 측정 완료했는지
    float stillSec  = 0.f;   // 카메라가 조용히 있었던 시간 (초)

    int curWindowSamples() const {
        return (testTech < 0) ? 0 : (int)win[subWindow].size();
    }

    void setApplicability(bool hasLod, bool hasInstancing) {
        techs[1].applicable = hasLod;
        techs[2].applicable = hasInstancing;
    }

    // 진행 중 테스트를 중단하고(플래그 원복) 모든 측정 기록을 무효화한다.
    // 씬 교체·스트레스 배율 변경·모드 토글 시 호출.
    void reset(OptFlags& flags) {
        abortTest(flags);
        for (Tech& t : techs) { t.tested = false; t.fpsGainPct = 0.f; t.discards = 0; }
        sweepDone    = false;
        sweepCursor  = 0;
        idleTimer    = 0.f;
        heartbeatSec = 0.f;
        stillSec     = 0.f;
    }

    // 매 프레임 호출. dtMs = 이번 프레임 시간(ms).
    // pauseNow = 측정이 무의미한 상황 (FPS 캡 고정 프레임 모드, 벤치마크 진행 등).
    void update(OptFlags& flags, float dtMs, bool pauseNow,
                const glm::vec3& camPos, float camYaw, float camPitch) {
        paused = pauseNow;
        if (!enabled) return;
        if (paused) { abortTest(flags); return; }
        curPos = camPos; curYaw = camYaw; curPitch = camPitch;

        if (testTech < 0) { idleStep(flags, dtMs); return; }

        Tech& t = techs[testTech];

        // 테스트 폐기 조건: 히치 프레임, 큰 회전/이동 — 부하 자체가 변해 A/B 비교가 무의미
        if (dtMs > HITCH_MS ||
            std::fabs(angleDiff(curYaw, startYaw)) > MAX_YAW_DRIFT ||
            std::fabs(curPitch - startPitch)       > MAX_PITCH_DRIFT ||
            glm::length(curPos - startPos)         > MAX_POS_DRIFT) {
            ++t.discards;
            abortTest(flags);
            return;
        }

        if (warmLeft > 0) { --warmLeft; phase = Phase::Warmup; winMs = 0.f; return; }
        phase = Phase::Measure;

        win[subWindow].push_back(dtMs);
        winMs += dtMs;

        // 조기 원상 판정: 반전(B) 상태가 명백히 나쁘면 남은 측정을 건너뛴다 —
        // 손해 보는 상태로 머무는 시간에 상한을 둔다 (탐색 비용 관리의 핵심).
        if ((subWindow & 1) &&
            (int)win[subWindow].size() >= EARLY_MIN_SAMPLES && winMs >= EARLY_MIN_MS) {
            float mA = poolMean(0), mB = poolMean(1);
            if (mA > 0.f && mB > mA * EARLY_RATIO) { decide(flags); return; }
        }

        // 프레임 수와 시간 둘 다 채워야 서브윈도 종료 (고FPS 노이즈 방지)
        if ((int)win[subWindow].size() < MEASURE_FRAMES || winMs < WINDOW_MIN_MS) return;

        ++subWindow;
        winMs = 0.f;
        if (subWindow >= 2 * ROUNDS) { decide(flags); return; }

        bool nextState = (subWindow & 1) ? !stateA : stateA;
        flags.*(t.flag) = nextState;
        warmLeft = warmFramesFor(testTech, nextState);
    }

private:
    std::vector<float> win[4]; // 서브윈도별 표본: [0]=A1 [1]=B1 [2]=A2 [3]=B2
    float winMs       = 0.f;   // 현재 서브윈도의 누적 측정 시간 (ms)
    int   sweepCursor = 0;     // 다음에 측정할 기법 (라운드로빈)
    int   warmLeft    = 0;     // 현재 서브윈도에서 남은 예열 프레임
    bool  stateA      = false; // 테스트 시작 시점의 플래그 값 (A 서브윈도 상태)
    float idleTimer   = 0.f;
    float heartbeatSec = 0.f;

    glm::vec3 curPos{};   float curYaw = 0.f,   curPitch = 0.f;   // 이번 프레임 카메라
    glm::vec3 startPos{}; float startYaw = 0.f, startPitch = 0.f; // 테스트 시작 시점
    glm::vec3 stillPos{}; float stillYaw = 0.f, stillPitch = 0.f; // 정지 감지 앵커
    glm::vec3 anchorPos{};float anchorYaw = 0.f,anchorPitch = 0.f;// 마지막 스윕 완료 시점

    static float angleDiff(float a, float b) {
        float d = std::fmod(a - b, 360.f);
        if (d >  180.f) d -= 360.f;
        if (d < -180.f) d += 360.f;
        return d;
    }

    static float mean(const std::vector<float>& v) {
        if (v.empty()) return 0.f;
        float s = 0.f;
        for (float x : v) s += x;
        return s / (float)v.size();
    }

    // A(짝수 윈도) 또는 B(홀수 윈도) 풀링 평균
    float poolMean(int side) const {
        float s = 0.f; size_t n = 0;
        for (int w = side; w < 4; w += 2) {
            for (float x : win[w]) s += x;
            n += win[w].size();
        }
        return n ? s / (float)n : 0.f;
    }

    int warmFramesFor(int idx, bool turningOn) const {
        // 오클루전 컬링은 켠 직후 쿼리 결과가 몇 프레임 늦게 도착하므로 더 길게 예열
        if (techs[idx].flag == &OptFlags::occlusionCulling && turningOn)
            return WARM_FRAMES_OCC;
        return WARM_FRAMES;
    }

    void idleStep(OptFlags& flags, float dtMs) {
        phase = Phase::Idle;

        // 정지 감지: 앵커에서 조금이라도 벗어나면 앵커를 옮기고 타이머 리셋
        if (glm::length(curPos - stillPos) > STILL_POS ||
            std::fabs(angleDiff(curYaw, stillYaw)) > STILL_ANG ||
            std::fabs(curPitch - stillPitch)       > STILL_ANG) {
            stillPos = curPos; stillYaw = curYaw; stillPitch = curPitch;
            stillSec = 0.f;
        } else {
            stillSec += dtMs * 0.001f;
        }

        idleTimer += dtMs * 0.001f;

        if (sweepDone) {
            // 시점이 크게 바뀌면(다른 공간으로 이동) 전체 재스윕 예약
            if (glm::length(curPos - anchorPos) > RESWEEP_POS ||
                std::fabs(angleDiff(curYaw, anchorYaw)) > RESWEEP_YAW ||
                std::fabs(curPitch - anchorPitch)       > RESWEEP_PITCH) {
                for (Tech& t : techs) t.tested = false;
                sweepDone = false;
            } else {
                // 같은 자리: 저빈도 하트비트 점검만 (낮/밤처럼 서서히 변하는 부하 대비)
                heartbeatSec += dtMs * 0.001f;
            }
        }

        bool wantTest = !sweepDone || heartbeatSec >= HEARTBEAT_SEC;
        if (wantTest && stillSec >= STILL_SEC && idleTimer >= SWEEP_GAP_SEC) {
            heartbeatSec = 0.f;
            beginNextTest(flags);
        }
    }

    void beginNextTest(OptFlags& flags) {
        int idx = -1;
        for (int i = 0; i < NUM_TECH; ++i) {
            int c = (sweepCursor + i) % NUM_TECH;
            if (techs[c].applicable && (sweepDone || !techs[c].tested)) { idx = c; break; }
        }
        if (idx < 0) { // 스윕 중인데 남은 미측정 기법이 없으면 아무거나 순환
            for (int i = 0; i < NUM_TECH; ++i) {
                int c = (sweepCursor + i) % NUM_TECH;
                if (techs[c].applicable) { idx = c; break; }
            }
        }
        if (idx < 0) return; // 적용 가능한 기법이 없음
        sweepCursor = (idx + 1) % NUM_TECH;

        testTech  = idx;
        stateA    = flags.*(techs[idx].flag);
        subWindow = 0;
        winMs     = 0.f;
        for (auto& w : win) w.clear();
        warmLeft  = warmFramesFor(idx, stateA);
        phase     = Phase::Warmup;
        startPos  = curPos;
        startYaw  = curYaw;
        startPitch = curPitch;
        idleTimer = 0.f;
    }

    void abortTest(OptFlags& flags) {
        if (testTech >= 0) flags.*(techs[testTech].flag) = stateA; // 원상 복구
        testTech  = -1;
        phase     = Phase::Idle;
        idleTimer = 0.f;
        winMs     = 0.f;
    }

    // 가운데 50% 절사 평균 — 스파이크/순간 최저치에 강인
    static float trimmedMean(std::vector<float>& v) {
        if (v.empty()) return 0.f;
        std::sort(v.begin(), v.end());
        size_t lo = v.size() / 4, hi = v.size() - v.size() / 4;
        if (lo >= hi) { lo = 0; hi = v.size(); }
        float sum = 0.f;
        for (size_t i = lo; i < hi; ++i) sum += v[i];
        return sum / float(hi - lo);
    }

    void decide(OptFlags& flags) {
        Tech& t = techs[testTech];

        std::vector<float> A = win[0], B = win[1];
        A.insert(A.end(), win[2].begin(), win[2].end());
        B.insert(B.end(), win[3].begin(), win[3].end());
        float tA = trimmedMean(A);
        float tB = trimmedMean(B);
        if (tA <= 0.f || tB <= 0.f) { abortTest(flags); return; }

        float tOn  = stateA ? tA : tB;
        float tOff = stateA ? tB : tA;

        // ON이 OFF 대비 몇 % 빠른가 (프레임 시간 비 → FPS 이득으로 환산)
        float gain   = (tOff / tOn - 1.f) * 100.f;
        t.fpsGainPct = gain;
        t.tested     = true;

        // 노이즈 필터 3종: 라운드 간 부호 일관성 + 상대 차이 + 절대 차이.
        // 하나라도 미달이면 테스트 전 상태 유지 (플래핑 방지).
        bool consistent = true;
        if (!win[2].empty() && !win[3].empty()) {
            float d1 = mean(win[0]) - mean(win[1]);
            float d2 = mean(win[2]) - mean(win[3]);
            consistent = (d1 * d2) > 0.f;
        }
        bool significant = consistent &&
                           std::fabs(gain) >= HYST_PCT &&
                           std::fabs(tOff - tOn) >= MIN_DELTA_MS;
        flags.*(t.flag) = significant ? (gain > 0.f) : stateA;

        testTech  = -1;
        phase     = Phase::Idle;
        idleTimer = 0.f;

        if (!sweepDone) {
            sweepDone = true;
            for (const Tech& tt : techs)
                if (tt.applicable && !tt.tested) { sweepDone = false; break; }
            if (sweepDone) {
                // 스윕 완료 시점의 시점(視點)을 앵커로 저장 —
                // 여기서 크게 벗어날 때만 재스윕한다.
                anchorPos = curPos; anchorYaw = curYaw; anchorPitch = curPitch;
            }
        }
    }
};
