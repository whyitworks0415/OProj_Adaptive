// ============================================================================
// PerformanceStats.cpp — FPS/CPU/RAM/GPU 실시간 수집 (Windows 전용 API)
// ============================================================================
// - FPS: dt 누적 롤링 평균 (sampleInterval=0.5s 마다 확정)
// - 시스템 CPU: GetSystemTimes 의 idle/kernel/user FILETIME 델타로 계산
// - 프로세스 CPU: GetProcessTimes 델타 / 시스템 총량 × 100
// - RAM: PSAPI GetProcessMemoryInfo 의 WorkingSetSize
// - GPU: PDH 성능 카운터 "GPU Engine(*engtype_3D) % Utilization" 인스턴스 합산
//   (드라이버/OS 가 지원하지 않으면 -1 로 표시 = HUD 에서 N/A)
// 폴링은 0.5초 주기 — 매 프레임 API 호출은 그 자체가 오버헤드이기 때문.
// ============================================================================
#include "PerformanceStats.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "pdh.lib")

#include <algorithm>
#include <cstdlib>

// init  –  앱 시작 시 1회 호출
// CPU 델타 계산을 위한 기준값(이전 FILETIME)을 지금 시점으로 초기화하고
// GPU PDH 쿼리를 열어 첫 번째 수집(시드)을 수행한다.
void PerformanceStats::init() {
    // 시스템 전체 CPU 기준값 시드
    // FILETIME 두 개(이번 - 이전)의 차이로 점유율을 계산하므로, 처음엔 현재 값을 저장
    FILETIME idle, kernel, user;
    GetSystemTimes(&idle, &kernel, &user);
    lastIdle   = (unsigned long long)idle.dwHighDateTime   << 32 | idle.dwLowDateTime;
    lastKernel = (unsigned long long)kernel.dwHighDateTime << 32 | kernel.dwLowDateTime;
    lastUser   = (unsigned long long)user.dwHighDateTime   << 32 | user.dwLowDateTime;
    lastSysTotal = (lastKernel + lastUser); // sampleProcessCPU() 가 재사용

    // 이 프로세스 CPU 기준값 시드
    FILETIME ct, et, kt, ut;
    GetProcessTimes(GetCurrentProcess(), &ct, &et, &kt, &ut);
    lastProcKernel = (unsigned long long)kt.dwHighDateTime << 32 | kt.dwLowDateTime;
    lastProcUser   = (unsigned long long)ut.dwHighDateTime << 32 | ut.dwLowDateTime;

    initGPU(); // PDH 쿼리 열기
}

// PDH 쿼리를 닫아 GPU 카운터 리소스를 해제한다
void PerformanceStats::cleanup() {
    if (pdhQuery) {
        PdhCloseQuery((PDH_HQUERY)pdhQuery);
        pdhQuery   = nullptr;
        gpuCounter = nullptr;
    }
}

// update  –  매 프레임 호출
// frameTimeMs 는 즉시 갱신, FPS/CPU/RAM/GPU 는 sampleInterval 마다 폴링
void PerformanceStats::update(float dt, float sampleInterval) {
    frameTimeMs = dt * 1000.0f; // 직전 프레임 시간을 밀리초로 변환

    // FPS 롤링 평균: 1/dt 를 누산하다가 sampleInterval 이 되면 평균 계산
    fpsAccum += 1.0f / dt;
    fpsCount++;
    pollAccum += dt;

    if (pollAccum >= sampleInterval) {
        fps      = fpsAccum / fpsCount; // 구간 평균 FPS
        fpsAccum = 0.0f;
        fpsCount = 0;

        // 각 항목을 OS API 로 폴링 (0.5초마다 -> 오버헤드 최소화)
        sampleCPU(); // 시스템 전체 CPU
        sampleProcessCPU(); // 이 프로세스 CPU
        sampleRAM(); // 메모리 사용량
        sampleGPU(); // GPU 점유율

        pollAccum = 0.0f;
    }
}

// sampleCPU  –  시스템 전체 CPU 점유율 계산 (모든 논리 코어 평균)
// 원리: GetSystemTimes 로 idle·kernel·user 시간의 델타를 구하면
// total = kernelDiff + userDiff  (idle 포함)
// busy  = total - idleDiff  (실제 사용 시간)
// cpuPercent = busy / total * 100  (CPU 점유율)
// 주의: kernel 시간에는 idle 시간이 포함되어 있으므로 idle 을 별도로 빼야 함
void PerformanceStats::sampleCPU() {
    FILETIME idle, kernel, user;
    GetSystemTimes(&idle, &kernel, &user);

    // FILETIME 을 64비트 정수로 변환 (100나노초 단위)
    unsigned long long curIdle   = (unsigned long long)idle.dwHighDateTime   << 32 | idle.dwLowDateTime;
    unsigned long long curKernel = (unsigned long long)kernel.dwHighDateTime << 32 | kernel.dwLowDateTime;
    unsigned long long curUser   = (unsigned long long)user.dwHighDateTime   << 32 | user.dwLowDateTime;

    // 이전 샘플과의 델타 계산
    unsigned long long idleDiff   = curIdle   - lastIdle;
    unsigned long long kernelDiff = curKernel - lastKernel;
    unsigned long long userDiff   = curUser   - lastUser;

    // 다음 호출을 위해 현재 값을 저장
    lastIdle   = curIdle;
    lastKernel = curKernel;
    lastUser   = curUser;

    unsigned long long total = kernelDiff + userDiff;
    lastSysTotal = total; // sampleProcessCPU() 에서 분모로 재사용

    if (total > 0)
        cpuPercent = (float)(total - idleDiff) / (float)total * 100.0f;
}

// sampleProcessCPU  –  이 프로세스만의 CPU 점유율 계산
// 원리: GetProcessTimes 로 이 프로세스의 kernel+user 시간 델타를 구하고
// sampleCPU() 가 방금 측정한 시스템 총 시간(lastSysTotal) 으로 나눈다.
// → 프로세스가 시스템 CPU 의 몇 %를 쓰는지 파악 가능
void PerformanceStats::sampleProcessCPU() {
    FILETIME ct, et, kt, ut;
    GetProcessTimes(GetCurrentProcess(), &ct, &et, &kt, &ut);

    unsigned long long curK = (unsigned long long)kt.dwHighDateTime << 32 | kt.dwLowDateTime;
    unsigned long long curU = (unsigned long long)ut.dwHighDateTime << 32 | ut.dwLowDateTime;

    // 프로세스가 이번 구간에 소모한 CPU 시간 (kernel + user)
    unsigned long long procDiff = (curK - lastProcKernel) + (curU - lastProcUser);
    lastProcKernel = curK;
    lastProcUser   = curU;

    // 시스템 총 시간 대비 비율 -> 퍼센트
    if (lastSysTotal > 0)
        processCpuPercent = (float)procDiff / (float)lastSysTotal * 100.0f;
}

// sampleRAM  –  이 프로세스의 Working Set(물리 RAM 사용량)을 MB 단위로 조회
void PerformanceStats::sampleRAM() {
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        ramMB = (float)(pmc.WorkingSetSize) / (1024.0f * 1024.0f);
}

// initGPU  –  PDH 를 이용해 GPU Engine 3D 점유율 카운터를 등록
// 와일드카드(*)로 모든 GPU 엔진 인스턴스를 한 번에 구독한다.
// PDH 레이트 카운터는 첫 번째 수집이 기준값이 되므로, 여기서 1회 시드한다.
void PerformanceStats::initGPU() {
    PDH_HQUERY q;
    if (PdhOpenQuery(nullptr, 0, &q) != ERROR_SUCCESS) return;

    PDH_HCOUNTER c;
    // (*engtype_3D) 와일드카드: 3D 작업을 처리하는 모든 GPU 엔진 캡처
    if (PdhAddEnglishCounterW(q, L"\\GPU Engine(*engtype_3D)\\Utilization Percentage",
                               0, &c) != ERROR_SUCCESS) {
        PdhCloseQuery(q);
        return;
    }

    // 첫 번째 수집: 레이트 카운터에 필요한 기준값 시드
    PdhCollectQueryData(q);

    pdhQuery   = q;
    gpuCounter = c;
}

// sampleGPU  –  등록된 GPU 카운터를 수집하고 모든 엔진 인스턴스의 값을 합산
// GPU 는 물리적으로 여러 엔진(컴퓨트·복사·디코드 등)을 가질 수 있고,
// PDH 는 각 엔진을 개별 인스턴스로 반환한다.
// 3D 엔진만 필터했으므로 합산해도 100% 를 넘지 않는다.
void PerformanceStats::sampleGPU() {
    if (!pdhQuery || !gpuCounter) return;

    PdhCollectQueryData((PDH_HQUERY)pdhQuery); // 이번 샘플 수집

    // 필요한 버퍼 크기와 인스턴스 수를 먼저 조회
    DWORD bufSize = 0, itemCount = 0;
    PdhGetFormattedCounterArrayW((PDH_HCOUNTER)gpuCounter,
                                  PDH_FMT_DOUBLE, &bufSize, &itemCount, nullptr);
    if (bufSize == 0 || itemCount == 0) return;

    // 실제 데이터 수신
    auto* items = (PDH_FMT_COUNTERVALUE_ITEM_W*)std::malloc(bufSize);
    if (!items) return;

    if (PdhGetFormattedCounterArrayW((PDH_HCOUNTER)gpuCounter,
                                      PDH_FMT_DOUBLE, &bufSize, &itemCount, items)
        == ERROR_SUCCESS)
    {
        // 모든 3D 엔진 인스턴스의 점유율을 합산 (최대 100% 로 클램프)
        double total = 0.0;
        for (DWORD i = 0; i < itemCount; ++i)
            total += items[i].FmtValue.doubleValue;
        gpuPercent = (float)std::min(total, 100.0);
    }
    std::free(items);
}
