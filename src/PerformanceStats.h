#pragma once

// PerformanceStats: 실시간 성능 지표 수집기
// 측정 항목:
// - FPS / 프레임 시간 (매 프레임 갱신)
// - 시스템 전체 CPU 점유율 (Windows GetSystemTimes)
// - 이 프로세스만의 CPU 점유율 (Windows GetProcessTimes)
// - 이 프로세스 RAM 사용량 (Windows PSAPI WorkingSetSize)
// - GPU 3D 엔진 점유율 (Windows PDH 카운터)
// 사용법:
// init()  -> 앱 시작 시 1회 호출 (기준값 시드, GPU PDH 쿼리 열기)
// update(dt) -> 매 프레임 호출 (sampleInterval 마다 CPU/GPU/RAM 폴링)
// cleanup() -> 앱 종료 시 1회 호출 (PDH 쿼리 닫기)
struct PerformanceStats {
    float fps               = 0.0f; // 최근 sampleInterval 동안의 평균 FPS
    float frameTimeMs       = 0.0f; // 직전 프레임 시간 (밀리초)
    float cpuPercent        = 0.0f; // 시스템 전체 CPU 점유율 (%)
    float processCpuPercent = 0.0f; // 이 프로세스만의 CPU 점유율 (%)
    float ramMB             = 0.0f; // 이 프로세스 메모리 사용량 (MB)
    float gpuPercent        = -1.0f; // GPU 3D 엔진 점유율 (%). -1 = 수집 불가

    // 앱 시작 시 1회: CPU·GPU 기준값 초기화, PDH 쿼리 열기
    void init();

    // 매 프레임 호출. sampleInterval(초) 주기로 CPU/RAM/GPU 를 폴링
    void update(float deltaTime, float sampleInterval = 0.5f);

    // 앱 종료 시 1회: PDH 쿼리 닫기
    void cleanup();

private:
    // FPS 롤링 평균용 누산기
    float fpsAccum  = 0.0f; // sampleInterval 동안 누적된 FPS 합
    int   fpsCount  = 0; // 누적 프레임 수
    float pollAccum = 0.0f; // 마지막 폴링 이후 경과 시간 (초)

    // 시스템 전체 CPU (GetSystemTimes)
    // 이전 샘플의 FILETIME 값을 저장해 델타로 점유율을 계산
    unsigned long long lastIdle   = 0;
    unsigned long long lastKernel = 0;
    unsigned long long lastUser   = 0;
    void sampleCPU(); // 시스템 전체 CPU 점유율 계산식 적용

    // 프로세스 CPU (GetProcessTimes)
    // procDiff / lastSysTotal × 100 으로 해당 프로세스만의 점유율 계산
    unsigned long long lastProcKernel = 0;
    unsigned long long lastProcUser   = 0;
    unsigned long long lastSysTotal   = 0; // sampleCPU() 에서 공유한 시스템 총량
    void sampleProcessCPU();

    // RAM: PSAPI GetProcessMemoryInfo 사용
    void sampleRAM();

    // GPU: PDH GPU Engine 3D 점유율 카운터 사용
    void* pdhQuery   = nullptr; // PDH_HQUERY  – PDH 쿼리 핸들
    void* gpuCounter = nullptr; // PDH_HCOUNTER – GPU 카운터 핸들
    void initGPU(); // PDH 쿼리 열기 + GPU Engine 카운터 등록
    void sampleGPU(); // 모든 GPU 엔진 인스턴스 점유율을 합산
};
