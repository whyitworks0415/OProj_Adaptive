#include "VulkanApp.h"
#include <iostream>
#include <stdexcept>

// 프로그램 진입점.
// VulkanApp 객체를 생성하고 run()을 호출한다.
// 초기화~메인루프~정리까지 모든 흐름이 run() 내부에서 처리된다.
// 사용법: VulkanRenderer.exe [시작 맵 경로] [--bench] [--classic]
//   예)  VulkanRenderer.exe maps/map/1.glb
//   --bench   : 시작 4초 후 5초 벤치마크를 자동 실행하고 CSV 저장 후 종료
//   --classic : 클래식 셰이딩(H OFF 상태)으로 시작
int main(int argc, char** argv) {
    try {
        VulkanApp app;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--bench")    app.setCliBench(true);
            else if (a == "--classic")  app.setClassicShading(true);
            else if (a == "--nolights") app.setNoLights(true);
            else                        app.setInitialMap(a); // 시작 맵 경로
        }
        app.run();
    } catch (const std::exception& e) {
        // Vulkan 초기화 실패, 셰이더 파일 없음 등 치명적 오류를 콘솔에 출력
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
