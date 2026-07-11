# Vulkan 3D Renderer — Adaptive Edition (Enhanced 복사본)

Vulkan API 기반 실시간 3D 렌더러. 다양한 렌더링 최적화 기법을 개별적으로 ON/OFF하여 성능 차이를 시각적으로 비교할 수 있다.

> 이 폴더는 `OProj_Enhanced` 의 복사본으로, **적응형 최적화 (Adaptive Optimizer)** 기능이 추가되어 있다.

## 적응형 최적화 (` 키)

**상황상 실제로 프레임이 오르는 기법만 자동으로 골라 켠다.** 각 기법은 탐색/처리 비용이
있어서 상황에 따라 이득일 수도 손해일 수도 있는데(예: 개활지에서 오클루전 컬링은 쿼리
비용만 내고 이득이 없음), 이 모드는 실행 중에 기법 하나씩 ON/OFF 를 번갈아 실측(A/B
테스트)해서 **프레임 시간이 실제로 짧아지는 쪽만 유지**한다.

| 키 | 동작 |
|---|---|
| ` (백쿼트) | 적응형 최적화 모드 ON/OFF (OFF 시 이전 수동 설정 복원) |

동작 방식:

- 기법당 A(현재 상태)/B(반전 상태) 측정 윈도를 **인터리브(A B A B)** 로 돌려 카메라 이동
  같은 점진적 부하 변화가 한쪽에만 쏠리는 것을 상쇄한다. 윈도 전환 직후 프레임은 예열로
  버리고, 가운데 50% 절사 평균으로 집계한다. 각 윈도는 **최소 프레임 수 + 최소 시간
  (40ms)** 을 모두 채워야 끝난다 — 수천 FPS 상황에서 표본이 몇 ms 만에 끝나 노이즈에
  휘둘리는 것을 방지.
- **판정 3중 필터** (셋 다 통과해야 상태 변경, 아니면 현 상태 유지):
  상대 차이 ≥ 2% + **절대 차이 ≥ 0.08ms** (고FPS %-노이즈 차단) + **라운드 간 부호
  일관성** (1·2라운드 우열이 반대면 보류).
- **조기 원상 판정**: 반전 상태가 명백히 나쁘면(프레임타임 1.3배↑) 남은 측정을 건너뛰고
  즉시 되돌린다 — 손해 상태로 머무는 시간에 상한.
- 큰 회전/이동·히치(>100ms)가 끼면 그 테스트는 폐기하고 원상 복구. 카메라가 **0.35초
  이상 조용할 때만** 테스트를 시작한다 (이동 중 테스트는 어차피 폐기되므로 순수 낭비).
- **재측정은 시점 변화 기반**: 스윕 완료 후 같은 자리에 있으면 재측정하지 않는다
  (탐색 비용 ≈ 0). 스윕 시점 대비 8m/30° 이상 벗어나면 전체 재스윕하고, 그 외에는
  30초에 기법 1개씩만 저빈도 점검한다 (낮/밤 주기처럼 서서히 변하는 부하 대비).
- 씬 전환(TAB)·스트레스 배율(T) 변경 시 측정 기록을 무효화하고 다시 스윕한다.
- LOD·인스턴싱은 씬에 해당 데이터가 없으면 측정 대상에서 제외(n/a)된다.
- **원거리 컬링(7)·소형 오브젝트 컬링(8)은 적응형 대상이 아니다** — 오브젝트가 눈에
  띄게 사라지는 품질 트레이드오프 기법이라, 화면 결과가 달라지지 않는 기법만 자동
  제어한다는 방향성에 맞지 않는다. 이 둘은 적응형 모드 중에도 수동 키로 조작한다.
- **우상단 오버레이**에 현재 상태가 표시된다: 어떤 기법이 켜져 있는지(초록 점), 지금
  무엇을 측정 중인지, 기법별 실측 이득(%; 초록 = 켜서 이득, 빨강 = 켜면 손해).
- 적응형 모드 중에는 컨트롤러가 관리하는 수동 토글 키(1~6, 9, 0)가 잠기고(7·8은 계속
  수동 가능), FPS 캡(I)·벤치마크 중에는 프레임 비교가 무의미하므로 측정이 일시정지된다.
  자동 벤치마크(M)를 시작하면 자동으로 꺼진다.

구현: [src/AdaptiveOpt.h](src/AdaptiveOpt.h) (컨트롤러), `VulkanApp::updateAdaptiveOptimizer` /
`VulkanApp::drawAdaptiveOverlay` (통합·오버레이).

## Enhanced Edition 추가 기능 (그래픽 품질 업그레이드)

원본 대비 다음이 추가/개선되었다.

| 항목 | 원본 | Enhanced |
|------|------|----------|
| 직접 조명 | Blinn-Phong | **PBR Cook-Torrance (GGX + Smith + Fresnel)** |
| 톤매핑 | Reinhard | **ACES 필름 톤매핑** |
| 그림자 | 3x3 PCF, 2048px | **회전 Poisson 16탭 소프트 섀도, 4096px + 노멀 오프셋 바이어스** |
| 배경 | 단색 클리어 | **절차적 하늘** (대기 그라데이션·태양/달 원반·FBM 구름·밤하늘 별) |
| 환경광 | 고정 0.08 | **반구 환경광** (하늘색/지면색, 시간대 연동) |
| 환경 반사 | 고정 Fresnel | **러프니스 기반 하늘 환경 스페큘러** |
| 시간대 | 없음 | **낮/밤 주기 시스템** (태양 고도/방위 애니메이션, 일출·일몰 색, 달빛, 별) |
| 안개 | 없음 | **지수 거리 안개** (하늘 지평선색과 일치, 황혼엔 주황 틴트) |
| 텍스처 필터링 | 이방성 4x | **이방성 16x** |

- 새 셰이더: `shaders/sky.frag` (하늘 전용 전체화면 패스), `shaders/common_env.glsl` (PBR·하늘·안개 공용 라이브러리 — forward/deferred 양쪽에서 #include).
- GLTF 방향광(태양)은 시간대에 따라 강도·색이 자동 감쇠/틴트되고, 밤에는 달빛으로 교체된다. GLTF 포인트/스팟(램프)은 그대로 유지된다.
- 낮/밤 주기(Y)가 켜져 있으면 GLTF 조명 대신 애니메이션 태양/달이 씬과 섀도맵을 주도한다.

### 환경 조작 키

| 키 | 동작 |
|---|---|
| H | **현실적 셰이더 ON/OFF** (OFF = 원본 Blinn-Phong + Reinhard 클래식 룩) |
| Y | 낮/밤 자동 주기 ON/OFF (하루 = 120초) |
| U / J | 시간 수동 스크럽 (앞으로/뒤로) |
| X | 거리 안개 ON/OFF (기본 OFF) |
| K | 구름량 순환 (0 → 25 → 45 → 70 → 95%) |

### Blender 재질·조명 임포트

GLB 로더가 Blender(glTF) 재질에서 다음을 읽어 적용한다.

- baseColor 팩터/텍스처, **metallicRoughness 텍스처**(G=러프니스, B=메탈릭), **emissive 텍스처 + KHR_materials_emissive_strength**, 알파 블렌드(투명) 재질, **버텍스 컬러(COLOR_0)**, doubleSided
- 조명: KHR_lights_punctual (Point/Directional/Spot, **스팟 내부/외부 콘 각도** 포함)
- GLB에 임베드되지 않은 외부 경로 텍스처(`C:/xxx.png` 등)는 파일명으로 `maps/texture/` 등에서 자동 복구
- 동반 MTL(`maps/mtl/N.mtl`)의 `map_Kd` 매칭은 재질 이름의 공백/언더스코어 차이를 자동 처리

### 오클루전 컬링 시연용 던전 맵

`maps/map/3.glb` — 뱀 모양으로 꺾이는 긴 던전 (방 7개 + 복도 6개, 경로 약 216m, 오브젝트 419개).
벽·천장으로 완전히 막혀 있어 벽 뒤 방들이 전부 오클루전 컬링 대상이 된다.

```bash
./VulkanRenderer.exe maps/map/3.glb
```

- 복도에서 안쪽을 바라보면 Frustum Culling만으로는 거의 컬링되지 않는다 (417/419 draw).
- `6` (Occlusion Culling) + `5` (Front-Back Sort)를 켜면 벽 뒤 오브젝트가 제거되며 draw call이 크게 떨어진다.
- 횃불(emissive strength)·금괴(metallic)·수정(파랑 발광)·KHR 조명 8개 포함 — PBR 데모 겸용.
- 재생성: `python tools/make_dungeon_glb.py` (프로젝트 루트에서)

**Blender에서 내보낼 때 체크리스트** (File > Export > glTF 2.0):
1. Format: `glTF Binary (.glb)`
2. Include > Data > **Punctual Lights 체크** — 이걸 켜야 Blender 조명이 맵에 포함된다
3. Material > Images: `Automatic` (텍스처가 GLB 안에 임베드됨)
4. 재질은 Principled BSDF 기준으로 Base Color / Metallic / Roughness / Emission 값·텍스처가 그대로 넘어온다

## 요구사항

| 항목 | 버전 |
|------|------|
| C++ 컴파일러 | C++17 이상 (MSVC 권장) |
| CMake | 3.20 이상 |
| Vulkan SDK | 1.3 이상 |
| vcpkg | 최신 |

### vcpkg 패키지

```
vcpkg install glfw3 glm imgui[core,glfw-binding,vulkan-binding] tinygltf
```

## 빌드

### 1. 셰이더 컴파일 (SPIR-V)

Vulkan SDK의 `glslc`를 사용하여 셰이더를 컴파일한다.

```bash
mkdir -p shaders/spv

glslc shaders/scene.vert           -o shaders/spv/scene.vert.spv
glslc shaders/scene.frag           -o shaders/spv/scene.frag.spv
glslc shaders/scene_instanced.vert -o shaders/spv/scene_instanced.vert.spv
glslc shaders/gbuffer.frag         -o shaders/spv/gbuffer.frag.spv
glslc shaders/deferred_light.vert  -o shaders/spv/deferred_light.vert.spv
glslc shaders/deferred_light.frag  -o shaders/spv/deferred_light.frag.spv
glslc shaders/gizmo.vert           -o shaders/spv/gizmo.vert.spv
glslc shaders/gizmo.frag           -o shaders/spv/gizmo.frag.spv
glslc shaders/sky.frag             -o shaders/spv/sky.frag.spv
```

또는 프로젝트 루트에서 `compile_shaders.bat` 실행 (glslc가 PATH에 있어야 한다).

Windows에서 `glslc`는 보통 `C:\VulkanSDK\<version>\Bin\glslc.exe`에 있다.

### 2. CMake 빌드

```bash
# vcpkg 툴체인 경로를 지정하여 CMake 구성
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake

# Release 빌드
cmake --build build --config Release
```

빌드 완료 시 `build/Release/VulkanRenderer.exe`가 생성되며, 셰이더/에셋/맵이 자동 복사된다.

### 3. 맵 파일 배치

**맵 형식은 GLB/GLTF 하나로 통일**되어 있다 (Blender 내보내기·Unity 비교 실험과 호환).
프로그램이 `maps/`를 재귀 탐색하며, `map/` 폴더의 번호 맵을 목록 맨 앞에 정렬한다(숫자 순).

```
maps/
  map/
    1.glb        # 저폴리 숲 (Blender 에셋)
    2.glb        # 용암 월드 (텍스처·발광 재질)
    3.glb        # 던전 (오클루전 컬링 시연 — tools/make_dungeon_glb.py)
    4.glb        # 소나무 숲 (인스턴싱/LOD 시연 — tools/make_forest_glb.py)
    1.blend      # (선택) 블렌더 원본 — 맵 탐색에서 무시됨
  mtl/           # (선택) 같은 번호 glb의 동반 머티리얼 (map_Kd 텍스처 파일명)
  texture/       # (선택) 외부 텍스처 — MTL의 map_Kd 파일명 그대로 배치
  test_city.glb  # 기본 맵 (시작 시 로드 — tools/make_test_glb.py)
```

- `.glb` 내장 텍스처는 그대로 쓰고, 외부 텍스처는 동반 MTL의 `map_Kd` 파일명을 `maps/texture/`에서 찾는다.
- **같은 메시를 여러 노드가 공유하면**(Blender 링크 복제, Alt+D) 로더가 지오메트리를 한 번만
  올리고 자동으로 GPU 인스턴싱 그룹을 만든다 — 나무·가로등처럼 반복되는 오브젝트에 유리.

## 실행

```bash
cd build/Release
./VulkanRenderer.exe

# 시작 맵을 지정할 수도 있다
./VulkanRenderer.exe maps/map/1.glb

# 무인 자동 벤치마크: 4초 워밍업 → 5초 측정 → results/ CSV 저장 → 종료
./VulkanRenderer.exe maps/map/2.glb --bench
./VulkanRenderer.exe maps/map/2.glb --bench --classic    # 클래식 셰이딩(H OFF)으로 측정
./VulkanRenderer.exe maps/map/2.glb --bench --nolights   # 씬 조명 OFF로 측정 (비용 분석)
```

## 조작법

### 카메라

| 키 | 동작 |
|---|---|
| W/A/S/D | 이동 |
| 마우스 | 시점 회전 |
| Shift | 빠른 이동 |
| 스크롤 휠 | 방향광 회전 |

### 맵 / 모드

| 키 | 동작 |
|---|---|
| TAB | 다음 맵으로 전환 |
| G | Ghost(관찰자) 모드 ON/OFF |
| C | 시네마틱 카메라 (부드러운 이동/회전) ON/OFF |
| F11 | 전체화면 토글 |
| ESC | 종료 |

### 최적화 기법 토글

| 키 | 기법 |
|---|---|
| 1 | Frustum Culling |
| 2 | LOD (Level of Detail) |
| 3 | GPU Instancing |
| 4 | Backface Culling |
| 5 | Front-Back Depth Sort |
| 6 | Occlusion Culling |
| 7 | View Distance Culling |
| 8 | Small Object Culling |
| 9 | Deferred Shading |
| 0 | 전체 ON/OFF 토글 |
| ` | **적응형 최적화** (A/B 실측 자동 선택 — 켜져 있는 동안 1~6/9/0 잠김, 7·8은 수동 유지) |

### 렌더링 설정

| 키 | 동작 |
|---|---|
| L | Scene Lights ON/OFF |
| N | Ambient Light ON/OFF |
| V | Emissive ON/OFF |

(원거리 클리핑은 상시 5000m 로 고정되어 있다.)

### HUD (Material UI)

좌측 패널은 페이지로 나뉘어 있으며 **Ctrl + 방향키**(←/→ 또는 ↑/↓)로 전환한다.
FPS·프레임타임·그래프 영역은 어느 페이지에서든 항상 표시된다.

| 페이지 | 내용 |
|---|---|
| 1 | System · Rendering (CPU/GPU/RAM 사용률, draw call, 컬링 비율) |
| 2 | Optimizations (최적화 기법 스위치 상태) |
| 3 | Lighting (조명 토글, GLTF 조명 목록) |
| 4 | Environment (시간·낮/밤·안개·구름) |
| 5 | Benchmark (FPS 캡, 자동 벤치마크, 스트레스) |
| 6 | Modes · Replay (고스트/시네마틱, 녹화/재생) |

### 녹화 / 리플레이 / 벤치마크

| 키 | 동작 |
|---|---|
| R | 카메라 경로 녹화 시작/정지 |
| P | 리플레이 재생/정지 |
| M | 벤치마크 시작 (리플레이 있으면 자동 벤치마크: 10실험 x 5회) |
| T | 스트레스 배율 순환 (1x / 2x / 4x / 8x / 16x) |
| I | FPS 캡 순환 (무제한 → 144 → 60 → 30) |

### 두 가지 벤치마크 방식 (I 키)

| 방식 | 설정 | 측정 대상 |
|------|------|----------|
| **최대 FPS 모드** | 캡 OFF (기본) | 하드웨어 한계에서 최대 프레임이 몇이 나오는지 — FPS/frametime 비교 |
| **고정 프레임 모드** | 캡 144/60/30 | 모든 실험이 같은 프레임 수를 그려 작업량이 동일 — CPU/GPU/RAM **사용률** 변화로 각 기법의 비용·이득 비교 |

- 고정 프레임 모드는 sleep+spin 하이브리드 리미터로 ±0.5ms 이내로 프레임 시간을 고정한다.
- 캡 설정은 수동(M 5초)·자동(리플레이) 벤치마크 모두에 적용되며, CSV 파일명(`_capN`)과
  메타데이터(`# fps_cap:`)에 기록된다. CSV에는 프레임별 `ram_mb` 컬럼과 `avg_ram_mb` 요약이 추가됐다.

### Unity 비교 실험 (같은 맵 + 같은 움직임)

같은 맵을 같은 카메라 경로로 이 렌더러와 Unity 에서 각각 벤치마크해 성능을 비교하는 워크플로:

1. **경로 녹화 (맵별 저장)**: 렌더러에서 원하는 맵을 띄우고 `R` 로 녹화 → 이동 → `R` 로 종료.
   `replays/<맵이름>_NNN.replay`(재생용) + **`<맵이름>_NNN.csv`(Unity용)** 가 함께 저장된다.
   리플레이는 **맵별로 분리 저장**되며, `P` 재생과 `M` 자동 벤치마크는 현재 맵 전용
   리플레이를 자동으로 골라 쓴다 (같은 맵 = 같은 movement 보장).
2. **렌더러 측 측정**: `M` → 자동 벤치마크 (10실험 × 5회) → `results/autobench_*.csv`
3. **Unity 측 측정**: 준비된 Unity 프로젝트 **`../UnityBenchmark`** 를 Unity Hub 로 열고
   Play → 맵 선택(자동으로 같은 이름의 movement CSV 매칭) → 벤치마크 시작.
   자세한 사용법은 [../UnityBenchmark/README_UNITY.md](../UnityBenchmark/README_UNITY.md) 참고.
   새로 녹화한 CSV 는 `UnityBenchmark/Assets/StreamingAssets/replays/` 에 복사하면 된다.

공정 비교 체크리스트:
- **해상도 동일** (렌더러 1280×720 창모드면 Unity Game 뷰도 1280×720 고정)
- **VSync OFF + FPS 캡 동일** (스크립트가 VSync 를 꺼 주고, `targetFps` 를 렌더러의 `I` 캡과 맞춤)
- Unity 에디터 Play 는 오버헤드가 크므로 **Standalone 빌드로 측정**
- 좌표계: glTFast 임포트면 스크립트의 `FlipX`(기본값) 그대로, 맵이 좌우 반전돼 보이면 `FlipZ`/`None` 으로 변경

### 벤치마크 결과

벤치마크 완료 시 `results/` 폴더에 CSV 파일이 저장된다.

- `bench_*.csv` - 수동 벤치마크 (5초 측정)
- `autobench_*.csv` - 자동 벤치마크 (10 실험별 baseline 대비 FPS 개선율 포함)

## 프로젝트 구조

```
src/
  VulkanApp.cpp/h    - 메인 렌더러 (Vulkan 초기화, 렌더 루프, UI)
  Camera.cpp/h       - FPS 카메라
  GLTFLoader.cpp/h   - GLB/GLTF 로더 (tinygltf) — 메시 공유 + 자동 인스턴싱 그룹
  MeshLoader.h       - MeshRange 구조체 (글로벌 버퍼 안의 메시 범위)
  SceneLoader.cpp/h  - maps/ 에서 GLB/GLTF 맵 탐색
  Replay.cpp/h       - 카메라 경로 녹화/재생 + Unity용 CSV 내보내기
  PerformanceStats.*  - FPS/CPU/GPU 통계
shaders/
  common_env.glsl           - PBR·하늘·안개·톤매핑 공용 함수 (#include)
  scene.vert/frag           - Forward 렌더링 (PBR + 소프트 섀도 + 안개)
  scene_instanced.vert      - GPU Instancing
  gbuffer.frag              - G-Buffer (Deferred)
  deferred_light.vert/frag  - Deferred Lighting (PBR, 배경은 절차적 하늘)
  sky.frag                  - 절차적 하늘 전체화면 패스 (forward 경로)
  gizmo.vert/frag           - Ghost 모드 시각화
maps/                       - 맵 파일 (GLB/GLTF 로 통일)
tools/
  make_test_glb.py          - 기본 도시 맵 생성기
  make_dungeon_glb.py       - 던전 맵 생성기 (오클루전 시연)
  make_forest_glb.py        - 소나무 숲 생성기 (인스턴싱/LOD 시연)
  unity/                    - Unity 비교 실험용 스크립트
```
