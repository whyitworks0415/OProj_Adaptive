// ============================================================================
// VulkanApp.cpp — 메인 렌더러 구현
// ============================================================================
// 이 파일 하나가 렌더러의 거의 모든 것을 담당한다. 크게 다음 구역으로 나뉜다:
//
//  [1] 유틸리티          : 디버그 메신저, 버텍스 레이아웃, 절차적 박스(오클루전 프록시)
//  [2] 씬 관리           : buildScene(GLB 로드) / reloadScene(GPU 안전 씬 교체)
//  [3] Vulkan 초기화     : initVulkan() 이 인스턴스→장치→스왑체인→파이프라인 순서로 생성
//  [4] 파이프라인        : forward 4종 + 인스턴싱 2종 + 디퍼드 + 하늘 + 기즈모 + 섀도
//  [5] 입력/메인 루프    : handleOptKeys(토글 키) / mainLoop(프레임 순환)
//  [6] 프레임 렌더링     : updateUniformBuffer → recordCommandBuffer → drawFrame
//  [7] 리소스(오프스크린): G-Buffer(디퍼드), 섀도맵, 텍스처
//  [8] 리플레이/벤치마크 : 카메라 경로 녹화·재생, 수동/자동 벤치마크 CSV
//  [9] HUD               : drawStatsOverlay (ImGui 좌측 패널)
//  [10] 정리             : cleanup() — 생성의 역순으로 Vulkan 리소스 해제
//
// 한 프레임의 흐름 (mainLoop 한 바퀴):
//   이벤트 폴링 → 키 입력 처리 → 리플레이/벤치마크 갱신 → 성능 통계 갱신
//   → ImGui HUD 구성 → drawFrame() → (FPS 캡이면 목표 시간까지 대기)
// drawFrame() 내부:
//   펜스 대기(CPU-GPU 동기화) → 스왑체인 이미지 획득 → 오클루전 쿼리 결과 읽기
//   → updateUniformBuffer(카메라/조명/환경 UBO) → recordCommandBuffer(커맨드 기록)
//   → 큐 제출 → 프리젠트
// ============================================================================
#include "VulkanApp.h"
#include "GLTFLoader.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <thread>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

// 상수와 검증 레이어 설정
static const std::vector<const char*> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation"
};
static const std::vector<const char*> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};
#ifdef ENABLE_VALIDATION_LAYERS
constexpr bool kEnableValidation = true;
#else
constexpr bool kEnableValidation = false;
#endif

// Vulkan 검증 메시지 콜백
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        std::cerr << "[Vulkan] " << data->pMessage << "\n";
    return VK_FALSE;
}

static VkResult CreateDebugMessenger(VkInstance inst,
    const VkDebugUtilsMessengerCreateInfoEXT* ci,
    VkDebugUtilsMessengerEXT* out)
{
    auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
              vkGetInstanceProcAddr(inst, "vkCreateDebugUtilsMessengerEXT");
    return fn ? fn(inst, ci, nullptr, out) : VK_ERROR_EXTENSION_NOT_PRESENT;
}
static void DestroyDebugMessenger(VkInstance inst, VkDebugUtilsMessengerEXT m)
{
    auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
              vkGetInstanceProcAddr(inst, "vkDestroyDebugUtilsMessengerEXT");
    if (fn) fn(inst, m, nullptr);
}

// 버텍스 입력 레이아웃
VkVertexInputBindingDescription Vertex::getBindingDesc() {
    return {0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
}
std::array<VkVertexInputAttributeDescription, 4> Vertex::getAttrDescs() {
    return {{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
        {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)},
        {3, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv)},
    }};
}

// 절차적 씬 지오메트리 생성
namespace {


// Y-up 원점 기준 축 정렬 박스를 만든다.
// hw/hh/hd는 각 축의 반 길이이며, 정점 순서는 Vulkan 전면 판정에 맞춘다.
// 각 면의 정점 순서가 바깥쪽 법선을 만들도록 유지한다.
static void addBoxFace(std::vector<Vertex>& verts, std::vector<uint32_t>& inds,
                       glm::vec3 n,
                       glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3,
                       glm::vec3 color)
{
    uint32_t base = static_cast<uint32_t>(verts.size());
    verts.push_back({v0, n, color});
    verts.push_back({v1, n, color});
    verts.push_back({v2, n, color});
    verts.push_back({v3, n, color});
    inds.insert(inds.end(), {base, base+1, base+2,  base, base+2, base+3});
}

static MeshRange addBox(std::vector<Vertex>& verts, std::vector<uint32_t>& inds,
                        float hw, float hh, float hd, glm::vec3 color)
{
    uint32_t iBase = static_cast<uint32_t>(inds.size());

    // 각 면의 정점 순서가 바깥쪽 법선을 만들도록 유지한다.
    //  +X 면 법선
    addBoxFace(verts,inds,{1,0,0},   {hw,hh,hd},{hw,-hh,hd},{hw,-hh,-hd},{hw,hh,-hd}, color);
    // -X 면 법선
    addBoxFace(verts,inds,{-1,0,0},  {-hw,hh,-hd},{-hw,-hh,-hd},{-hw,-hh,hd},{-hw,hh,hd}, color);
    //  +Y 면 법선
    addBoxFace(verts,inds,{0,1,0},   {-hw,hh,hd},{hw,hh,hd},{hw,hh,-hd},{-hw,hh,-hd}, color);
    // -Y 면 법선
    addBoxFace(verts,inds,{0,-1,0},  {-hw,-hh,-hd},{hw,-hh,-hd},{hw,-hh,hd},{-hw,-hh,hd}, color);
    //  +Z 면 법선
    addBoxFace(verts,inds,{0,0,1},   {hw,hh,hd},{-hw,hh,hd},{-hw,-hh,hd},{hw,-hh,hd}, color);
    // -Z 면 법선
    addBoxFace(verts,inds,{0,0,-1},  {-hw,hh,-hd},{hw,hh,-hd},{hw,-hh,-hd},{-hw,-hh,-hd}, color);

    MeshRange r{};
    r.indexStart = iBase;
    r.indexCount = static_cast<uint32_t>(inds.size()) - iBase;
    r.bboxMin = {-hw, -hh, -hd};
    r.bboxMax = { hw,  hh,  hd};
    return r;
}




} // 익명 namespace 종료

bool VulkanApp::sampleObjectFrontFaceIsClockwise(const DrawObject& obj,
                                                 const Camera&     cam,
                                                 bool&             outClockwise) const
{
    if (indices.empty() || vertices.empty() || obj.indexCount < 3) return false;

    glm::mat4 view = cam.getViewMatrix();
    glm::mat4 proj = glm::perspective(glm::radians(60.f),
                                      scExtent.width / (float)scExtent.height,
                                      0.1f, getFarPlane());
    proj[1][1] *= -1.f;
    glm::mat4 vp = proj * view;

    int cwCount  = 0;
    int ccwCount = 0;
    int samples  = 0;

    for (uint32_t i = 0; i + 2 < obj.indexCount && samples < 128; i += 3) {
        uint32_t ia = indices[obj.indexStart + i + 0];
        uint32_t ib = indices[obj.indexStart + i + 1];
        uint32_t ic = indices[obj.indexStart + i + 2];
        if (ia >= vertices.size() || ib >= vertices.size() || ic >= vertices.size()) continue;

        glm::vec3 wa = glm::vec3(obj.push.model * glm::vec4(vertices[ia].pos, 1.f));
        glm::vec3 wb = glm::vec3(obj.push.model * glm::vec4(vertices[ib].pos, 1.f));
        glm::vec3 wc = glm::vec3(obj.push.model * glm::vec4(vertices[ic].pos, 1.f));

        glm::vec3 center = (wa + wb + wc) / 3.f;
        glm::mat3 normalM = glm::transpose(glm::inverse(glm::mat3(obj.push.model)));
        glm::vec3 avgN = vertices[ia].normal + vertices[ib].normal + vertices[ic].normal;
        glm::vec3 viewDir = cam.position - center;
        if (glm::dot(avgN, avgN) > 1e-10f) {
            glm::vec3 worldN = glm::normalize(normalM * avgN);
            if (glm::dot(worldN, viewDir) <= 1e-5f) continue;
        } else {
            glm::vec3 faceN = glm::cross(wb - wa, wc - wa);
            if (glm::dot(faceN, faceN) <= 1e-10f) continue;
            if (glm::dot(faceN, viewDir) <= 1e-5f) continue;
        }

        glm::vec4 ca = vp * glm::vec4(wa, 1.f);
        glm::vec4 cb = vp * glm::vec4(wb, 1.f);
        glm::vec4 cc = vp * glm::vec4(wc, 1.f);
        if (ca.w <= 1e-5f || cb.w <= 1e-5f || cc.w <= 1e-5f) continue;

        glm::vec2 aNdc = glm::vec2(ca) / ca.w;
        glm::vec2 bNdc = glm::vec2(cb) / cb.w;
        glm::vec2 cNdc = glm::vec2(cc) / cc.w;

        // Vulkan 화면 좌표계의 전면 판정과 맞춰 winding을 판별한다.
        // Y-down 화면 좌표에서 signed area가 양수이면 시계 방향이다.
        glm::vec2 a((aNdc.x * 0.5f + 0.5f) * scExtent.width,
                    (aNdc.y * 0.5f + 0.5f) * scExtent.height);
        glm::vec2 b((bNdc.x * 0.5f + 0.5f) * scExtent.width,
                    (bNdc.y * 0.5f + 0.5f) * scExtent.height);
        glm::vec2 c((cNdc.x * 0.5f + 0.5f) * scExtent.width,
                    (cNdc.y * 0.5f + 0.5f) * scExtent.height);
        float area2 = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if (std::abs(area2) <= 1e-7f) continue;

        if (area2 > 0.f) ++cwCount;
        else             ++ccwCount;
        ++samples;
    }

    if (samples == 0) return false;
    outClockwise = cwCount >= ccwCount;
    return true;
}

void VulkanApp::refreshFrontFaceHints(const Camera& cam)
{
    for (auto& obj : drawObjects) {
        if (obj.twoSided) {
            obj.reverseFrontFace = false;
            continue;
        }

        bool clockwise = false;
        if (sampleObjectFrontFaceIsClockwise(obj, cam, clockwise)) {
            obj.reverseFrontFace = !clockwise; // 기본 파이프라인은 시계 방향을 전면으로 본다.
        } else {
            obj.reverseFrontFace = glm::determinant(glm::mat3(obj.push.model)) < 0.f;
        }
    }
}


// ============================================================================
// [2] 씬 관리
// ============================================================================

// buildScene — 현재 맵(GLB)을 파싱해 CPU 측 씬 데이터를 구성한다.
// 하는 일 (순서대로):
//  1. GLTFLoader 로 vertices/indices/drawObjects/sceneTextures/sceneLights 채우기
//  2. 시작 카메라를 GLB 의 카메라 노드 위치로 이동
//  3. 오클루전 쿼리용 단위 박스 프록시 메시 추가 (모든 씬 공용)
//  4. 인스턴싱 그룹 정의 재구성 + 스트레스 배율 기준 복사본 저장
//  5. 씬 전체 바운딩 구 계산 (방향광 섀도맵의 ortho 절두체를 씬 크기에 맞추는 데 사용)
// 주의: GPU 버퍼 업로드는 여기서 하지 않는다 — createSceneBuffers() 가 담당.
void VulkanApp::buildScene() {
    using Clock = std::chrono::steady_clock;
    auto elapsed = [](Clock::time_point t0) {
        return std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
    };

    // 확장자에 따라 적절한 로더로 씬 파일 로드
    auto t_parse = Clock::now();
    GLTFSceneDesc gDesc;
    sceneTextures.clear();

    // 확장자 추출 (소문자)
    auto extOf = [](const std::string& p) {
        auto dot = p.rfind('.');
        if (dot == std::string::npos) return std::string{};
        std::string e = p.substr(dot);
        std::transform(e.begin(), e.end(), e.begin(), ::tolower);
        return e;
    };
    std::string ext = extOf(currentMapFile);

    // 맵 형식은 GLB/GLTF 로 통일한다 (Blender 내보내기 + Unity 비교 실험과 호환).
    bool loadOk = false;
    if (ext == ".glb" || ext == ".gltf") {
        loadOk = loadGLTFScene(currentMapFile, vertices, indices, drawObjects, gDesc, sceneTextures);
    } else {
        std::cerr << "[Scene] Unsupported map format (only .glb/.gltf): " << currentMapFile << "\n";
    }
    if (!loadOk)
        throw std::runtime_error("Failed to load scene: " + currentMapFile);
    lastLoadTiming.sceneParseMs = elapsed(t_parse);

    sceneLights     = gDesc.lights; // KHR_lights_punctual 조명 저장
    sceneLightDirty = true; // UBO 갱신 필요

    camera.position = gDesc.cameraPos;
    camera.yaw      = gDesc.cameraYaw;
    camera.pitch    = gDesc.cameraPitch;
    camera.syncTarget(); // 시네마틱 보간 타겟을 새 씬 위치에 다시 맞춤
    refreshFrontFaceHints(camera);

    // 오클루전 쿼리용 단위 박스 프록시를 만든다.
    occBBoxMesh = addBox(vertices, indices, 1.f, 1.f, 1.f, {0.f, 0.f, 0.f});

    // 오클루전 쿼리 결과와 안정화 상태를 초기화한다.
    resetOcclusionState();

    rebuildInstancingGroups();

    // 스트레스 배율 기준 복사본 저장 (씬 로드마다 리셋)
    baseDrawObjects = drawObjects;
    stressLevel     = 0;

    // 씬 전체 바운딩 구 계산 — 방향광 섀도맵 ortho 절두체를 씬에 맞춘다.
    if (!drawObjects.empty()) {
        glm::vec3 bmin( std::numeric_limits<float>::max());
        glm::vec3 bmax(-std::numeric_limits<float>::max());
        for (const DrawObject& o : drawObjects) {
            bmin = glm::min(bmin, o.boundCenter - glm::vec3(o.boundRadius));
            bmax = glm::max(bmax, o.boundCenter + glm::vec3(o.boundRadius));
        }
        sceneBoundsCenter = (bmin + bmax) * 0.5f;
        sceneBoundsRadius = glm::max(glm::length(bmax - sceneBoundsCenter), 1.0f);
    } else {
        sceneBoundsCenter = glm::vec3(0.0f);
        sceneBoundsRadius = 50.0f;
    }

    std::cout << "Scene built: "
              << drawObjects.size() << " draw objects, "
              << vertices.size()    << " vertices, "
              << indices.size() / 3 << " triangles\n";
}

void VulkanApp::rebuildInstancingGroups() {
    instGroupDefs.clear();
    std::map<std::tuple<int, uint32_t, uint32_t, int, bool, bool>, int> groupIdxMap;
    for (int i = 0; i < (int)drawObjects.size(); ++i) {
        const DrawObject& obj = drawObjects[i];
        int gid = obj.instanceGroupId;
        if (gid < 0) continue;
        auto key = std::make_tuple(gid, obj.indexStart, obj.indexCount,
                                   obj.textureIndex, obj.twoSided, obj.reverseFrontFace);
        auto it = groupIdxMap.find(key);
        if (it == groupIdxMap.end()) {
            groupIdxMap[key] = (int)instGroupDefs.size();
            InstGroupDef def;
            def.groupId    = gid;
            def.indexStart = obj.indexStart;
            def.indexCount = obj.indexCount;
            def.push       = obj.push;
            def.members.push_back(i);
            instGroupDefs.push_back(std::move(def));
        } else {
            instGroupDefs[it->second].members.push_back(i);
        }
    }
}

// reloadScene — 실행 중 맵 교체 (TAB 키). GPU 가 쓰는 버퍼를 안전하게 갈아끼운다.
//  1. 새 씬을 CPU 배열에 먼저 파싱 (실패하면 이전 맵으로 되돌림 — GPU 는 무사)
//  2. vkDeviceWaitIdle 로 GPU 드레인 → 기존 버퍼/텍스처 파괴
//  3. 새 버텍스/인덱스/텍스처 업로드 + 오클루전 쿼리 풀 재생성
// 로드 단계별 시간(파싱/업로드)을 재서 HUD 의 "Last scene load" 에 표시한다.
void VulkanApp::reloadScene() {
    using Clock = std::chrono::steady_clock;
    auto t_total = Clock::now();
    auto elapsed = [](Clock::time_point t0) {
        return std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
    };

    lastLoadTiming = {};
    lastLoadTiming.mapName = currentMapFile;
    adaptiveSceneDirty = true; // 새 씬 → 적응형 최적화 측정 기록 무효화

    std::string prevMapFile = currentMapFile;

    // 1단계: 새 CPU 씬 데이터를 먼저 구성한다.
    // 파일 파싱 중에는 기존 GPU 버퍼를 유지하고 CPU 배열만 새로 채운다.
    vertices.clear();
    indices.clear();
    drawObjects.clear();
    occBBoxMesh = {};

    bool buildOk = false;
    try {
        buildScene();
        buildOk = true;
    } catch (const std::exception& e) {
        std::cerr << "[reloadScene] Failed to load '" << currentMapFile
                  << "': " << e.what() << "\n";

        if (currentMapFile != prevMapFile) {
            std::cerr << "[reloadScene] Reverting to: " << prevMapFile << "\n";
            currentMapFile = prevMapFile;
            for (int i = 0; i < (int)availableMaps.size(); ++i)
                if (availableMaps[i] == prevMapFile) { currentMapIndex = i; break; }
            vertices.clear(); indices.clear(); drawObjects.clear();
            try {
                buildScene();
                buildOk = true;
            } catch (...) {
                std::cerr << "[reloadScene] Fallback scene also failed. Running empty.\n";
            }
        }
    }
    (void)buildOk;

    // 2단계: GPU 작업을 비운 뒤 이전 GPU 버퍼를 제거한다.
    // 진행 중인 프레임이 이전 버퍼를 참조할 수 있으므로 GPU idle을 기다린다.
    // 느린 CPU 파싱은 이미 끝났기 때문에 여기서는 남은 프레임 작업만 비운다.
    vkDeviceWaitIdle(device);

    vkDestroyBuffer(device, indexBuffer,  nullptr);
    vkFreeMemory(device,    indexBufferMemory,  nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device,    vertexBufferMemory, nullptr);
    indexBuffer        = VK_NULL_HANDLE;  indexBufferMemory  = VK_NULL_HANDLE;
    vertexBuffer       = VK_NULL_HANDLE;  vertexBufferMemory = VK_NULL_HANDLE;

    if (occlusionQueryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device, occlusionQueryPool, nullptr);
        occlusionQueryPool = VK_NULL_HANDLE;
        occQueryCount      = 0;
    }
    occWarmupFrames = 0;

    // 3단계: 새 씬 데이터를 한 번의 배치 전송으로 업로드한다.
    // createSceneBuffers()는 버텍스와 인덱스 복사를 하나의 커맨드 버퍼로 기록한다.
    // 별도 queue idle 두 번 대신 하나의 fence만 기다린다.
    auto t_upload = Clock::now();
    if (!vertices.empty() && !indices.empty())
        createSceneBuffers();
    else {
        if (!vertices.empty()) createVertexBuffer();
        if (!indices.empty())  createIndexBuffer();
    }
    lastLoadTiming.uploadMs = elapsed(t_upload);

    if (!drawObjects.empty()) createOcclusionQueryPool();
    resetOcclusionState();

    createTextureResources();

    lastLoadTiming.totalMs = elapsed(t_total);
    lastLoadTiming.valid   = true;

    printf("[Load] %-32s  total %6.1f ms  parse %6.1f ms  upload %5.1f ms\n",
           currentMapFile.c_str(),
           lastLoadTiming.totalMs,
           lastLoadTiming.sceneParseMs,
           lastLoadTiming.uploadMs);
}



void VulkanApp::run() {
    // 기존 리플레이 중 CSV 가 없는 것을 일괄 내보낸다 (Unity 비교 실험용).
    exportMissingReplayCsvs(replayDir);

    // maps/ 디렉터리에서 렌더링 가능한 씬 파일을 모두 찾는다.
    availableMaps = findMapFiles("maps");
    if (availableMaps.empty()) {
        // 탐색에 실패하면 기본 맵 경로를 그대로 사용한다.
        if (currentMapFile.empty())
            currentMapFile = "maps/test_city.glb";
        availableMaps.push_back(currentMapFile);
    }

    // currentMapFile과 일치하는 항목으로 현재 맵 인덱스를 맞춘다.
    // 비어 있거나 목록에 없으면 가벼운 기본 맵(test_city)을 우선 사용한다.
    auto inOrganizedMapDir = [](const std::string& p) {
        std::string lower = p;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return lower.find("/map/") != std::string::npos ||
               lower.find("\\map\\") != std::string::npos;
    };
    bool hasOrganizedMaps = false;
    for (const auto& m : availableMaps) {
        if (inOrganizedMapDir(m)) {
            hasOrganizedMaps = true;
            break;
        }
    }

    int selectedIndex = -1;
    if (!hasOrganizedMaps || currentMapFile != "maps/test_city.glb") {
        for (int i = 0; i < (int)availableMaps.size(); ++i) {
            if (availableMaps[i] == currentMapFile) {
                selectedIndex = i;
                break;
            }
        }
    }
    if (selectedIndex < 0) {
        for (int i = 0; i < (int)availableMaps.size(); ++i) {
            if (availableMaps[i] == "maps/test_city.glb") {
                selectedIndex = i;
                break;
            }
        }
    }
    if (selectedIndex < 0)
        selectedIndex = 0;
    if (!availableMaps.empty()) {
        currentMapIndex = selectedIndex;
        currentMapFile  = availableMaps[currentMapIndex];
    }

    std::cout << "Found " << availableMaps.size() << " map(s):\n";
    for (auto& m : availableMaps) std::cout << "  " << m << "\n";

    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

// 윈도우와 입력 콜백
void VulkanApp::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Debug Scene", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void VulkanApp::framebufferResizeCallback(GLFWwindow* w, int, int) {
    auto app = reinterpret_cast<VulkanApp*>(glfwGetWindowUserPointer(w));
    app->framebufferResized = true;
}

void VulkanApp::cursorPosCallback(GLFWwindow* w, double xpos, double ypos) {
    auto app = reinterpret_cast<VulkanApp*>(glfwGetWindowUserPointer(w));
    if (app->firstMouse) {
        app->lastMouseX = xpos;
        app->lastMouseY = ypos;
        app->firstMouse = false;
        return;
    }
    float dx = static_cast<float>(xpos - app->lastMouseX);
    float dy = static_cast<float>(ypos - app->lastMouseY);
    app->lastMouseX = xpos;
    app->lastMouseY = ypos;

    if (app->ghostMode) {
        // ghost 모드에서 가운데 버튼을 누르면 배치 카메라를 회전한다.
        bool midHeld = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
        if (midHeld)
            app->camera.processMouseDelta(dx, dy);
        else
            app->observerCamera.processMouseDelta(dx, dy);
    } else {
        // 리플레이 중에는 카메라를 재생 데이터가 제어한다.
        if (!app->isReplaying)
            app->camera.processMouseDelta(dx, dy);
    }
}

void VulkanApp::mouseButtonCallback(GLFWwindow* w, int button, int action, int) {
    // 필요하면 클릭 포커스 처리를 여기에 추가할 수 있다.
}

void VulkanApp::scrollCallback(GLFWwindow* w, double /*xoffset*/, double yoffset) {
    auto app = reinterpret_cast<VulkanApp*>(glfwGetWindowUserPointer(w));
    // 휠 한 칸마다 방향광을 Y축 기준 12도 회전한다.
    app->lightYaw += static_cast<float>(yoffset) * 12.0f;
}

void VulkanApp::toggleFullscreen() {
    if (!isFullscreen) {
        // 창 모드 위치와 크기를 저장한다.
        glfwGetWindowPos(window,  &windowedX, &windowedY);
        glfwGetWindowSize(window, &windowedW, &windowedH);
        // 주 모니터의 보더리스 전체화면으로 전환한다.
        GLFWmonitor*       monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode    = glfwGetVideoMode(monitor);
        glfwSetWindowMonitor(window, monitor, 0, 0,
                             mode->width, mode->height, mode->refreshRate);
        isFullscreen = true;
    } else {
        // 저장해 둔 창 모드로 복귀한다.
        glfwSetWindowMonitor(window, nullptr,
                             windowedX, windowedY, windowedW, windowedH, 0);
        isFullscreen = false;
    }
    // GLFW resize 콜백이 스왑체인 재생성을 유도한다.
}

// Vulkan 초기화 순서
// ============================================================================
// [3] Vulkan 초기화
// ============================================================================
// Vulkan 은 OpenGL 과 달리 "전부 명시적"이다. 아래 순서 하나하나가 의존 관계다:
//  인스턴스 → 서피스 → 물리 장치(GPU) 선택 → 논리 장치+큐 → 스왑체인 → 이미지 뷰
//  → 렌더패스 → 디스크립터 레이아웃 → 파이프라인들 → 깊이 버퍼 → 프레임버퍼
//  → 섀도맵 → 커맨드 풀 → 씬 로드/업로드 → UBO/디스크립터 → 동기화 오브젝트
void VulkanApp::initVulkan() {
    createInstance();
    if (kEnableValidation) setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createDepthResources();
    createFramebuffers();
    createShadowResources(); // 섀도맵은 descriptor set이 binding 3으로 참조하므로 그 전에 만든다.
    createShadowPipeline();
    createCommandPool();
    {
        using Clock = std::chrono::steady_clock;
        auto t_total = Clock::now();
        if (currentMapFile.empty()) {
            if (!availableMaps.empty()) {
                if (currentMapIndex < 0 || currentMapIndex >= (int)availableMaps.size())
                    currentMapIndex = 0;
                currentMapFile = availableMaps[currentMapIndex];
            } else {
                currentMapFile = "maps/test_city.glb";
            }
        }
        lastLoadTiming = {};
        lastLoadTiming.mapName = currentMapFile;
        buildScene();
        auto t_upload = Clock::now();
        createSceneBuffers(); // 하나의 커맨드 버퍼와 펜스로 일괄 업로드
        lastLoadTiming.uploadMs = std::chrono::duration<float, std::milli>(Clock::now() - t_upload).count();
        lastLoadTiming.totalMs  = std::chrono::duration<float, std::milli>(Clock::now() - t_total).count();
        lastLoadTiming.valid    = true;
        printf("[Load] %-32s  total %6.1f ms  parse %6.1f ms  upload %5.1f ms\n",
               currentMapFile.c_str(),
               lastLoadTiming.totalMs, lastLoadTiming.sceneParseMs,
               lastLoadTiming.uploadMs);
    }
    createUniformBuffers();
    createDefaultTextureAndSampler();
    createDescriptorPool();
    createDescriptorSets();
    createTextureResources();
    createCommandBuffers();
    createSyncObjects();
    createInstanceResources();
    createOcclusionQueryPool();
    createGizmoPipeline();
    createGizmoBuffers();
    createDeferredResources();
    createBloomPipelines(); // 블룸 렌더패스/파이프라인 (1회)
    createBloomResources(); // 하프 해상도 타깃 + 디스크립터 (리사이즈마다)
    initImGui();
    perfStats.init();
}

// Vulkan 인스턴스 생성
void VulkanApp::createInstance() {
    if (kEnableValidation) {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> layers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, layers.data());
        for (auto* name : kValidationLayers) {
            bool found = false;
            for (auto& l : layers) if (strcmp(l.layerName, name) == 0) { found = true; break; }
            if (!found) throw std::runtime_error(std::string("Validation layer not found: ") + name);
        }
    }

    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName   = "VulkanDebugScene";
    appInfo.applicationVersion = VK_MAKE_VERSION(1,0,0);
    appInfo.pEngineName        = "None";
    appInfo.apiVersion         = VK_API_VERSION_1_2;

    uint32_t glfwExtCount;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char*> extensions(glfwExts, glfwExts + glfwExtCount);
    if (kEnableValidation) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo        = &appInfo;
    ci.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();
    if (kEnableValidation) {
        ci.enabledLayerCount   = static_cast<uint32_t>(kValidationLayers.size());
        ci.ppEnabledLayerNames = kValidationLayers.data();
    }
    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("vkCreateInstance failed");
}

void VulkanApp::setupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT ci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = debugCallback;
    CreateDebugMessenger(instance, &ci, &debugMessenger);
}

void VulkanApp::createSurface() {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("glfwCreateWindowSurface failed");
}

// 물리/논리 디바이스 선택
QueueFamilyIndices VulkanApp::findQueueFamilies(VkPhysicalDevice dev) {
    uint32_t count;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, props.data());

    QueueFamilyIndices idx;
    for (uint32_t i = 0; i < count; ++i) {
        if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) idx.graphics = i;
        VkBool32 presentSupport;
        vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &presentSupport);
        if (presentSupport) idx.present = i;
        if (idx.isComplete()) break;
    }
    return idx;
}

SwapChainSupportDetails VulkanApp::querySwapChainSupport(VkPhysicalDevice dev) {
    SwapChainSupportDetails d;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, surface, &d.capabilities);
    uint32_t count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &count, nullptr);
    if (count) { d.formats.resize(count); vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &count, d.formats.data()); }
    vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &count, nullptr);
    if (count) { d.presentModes.resize(count); vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &count, d.presentModes.data()); }
    return d;
}

bool VulkanApp::isDeviceSuitable(VkPhysicalDevice dev) {
    auto idx = findQueueFamilies(dev);
    if (!idx.isComplete()) return false;

    uint32_t extCount;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, exts.data());
    for (auto* req : kDeviceExtensions) {
        bool found = false;
        for (auto& e : exts) if (strcmp(e.extensionName, req) == 0) { found = true; break; }
        if (!found) return false;
    }

    auto sc = querySwapChainSupport(dev);
    return !sc.formats.empty() && !sc.presentModes.empty();
}

void VulkanApp::pickPhysicalDevice() {
    uint32_t count;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (!count) throw std::runtime_error("No Vulkan GPUs found");
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(instance, &count, devs.data());

    // 가능하면 외장 GPU를 우선 선택한다.
    for (auto dev : devs) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(dev, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && isDeviceSuitable(dev)) {
            physDevice = dev; return;
        }
    }
    for (auto dev : devs) if (isDeviceSuitable(dev)) { physDevice = dev; return; }
    throw std::runtime_error("No suitable GPU found");
}

void VulkanApp::createLogicalDevice() {
    auto idx = findQueueFamilies(physDevice);
    std::set<uint32_t> unique = {idx.graphics.value(), idx.present.value()};
    std::vector<VkDeviceQueueCreateInfo> queueCIs;
    float priority = 1.0f;
    for (uint32_t qf : unique) {
        VkDeviceQueueCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        ci.queueFamilyIndex = qf; ci.queueCount = 1; ci.pQueuePriorities = &priority;
        queueCIs.push_back(ci);
    }

    VkPhysicalDeviceFeatures features{};
    features.samplerAnisotropy                       = VK_TRUE;
    features.shaderSampledImageArrayDynamicIndexing  = VK_TRUE; // textures[] 동적 인덱싱

    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.queueCreateInfoCount    = static_cast<uint32_t>(queueCIs.size());
    ci.pQueueCreateInfos       = queueCIs.data();
    ci.enabledExtensionCount   = static_cast<uint32_t>(kDeviceExtensions.size());
    ci.ppEnabledExtensionNames = kDeviceExtensions.data();
    ci.pEnabledFeatures        = &features;
    if (kEnableValidation) {
        ci.enabledLayerCount   = static_cast<uint32_t>(kValidationLayers.size());
        ci.ppEnabledLayerNames = kValidationLayers.data();
    }
    if (vkCreateDevice(physDevice, &ci, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDevice failed");

    vkGetDeviceQueue(device, idx.graphics.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, idx.present.value(),  0, &presentQueue);
}

// 스왑체인 생성
// createSwapChain — 화면에 표시할 이미지 체인을 만든다.
// 결정하는 것 3가지:
//  - 포맷: B8G8R8A8_SRGB 우선 (감마 보정된 색 공간)
//  - 프리젠트 모드: MAILBOX(삼중 버퍼링, 티어링 없음 + 무제한 FPS) 우선, 없으면 FIFO(VSync)
//  - 해상도: 서피스가 알려주는 currentExtent, 없으면 프레임버퍼 크기를 클램프
// 창 크기가 바뀌면 recreateSwapChain() 이 이 함수를 다시 호출한다.
void VulkanApp::createSwapChain() {
    auto sc = querySwapChainSupport(physDevice);

    // 스왑체인 색상 포맷 선택
    VkSurfaceFormatKHR fmt = sc.formats[0];
    for (auto& f : sc.formats)
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            { fmt = f; break; }

    // mailbox present mode를 우선 사용하고 없으면 FIFO를 쓴다.
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    for (auto m : sc.presentModes) if (m == VK_PRESENT_MODE_MAILBOX_KHR) { mode = m; break; }

    // 프레임버퍼 크기 결정
    VkExtent2D extent;
    if (sc.capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        extent = sc.capabilities.currentExtent;
    } else {
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        extent = {std::clamp((uint32_t)w, sc.capabilities.minImageExtent.width,  sc.capabilities.maxImageExtent.width),
                  std::clamp((uint32_t)h, sc.capabilities.minImageExtent.height, sc.capabilities.maxImageExtent.height)};
    }

    uint32_t imgCount = sc.capabilities.minImageCount + 1;
    if (sc.capabilities.maxImageCount > 0) imgCount = std::min(imgCount, sc.capabilities.maxImageCount);

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface            = surface;
    ci.minImageCount      = imgCount;
    ci.imageFormat        = fmt.format;
    ci.imageColorSpace    = fmt.colorSpace;
    ci.imageExtent        = extent;
    ci.imageArrayLayers   = 1;
    // SAMPLED: 블룸 포스트프로세스가 씬 결과를 샘플링할 수 있게 한다.
    ci.imageUsage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                          | VK_IMAGE_USAGE_SAMPLED_BIT;

    auto idx = findQueueFamilies(physDevice);
    uint32_t queueFamilies[] = {idx.graphics.value(), idx.present.value()};
    if (idx.graphics != idx.present) {
        ci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices   = queueFamilies;
    } else {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    ci.preTransform   = sc.capabilities.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode    = mode;
    ci.clipped        = VK_TRUE;

    if (vkCreateSwapchainKHR(device, &ci, nullptr, &swapchain) != VK_SUCCESS)
        throw std::runtime_error("vkCreateSwapchainKHR failed");

    uint32_t n;
    vkGetSwapchainImagesKHR(device, swapchain, &n, nullptr);
    scImages.resize(n);
    vkGetSwapchainImagesKHR(device, swapchain, &n, scImages.data());
    scFormat = fmt.format;
    scExtent = extent;
}

void VulkanApp::createImageViews() {
    scImageViews.resize(scImages.size());
    for (size_t i = 0; i < scImages.size(); ++i)
        scImageViews[i] = createImageView(scImages[i], scFormat, VK_IMAGE_ASPECT_COLOR_BIT);
}

// 기본 렌더 패스
void VulkanApp::createRenderPass() {
    VkAttachmentDescription color{};
    color.format         = scFormat;
    color.samples        = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depth{};
    depth.format         = findDepthFormat();
    depth.samples        = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 2> attachments = {color, depth};
    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = static_cast<uint32_t>(attachments.size());
    ci.pAttachments    = attachments.data();
    ci.subpassCount    = 1;
    ci.pSubpasses      = &subpass;
    ci.dependencyCount = 1;
    ci.pDependencies   = &dep;

    if (vkCreateRenderPass(device, &ci, nullptr, &renderPass) != VK_SUCCESS)
        throw std::runtime_error("vkCreateRenderPass failed");
}

// 디스크립터 세트 레이아웃
void VulkanApp::createDescriptorSetLayout() {
    // set 0 binding 0은 카메라 UBO다.
    VkDescriptorSetLayoutBinding uboBinding{};
    uboBinding.binding         = 0;
    uboBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboBinding.descriptorCount = 1;
    uboBinding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    // set 0 binding 1은 씬 텍스처 배열이다.
    // 사용하지 않는 슬롯은 1×1 흰색 기본 텍스처로 채워서 모든 슬롯이 항상 유효하다.
    VkDescriptorSetLayoutBinding samplerBinding{};
    samplerBinding.binding         = 1;
    samplerBinding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerBinding.descriptorCount = MAX_TEXTURES;
    samplerBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    // set 0 binding 2는 SceneLightUBO다.
    VkDescriptorSetLayoutBinding lightBinding{};
    lightBinding.binding         = 2;
    lightBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    lightBinding.descriptorCount = 1;
    lightBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    // set 0 binding 3은 방향광 섀도맵 샘플러다(forward·deferred 양쪽이 공유).
    VkDescriptorSetLayoutBinding shadowBinding{};
    shadowBinding.binding         = 3;
    shadowBinding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    shadowBinding.descriptorCount = 1;
    shadowBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding bindings[] = {uboBinding, samplerBinding, lightBinding, shadowBinding};

    VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.bindingCount = 4;
    ci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &descSetLayout) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDescriptorSetLayout failed");

    // set 1 binding 0은 인스턴스 모델 행렬 SSBO다.
    VkDescriptorSetLayoutBinding ssboBinding{};
    ssboBinding.binding         = 0;
    ssboBinding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ssboBinding.descriptorCount = 1;
    ssboBinding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    ci.bindingCount = 1;
    ci.pBindings = &ssboBinding;
    if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &instanceDescSetLayout) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDescriptorSetLayout (instance SSBO) failed");
}

// 그래픽 파이프라인 생성
std::vector<char> VulkanApp::readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open file: " + path);
    size_t size = file.tellg();
    std::vector<char> buf(size);
    file.seekg(0);
    file.read(buf.data(), size);
    return buf;
}

VkShaderModule VulkanApp::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size();
    ci.pCode    = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule m;
    if (vkCreateShaderModule(device, &ci, nullptr, &m) != VK_SUCCESS)
        throw std::runtime_error("vkCreateShaderModule failed");
    return m;
}

// ============================================================================
// [4] 그래픽 파이프라인 생성
// ============================================================================
// Vulkan 파이프라인은 셰이더+고정기능 상태(래스터라이저·블렌딩·깊이)를 통째로
// 굽는(bake) 불변 객체다. 상태 조합마다 별도 파이프라인이 필요해서 종류가 많다:
//  - graphicsPipeline           : 기본 (백페이스 컬링 ON, 시계방향 전면)
//  - graphicsPipelineFlippedCull: 전면 판정 반전 (거울 변환된 오브젝트용)
//  - graphicsPipelineNoCull     : 컬링 OFF (양면 재질, 키 4 OFF)
//  - graphicsPipelineAlpha      : 알파 블렌딩 (투명 오브젝트, 깊이 쓰기 OFF)
//  - graphicsPipelineQueryOnly  : 오클루전 쿼리 전용 (색/깊이 쓰기 없음)
//  - graphicsPipelineInst(+NoCull): 인스턴싱 — SSBO 에서 모델 행렬을 읽음
//  - gbufPipeline(+NoCull)      : 디퍼드 G-Buffer 채우기 (출력 4장)
//  - deferredLightPipeline      : 전체화면 조명 합성 (버텍스 버퍼 없음)
//  - skyPipeline                : 절차적 하늘 (깊이 테스트/쓰기 OFF, 배경 전용)
void VulkanApp::createGraphicsPipeline() {
    // SPIR-V 셰이더 모듈을 읽고 생성한다.
    auto vertCode     = readFile("shaders/spv/scene.vert.spv");
    auto fragCode     = readFile("shaders/spv/scene.frag.spv");
    auto vertInstCode = readFile("shaders/spv/scene_instanced.vert.spv");

    VkShaderModule vertM     = createShaderModule(vertCode);
    VkShaderModule fragM     = createShaderModule(fragCode);
    VkShaderModule vertInstM = createShaderModule(vertInstCode);

    VkPipelineShaderStageCreateInfo vertStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    vertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertM;
    vertStage.pName  = "main";

    VkPipelineShaderStageCreateInfo fragStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    fragStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragM;
    fragStage.pName  = "main";

    VkPipelineShaderStageCreateInfo stages[] = {vertStage, fragStage};

    // 고정 기능 파이프라인 상태를 설정한다.
    auto binding  = Vertex::getBindingDesc();
    auto attrs    = Vertex::getAttrDescs();
    VkPipelineVertexInputStateCreateInfo vertInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertInput.vertexBindingDescriptionCount   = 1;
    vertInput.pVertexBindingDescriptions      = &binding;
    vertInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
    vertInput.pVertexAttributeDescriptions    = attrs.data();

    std::array<VkVertexInputBindingDescription, 2> instBindings{};
    instBindings[0] = binding;
    instBindings[1].binding   = 1;
    instBindings[1].stride    = sizeof(glm::mat4);
    instBindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    std::array<VkVertexInputAttributeDescription, 8> instAttrs{};
    for (size_t i = 0; i < attrs.size(); ++i)
        instAttrs[i] = attrs[i];
    for (uint32_t col = 0; col < 4; ++col) {
        instAttrs[4 + col].location = 4 + col;
        instAttrs[4 + col].binding  = 1;
        instAttrs[4 + col].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
        instAttrs[4 + col].offset   = sizeof(glm::vec4) * col;
    }

    VkPipelineVertexInputStateCreateInfo instVertInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    instVertInput.vertexBindingDescriptionCount   = static_cast<uint32_t>(instBindings.size());
    instVertInput.pVertexBindingDescriptions      = instBindings.data();
    instVertInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(instAttrs.size());
    instVertInput.pVertexAttributeDescriptions    = instAttrs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // 뷰포트와 시저는 프레임마다 동적으로 설정한다.
    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynState.dynamicStateCount = 2;
    dynState.pDynamicStates    = dynStates;

    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo msaa{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthStencil.depthTestEnable  = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendAttach{};
    blendAttach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAttach.blendEnable    = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1; blend.pAttachments = &blendAttach;

    // 모든 파이프라인이 같은 push constant 범위를 공유한다.
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(PushConstants);

    // 일반/인스턴싱 파이프라인 레이아웃을 만든다.
    // 일반 렌더링은 set 0의 카메라/조명/텍스처 디스크립터만 사용한다.
    VkPipelineLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutCI.setLayoutCount         = 1;
    layoutCI.pSetLayouts            = &descSetLayout;
    layoutCI.pushConstantRangeCount = 1;
    layoutCI.pPushConstantRanges    = &pcRange;
    if (vkCreatePipelineLayout(device, &layoutCI, nullptr, &pipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("vkCreatePipelineLayout failed");

    // 인스턴싱 렌더링은 set 0에 더해 set 1의 인스턴스 행렬 SSBO를 사용한다.
    VkDescriptorSetLayout instLayouts[] = { descSetLayout, instanceDescSetLayout };
    layoutCI.setLayoutCount = 2;
    layoutCI.pSetLayouts    = instLayouts;
    if (vkCreatePipelineLayout(device, &layoutCI, nullptr, &instancePipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("vkCreatePipelineLayout (instanced) failed");

    // 공통 그래픽 파이프라인 생성 정보를 준비한다.
    VkGraphicsPipelineCreateInfo pipeCI{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeCI.stageCount          = 2;
    pipeCI.pStages             = stages;
    pipeCI.pVertexInputState   = &vertInput;
    pipeCI.pInputAssemblyState = &inputAssembly;
    pipeCI.pViewportState      = &viewportState;
    pipeCI.pRasterizationState = &raster;
    pipeCI.pMultisampleState   = &msaa;
    pipeCI.pDepthStencilState  = &depthStencil;
    pipeCI.pColorBlendState    = &blend;
    pipeCI.pDynamicState       = &dynState;
    pipeCI.renderPass          = renderPass;
    pipeCI.subpass             = 0;

    // 일반 파이프라인을 생성한다.
    raster.cullMode = VK_CULL_MODE_BACK_BIT;
    pipeCI.layout   = pipelineLayout;
    stages[0].module = vertM;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &graphicsPipeline) != VK_SUCCESS)
        throw std::runtime_error("vkCreateGraphicsPipelines failed");

    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &graphicsPipelineFlippedCull) != VK_SUCCESS)
        throw std::runtime_error("vkCreateGraphicsPipelines (flipped cull) failed");
    raster.frontFace = VK_FRONT_FACE_CLOCKWISE;

    // 일반 파이프라인을 생성한다.
    raster.cullMode = VK_CULL_MODE_NONE;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &graphicsPipelineNoCull) != VK_SUCCESS)
        throw std::runtime_error("vkCreateGraphicsPipelines (no-cull) failed");

    // 인스턴싱 파이프라인을 생성한다.
    raster.cullMode  = VK_CULL_MODE_BACK_BIT;
    pipeCI.layout    = instancePipelineLayout;
    pipeCI.pVertexInputState = &instVertInput;
    stages[0].module = vertInstM;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &graphicsPipelineInst) != VK_SUCCESS)
        throw std::runtime_error("vkCreateGraphicsPipelines (instanced) failed");

    // 인스턴싱 파이프라인을 생성한다.
    raster.cullMode = VK_CULL_MODE_NONE;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &graphicsPipelineInstNoCull) != VK_SUCCESS)
        throw std::runtime_error("vkCreateGraphicsPipelines (instanced no-cull) failed");
    pipeCI.pVertexInputState = &vertInput;

    // 투명 오브젝트용 알파 블렌딩 파이프라인을 만든다.
    // 일반 버텍스/프래그먼트 셰이더를 그대로 재사용한다.
    // 투명 오브젝트는 모든 불투명 오브젝트 뒤에 뒤에서 앞으로 그린다.
    {
        blendAttach.blendEnable         = VK_TRUE;
        blendAttach.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blendAttach.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAttach.colorBlendOp        = VK_BLEND_OP_ADD;
        blendAttach.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttach.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blendAttach.alphaBlendOp        = VK_BLEND_OP_ADD;

        depthStencil.depthWriteEnable = VK_FALSE; // 투명 오브젝트는 깊이를 쓰지 않는다.
        raster.cullMode               = VK_CULL_MODE_NONE; // 유리처럼 양면이 보여야 하는 재질을 위해 컬링하지 않는다.

        pipeCI.layout    = pipelineLayout;
        stages[0].module = vertM;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &graphicsPipelineAlpha) != VK_SUCCESS)
            throw std::runtime_error("vkCreateGraphicsPipelines (alpha) failed");

        // 이후 파이프라인 생성에 영향을 주지 않도록 상태를 되돌린다.
        blendAttach.blendEnable       = VK_FALSE;
        depthStencil.depthWriteEnable = VK_TRUE;
    }

    // 바운딩 박스 오클루전 쿼리 전용 파이프라인을 만든다.
    {
        blendAttach.colorWriteMask    = 0; // 색상 버퍼에는 쓰지 않는다.
        blendAttach.blendEnable       = VK_FALSE;
        depthStencil.depthWriteEnable = VK_FALSE; // 실제 깊이 버퍼를 오염시키지 않는다.
        depthStencil.depthTestEnable  = VK_TRUE;
        depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;
        raster.cullMode               = VK_CULL_MODE_NONE;
        pipeCI.layout    = pipelineLayout;
        stages[0].module = vertM;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &graphicsPipelineQueryOnly) != VK_SUCCESS)
            throw std::runtime_error("vkCreateGraphicsPipelines (query-only) failed");
        // 상태 복원
        blendAttach.colorWriteMask    = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;
    }

    vkDestroyShaderModule(device, vertM,     nullptr);
    vkDestroyShaderModule(device, fragM,     nullptr);
    vkDestroyShaderModule(device, vertInstM, nullptr);

    // 디퍼드 렌더링 파이프라인을 만든다.

    // G-Buffer 렌더패스는 컬러 attachment들과 깊이 attachment를 사용한다.
    {
        VkFormat depthFmt = findDepthFormat();
        VkAttachmentDescription gbAtts[GBUFFER_COLOR_ATTACHMENTS + 1] = {};
        gbAtts[0].format         = VK_FORMAT_R8G8B8A8_UNORM;
        gbAtts[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        gbAtts[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        gbAtts[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        gbAtts[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        gbAtts[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        gbAtts[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        gbAtts[0].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        gbAtts[1]        = gbAtts[0];
        gbAtts[1].format = VK_FORMAT_R16G16B16A16_SFLOAT;
        gbAtts[2]        = gbAtts[1];
        gbAtts[3]        = gbAtts[1];
        gbAtts[4].format         = depthFmt;
        gbAtts[4].samples        = VK_SAMPLE_COUNT_1_BIT;
        gbAtts[4].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        gbAtts[4].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        gbAtts[4].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        gbAtts[4].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        gbAtts[4].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        gbAtts[4].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference cRefs[GBUFFER_COLOR_ATTACHMENTS] = {
            {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            {3, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        };
        VkAttachmentReference dRef = {GBUFFER_COLOR_ATTACHMENTS, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount    = GBUFFER_COLOR_ATTACHMENTS;
        sp.pColorAttachments       = cRefs;
        sp.pDepthStencilAttachment = &dRef;
        VkSubpassDependency dep{};
        dep.srcSubpass    = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
        dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = 0;
        dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo rpCI{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rpCI.attachmentCount = GBUFFER_COLOR_ATTACHMENTS + 1; rpCI.pAttachments = gbAtts;
        rpCI.subpassCount    = 1; rpCI.pSubpasses   = &sp;
        rpCI.dependencyCount = 1; rpCI.pDependencies = &dep;
        if (vkCreateRenderPass(device, &rpCI, nullptr, &gbufRenderPass) != VK_SUCCESS)
            throw std::runtime_error("vkCreateRenderPass (G-Buffer) failed");
    }

    // 지오메트리 패스용 파이프라인을 만든다.
    {
        auto vertCode2    = readFile("shaders/spv/scene.vert.spv");
        auto gbufFragCode = readFile("shaders/spv/gbuffer.frag.spv");
        VkShaderModule vertM2    = createShaderModule(vertCode2);
        VkShaderModule gbufFragM = createShaderModule(gbufFragCode);
        VkPipelineShaderStageCreateInfo gbStages[2]{};
        gbStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        gbStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        gbStages[0].module = vertM2; gbStages[0].pName = "main";
        gbStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        gbStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        gbStages[1].module = gbufFragM; gbStages[1].pName = "main";
        VkPipelineColorBlendAttachmentState gbAtt{};
        gbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        gbAtt.blendEnable = VK_FALSE;
        VkPipelineColorBlendAttachmentState gbAtts3[GBUFFER_COLOR_ATTACHMENTS] = {gbAtt, gbAtt, gbAtt, gbAtt};
        VkPipelineColorBlendStateCreateInfo gbBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        gbBlend.attachmentCount = GBUFFER_COLOR_ATTACHMENTS; gbBlend.pAttachments = gbAtts3;
        VkPipelineDepthStencilStateCreateInfo gbDs{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        gbDs.depthTestEnable  = VK_TRUE;
        gbDs.depthWriteEnable = VK_TRUE;
        gbDs.depthCompareOp   = VK_COMPARE_OP_LESS;
        VkGraphicsPipelineCreateInfo gbCI{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gbCI.stageCount = 2; gbCI.pStages = gbStages;
        gbCI.pVertexInputState   = &vertInput;
        gbCI.pInputAssemblyState = &inputAssembly;
        gbCI.pViewportState      = &viewportState;
        gbCI.pRasterizationState = &raster;
        gbCI.pMultisampleState   = &msaa;
        gbCI.pDepthStencilState  = &gbDs;
        gbCI.pColorBlendState    = &gbBlend;
        gbCI.pDynamicState       = &dynState;
        gbCI.layout              = pipelineLayout;
        gbCI.renderPass          = gbufRenderPass;
        gbCI.subpass             = 0;
        raster.cullMode  = VK_CULL_MODE_BACK_BIT;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gbCI, nullptr, &gbufPipeline) != VK_SUCCESS)
            throw std::runtime_error("vkCreateGraphicsPipelines (gbuf) failed");
        raster.cullMode = VK_CULL_MODE_NONE;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gbCI, nullptr, &gbufPipelineNoCull) != VK_SUCCESS)
            throw std::runtime_error("vkCreateGraphicsPipelines (gbuf no-cull) failed");
        vkDestroyShaderModule(device, gbufFragM, nullptr);
        vkDestroyShaderModule(device, vertM2,    nullptr);
    }

    // 디퍼드 조명 패스가 읽을 G-Buffer 샘플러 레이아웃을 만든다.
    {
        VkDescriptorSetLayoutBinding binds[GBUFFER_COLOR_ATTACHMENTS] = {};
        for (int i = 0; i < GBUFFER_COLOR_ATTACHMENTS; i++) {
            binds[i].binding         = (uint32_t)i;
            binds[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = GBUFFER_COLOR_ATTACHMENTS; ci.pBindings = binds;
        if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &deferredDescSetLayout) != VK_SUCCESS)
            throw std::runtime_error("vkCreateDescriptorSetLayout (deferred) failed");
    }

    // 전체화면 조명 패스는 씬 UBO와 G-Buffer 디스크립터를 함께 사용한다.
    {
        VkDescriptorSetLayout sets[] = {descSetLayout, deferredDescSetLayout};
        VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        ci.setLayoutCount = 2; ci.pSetLayouts = sets;
        if (vkCreatePipelineLayout(device, &ci, nullptr, &deferredLightLayout) != VK_SUCCESS)
            throw std::runtime_error("vkCreatePipelineLayout (deferred light) failed");
    }

    // 버텍스 버퍼 없이 전체화면 삼각형을 그린다.
    {
        auto dLVCode = readFile("shaders/spv/deferred_light.vert.spv");
        auto dLFCode = readFile("shaders/spv/deferred_light.frag.spv");
        VkShaderModule dLVM = createShaderModule(dLVCode);
        VkShaderModule dLFM = createShaderModule(dLFCode);
        VkPipelineShaderStageCreateInfo dStages[2]{};
        dStages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                      VK_SHADER_STAGE_VERTEX_BIT,   dLVM, "main", nullptr};
        dStages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                      VK_SHADER_STAGE_FRAGMENT_BIT, dLFM, "main", nullptr};
        VkPipelineVertexInputStateCreateInfo emptyVI{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState dAtt{};
        dAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        dAtt.blendEnable = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo dBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        dBlend.attachmentCount = 1; dBlend.pAttachments = &dAtt;
        VkPipelineDepthStencilStateCreateInfo dDs{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        dDs.depthTestEnable  = VK_FALSE;
        dDs.depthWriteEnable = VK_FALSE;
        raster.cullMode  = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkGraphicsPipelineCreateInfo dCI{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        dCI.stageCount = 2; dCI.pStages = dStages;
        dCI.pVertexInputState   = &emptyVI;
        dCI.pInputAssemblyState = &inputAssembly;
        dCI.pViewportState      = &viewportState;
        dCI.pRasterizationState = &raster;
        dCI.pMultisampleState   = &msaa;
        dCI.pDepthStencilState  = &dDs;
        dCI.pColorBlendState    = &dBlend;
        dCI.pDynamicState       = &dynState;
        dCI.layout              = deferredLightLayout;
        dCI.renderPass          = renderPass;
        dCI.subpass             = 0;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &dCI, nullptr, &deferredLightPipeline) != VK_SUCCESS)
            throw std::runtime_error("vkCreateGraphicsPipelines (deferred light) failed");
        vkDestroyShaderModule(device, dLVM, nullptr);
        vkDestroyShaderModule(device, dLFM, nullptr);
    }

    // 절차적 하늘 파이프라인: 전체화면 삼각형(deferred_light.vert 재사용) + sky.frag
    // 깊이 테스트/쓰기 없이 forward 렌더패스 맨 앞에서 배경을 채운다.
    {
        auto skyVCode = readFile("shaders/spv/deferred_light.vert.spv");
        auto skyFCode = readFile("shaders/spv/sky.frag.spv");
        VkShaderModule skyVM = createShaderModule(skyVCode);
        VkShaderModule skyFM = createShaderModule(skyFCode);
        VkPipelineShaderStageCreateInfo sStages[2]{};
        sStages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                      VK_SHADER_STAGE_VERTEX_BIT,   skyVM, "main", nullptr};
        sStages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                      VK_SHADER_STAGE_FRAGMENT_BIT, skyFM, "main", nullptr};
        VkPipelineVertexInputStateCreateInfo skyVI{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState sAtt{};
        sAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        sAtt.blendEnable = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo sBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        sBlend.attachmentCount = 1; sBlend.pAttachments = &sAtt;
        VkPipelineDepthStencilStateCreateInfo sDs{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        sDs.depthTestEnable  = VK_FALSE;
        sDs.depthWriteEnable = VK_FALSE;
        raster.cullMode  = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkGraphicsPipelineCreateInfo sCI{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        sCI.stageCount = 2; sCI.pStages = sStages;
        sCI.pVertexInputState   = &skyVI;
        sCI.pInputAssemblyState = &inputAssembly;
        sCI.pViewportState      = &viewportState;
        sCI.pRasterizationState = &raster;
        sCI.pMultisampleState   = &msaa;
        sCI.pDepthStencilState  = &sDs;
        sCI.pColorBlendState    = &sBlend;
        sCI.pDynamicState       = &dynState;
        sCI.layout              = pipelineLayout; // set 0(카메라 UBO)만 사용
        sCI.renderPass          = renderPass;
        sCI.subpass             = 0;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &sCI, nullptr, &skyPipeline) != VK_SUCCESS)
            throw std::runtime_error("vkCreateGraphicsPipelines (sky) failed");
        vkDestroyShaderModule(device, skyVM, nullptr);
        vkDestroyShaderModule(device, skyFM, nullptr);
    }

}

// 깊이 이미지
VkFormat VulkanApp::findSupportedFormat(const std::vector<VkFormat>& candidates,
                                         VkImageTiling tiling, VkFormatFeatureFlags features) {
    for (auto fmt : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physDevice, fmt, &props);
        if (tiling == VK_IMAGE_TILING_LINEAR  && (props.linearTilingFeatures  & features) == features) return fmt;
        if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) return fmt;
    }
    throw std::runtime_error("No supported depth format");
}

VkFormat VulkanApp::findDepthFormat() {
    return findSupportedFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

void VulkanApp::createDepthResources() {
    auto fmt = findDepthFormat();
    createImage(scExtent.width, scExtent.height, fmt, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
    depthImageView = createImageView(depthImage, fmt, VK_IMAGE_ASPECT_DEPTH_BIT);
}

// 프레임버퍼
void VulkanApp::createFramebuffers() {
    framebuffers.resize(scImageViews.size());
    for (size_t i = 0; i < scImageViews.size(); ++i) {
        std::array<VkImageView, 2> attachments = {scImageViews[i], depthImageView};
        VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        ci.renderPass      = renderPass;
        ci.attachmentCount = static_cast<uint32_t>(attachments.size());
        ci.pAttachments    = attachments.data();
        ci.width           = scExtent.width;
        ci.height          = scExtent.height;
        ci.layers          = 1;
        if (vkCreateFramebuffer(device, &ci, nullptr, &framebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("vkCreateFramebuffer failed");
    }
}

// 커맨드 풀과 커맨드 버퍼
void VulkanApp::createCommandPool() {
    auto idx = findQueueFamilies(physDevice);
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = idx.graphics.value();
    if (vkCreateCommandPool(device, &ci, nullptr, &commandPool) != VK_SUCCESS)
        throw std::runtime_error("vkCreateCommandPool failed");
}

void VulkanApp::createCommandBuffers() {
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
    if (vkAllocateCommandBuffers(device, &ai, commandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateCommandBuffers failed");
}

// 버퍼 헬퍼
uint32_t VulkanApp::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("No suitable memory type");
}

void VulkanApp::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                              VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size        = size;
    ci.usage       = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &ci, nullptr, &buf) != VK_SUCCESS)
        throw std::runtime_error("vkCreateBuffer failed");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, buf, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
    if (vkAllocateMemory(device, &ai, nullptr, &mem) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateMemory failed");
    vkBindBufferMemory(device, buf, mem, 0);
}


void VulkanApp::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &ai, &cmd);

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    VkBufferCopy copy{0, 0, size};
    vkCmdCopyBuffer(cmd, src, dst, 1, &copy);
    vkEndCommandBuffer(cmd);

    // 전체 큐를 멈추지 않도록 fence로 복사 완료만 기다린다.
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    vkCreateFence(device, &fci, nullptr, &fence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(graphicsQueue, 1, &si, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

// 버텍스/인덱스 버퍼
void VulkanApp::createVertexBuffer() {
    VkDeviceSize size = sizeof(Vertex) * vertices.size();
    VkBuffer staging; VkDeviceMemory stagingMem;
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging, stagingMem);
    void* data;
    vkMapMemory(device, stagingMem, 0, size, 0, &data);
    memcpy(data, vertices.data(), size);
    vkUnmapMemory(device, stagingMem);
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
    copyBuffer(staging, vertexBuffer, size);
    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);
}

void VulkanApp::createIndexBuffer() {
    VkDeviceSize size = sizeof(uint32_t) * indices.size();
    VkBuffer staging; VkDeviceMemory stagingMem;
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging, stagingMem);
    void* data;
    vkMapMemory(device, stagingMem, 0, size, 0, &data);
    memcpy(data, indices.data(), size);
    vkUnmapMemory(device, stagingMem);
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);
    copyBuffer(staging, indexBuffer, size);
    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);
}

// 버텍스와 인덱스 버퍼를 하나의 커맨드 버퍼/펜스로 함께 업로드한다.
// 씬 재로드 때 버텍스/인덱스 업로드가 큐 제출 두 번으로 쪼개지는 것을 피한다.
void VulkanApp::createSceneBuffers() {
    VkDeviceSize vSize = sizeof(Vertex)   * vertices.size();
    VkDeviceSize iSize = sizeof(uint32_t) * indices.size();

    // 버텍스 스테이징 버퍼를 준비한다.
    VkBuffer vStaging; VkDeviceMemory vStagingMem;
    createBuffer(vSize,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 vStaging, vStagingMem);
    void* vData;
    vkMapMemory(device, vStagingMem, 0, vSize, 0, &vData);
    memcpy(vData, vertices.data(), vSize);
    vkUnmapMemory(device, vStagingMem);
    createBuffer(vSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vertexBuffer, vertexBufferMemory);

    // 인덱스 스테이징 버퍼를 준비한다.
    VkBuffer iStaging; VkDeviceMemory iStagingMem;
    createBuffer(iSize,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 iStaging, iStagingMem);
    void* iData;
    vkMapMemory(device, iStagingMem, 0, iSize, 0, &iData);
    memcpy(iData, indices.data(), iSize);
    vkUnmapMemory(device, iStagingMem);
    createBuffer(iSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 indexBuffer, indexBufferMemory);

    // 하나의 커맨드 버퍼에 복사를 모두 기록한다.
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &ai, &cmd);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);
    VkBufferCopy vCopy{0, 0, vSize};
    vkCmdCopyBuffer(cmd, vStaging, vertexBuffer, 1, &vCopy);
    VkBufferCopy iCopy{0, 0, iSize};
    vkCmdCopyBuffer(cmd, iStaging, indexBuffer,  1, &iCopy);
    vkEndCommandBuffer(cmd);

    // 한 번 제출하고 하나의 fence만 기다린다.
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    vkCreateFence(device, &fci, nullptr, &fence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(graphicsQueue, 1, &si, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, commandPool, 1, &cmd);

    vkDestroyBuffer(device, vStaging, nullptr); vkFreeMemory(device, vStagingMem, nullptr);
    vkDestroyBuffer(device, iStaging, nullptr); vkFreeMemory(device, iStagingMem, nullptr);
}


// 프레임별 UBO
void VulkanApp::createUniformBuffers() {
    VkDeviceSize camSize   = sizeof(CameraUBO);
    VkDeviceSize lightSize = sizeof(SceneLightUBO);

    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBufferMemories.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
    cullUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    cullUniformBufferMemories.resize(MAX_FRAMES_IN_FLIGHT);
    cullUniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
    sceneLightUBOs.resize(MAX_FRAMES_IN_FLIGHT);
    sceneLightUBOMemories.resize(MAX_FRAMES_IN_FLIGHT);
    sceneLightUBOMapped.resize(MAX_FRAMES_IN_FLIGHT);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        createBuffer(camSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     uniformBuffers[i], uniformBufferMemories[i]);
        vkMapMemory(device, uniformBufferMemories[i], 0, camSize, 0, &uniformBuffersMapped[i]);

        createBuffer(camSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     cullUniformBuffers[i], cullUniformBufferMemories[i]);
        vkMapMemory(device, cullUniformBufferMemories[i], 0, camSize, 0, &cullUniformBuffersMapped[i]);

        createBuffer(lightSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     sceneLightUBOs[i], sceneLightUBOMemories[i]);
        vkMapMemory(device, sceneLightUBOMemories[i], 0, lightSize, 0, &sceneLightUBOMapped[i]);
        // 초기값: 조명 없음, fallback 모드
        memset(sceneLightUBOMapped[i], 0, lightSize);
    }
}

// 디스크립터 풀과 세트
void VulkanApp::createDescriptorPool() {
    // 렌더 카메라와 컬링 카메라가 각각 프레임 수만큼 디스크립터 세트를 가진다.
    // 각 세트는 카메라 UBO, 조명 UBO, 텍스처 샘플러 배열을 포함한다.
    uint32_t setCount = 2u * (uint32_t)MAX_FRAMES_IN_FLIGHT;
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         setCount * 2}, // binding 0=카메라, binding 2=씬 조명
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         (MAX_TEXTURES + 1) * setCount}, // binding 1=텍스처 배열, binding 3=섀도맵
    };
    VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.maxSets       = setCount;
    ci.poolSizeCount = 2;
    ci.pPoolSizes    = poolSizes;
    if (vkCreateDescriptorPool(device, &ci, nullptr, &descPool) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDescriptorPool failed");
}

void VulkanApp::createDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descSetLayout);
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = descPool;
    ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    ai.pSetLayouts        = layouts.data();
    descSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(device, &ai, descSets.data()) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateDescriptorSets failed");

    cullDescSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(device, &ai, cullDescSets.data()) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateDescriptorSets (cull) failed");

    // 텍스처 배열 (binding 1): 모든 슬롯을 기본 1×1 흰색 텍스처로 채움.
    // 실제 씬 텍스처는 createTextureResources() 에서 갱신된다.
    std::vector<VkDescriptorImageInfo> defaultImgInfos(MAX_TEXTURES);
    for (uint32_t s = 0; s < MAX_TEXTURES; ++s) {
        defaultImgInfos[s].sampler     = texSampler;
        defaultImgInfos[s].imageView   = defaultTexView;
        defaultImgInfos[s].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        // binding 0: 카메라 UBO
        VkDescriptorBufferInfo bufInfo{uniformBuffers[i], 0, sizeof(CameraUBO)};
        VkWriteDescriptorSet uboWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        uboWrite.dstSet          = descSets[i];
        uboWrite.dstBinding      = 0;
        uboWrite.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboWrite.descriptorCount = 1;
        uboWrite.pBufferInfo     = &bufInfo;

        // binding 1: 텍스처 배열(기본 흰색으로 초기화)
        VkWriteDescriptorSet texWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        texWrite.dstSet          = descSets[i];
        texWrite.dstBinding      = 1;
        texWrite.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        texWrite.descriptorCount = MAX_TEXTURES;
        texWrite.pImageInfo      = defaultImgInfos.data();

        // binding 2: 씬 조명 UBO
        VkDescriptorBufferInfo lightBufInfo{sceneLightUBOs[i], 0, sizeof(SceneLightUBO)};
        VkWriteDescriptorSet lightWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        lightWrite.dstSet          = descSets[i];
        lightWrite.dstBinding      = 2;
        lightWrite.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        lightWrite.descriptorCount = 1;
        lightWrite.pBufferInfo     = &lightBufInfo;

        // binding 3: 방향광 섀도맵 (프레임 슬롯별 깊이 이미지)
        VkDescriptorImageInfo shadowInfo{shadowSampler, shadowImageViews[i],
                                         VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet shadowWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        shadowWrite.dstSet          = descSets[i];
        shadowWrite.dstBinding      = 3;
        shadowWrite.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        shadowWrite.descriptorCount = 1;
        shadowWrite.pImageInfo      = &shadowInfo;

        VkWriteDescriptorSet writes[] = {uboWrite, texWrite, lightWrite, shadowWrite};
        vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);

        // 컬링용 세트도 같은 레이아웃을 쓰므로 텍스처 배열까지 유효하게 채운다.
        VkDescriptorBufferInfo cullBufInfo{cullUniformBuffers[i], 0, sizeof(CameraUBO)};
        VkWriteDescriptorSet cullUboWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        cullUboWrite.dstSet          = cullDescSets[i];
        cullUboWrite.dstBinding      = 0;
        cullUboWrite.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        cullUboWrite.descriptorCount = 1;
        cullUboWrite.pBufferInfo     = &cullBufInfo;

        VkWriteDescriptorSet cullTexWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        cullTexWrite.dstSet          = cullDescSets[i];
        cullTexWrite.dstBinding      = 1;
        cullTexWrite.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        cullTexWrite.descriptorCount = MAX_TEXTURES;
        cullTexWrite.pImageInfo      = defaultImgInfos.data();

        VkDescriptorBufferInfo cullLightBufInfo{sceneLightUBOs[i], 0, sizeof(SceneLightUBO)};
        VkWriteDescriptorSet cullLightWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        cullLightWrite.dstSet          = cullDescSets[i];
        cullLightWrite.dstBinding      = 2;
        cullLightWrite.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        cullLightWrite.descriptorCount = 1;
        cullLightWrite.pBufferInfo     = &cullLightBufInfo;

        VkDescriptorImageInfo cullShadowInfo{shadowSampler, shadowImageViews[i],
                                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet cullShadowWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        cullShadowWrite.dstSet          = cullDescSets[i];
        cullShadowWrite.dstBinding      = 3;
        cullShadowWrite.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        cullShadowWrite.descriptorCount = 1;
        cullShadowWrite.pImageInfo      = &cullShadowInfo;

        VkWriteDescriptorSet cullWrites[] = {cullUboWrite, cullTexWrite, cullLightWrite, cullShadowWrite};
        vkUpdateDescriptorSets(device, 4, cullWrites, 0, nullptr);
    }
}

// 프레임 동기화 객체
void VulkanApp::createSyncObjects() {
    imageAvailableSems.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo     fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkCreateSemaphore(device, &sci, nullptr, &imageAvailableSems[i]);
        vkCreateFence(device, &fci, nullptr, &inFlightFences[i]);
    }
    recreatePresentSemaphores();
}

void VulkanApp::recreatePresentSemaphores() {
    for (VkSemaphore sem : renderFinishedSems)
        if (sem != VK_NULL_HANDLE)
            vkDestroySemaphore(device, sem, nullptr);

    renderFinishedSems.assign(scImages.size(), VK_NULL_HANDLE);
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (size_t i = 0; i < renderFinishedSems.size(); ++i)
        vkCreateSemaphore(device, &sci, nullptr, &renderFinishedSems[i]);
}

// GPU 인스턴싱 리소스
void VulkanApp::createInstanceResources() {
    const VkDeviceSize ssboSize = MAX_INSTANCES * sizeof(glm::mat4);

    // 프레임 슬롯마다 host-visible SSBO를 하나씩 둔다.
    instanceSSBOs.resize(MAX_FRAMES_IN_FLIGHT);
    instanceSSBOMemories.resize(MAX_FRAMES_IN_FLIGHT);
    instanceStagingBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    instanceStagingMemories.resize(MAX_FRAMES_IN_FLIGHT);
    instanceSSBOMapped.resize(MAX_FRAMES_IN_FLIGHT);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        createBuffer(ssboSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     instanceSSBOs[i], instanceSSBOMemories[i]);
        createBuffer(ssboSize,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     instanceStagingBuffers[i], instanceStagingMemories[i]);
        vkMapMemory(device, instanceStagingMemories[i], 0, ssboSize, 0, &instanceSSBOMapped[i]);
    }

    // 프레임별 SSBO descriptor set용 풀을 만든다.
    VkDescriptorPoolSize ssboPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)MAX_FRAMES_IN_FLIGHT};
    VkDescriptorPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolCI.maxSets       = MAX_FRAMES_IN_FLIGHT;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes    = &ssboPoolSize;
    if (vkCreateDescriptorPool(device, &poolCI, nullptr, &instanceDescPool) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDescriptorPool (instance) failed");

    // 인스턴싱 파이프라인 set 1을 프레임별로 할당한다.
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, instanceDescSetLayout);
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = instanceDescPool;
    ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    ai.pSetLayouts        = layouts.data();
    instanceDescSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(device, &ai, instanceDescSets.data()) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateDescriptorSets (instance) failed");

    // 각 descriptor set이 자기 프레임의 SSBO를 가리키게 한다.
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        VkDescriptorBufferInfo bufInfo{instanceSSBOs[i], 0, ssboSize};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet          = instanceDescSets[i];
        write.dstBinding      = 0;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo     = &bufInfo;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
}

// 오클루전 쿼리 풀
void VulkanApp::createOcclusionQueryPool() {
    // 프레임 슬롯마다 draw object 수만큼 오클루전 쿼리를 확보한다.
    occQueryCount = static_cast<uint32_t>(drawObjects.size());
    VkQueryPoolCreateInfo ci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    ci.queryType  = VK_QUERY_TYPE_OCCLUSION;
    ci.queryCount = occQueryCount * MAX_FRAMES_IN_FLIGHT;
    if (vkCreateQueryPool(device, &ci, nullptr, &occlusionQueryPool) != VK_SUCCESS)
        throw std::runtime_error("vkCreateQueryPool failed");
}

void VulkanApp::resetOcclusionState() {
    const size_t objectCount = drawObjects.size();
    occResults.assign(objectCount, 1u);
    occQueryBuf.assign(objectCount * 2, 0u);
    occVisible.assign(objectCount, 1u);
    occZeroStreak.assign(objectCount, 0u);
    occWarmupFrames = MAX_FRAMES_IN_FLIGHT + OCC_ZERO_FRAMES_TO_HIDE;
}

bool VulkanApp::isOcclusionHidden(int objectIndex) const {
    if (objectIndex < 0) return false;
    const size_t idx = static_cast<size_t>(objectIndex);
    return idx < occVisible.size() && occVisible[idx] == 0;
}

// 기즈모 라인 파이프라인
void VulkanApp::createGizmoPipeline() {
    auto vertCode = readFile("shaders/spv/gizmo.vert.spv");
    auto fragCode = readFile("shaders/spv/gizmo.frag.spv");
    VkShaderModule vertM = createShaderModule(vertCode);
    VkShaderModule fragM = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertM; stages[0].pName = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragM; stages[1].pName = "main";

    // 기즈모도 동일한 버텍스 포맷(pos+normal+color)을 사용한다.
    auto binding = Vertex::getBindingDesc();
    auto attrs   = Vertex::getAttrDescs();
    VkPipelineVertexInputStateCreateInfo vertInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertInput.vertexBindingDescriptionCount   = 1;
    vertInput.pVertexBindingDescriptions      = &binding;
    vertInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
    vertInput.pVertexAttributeDescriptions    = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; // LINE_LIST 토폴로지

    VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vs.viewportCount = 1;
    vs.scissorCount  = 1;

    VkDynamicState gizmoDynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo gizmoDyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    gizmoDyn.dynamicStateCount = 2;
    gizmoDyn.pDynamicStates    = gizmoDynStates;

    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode    = VK_CULL_MODE_NONE;
    raster.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo msaa{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // 깊이 테스트를 꺼서 기즈모를 항상 위에 그린다.
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable  = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|
                        VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
    ba.blendEnable    = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1; blend.pAttachments = &ba;

    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount          = 2;
    ci.pStages             = stages;
    ci.pVertexInputState   = &vertInput;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState      = &vs;
    ci.pRasterizationState = &raster;
    ci.pMultisampleState   = &msaa;
    ci.pDepthStencilState  = &ds;
    ci.pColorBlendState    = &blend;
    ci.pDynamicState       = &gizmoDyn;
    ci.layout              = pipelineLayout;
    ci.renderPass          = renderPass;
    ci.subpass             = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &gizmoPipeline) != VK_SUCCESS)
        throw std::runtime_error("vkCreateGraphicsPipelines (gizmo) failed");

    vkDestroyShaderModule(device, vertM, nullptr);
    vkDestroyShaderModule(device, fragM, nullptr);
}

// 프레임별 기즈모 버텍스 버퍼
void VulkanApp::createGizmoBuffers() {
    const VkDeviceSize sz = GIZMO_MAX_VERTS * sizeof(Vertex);
    gizmoVBs.resize(MAX_FRAMES_IN_FLIGHT);
    gizmoVBMemories.resize(MAX_FRAMES_IN_FLIGHT);
    gizmoVBMapped.resize(MAX_FRAMES_IN_FLIGHT);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        createBuffer(sz,
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     gizmoVBs[i], gizmoVBMemories[i]);
        vkMapMemory(device, gizmoVBMemories[i], 0, sz, 0, &gizmoVBMapped[i]);
    }
}

// 배치 카메라 기즈모 라인 생성
void VulkanApp::buildGizmoGeometry(uint32_t frame) {
    std::vector<Vertex> verts;
    verts.reserve(GIZMO_MAX_VERTS);

    auto addLine = [&](glm::vec3 a, glm::vec3 b, glm::vec3 col) {
        verts.push_back({a, {0,1,0}, col});
        verts.push_back({b, {0,1,0}, col});
    };

    glm::vec3 pos   = camera.position;
    glm::vec3 fwd   = camera.getCameraFront();
    glm::vec3 right = camera.getRight();
    glm::vec3 up    = glm::normalize(glm::cross(right, fwd));

    // 카메라 본체를 방향이 있는 박스로 그린다.
    const float hw = 0.22f, hh = 0.14f, hd = 0.16f;
    glm::vec3 bodyCol = {1.0f, 0.55f, 0.05f}; // 주황색

    // 뒤/앞 면의 8개 꼭짓점을 계산한다.
    glm::vec3 c[8] = {
        pos + right*(-hw) + up*(-hh) + fwd*(-hd), // 0 BLB
        pos + right*(+hw) + up*(-hh) + fwd*(-hd), // 1 BRB
        pos + right*(+hw) + up*(+hh) + fwd*(-hd), // 2 BRT
        pos + right*(-hw) + up*(+hh) + fwd*(-hd), // 3 BLT
        pos + right*(-hw) + up*(-hh) + fwd*(+hd), // 4 FLB
        pos + right*(+hw) + up*(-hh) + fwd*(+hd), // 5 FRB
        pos + right*(+hw) + up*(+hh) + fwd*(+hd), // 6 FRT
        pos + right*(-hw) + up*(+hh) + fwd*(+hd), // 7 FLT
    };
    // 박스의 12개 모서리를 연결한다.
    int edges[12][2] = {
        {0,1},{1,2},{2,3},{3,0}, // 뒤쪽 면
        {4,5},{5,6},{6,7},{7,4}, // 앞쪽 면
        {0,4},{1,5},{2,6},{3,7} // 앞뒤 연결 모서리
    };
    for (auto& e : edges) addLine(c[e[0]], c[e[1]], bodyCol);

    // 위쪽 방향을 청록색 선으로 표시한다.
    glm::vec3 topMid = (c[3] + c[2]) * 0.5f;
    addLine(topMid, topMid + up * 0.18f, {0.4f, 0.85f, 1.0f});

    // 렌즈 방향을 앞쪽 꼭짓점에서 뻗는 선으로 표시한다.
    glm::vec3 lensColor = {1.0f, 0.9f, 0.2f}; // 노란색
    glm::vec3 lensTip   = pos + fwd * (hd + 0.18f);
    addLine(c[4], lensTip, lensColor);
    addLine(c[5], lensTip, lensColor);
    addLine(c[6], lensTip, lensColor);
    addLine(c[7], lensTip, lensColor);

    // 카메라 시야 절두체를 선으로 표시한다.
    float fovY   = glm::radians(60.f);
    float aspect = scExtent.width / (float)scExtent.height;
    glm::vec3 frustCol = {0.85f, 1.0f, 0.3f}; // 연두색

    auto makeRect = [&](float dist) {
        float hY = dist * std::tan(fovY * 0.5f);
        float hX = hY * aspect;
        glm::vec3 center = pos + fwd * dist;
        return std::array<glm::vec3,4>{
            center + right*(-hX) + up*(-hY),
            center + right*(+hX) + up*(-hY),
            center + right*(+hX) + up*(+hY),
            center + right*(-hX) + up*(+hY),
        };
    };

    auto nearR = makeRect(0.5f);
    auto farR  = makeRect(7.0f);

    // 카메라 위치에서 원평면 네 꼭짓점으로 선을 뻗는다.
    for (int i = 0; i < 4; ++i)
        addLine(pos, farR[i], frustCol);
    // near plane 사각형
    for (int i = 0; i < 4; ++i)
        addLine(nearR[i], nearR[(i+1)%4], frustCol);
    // far plane 사각형
    for (int i = 0; i < 4; ++i)
        addLine(farR[i], farR[(i+1)%4], frustCol);
    // near/far plane 연결선
    for (int i = 0; i < 4; ++i)
        addLine(nearR[i], farR[i], frustCol);

    // LOD/원거리 거리 링 (수평 원, 32 세그먼트)
    constexpr int   RING_SEGS = 32;
    constexpr float TWO_PI    = 6.28318530718f;
    auto addRing = [&](float radius, float yHeight, glm::vec3 col) {
        glm::vec3 center = glm::vec3(pos.x, yHeight, pos.z);
        for (int i = 0; i < RING_SEGS; ++i) {
            float a0 = TWO_PI * i       / RING_SEGS;
            float a1 = TWO_PI * (i + 1) / RING_SEGS;
            glm::vec3 p0 = center + glm::vec3(std::cos(a0)*radius, 0.f, std::sin(a0)*radius);
            glm::vec3 p1 = center + glm::vec3(std::cos(a1)*radius, 0.f, std::sin(a1)*radius);
            addLine(p0, p1, col);
        }
    };
    addRing(18.f, pos.y, {1.0f, 0.60f, 0.05f}); // 주황 = LOD1 전환 거리 (18 m)
    addRing(36.f, pos.y, {1.0f, 0.20f, 0.05f}); // 빨강 = LOD2 전환 거리 (36 m)
    addRing(viewDistMax, pos.y, {0.2f, 0.8f, 1.0f}); // 하늘색 = 원거리 컬링 한계

    gizmoVertCount = static_cast<uint32_t>(verts.size());
    memcpy(gizmoVBMapped[frame], verts.data(), verts.size() * sizeof(Vertex));
}

// 정적 렌더링 헬퍼
Frustum VulkanApp::extractFrustum(const glm::mat4& vp) {
    // Vulkan 클립 공간 규칙에 맞춰 평면을 추출한다.
    // -w <= x <= w, -w <= y <= w, 0 <= z <= w
    // GLM은 열 우선 저장이므로 명시적으로 행 벡터를 꺼낸다.
    auto row = [&](int r) {
        return glm::vec4(vp[0][r], vp[1][r], vp[2][r], vp[3][r]);
    };
    const glm::vec4 r0 = row(0);
    const glm::vec4 r1 = row(1);
    const glm::vec4 r2 = row(2);
    const glm::vec4 r3 = row(3);

    Frustum f;
    f.planes[0] = r3 + r0; // 왼쪽
    f.planes[1] = r3 - r0; // 오른쪽
    f.planes[2] = r3 + r1; // 아래
    f.planes[3] = r3 - r1; // 위
    f.planes[4] = r2;      // 가까운 평면
    f.planes[5] = r3 - r2; // 먼 평면
    // 평면 법선을 정규화한다.
    for (auto& p : f.planes) {
        float len = glm::length(glm::vec3(p));
        if (len > 0.f) p /= len;
    }
    return f;
}

bool VulkanApp::sphereInFrustum(const Frustum& f, glm::vec3 c, float r) {
    constexpr float kCullEpsilon = 1e-4f;
    for (const auto& p : f.planes)
        if (glm::dot(glm::vec3(p), c) + p.w < -r - kCullEpsilon)
            return false;
    return true;
}

void VulkanApp::computeBoundSphere(DrawObject& obj, glm::vec3 bmin, glm::vec3 bmax) {
    // AABB의 8개 꼭짓점을 모델 행렬로 월드 공간에 옮긴다.
    const glm::mat4& M = obj.push.model;
    glm::vec3 corners[8];
    int k = 0;
    for (float x : {bmin.x, bmax.x})
        for (float y : {bmin.y, bmax.y})
            for (float z : {bmin.z, bmax.z})
                corners[k++] = glm::vec3(M * glm::vec4(x, y, z, 1.f));

    // 바운딩 구 중심은 변환된 꼭짓점들의 평균으로 잡는다.
    glm::vec3 center{};
    for (auto& c : corners) center += c;
    center /= 8.f;

    // 반지름은 중심에서 가장 먼 꼭짓점까지의 거리다.
    float radius = 0.f;
    for (auto& c : corners)
        radius = std::max(radius, glm::distance(c, center));

    obj.boundCenter = center;
    obj.boundRadius = radius;
}

// 이미지와 이미지 뷰 헬퍼
void VulkanApp::createImage(uint32_t w, uint32_t h, VkFormat fmt, VkImageTiling tiling,
                             VkImageUsageFlags usage, VkMemoryPropertyFlags props,
                             VkImage& img, VkDeviceMemory& mem) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType     = VK_IMAGE_TYPE_2D;
    ci.extent        = {w, h, 1};
    ci.mipLevels     = 1;
    ci.arrayLayers   = 1;
    ci.format        = fmt;
    ci.tiling        = tiling;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ci.usage         = usage;
    ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(device, &ci, nullptr, &img) != VK_SUCCESS)
        throw std::runtime_error("vkCreateImage failed");

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, img, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
    if (vkAllocateMemory(device, &ai, nullptr, &mem) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateMemory (image) failed");
    vkBindImageMemory(device, img, mem, 0);
}

VkImageView VulkanApp::createImageView(VkImage img, VkFormat fmt, VkImageAspectFlags aspect) {
    VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ci.image                           = img;
    ci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    ci.format                          = fmt;
    ci.subresourceRange.aspectMask     = aspect;
    ci.subresourceRange.baseMipLevel   = 0;
    ci.subresourceRange.levelCount     = 1;
    ci.subresourceRange.baseArrayLayer = 0;
    ci.subresourceRange.layerCount     = 1;
    VkImageView view;
    if (vkCreateImageView(device, &ci, nullptr, &view) != VK_SUCCESS)
        throw std::runtime_error("vkCreateImageView failed");
    return view;
}

// 메인 루프
// 최적화 토글 입력 처리

// 텍스처 업로드용 일회성 커맨드 버퍼 헬퍼
static VkCommandBuffer beginSingleCmd(VkDevice device, VkCommandPool pool) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cmd; vkAllocateCommandBuffers(device, &ai, &cmd);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}
static void endSingleCmd(VkDevice dev, VkCommandPool pool, VkQueue q, VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(q);
    vkFreeCommandBuffers(dev, pool, 1, &cmd);
}
static void transitionImageLayout(VkCommandBuffer cmd, VkImage image,
                                   VkImageLayout oldL, VkImageLayout newL)
{
    VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    bar.oldLayout = oldL; bar.newLayout = newL;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.image = image;
    bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkPipelineStageFlags src, dst;
    if (oldL == VK_IMAGE_LAYOUT_UNDEFINED) {
        bar.srcAccessMask = 0;
        bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        src = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT; dst = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else {
        bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        src = VK_PIPELINE_STAGE_TRANSFER_BIT; dst = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    vkCmdPipelineBarrier(cmd, src, dst, 0, 0, nullptr, 0, nullptr, 1, &bar);
}

void VulkanApp::createDefaultTextureAndSampler() {
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_LINEAR; sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.anisotropyEnable = VK_TRUE; sci.maxAnisotropy = 16.0f;
    sci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    if (vkCreateSampler(device, &sci, nullptr, &texSampler) != VK_SUCCESS)
        throw std::runtime_error("vkCreateSampler failed");

    const uint8_t white[4] = {255,255,255,255};
    createImage(1, 1, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, defaultTexImage, defaultTexMemory);
    VkBuffer sb; VkDeviceMemory sm;
    createBuffer(4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, sb, sm);
    void* mp; vkMapMemory(device, sm, 0, 4, 0, &mp); memcpy(mp, white, 4); vkUnmapMemory(device, sm);
    VkCommandBuffer cmd = beginSingleCmd(device, commandPool);
    transitionImageLayout(cmd, defaultTexImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy reg{}; reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; reg.imageExtent={1,1,1};
    vkCmdCopyBufferToImage(cmd, sb, defaultTexImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
    transitionImageLayout(cmd, defaultTexImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    endSingleCmd(device, commandPool, graphicsQueue, cmd);
    vkDestroyBuffer(device, sb, nullptr); vkFreeMemory(device, sm, nullptr);
    defaultTexView = createImageView(defaultTexImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
}

void VulkanApp::cleanupTextureResources() {
    for (auto& v : texViews)    vkDestroyImageView(device, v, nullptr);
    for (auto& i : texImages)   vkDestroyImage(device, i, nullptr);
    for (auto& mm : texMemories) vkFreeMemory(device, mm, nullptr);
    texViews.clear(); texImages.clear(); texMemories.clear();
}

void VulkanApp::createTextureResources() {
    cleanupTextureResources();
    if (!sceneTextures.empty()) {
        std::vector<VkBuffer>       sbs(sceneTextures.size(), VK_NULL_HANDLE);
        std::vector<VkDeviceMemory> sms(sceneTextures.size(), VK_NULL_HANDLE);
        texImages.resize(sceneTextures.size(),   VK_NULL_HANDLE);
        texMemories.resize(sceneTextures.size(), VK_NULL_HANDLE);
        for (size_t i = 0; i < sceneTextures.size(); ++i) {
            const auto& td = sceneTextures[i];
            VkDeviceSize sz = (VkDeviceSize)td.width * td.height * 4;
            createImage((uint32_t)td.width,(uint32_t)td.height,VK_FORMAT_R8G8B8A8_SRGB,
                        VK_IMAGE_TILING_OPTIMAL,
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, texImages[i], texMemories[i]);
            createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         sbs[i], sms[i]);
            void* p; vkMapMemory(device, sms[i], 0, sz, 0, &p);
            memcpy(p, td.pixels.data(), (size_t)sz); vkUnmapMemory(device, sms[i]);
        }
        VkCommandBuffer cmd = beginSingleCmd(device, commandPool);
        for (size_t i = 0; i < sceneTextures.size(); ++i) {
            transitionImageLayout(cmd, texImages[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
            r.imageExtent = {(uint32_t)sceneTextures[i].width,(uint32_t)sceneTextures[i].height,1};
            vkCmdCopyBufferToImage(cmd, sbs[i], texImages[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
            transitionImageLayout(cmd, texImages[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        endSingleCmd(device, commandPool, graphicsQueue, cmd);
        for (size_t i = 0; i < sceneTextures.size(); ++i) {
            vkDestroyBuffer(device, sbs[i], nullptr); vkFreeMemory(device, sms[i], nullptr);
        }
        texViews.resize(sceneTextures.size());
        for (size_t i = 0; i < sceneTextures.size(); ++i)
            texViews[i] = createImageView(texImages[i], VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
    }
    // descriptor set의 텍스처 배열을 새 이미지 뷰로 갱신한다.
    std::vector<VkDescriptorImageInfo> imgInfos(MAX_TEXTURES);
    for (uint32_t s = 0; s < MAX_TEXTURES; ++s) {
        imgInfos[s].sampler     = texSampler;
        imgInfos[s].imageView   = (s < texViews.size()) ? texViews[s] : defaultTexView;
        imgInfos[s].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    for (int f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f) {
        auto doWrite = [&](VkDescriptorSet dst) {
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = dst; w.dstBinding = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.descriptorCount = MAX_TEXTURES; w.pImageInfo = imgInfos.data();
            vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
        };
        doWrite(descSets[f]); doWrite(cullDescSets[f]);
    }
}

// ============================================================================
// [5] 입력 / 메인 루프
// ============================================================================

// handleOptKeys — 매 프레임 폴링 방식으로 토글 키를 처리한다.
// 모든 키는 "이전 프레임에 안 눌려 있었고 지금 눌림" (rising edge) 일 때만
// 1회 토글된다 — 꾹 누르고 있어도 연타되지 않게 하는 디바운스 패턴.
//  숫자 1~9 : 최적화 기법 개별 토글  |  0 : 전체 ON/OFF
//  B 바닥 밝기  F 원거리 클리핑  L 씬 조명  N 환경광  V 발광
//  H 현실적 셰이더  Y 낮/밤 주기  X 안개  K 구름량  I FPS 캡
void VulkanApp::handleOptKeys() {
    // 키를 누르는 순간에만 토글되도록 이전 상태를 기억한다.
    static bool prevKeys[10] = {};
    struct { int key; bool* flag; } binds[] = {
        { GLFW_KEY_1, &optFlags.frustumCulling   },
        { GLFW_KEY_2, &optFlags.lod              },
        { GLFW_KEY_3, &optFlags.instancing       },
        { GLFW_KEY_4, &optFlags.backfaceCulling  },
        { GLFW_KEY_5, &optFlags.depthSort        },
        { GLFW_KEY_6, &optFlags.occlusionCulling },
        { GLFW_KEY_7, &optFlags.viewDistCulling  },
        { GLFW_KEY_8, &optFlags.smallCulling     },
        { GLFW_KEY_9, &optFlags.deferredShading  },
    };
    for (int i = 0; i < 9; ++i) {
        bool pressed = (glfwGetKey(window, binds[i].key) == GLFW_PRESS);
        // 적응형 모드(` 키) 중에는 컨트롤러가 소유한 플래그의 수동 토글을 잠근다.
        // 7(원거리)·8(소형)은 적응형 대상이 아니므로(품질 트레이드오프 기법) 항상 수동 가능.
        bool adaptiveLocked = adaptive.enabled &&
                              binds[i].key != GLFW_KEY_7 && binds[i].key != GLFW_KEY_8;
        if (pressed && !prevKeys[i] && !adaptiveLocked) {
            *binds[i].flag = !*binds[i].flag;
            if (binds[i].key == GLFW_KEY_6) {
                if (*binds[i].flag) // 방금 켜진 경우에만 워밍업
                    resetOcclusionState();
            }
        }
        prevKeys[i] = pressed;
    }

    // Ctrl + 방향키로 HUD 페이지를 넘긴다 (좌/위 = 이전, 우/아래 = 다음).
    {
        bool ctrl = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL)  == GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
        static bool prevPgL = false, prevPgR = false;
        bool pgL = ctrl && (glfwGetKey(window, GLFW_KEY_LEFT)  == GLFW_PRESS ||
                            glfwGetKey(window, GLFW_KEY_UP)    == GLFW_PRESS);
        bool pgR = ctrl && (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS ||
                            glfwGetKey(window, GLFW_KEY_DOWN)  == GLFW_PRESS);
        if (pgL && !prevPgL) hudPage = (hudPage + HUD_PAGE_COUNT - 1) % HUD_PAGE_COUNT;
        if (pgR && !prevPgR) hudPage = (hudPage + 1) % HUD_PAGE_COUNT;
        prevPgL = pgL;
        prevPgR = pgR;
    }

    // L 키로 씬 조명 전체를 켜고 끈다.
    static bool prevL = false;
    bool lPressed = (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS);
    if (lPressed && !prevL) {
        sceneLightsOn = !sceneLightsOn;
        sceneLightDirty = true;
    }
    prevL = lPressed;

    // N 키로 환경광을 켜고 끈다.
    static bool prevN = false;
    bool nPressed = (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS);
    if (nPressed && !prevN) {
        ambientOn = !ambientOn;
        sceneLightDirty = true;
    }
    prevN = nPressed;

    // V 키로 발광 재질을 켜고 끈다.
    static bool prevV = false;
    bool vPressed = (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS);
    if (vPressed && !prevV) {
        emissiveOn = !emissiveOn;
        sceneLightDirty = true;
    }
    prevV = vPressed;

    // H 키로 현실적 셰이더(PBR·하늘·안개·ACES) <-> 클래식(원본)을 전환한다.
    static bool prevH = false;
    bool hPressed = (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS);
    if (hPressed && !prevH)
        realisticShading = !realisticShading;
    prevH = hPressed;

    // Y 키로 낮/밤 자동 주기를 토글한다.
    static bool prevY = false;
    bool yPressed = (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS);
    if (yPressed && !prevY) {
        dayNightCycle = !dayNightCycle;
        sceneLightDirty = true; // useSceneLights 가 주기 여부에 따라 달라진다.
    }
    prevY = yPressed;

    // X 키로 거리 안개를 토글한다.
    static bool prevX = false;
    bool xPressed = (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS);
    if (xPressed && !prevX)
        fogOn = !fogOn;
    prevX = xPressed;

    // I 키로 FPS 캡을 순환한다 (무제한 -> 144 -> 60 -> 30).
    // 캡 ON = 고정 프레임 모드: 같은 작업량에서 CPU/GPU/RAM 사용률을 비교한다.
    static bool prevI = false;
    bool iPressed = (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS);
    if (iPressed && !prevI) {
        fpsCapIndex = (fpsCapIndex + 1) % 4;
        if (currentFpsCap() > 0)
            printf("[FPS Cap] %d fps (fixed-frame mode)\n", currentFpsCap());
        else
            printf("[FPS Cap] OFF (max-FPS mode)\n");
    }
    prevI = iPressed;

    // K 키로 구름량을 순환한다 (0 -> 25 -> 45 -> 70 -> 95%).
    static bool prevKCloud = false;
    bool kPressed = (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS);
    if (kPressed && !prevKCloud) {
        if      (cloudiness < 0.10f) cloudiness = 0.25f;
        else if (cloudiness < 0.30f) cloudiness = 0.45f;
        else if (cloudiness < 0.50f) cloudiness = 0.70f;
        else if (cloudiness < 0.80f) cloudiness = 0.95f;
        else                         cloudiness = 0.0f;
    }
    prevKCloud = kPressed;

    // 0 key: 최적화 전체 ON/OFF 토글
    static bool prev0 = false;
    bool k0Pressed = (glfwGetKey(window, GLFW_KEY_0) == GLFW_PRESS);
    if (k0Pressed && !prev0 && !adaptive.enabled) {
        bool anyOn = optFlags.frustumCulling || optFlags.lod || optFlags.instancing
                  || optFlags.backfaceCulling || optFlags.depthSort || optFlags.occlusionCulling
                  || optFlags.viewDistCulling || optFlags.smallCulling || optFlags.deferredShading;
        bool val = !anyOn;
        optFlags.frustumCulling  = val;
        optFlags.lod             = val;
        optFlags.instancing      = val;
        optFlags.backfaceCulling = val;
        optFlags.depthSort       = val;
        optFlags.occlusionCulling = val;
        optFlags.viewDistCulling = val;
        optFlags.smallCulling    = val;
        optFlags.deferredShading = val;
        if (val) {
            resetOcclusionState();
        }
    }
    prev0 = k0Pressed;

    // ` (grave) 키: 적응형 최적화 모드 토글
    // ON : 현재 플래그를 저장하고, A/B 실측 컨트롤러가 플래그를 소유한다.
    // OFF: 저장해 둔 사용자 플래그로 복원한다.
    static bool prevGrave = false;
    bool gravePressed = (glfwGetKey(window, GLFW_KEY_GRAVE_ACCENT) == GLFW_PRESS);
    if (gravePressed && !prevGrave) {
        if (!adaptive.enabled) {
            adaptiveSavedFlags = optFlags;
            adaptive.reset(optFlags);
            adaptive.enabled   = true;
            adaptiveSceneDirty = true;
            printf("[Adaptive] ON - A/B measuring each technique (keys 1-6/9/0 locked)\n");
        } else {
            adaptive.reset(optFlags); // 진행 중 테스트가 있으면 플래그 원복
            adaptive.enabled = false;
            bool occWasOn = optFlags.occlusionCulling;
            // 7/8은 적응형이 관리하지 않으므로 모드 중 수동으로 바꾼 값을 유지한다.
            bool keepViewDist = optFlags.viewDistCulling;
            bool keepSmall    = optFlags.smallCulling;
            optFlags = adaptiveSavedFlags;
            optFlags.viewDistCulling = keepViewDist;
            optFlags.smallCulling    = keepSmall;
            if (!occWasOn && optFlags.occlusionCulling) resetOcclusionState();
            printf("[Adaptive] OFF - user flags restored\n");
        }
    }
    prevGrave = gravePressed;
}

// mainLoop — 창이 닫힐 때까지 도는 프레임 순환의 심장부.
// 한 바퀴에서 하는 일 (위에서 아래 순서):
//  1. 이벤트 폴링 + dt 계산, 낮/밤 시간 진행(Y/U/J)
//  2. 전역 키 처리: ESC 종료, F11 전체화면, TAB 맵 전환, G 고스트, C 시네마틱
//  3. 리플레이: 녹화 중이면 프레임 기록 / 재생 중이면 카메라를 경로에 맞춰 보간 이동
//  4. 카메라 키 입력 (리플레이 중에는 재생 데이터가 카메라를 소유)
//  5. M 벤치마크 시작/중단, T 스트레스 배율, handleOptKeys(토글 키)
//  6. 성능 통계 갱신 + 벤치마크 샘플 수집
//  7. ImGui HUD 구성 → drawFrame() 으로 실제 렌더링
//  8. FPS 캡이 켜져 있으면 목표 프레임 시간까지 대기 (고정 프레임 모드)
void VulkanApp::mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        float now = static_cast<float>(glfwGetTime());
        float dt  = now - lastFrameTime;
        lastFrameTime = now;

        // 낮/밤 주기 진행 + U/J 키로 시간 수동 스크럽
        if (dayNightCycle)
            timeOfDay = std::fmod(timeOfDay + dt / dayLengthSec, 1.0f);
        if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
            timeOfDay = std::fmod(timeOfDay + dt * 0.05f, 1.0f);
        if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS)
            timeOfDay = std::fmod(timeOfDay - dt * 0.05f + 1.0f, 1.0f);

        // ESC를 누르면 앱을 종료한다.
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);

        // F11로 전체화면을 토글한다.
        {
            static bool prevF11 = false;
            bool f11 = (glfwGetKey(window, GLFW_KEY_F11) == GLFW_PRESS);
            if (f11 && !prevF11) toggleFullscreen();
            prevF11 = f11;
        }

        // Tab으로 다음 맵을 로드한다.
        {
            static bool prevTab = false;
            bool tab = (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS);
            if (tab && !prevTab && availableMaps.size() > 1) {
                int nextIndex  = (currentMapIndex + 1) % (int)availableMaps.size();
                currentMapFile = availableMaps[nextIndex];
                currentMapIndex = nextIndex;
                reloadScene(); // 내부에서 오류를 처리하고 실패 시 이전 씬으로 되돌린다.
                if (optFlags.occlusionCulling) resetOcclusionState();
            }
            prevTab = tab;
        }

        // G 키로 ghost 관찰자 모드를 토글한다.
        {
            bool gPressed = (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS);
            if (gPressed && !prevGhostKey) {
                ghostMode = !ghostMode;
                if (ghostMode) {
                    // ghost 카메라를 배치 카메라 뒤쪽과 위쪽에 초기화한다.
                    glm::vec3 back = -camera.getCameraFront();
                    observerCamera.position = camera.position
                                           + back * 5.0f
                                           + glm::vec3(0.f, 2.0f, 0.f);
                    // ghost 카메라가 배치 카메라를 바라보게 한다.
                    glm::vec3 toPlaced = camera.position - observerCamera.position;
                    float hDist = glm::length(glm::vec2(toPlaced.x, toPlaced.z));
                    observerCamera.yaw   = glm::degrees(std::atan2f(toPlaced.z, toPlaced.x));
                    observerCamera.pitch = glm::degrees(std::atan2f(toPlaced.y, hDist));
                    observerCamera.normalSpeed    = camera.normalSpeed;
                    observerCamera.fastSpeed      = camera.fastSpeed;
                    observerCamera.mouseSensitivity = camera.mouseSensitivity;
                    observerCamera.cinematic      = camera.cinematic; // 시네마틱 모드를 같이 따라감
                    observerCamera.syncTarget();
                }
            }
            prevGhostKey = gPressed;
        }

        // C 키로 시네마틱 카메라(부드러운 이동/회전) 모드를 토글한다.
        {
            static bool prevC = false;
            bool cPressed = (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS);
            if (cPressed && !prevC) {
                bool newMode = !camera.cinematic;
                camera.cinematic         = newMode;
                observerCamera.cinematic = newMode;
                // 토글 순간 잔여 속도/타겟 보간이 카메라를 끌고가지 않도록 동기화
                camera.syncTarget();
                observerCamera.syncTarget();
            }
            prevC = cPressed;
        }

        // R/P 키 + 리플레이/녹화 로직: 자동 벤치마크 중에는 전부 건너뜀
        if (!autoBenchActive) {
            // R 키로 리플레이 녹화를 시작하거나 종료한다.
            static bool prevR = false;
            bool r = (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS);
            if (r && !prevR && !isReplaying) {
                if (!isRecording) startRecording();
                else              stopRecording();
            }
            prevR = r;

            // P 키로 리플레이 재생을 시작하거나 종료한다.
            static bool prevP = false;
            bool p = (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS);
            if (p && !prevP) {
                if (!isReplaying) startReplay();
                else              stopReplay();
            }
            prevP = p;
        }

        // 녹화 중에는 일정 간격으로 카메라 상태를 저장한다.
        if (isRecording) {
            float elapsed = now - recordStartTime;
            if (recordedFrames.empty() || elapsed - recordedFrames.back().time >= 0.014f) {
                recordedFrames.push_back({elapsed, camera.position,
                                          camera.yaw, camera.pitch});
            }
        }

        // 리플레이 중에는 시간에 맞춰 카메라 상태를 보간한다.
        bool prevIsReplaying = isReplaying;
        if (isReplaying && !replayFrames.empty()) {
            float elapsed = now - replayStartTime;

            // 현재 재생 시간보다 늦지 않은 마지막 프레임까지 인덱스를 진행한다.
            while (replayFrameIdx + 1 < (int)replayFrames.size() &&
                   replayFrames[replayFrameIdx + 1].time <= elapsed)
                ++replayFrameIdx;

            if (replayFrameIdx + 1 < (int)replayFrames.size()) {
                // 인접한 두 리플레이 프레임 사이를 선형 보간한다.
                const auto& f0 = replayFrames[replayFrameIdx];
                const auto& f1 = replayFrames[replayFrameIdx + 1];
                float span = f1.time - f0.time;
                float t    = (span > 0.f) ? glm::clamp((elapsed - f0.time) / span, 0.f, 1.f) : 1.f;
                camera.position = glm::mix(f0.pos,   f1.pos,   t);
                camera.yaw      = f0.yaw   + t * (f1.yaw   - f0.yaw);
                camera.pitch    = f0.pitch + t * (f1.pitch - f0.pitch);
                camera.syncTarget(); // 리플레이가 카메라를 점프시켜도 보간이 따라가지 않도록
            } else {
                // 마지막 프레임에 도달하면 리플레이를 종료한다.
                const auto& last = replayFrames.back();
                camera.position = last.pos;
                camera.yaw      = last.yaw;
                camera.pitch    = last.pitch;
                camera.syncTarget();
                stopReplay();
            }
        }

        // 자동 벤치마크: 리플레이 종료 감지 -> 다음 실험/반복으로 진행
        if (autoBenchActive && prevIsReplaying && !isReplaying)
            onAutoBenchRunEnd();

        // 카메라 입력 처리
        if (ghostMode) {
            observerCamera.processKeyboard(window, dt);
            // 리플레이 중이 아닐 때만 방향키로 배치 카메라를 움직인다.
            if (!isReplaying) camera.processArrowKeys(window, dt);
        } else {
            // 리플레이 중에는 카메라를 재생 데이터가 제어한다.
            if (!isReplaying) camera.processKeyboard(window, dt);
        }

        // M 키는 현재 맵을 벤치마크한다:
        //  - 리플레이가 있으면 자동 벤치마크(실험×반복), 없으면 5초 벤치마크.
        // B 키는 모든 맵을 처음부터 끝까지 순서대로 자동 벤치마크한다.
        // 벤치마크(자동/전체 맵) 진행 중에는 M 또는 B 로 중단한다.
        if (!autoBenchActive && !benchAllMapsActive) {
            static bool prevM = false;
            bool m = (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS);
            if (m && !prevM) {
                if (!replaysForCurrentMap().empty()) {
                    startAutoBenchmark();
                } else {
                    if (!benchmarkActive) startBenchmark();
                    else                  finishBenchmark();
                }
            }
            prevM = m;

            static bool prevB = false;
            bool b = (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS);
            if (b && !prevB) startBenchmarkAllMaps();
            prevB = b;
        } else {
            // 자동 벤치마크 / 전체 맵 벤치마크 진행 중: M 또는 B 키로 중단
            static bool prevM = false, prevB = false;
            bool m = (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS);
            bool b = (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS);
            if ((m && !prevM) || (b && !prevB)) abortAutoBenchmark();
            prevM = m;
            prevB = b;
        }

        // T 키로 스트레스 배율을 순환한다.
        if (!autoBenchActive) {
            static bool prevT = false;
            bool tk = (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS);
            if (tk && !prevT) {
                stressLevel = (stressLevel + 1) % 5;
                applyStress();
            }
            prevT = tk;
        }

        // 자동 벤치마크 중에는 숫자 키(1~9) 토글 비활성화
        if (!autoBenchActive) handleOptKeys();

        // 적응형 최적화 컨트롤러 (` 키): A/B 실측으로 최적화 플래그를 자동 결정
        updateAdaptiveOptimizer(dt);

        perfStats.update(dt);
        const bool validBenchFrame = dt > 0.0f && dt < 0.25f;
        const float frameFps = (perfStats.frameTimeMs > 0.0f)
            ? 1000.0f / perfStats.frameTimeMs
            : 0.0f;

        // 프레임 시간 히스토리 갱신 (고정 링버퍼)
        frameTimeHistBuf[frameTimeHistIdx] = dt * 1000.f;
        frameTimeHistIdx = (frameTimeHistIdx + 1) % FRAME_HISTORY_SIZE;
        if (frameTimeHistCount < FRAME_HISTORY_SIZE) ++frameTimeHistCount;

        // 수동 벤치마크 샘플 수집 (5초 단순 측정)
        if (benchmarkActive) {
            if (validBenchFrame) {
                benchmarkElapsed += dt;
                benchmarkSamples.push_back({
                    frameFps,
                    perfStats.frameTimeMs,
                    perfStats.cpuPercent,
                    perfStats.gpuPercent,
                    perfStats.ramMB,
                    renderedCount,
                    culledCount
                });
            }
            if (benchmarkElapsed >= benchmarkDuration)
                finishBenchmark();
        }

        // --bench: 워밍업(4초) 후 자동 벤치마크 시작, 완료되면 CSV 저장 후 종료
        if (cliBench) {
            if (!cliBenchStarted && now > 4.0f) {
                cliBenchStarted = true;
                startBenchmark();
            } else if (cliBenchStarted && !benchmarkActive) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
        }

        // 자동 벤치마크 샘플 수집 (리플레이 재생 중)
        if (autoBenchActive && isReplaying &&
            autoBenchExpIdx < (int)autoBenchExps.size()) {
            if (autoBenchSkipFrames > 0) {
                --autoBenchSkipFrames;
            } else if (validBenchFrame) {
                autoBenchExps[autoBenchExpIdx].current.push_back({
                    frameFps,
                    perfStats.frameTimeMs,
                    perfStats.cpuPercent,
                    perfStats.gpuPercent,
                    perfStats.ramMB,
                    renderedCount,
                    culledCount
                });
            }
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        drawStatsOverlay();
        drawAdaptiveOverlay();
        ImGui::Render();

        drawFrame();

        // FPS 캡 (고정 프레임 모드): 목표 프레임 시간까지 대기한다.
        // 대부분은 1ms sleep으로 CPU를 양보하고, 마지막 ~2ms는 spin으로 정밀도를 확보한다.
        // 누적 목표 시각 방식이라 프레임 간 오차가 다음 프레임으로 전파되지 않는다.
        {
            static double nextFrameTarget = 0.0;
            const int cap = currentFpsCap();
            if (cap > 0) {
                const double period = 1.0 / cap;
                const double tNow   = glfwGetTime();
                if (nextFrameTarget < tNow - period)
                    nextFrameTarget = tNow; // 크게 밀렸으면 리셋 (씬 로드 직후 등)
                nextFrameTarget += period;
                while (glfwGetTime() < nextFrameTarget) {
                    if (nextFrameTarget - glfwGetTime() > 0.002)
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            } else {
                nextFrameTarget = 0.0;
            }
        }
    }
    vkDeviceWaitIdle(device);
}

// 프레임 갱신과 커맨드 기록
void VulkanApp::updateSunState() {
    const float TWO_PI = 6.2831853f;
    float sunAng = (timeOfDay - 0.25f) * TWO_PI; // 0.25=일출, 0.5=정오, 0.75=일몰
    float elev   = std::sin(sunAng) * glm::radians(62.f); // 태양 고도각
    float az     = glm::radians(lightYaw) - sunAng; // 하루 동안 동->서로 이동
    glm::vec3 toSun(std::cos(elev) * std::cos(az),
                    std::sin(elev),
                    std::cos(elev) * std::sin(az));
    isNightNow = toSun.y < 0.f;
    // 밤에는 태양의 고도 반전 위치에 있는 달을 활성 광원으로 쓴다.
    glm::vec3 toLight = isNightNow ? glm::vec3(toSun.x, -toSun.y, toSun.z) : toSun;
    activeLightDir = -glm::normalize(toLight);
}

glm::mat4 VulkanApp::computeLightMatrix() const {
    // 그림자를 드리울 방향광을 고른다.
    // GLTF 조명이 활성화돼 있으면 첫 방향광을, 아니면 셰이더 기본 태양(lightYaw 반영)을 쓴다.
    glm::vec3 dir(0.0f);
    bool haveDir = false;
    // 낮/밤 주기가 켜져 있으면 GLTF 조명 대신 애니메이션 태양/달을 따른다.
    if (sceneLightsOn && !dayNightCycle && !sceneLights.empty()) {
        for (const SceneLight& sl : sceneLights) {
            if (sl.type == SceneLight::Directional && sl.enabled) {
                dir = glm::normalize(sl.direction);
                haveDir = true;
                break;
            }
        }
    }
    if (!haveDir) {
        dir = activeLightDir; // updateSunState() 가 계산한 태양/달 방향
    }

    // 씬 바운딩 구를 감싸는 ortho 절두체를 라이트 방향으로 정렬한다.
    const glm::vec3 center = sceneBoundsCenter;
    const float     r      = sceneBoundsRadius;
    const glm::vec3 eye    = center - dir * (r * 2.0f); // 광원을 구 밖(방향 반대쪽)에 둔다.
    const glm::vec3 up     = (std::abs(dir.y) > 0.99f) ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);

    glm::mat4 view = glm::lookAt(eye, center, up);
    // GLM_FORCE_DEPTH_ZERO_TO_ONE 가 정의돼 ortho 가 0..1 깊이(Vulkan 규칙)를 만든다.
    glm::mat4 proj = glm::ortho(-r, r, -r, r, 0.1f, r * 4.0f);
    return proj * view;
}

// ============================================================================
// [6] 프레임 렌더링
// ============================================================================

// updateUniformBuffer — 이번 프레임의 UBO(셰이더 전역 데이터)를 CPU 에서 채운다.
// 채우는 것:
//  - CameraUBO: 뷰/투영 행렬, 카메라 위치, 활성 광원(태양/달) 방향, 섀도 VP,
//               skyParams(시간·구름·안개), envParams(현실적 셰이더 스위치)
//    -> 렌더용(uniformBuffers)과 컬링용(cullUniformBuffers) 두 벌을 쓴다.
//       고스트 모드에서 "관찰 시점"과 "컬링 기준 시점"이 다르기 때문.
//  - SceneLightUBO: GLTF 조명 배열 (dirty 플래그가 섰을 때만 memcpy)
// 버퍼는 영구 매핑(persistently mapped)이라 map/unmap 없이 바로 쓴다.
void VulkanApp::updateUniformBuffer(uint32_t frameIndex) {
    // ghost 모드에서는 observerCamera, 일반 모드에서는 배치 카메라로 렌더링한다.
    const Camera& renderCam = ghostMode ? observerCamera : camera;
    const Camera& cullCam   = camera;

    // 태양/달 상태를 먼저 갱신한 뒤, 이번 프레임의 라이트 뷰·투영을 한 번 계산해
    // 섀도 패스(recordCommandBuffer)에서 재사용한다.
    updateSunState();
    currentLightVP = computeLightMatrix();

    auto writeCameraUbo = [&](void* dst, const Camera& cam) {
        CameraUBO ubo;
        ubo.view      = cam.getViewMatrix();
        ubo.proj      = glm::perspective(glm::radians(60.0f),
                                         scExtent.width / (float)scExtent.height,
                                         0.1f, getFarPlane());
        ubo.proj[1][1] *= -1; // Vulkan NDC 규칙에 맞춰 Y축을 뒤집는다.
        ubo.cameraPos  = glm::vec4(cam.position, 0.0f);

        ubo.lightDir  = glm::vec4(activeLightDir, isNightNow ? 1.f : 0.f);
        ubo.lightVP   = currentLightVP;
        ubo.skyParams = glm::vec4(static_cast<float>(glfwGetTime()), cloudiness,
                                  fogOn ? fogDensity : 0.f, timeOfDay);
        ubo.envParams = glm::vec4(realisticShading ? 1.f : 0.f, 1.f, 0.f, 0.f);

        memcpy(dst, &ubo, sizeof(ubo));
    };

    writeCameraUbo(uniformBuffersMapped[frameIndex], renderCam);
    writeCameraUbo(cullUniformBuffersMapped[frameIndex], cullCam);

    // SceneLightUBO 갱신 (dirty flag: 변경 시에만 memcpy)
    if (sceneLightDirty) {
        SceneLightUBO lightUbo{};
        // useSceneLights 는 "진짜 GLTF 조명(자동 발광 조명 제외)" 이 있을 때만 1.
        // 0 이면 셰이더가 기본 태양/달 + 도시 램프를 추가로 적용한다
        // (조명 배열 자체는 값에 관계없이 항상 적용됨 — 자동 발광 조명 포함).
        // 낮/밤 주기 중에는 셰이더 기본 태양/달을 쓰도록 GLTF 조명 모드를 끈다.
        bool hasRealLights = false;
        for (const SceneLight& sl : sceneLights)
            if (!sl.fromEmissive) { hasRealLights = true; break; }
        lightUbo.useSceneLights = (hasRealLights && sceneLightsOn && !dayNightCycle) ? 1 : 0;
        lightUbo.ambientOn      = ambientOn  ? 1 : 0;
        lightUbo.emissiveOn     = emissiveOn ? 1 : 0;

        int gpuIdx = 0;
        for (const SceneLight& sl : sceneLights) {
            if (gpuIdx >= MAX_SCENE_LIGHTS) break;
            GpuSceneLight& gl = lightUbo.lights[gpuIdx++];
            if (sl.type == SceneLight::Directional) {
                gl.posRange  = glm::vec4(sl.direction, 0.f);
                gl.dirType   = glm::vec4(sl.direction, 1.f);
            } else {
                gl.posRange  = glm::vec4(sl.position, sl.range);
                gl.dirType   = glm::vec4(sl.direction, (sl.type == SceneLight::Spot) ? 2.f : 0.f);
            }
            // 자동 발광 조명(fromEmissive)은 V 키(emissiveOn)에 연동된다.
            bool on = sceneLightsOn && sl.enabled &&
                      (!sl.fromEmissive || emissiveOn);
            gl.colorEnab = glm::vec4(sl.color * sl.intensity, on ? 1.f : 0.f);
            // Blender 스팟 콘 각도 (Point/Directional 은 사용하지 않음)
            gl.coneParams = glm::vec4(std::cos(sl.innerConeAngle),
                                      std::cos(sl.outerConeAngle), 0.f, 0.f);
        }
        lightUbo.numLights = gpuIdx;
        // 더블 버퍼 모두 업데이트
        for (int fi = 0; fi < MAX_FRAMES_IN_FLIGHT; ++fi)
            memcpy(sceneLightUBOMapped[fi], &lightUbo, sizeof(lightUbo));
        sceneLightDirty = false;
    }
}

// recordCommandBuffer — 이번 프레임의 GPU 작업 전체를 커맨드 버퍼에 기록한다.
// 패스 실행 순서:
//  1. 섀도 패스     : 라이트 시점에서 불투명 캐스터의 깊이만 기록 (섀도맵)
//  2. (고스트+오클루전) 관찰 시점 프리패스 + 쿼리 발행
//  3-a. 디퍼드 경로 : G-Buffer 4장 기록 → 배리어 → 전체화면 조명 삼각형
//  3-b. forward 경로: 하늘 → (인스턴싱 묶음) → 불투명 오브젝트 → 오클루전 쿼리
//  4. 고스트 오버레이(컬링된 오브젝트 반투명 표시) → 투명 오브젝트(뒤→앞 정렬)
//  5. 기즈모(라인) → ImGui HUD → 렌더패스 종료
// 컬링은 CPU 에서 이 함수 안에서 수행된다:
//  frustum(절두체) / viewDist(거리) / small(화면 크기) / occlusion(이전 프레임 쿼리)
void VulkanApp::recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &begin);

    const bool occlusionEnabled = optFlags.occlusionCulling
                               && !optFlags.deferredShading
                               && occlusionQueryPool != VK_NULL_HANDLE
                               && occQueryCount > 0;
    const Camera& renderCam = ghostMode ? observerCamera : camera;
    const Camera& cullCam   = camera;

    if (occlusionEnabled) {
        uint32_t base = currentFrame * occQueryCount;
        vkCmdResetQueryPool(cmd, occlusionQueryPool, base, occQueryCount);
    }

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color        = {{0.53f, 0.68f, 0.85f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpBegin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpBegin.renderPass        = renderPass;
    rpBegin.framebuffer       = framebuffers[imageIndex];
    rpBegin.renderArea.extent = scExtent;
    rpBegin.clearValueCount   = static_cast<uint32_t>(clearValues.size());
    rpBegin.pClearValues      = clearValues.data();

    VkViewport vp{0.f, 0.f, (float)scExtent.width, (float)scExtent.height, 0.f, 1.f};
    VkRect2D   sc{{0, 0}, scExtent};
    VkBuffer     vbs[]  = {vertexBuffer};
    VkDeviceSize offs[] = {0};

    // 방향광 섀도맵: 라이트 시점에서 불투명 캐스터의 깊이를 먼저 기록한다.
    // forward·deferred 양쪽 모두 set0 binding3로 이 결과를 샘플링한다.
    if (shadowPipeline != VK_NULL_HANDLE && !drawObjects.empty()) {
        VkClearValue shadowClear{};
        shadowClear.depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo spBegin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        spBegin.renderPass        = shadowRenderPass;
        spBegin.framebuffer       = shadowFramebuffers[currentFrame];
        spBegin.renderArea.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};
        spBegin.clearValueCount   = 1;
        spBegin.pClearValues      = &shadowClear;

        vkCmdBeginRenderPass(cmd, &spBegin, VK_SUBPASS_CONTENTS_INLINE);
        VkViewport sVp{0.f, 0.f, (float)SHADOW_MAP_SIZE, (float)SHADOW_MAP_SIZE, 0.f, 1.f};
        VkRect2D   sSc{{0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}};
        vkCmdSetViewport(cmd, 0, 1, &sVp);
        vkCmdSetScissor(cmd, 0, 1, &sSc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);
        vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offs);
        vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        for (const DrawObject& obj : drawObjects) {
            if (obj.push.baseColor.w < 0.999f) continue; // 투명체는 캐스터에서 제외한다.
            glm::mat4 lightMVP = currentLightVP * obj.push.model;
            vkCmdPushConstants(cmd, shadowPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(glm::mat4), &lightMVP);
            vkCmdDrawIndexed(cmd, obj.indexCount, 1, obj.indexStart, 0, 0);
        }
        vkCmdEndRenderPass(cmd);
    }

    Frustum frustum{};
    if (optFlags.frustumCulling || optFlags.occlusionCulling) {
        glm::mat4 view = cullCam.getViewMatrix();
        glm::mat4 proj = glm::perspective(glm::radians(60.f),
                                          scExtent.width / (float)scExtent.height,
                                          0.1f, getFarPlane());
        proj[1][1] *= -1;
        frustum = extractFrustum(proj * view);
    }

    // 프레임마다 힙 할당하지 않도록 frameOrder를 재사용한다.
    frameOrder.resize(drawObjects.size());
    std::iota(frameOrder.begin(), frameOrder.end(), 0);
    if (optFlags.depthSort || optFlags.occlusionCulling) {
        glm::vec3 camPos = cullCam.position;
        // sqrt 없이 제곱 거리로 같은 정렬 순서를 얻는다.
        std::sort(frameOrder.begin(), frameOrder.end(), [&](int a, int b) {
            glm::vec3 da = drawObjects[a].boundCenter - camPos;
            glm::vec3 db = drawObjects[b].boundCenter - camPos;
            return glm::dot(da, da) < glm::dot(db, db);
        });
    }
    const std::vector<int>& order = frameOrder;
    frameInstancedMask.clear();

    // 통합 컬링 체크: frustum + viewDist + smallObj 를 한 번에 수행
    // 3개 lambda 캡처·호출 오버헤드 제거
    constexpr float kTanHalfFov = 0.57735f; // tan(30°)
    const bool doFrustum  = optFlags.frustumCulling;
    const bool doViewDist = optFlags.viewDistCulling;
    const bool doSmall    = optFlags.smallCulling;
    const glm::vec3 cullPos = cullCam.position;
    const float halfH = scExtent.height * 0.5f;

    auto isCulled = [&](const DrawObject& obj) -> bool {
        if (doFrustum && !sphereInFrustum(frustum, obj.boundCenter, obj.boundRadius))
            return true;
        if (doViewDist) {
            float dist = glm::length(cullPos - obj.boundCenter) - obj.boundRadius;
            if (dist > viewDistMax) return true;
        }
        if (doSmall) {
            float dist = glm::length(cullPos - obj.boundCenter);
            if (dist >= 1e-4f && obj.boundRadius / dist * halfH / kTanHalfFov < smallCullPx)
                return true;
        }
        return false;
    };

    // Ghost 모드에서 컬링 이유별 색상을 반환한다. 컬링되지 않으면 alpha=0이다.
    auto ghostCullColor = [&](const DrawObject& obj) -> glm::vec4 {
        if (doFrustum && !sphereInFrustum(frustum, obj.boundCenter, obj.boundRadius))
            return {1.0f, 0.15f, 0.15f, 0.13f};
        if (doViewDist) {
            float dist = glm::length(cullPos - obj.boundCenter) - obj.boundRadius;
            if (dist > viewDistMax) return {1.0f, 0.55f, 0.05f, 0.13f};
        }
        if (doSmall) {
            float dist = glm::length(cullPos - obj.boundCenter);
            if (dist >= 1e-4f && obj.boundRadius / dist * halfH / kTanHalfFov < smallCullPx)
                return {1.0f, 0.95f, 0.05f, 0.13f};
        }
        return {0.f, 0.f, 0.f, 0.f};
    };

    auto beginScenePass = [&](VkDescriptorSet camSet, VkPipeline pipe) {
        vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offs);
        vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 0, 1, &camSet, 0, nullptr);
    };

    auto selectLod = [&](const DrawObject& obj,
                         uint32_t& idxStart,
                         uint32_t& idxCount,
                         glm::mat4& model,
                         int* outLodLevel = nullptr) -> bool {
        idxStart = obj.indexStart;
        idxCount = obj.indexCount;
        model    = obj.push.model;
        int level = 0;

        if (optFlags.lod && obj.numLods > 0) {
            // 큰 오브젝트가 바로 앞에 있어도 중심점이 멀다는 이유로 LOD가 바뀌지 않도록
            // 바운딩 구 표면까지의 거리로 전환 임계값을 비교한다.
            float dist = glm::length(cullCam.position - obj.boundCenter) - obj.boundRadius;
            dist = std::max(0.0f, dist);
            if (dist > obj.lodDist[1] && obj.numLods >= 2) {
                if (obj.lods[1].count == 0) return false;
                idxStart = obj.lods[1].start;
                idxCount = obj.lods[1].count;
                model    = obj.lods[1].model;
                level    = 2;
            } else if (dist > obj.lodDist[0]) {
                idxStart = obj.lods[0].start;
                idxCount = obj.lods[0].count;
                model    = obj.lods[0].model;
                level    = 1;
            }
        }
        if (outLodLevel) *outLodLevel = level;
        return true;
    };

    auto issueOcclusionQueries = [&](VkDescriptorSet camSet) {
        if (!occlusionEnabled) return;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineQueryOnly);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 0, 1, &camSet, 0, nullptr);

        for (int idx : order) {
            auto& obj = drawObjects[idx];
            if (obj.skipOcclusion) continue;
            if (obj.push.baseColor.w < 0.999f) continue;
            if (optFlags.frustumCulling &&
                !sphereInFrustum(frustum, obj.boundCenter, obj.boundRadius)) continue;

            // 바운딩 구 프록시: 단위 큐브를 boundCenter 로 이동, boundRadius 로 스케일
            PushConstants proxyPc = obj.push;
            proxyPc.model = glm::translate(glm::mat4(1.f), obj.boundCenter)
                          * glm::scale(glm::mat4(1.f), glm::vec3(obj.boundRadius));
            vkCmdPushConstants(cmd, pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(PushConstants), &proxyPc);

            uint32_t queryIndex = currentFrame * occQueryCount + (uint32_t)idx;
            vkCmdBeginQuery(cmd, occlusionQueryPool, queryIndex, 0);
            vkCmdDrawIndexed(cmd, occBBoxMesh.indexCount, 1, occBBoxMesh.indexStart, 0, 0);
            vkCmdEndQuery(cmd, occlusionQueryPool, queryIndex);
        }
    };

    if (ghostMode && occlusionEnabled) {
        beginScenePass(cullDescSets[currentFrame], graphicsPipelineNoCull);
        VkPipeline boundGhostPipe = VK_NULL_HANDLE;

        for (int idx : order) {
            auto& obj = drawObjects[idx];
            if (obj.push.baseColor.w < 0.999f) continue;
            if (doFrustum && !sphereInFrustum(frustum, obj.boundCenter, obj.boundRadius))
                continue;

            uint32_t idxStart = obj.indexStart;
            uint32_t idxCount = obj.indexCount;
            glm::mat4 model   = obj.push.model;
            if (!selectLod(obj, idxStart, idxCount, model)) continue;

            PushConstants pc = obj.push;
            pc.model = model;
            VkPipeline pipe = (!optFlags.backfaceCulling || obj.twoSided)
                ? graphicsPipelineNoCull
                : (obj.reverseFrontFace ? graphicsPipelineFlippedCull : graphicsPipeline);
            if (pipe != boundGhostPipe) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                boundGhostPipe = pipe;
            }
            vkCmdPushConstants(cmd, pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(PushConstants), &pc);
            vkCmdDrawIndexed(cmd, idxCount, 1, idxStart, 0, 0);
        }

        issueOcclusionQueries(cullDescSets[currentFrame]);
        vkCmdEndRenderPass(cmd);
    }

    VkPipeline boundOpaquePipe = VK_NULL_HANDLE; // forward/deferred 분기에서 공유하는 현재 파이프라인 캐시
    // 디퍼드 렌더링 파이프라인을 만든다.
    if (optFlags.deferredShading && gbufRenderPass != VK_NULL_HANDLE) {
        std::array<VkClearValue, GBUFFER_COLOR_ATTACHMENTS + 1> gbClear{};
        gbClear[0].color = {{0,0,0,0}};
        gbClear[1].color = {{0,0,0,0}};
        gbClear[2].color = {{0,0,0,0}};
        gbClear[3].color = {{0,0,0,0}};
        gbClear[4].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo gbRpBegin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        gbRpBegin.renderPass        = gbufRenderPass;
        gbRpBegin.framebuffer       = gbufFramebuffers[currentFrame];
        gbRpBegin.renderArea.extent = scExtent;
        gbRpBegin.clearValueCount   = static_cast<uint32_t>(gbClear.size());
        gbRpBegin.pClearValues      = gbClear.data();

        vkCmdBeginRenderPass(cmd, &gbRpBegin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offs);
        vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        VkPipeline gbCurPipe = VK_NULL_HANDLE;
        VkPipeline gbDefault = optFlags.backfaceCulling ? gbufPipeline : gbufPipelineNoCull;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gbDefault);
        gbCurPipe = gbDefault;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 0, 1, &descSets[currentFrame], 0, nullptr);

        for (int idx : order) {
            const auto& obj = drawObjects[idx];
            if (obj.push.baseColor.w < 0.999f) continue; // 투명 오브젝트는 이 패스에서 제외한다.
            if (isCulled(obj)) continue;

            uint32_t  gbIdxStart = obj.indexStart;
            uint32_t  gbIdxCount = obj.indexCount;
            glm::mat4 gbModel    = obj.push.model;
            if (!selectLod(obj, gbIdxStart, gbIdxCount, gbModel)) continue;

            PushConstants gbPc = obj.push;
            gbPc.model = gbModel;

            VkPipeline wantPipe = (!optFlags.backfaceCulling || obj.twoSided || obj.reverseFrontFace)
                                  ? gbufPipelineNoCull : gbufPipeline;
            if (wantPipe != gbCurPipe) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, wantPipe);
                gbCurPipe = wantPipe;
            }
            vkCmdPushConstants(cmd, pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(PushConstants), &gbPc);
            vkCmdDrawIndexed(cmd, gbIdxCount, 1, gbIdxStart, 0, 0);
        }
        vkCmdEndRenderPass(cmd);

        // G-Buffer 컬러 attachment를 조명 패스에서 샘플링 가능한 레이아웃으로 전환한다.
        std::array<VkImageMemoryBarrier, GBUFFER_COLOR_ATTACHMENTS> gbBarriers{};
        for (int i = 0; i < GBUFFER_COLOR_ATTACHMENTS; i++) {
            gbBarriers[i].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            gbBarriers[i].oldLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            gbBarriers[i].newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            gbBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            gbBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            gbBarriers[i].image               = gbufImages[i][currentFrame];
            gbBarriers[i].subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            gbBarriers[i].srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            gbBarriers[i].dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        }
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr,
            GBUFFER_COLOR_ATTACHMENTS, gbBarriers.data());
    }

    // 메인 렌더패스: 디퍼드 조명 삼각형 또는 forward 씬을 그린다.
    if (optFlags.deferredShading && gbufRenderPass != VK_NULL_HANDLE) {
        // 메인 렌더 패스를 시작하고 전체화면 조명 삼각형을 그린다.
        vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, deferredLightPipeline);
        VkDescriptorSet dSets[] = {descSets[currentFrame], deferredDescSets[currentFrame]};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                deferredLightLayout, 0, 2, dSets, 0, nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0); // 버텍스 버퍼 없이 전체화면 삼각형을 그린다.

        // 후속 투명/ghost/기즈모 패스를 위해 버텍스 버퍼를 다시 바인딩한다.
        vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offs);
        vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineNoCull);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 0, 1, &descSets[currentFrame], 0, nullptr);

        renderedCount       = (int)drawObjects.size();
        culledCount         = 0;
        instancedGroupCount = 0;
     } else {
    renderedCount       = 0;
    culledCount         = 0;
    instancedGroupCount = 0;
    frameInstMats.clear();
    frameInstObjectIndices.clear();
    frameInstDrawEntries.clear();
    if (optFlags.instancing)
        frameInstancedMask.assign(drawObjects.size(), 0u);

    if (optFlags.instancing && !instGroupDefs.empty()) {
        // 미리 만든 인스턴스 그룹 멤버십을 사용한다.
        // 매 프레임 힙 할당을 줄이기 위해 frameInstMats를 재사용한다.
        frameInstDrawEntries.reserve(instGroupDefs.size());

        bool instanceBufferFull = false;
        for (uint32_t di = 0; di < (uint32_t)instGroupDefs.size() && !instanceBufferFull; ++di) {
            const InstGroupDef& grp = instGroupDefs[di];
            uint32_t matrixOffset = (uint32_t)frameInstMats.size();
            uint32_t objectOffset = (uint32_t)frameInstObjectIndices.size();
            uint32_t count = 0;
            uint32_t groupCulled = 0;

            for (int idx : grp.members) {
                auto& obj = drawObjects[idx];
                if (obj.instanceGroupId < 0 || obj.twoSided || obj.reverseFrontFace) continue;
                if (obj.push.baseColor.w < 0.999f) continue;
                if (isCulled(obj)) {
                    groupCulled++;
                    continue;
                }
                if (occlusionEnabled && !obj.skipOcclusion && isOcclusionHidden(idx)) {
                    groupCulled++;
                    continue;
                }
                if ((uint32_t)frameInstMats.size() >= (uint32_t)MAX_INSTANCES) {
                    instanceBufferFull = true;
                    break;
                }
                frameInstMats.push_back(obj.push.model);
                frameInstObjectIndices.push_back(idx);
                ++count;
            }

            if (count >= (uint32_t)MIN_INSTANCES_PER_DRAW) {
                frameInstDrawEntries.push_back({matrixOffset, count, di});
                culledCount += (int)groupCulled;
                for (uint32_t oi = objectOffset; oi < (uint32_t)frameInstObjectIndices.size(); ++oi)
                    frameInstancedMask[frameInstObjectIndices[oi]] = 1u;
            } else {
                frameInstMats.resize(matrixOffset);
                frameInstObjectIndices.resize(objectOffset);
            }
        }

        if (!frameInstMats.empty()) {
            const VkDeviceSize instBytes = frameInstMats.size() * sizeof(glm::mat4);
            memcpy(instanceSSBOMapped[currentFrame],
                   frameInstMats.data(),
                   static_cast<size_t>(instBytes));

            VkBufferCopy instCopy{0, 0, instBytes};
            vkCmdCopyBuffer(cmd, instanceStagingBuffers[currentFrame],
                            instanceSSBOs[currentFrame], 1, &instCopy);
            VkBufferMemoryBarrier instBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            instBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            instBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
            instBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            instBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            instBarrier.buffer = instanceSSBOs[currentFrame];
            instBarrier.offset = 0;
            instBarrier.size = instBytes;
            vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                                 0, 0, nullptr, 1, &instBarrier, 0, nullptr);
        }
    }

    beginScenePass(descSets[currentFrame], graphicsPipelineNoCull);

    // 절차적 하늘: 깊이 쓰기 없는 전체화면 삼각형으로 배경을 먼저 채운다.
    // 이후 그려지는 불투명 오브젝트가 깊이 테스트로 하늘을 덮는다.
    // 클래식 셰이딩 모드에서는 원본 클리어 색상을 그대로 배경으로 쓴다.
    if (skyPipeline != VK_NULL_HANDLE && realisticShading) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipeline);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineNoCull);
    }

    if (!frameInstMats.empty()) {
        VkBuffer instVbs[] = { vertexBuffer, instanceSSBOs[currentFrame] };
        VkDeviceSize instOffs[] = { 0, 0 };
        vkCmdBindVertexBuffers(cmd, 0, 2, instVbs, instOffs);

        VkPipeline instPipe = optFlags.backfaceCulling
                            ? graphicsPipelineInst : graphicsPipelineInstNoCull;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, instPipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                instancePipelineLayout, 0, 1,
                                &descSets[currentFrame], 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                instancePipelineLayout, 1, 1,
                                &instanceDescSets[currentFrame], 0, nullptr);

        for (const InstDrawEntry& e : frameInstDrawEntries) {
            const InstGroupDef& grp = instGroupDefs[e.groupDefIndex];
            vkCmdPushConstants(cmd, instancePipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(PushConstants), &grp.push);
            vkCmdDrawIndexed(cmd, grp.indexCount, e.instanceCount,
                             grp.indexStart, 0, e.matrixOffset);
            renderedCount++;
            instancedGroupCount++;
        }

        // 남은 개별 오브젝트 렌더링을 위해 일반 파이프라인으로 되돌린다.
        vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offs);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineNoCull);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 0, 1,
                                &descSets[currentFrame], 0, nullptr);
        boundOpaquePipe = VK_NULL_HANDLE;
    }

    for (int idx : order) {
        auto& obj = drawObjects[idx];
        if (optFlags.instancing && idx < (int)frameInstancedMask.size() && frameInstancedMask[idx])
            continue;
        if (obj.push.baseColor.w < 0.999f) continue;

        if (isCulled(obj)) {
            culledCount++;
            continue;
        }

        if (occlusionEnabled && !obj.skipOcclusion && isOcclusionHidden(idx)) {
            culledCount++;
            continue;
        }

        uint32_t  idxStart = obj.indexStart;
        uint32_t  idxCount = obj.indexCount;
        glm::mat4 model    = obj.push.model;
        int lodLevel = 0;
        if (!selectLod(obj, idxStart, idxCount, model, &lodLevel)) {
            culledCount++;
            continue;
        }

        PushConstants pc = obj.push;
        pc.model = model;

        // Ghost 모드에서는 LOD 레벨에 따라 색상을 구분한다(LOD1=주황, LOD2=빨강).
        if (ghostMode && optFlags.lod && lodLevel > 0) {
            pc.textureIndex = -1.0f;
            pc.baseColor = (lodLevel == 1)
                ? glm::vec4(1.0f, 0.60f, 0.05f, 1.0f) // 주황 = LOD1
                : glm::vec4(1.0f, 0.20f, 0.05f, 1.0f); // 빨강 = LOD2
        }

        VkPipeline pipe = (!optFlags.backfaceCulling || obj.twoSided)
            ? graphicsPipelineNoCull
            : (obj.reverseFrontFace ? graphicsPipelineFlippedCull : graphicsPipeline);
        if (pipe != boundOpaquePipe) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            boundOpaquePipe = pipe;
        }
        vkCmdPushConstants(cmd, pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, idxCount, 1, idxStart, 0, 0);
        renderedCount++;
    }

    } // forward 전용 렌더링 블록 끝

    // Ghost 모드에서는 컬링된 오브젝트를 반투명 컬러 오버레이로 표시한다.
    // frustum 컬링 -> 빨강 / 원거리 컬링 -> 주황 / 소형 컬링 -> 노랑
    if (ghostMode && (optFlags.frustumCulling || optFlags.viewDistCulling || optFlags.smallCulling)) {
        // 알파 블렌딩이 맞도록 뒤에서 앞으로 정렬한다.
        std::vector<int> culledOverlay;
        for (int idx : order) {
            const auto& obj = drawObjects[idx];
            if (obj.push.baseColor.w < 0.999f) continue;
            if (optFlags.instancing && idx < (int)frameInstancedMask.size() && frameInstancedMask[idx])
                continue;
            glm::vec4 col = ghostCullColor(obj);
            if (col.a > 0.f) culledOverlay.push_back(idx);
        }
        if (!culledOverlay.empty()) {
            glm::vec3 obsPos = observerCamera.position;
            std::sort(culledOverlay.begin(), culledOverlay.end(), [&](int a, int b) {
                return glm::distance(drawObjects[a].boundCenter, obsPos)
                     > glm::distance(drawObjects[b].boundCenter, obsPos);
            });
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineAlpha);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipelineLayout, 0, 1, &descSets[currentFrame], 0, nullptr);
            boundOpaquePipe = VK_NULL_HANDLE;
            for (int idx : culledOverlay) {
                const auto& obj = drawObjects[idx];
                PushConstants pc = obj.push;
                pc.baseColor     = ghostCullColor(obj);
                pc.textureIndex  = -1.0f;
                vkCmdPushConstants(cmd, pipelineLayout,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(PushConstants), &pc);
                vkCmdDrawIndexed(cmd, obj.indexCount, 1, obj.indexStart, 0, 0);
            }
        }
    }

    if (!ghostMode && occlusionEnabled)
        issueOcclusionQueries(descSets[currentFrame]);

    {
        std::vector<int> transOrder;
        for (int i = 0; i < (int)drawObjects.size(); ++i) {
            if (drawObjects[i].push.baseColor.w < 0.999f)
                transOrder.push_back(i);
        }
        if (!transOrder.empty()) {
            glm::vec3 camPos = renderCam.position;
            std::sort(transOrder.begin(), transOrder.end(), [&](int a, int b) {
                float da = glm::distance(drawObjects[a].boundCenter, camPos);
                float db = glm::distance(drawObjects[b].boundCenter, camPos);
                return da > db;
            });

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineAlpha);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipelineLayout, 0, 1, &descSets[currentFrame], 0, nullptr);

            for (int idx : transOrder) {
                auto& obj = drawObjects[idx];
                if (doFrustum && !sphereInFrustum(frustum, obj.boundCenter, obj.boundRadius)) {
                    culledCount++;
                    continue;
                }

                vkCmdPushConstants(cmd, pipelineLayout,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(PushConstants), &obj.push);
                vkCmdDrawIndexed(cmd, obj.indexCount, 1, obj.indexStart, 0, 0);
            }
        }
    }

    // 씬 패스 종료. 이 시점의 스왑체인 이미지 레이아웃은 PRESENT_SRC 다.
    vkCmdEndRenderPass(cmd);

    // ---- 블룸 포스트프로세스 ------------------------------------------------
    // 씬 결과에서 밝은 부분을 뽑아 블러 후 가산 합성한다 (발광체가 화면에서 번짐).
    // 클래식 셰이딩(H OFF)에서는 원본 룩 유지를 위해 생략한다.
    const bool bloomActive = realisticShading && brightPipeline != VK_NULL_HANDLE
                          && imageIndex < brightSets.size();
    {
        // 스왑체인 이미지를 샘플링 가능 레이아웃으로 전환한다.
        // (블룸을 건너뛰어도 loadRenderPass 의 initialLayout 과 맞춰야 한다.)
        VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        bar.oldLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        bar.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.image               = scImages[imageIndex];
        bar.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        bar.srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        bar.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &bar);
    }

    if (bloomActive) {
        struct { float p0[4]; float p1[4]; } pp{};
        VkViewport bVp{0.f, 0.f, (float)bloomExtent.width, (float)bloomExtent.height, 0.f, 1.f};
        VkRect2D   bSc{{0, 0}, bloomExtent};

        // 하프 해상도 풀스크린 패스 실행 헬퍼 (bright / blurH / blurV 공용)
        auto fsPass = [&](VkFramebuffer fb, VkPipeline pipe, VkDescriptorSet set,
                          float dirX, float dirY) {
            VkClearValue cv{};
            VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            bi.renderPass = bloomRenderPass;
            bi.framebuffer = fb;
            bi.renderArea.extent = bloomExtent;
            bi.clearValueCount = 1; bi.pClearValues = &cv;
            vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdSetViewport(cmd, 0, 1, &bVp);
            vkCmdSetScissor(cmd, 0, 1, &bSc);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    postPipelineLayout, 0, 1, &set, 0, nullptr);
            pp.p0[0] = 1.f / bloomExtent.width;
            pp.p0[1] = 1.f / bloomExtent.height;
            pp.p0[2] = dirX; pp.p0[3] = dirY;
            pp.p1[0] = bloomThreshold; pp.p1[1] = bloomStrength;
            vkCmdPushConstants(cmd, postPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(pp), &pp);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRenderPass(cmd);
        };
        fsPass(bloomFramebuffers[0], brightPipeline, brightSets[imageIndex], 0.f, 0.f); // 밝기 추출 -> A
        fsPass(bloomFramebuffers[1], blurPipeline,   blurSets[0],            1.f, 0.f); // A 가로 블러 -> B
        fsPass(bloomFramebuffers[0], blurPipeline,   blurSets[1],            0.f, 1.f); // B 세로 블러 -> A
    }

    // ---- 합성 + 기즈모 + HUD 패스 (스왑체인 LOAD) ---------------------------
    {
        VkRenderPassBeginInfo li{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        li.renderPass        = loadRenderPass;
        li.framebuffer       = framebuffers[imageIndex];
        li.renderArea.extent = scExtent;
        vkCmdBeginRenderPass(cmd, &li, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);

        if (bloomActive) {
            // 블러 결과(A)를 화면에 가산 합성한다.
            struct { float p0[4]; float p1[4]; } pp{};
            pp.p1[0] = bloomThreshold; pp.p1[1] = bloomStrength;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bloomCompositePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    postPipelineLayout, 0, 1, &blurSets[0], 0, nullptr);
            vkCmdPushConstants(cmd, postPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(pp), &pp);
            vkCmdDraw(cmd, 3, 1, 0, 0);
        }

        if (ghostMode && gizmoVertCount > 0) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gizmoPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipelineLayout, 0, 1, &descSets[currentFrame], 0, nullptr);
            VkBuffer     gVbs[]  = { gizmoVBs[currentFrame] };
            VkDeviceSize gOffs[] = { 0 };
            vkCmdBindVertexBuffers(cmd, 0, 1, gVbs, gOffs);
            vkCmdDraw(cmd, gizmoVertCount, 1, 0, 0);
        }

        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
        vkCmdEndRenderPass(cmd);
    }
    vkEndCommandBuffer(cmd);
}

// drawFrame — 한 프레임의 제출 사이클. CPU 와 GPU 가 겹쳐 돌도록
// MAX_FRAMES_IN_FLIGHT(2) 개의 프레임 슬롯을 번갈아 쓴다 (더블 버퍼링).
//  1. 이 슬롯의 펜스 대기  : 2프레임 전 GPU 작업이 끝났음을 보장 (CPU-GPU 동기화)
//  2. 스왑체인 이미지 획득 : 세마포어가 "이미지 사용 가능" 신호를 GPU 에 전달
//  3. 오클루전 쿼리 결과 읽기: 펜스 덕분에 안전 — 다음 프레임 컬링에 사용
//  4. UBO 갱신 → 커맨드 기록 → 큐 제출(세마포어 체인) → 프리젠트
//  5. OUT_OF_DATE(창 크기 변경 등)면 스왑체인 재생성
void VulkanApp::drawFrame() {
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                                             imageAvailableSems[currentFrame], VK_NULL_HANDLE, &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapChain(); return; }

    vkResetFences(device, 1, &inFlightFences[currentFrame]);
    vkResetCommandBuffer(commandBuffers[currentFrame], 0);

    // 현재 프레임 슬롯의 이전 오클루전 쿼리 결과를 읽는다.
    // 위의 fence 대기로 해당 GPU 작업이 끝났음을 보장한다.
    if (optFlags.occlusionCulling
        && occlusionQueryPool != VK_NULL_HANDLE
        && occQueryCount > 0
        && !occResults.empty()) {
        if (occVisible.size() != drawObjects.size() ||
            occZeroStreak.size() != drawObjects.size() ||
            occQueryBuf.size() != drawObjects.size() * 2) {
            resetOcclusionState();
        }
        if (occWarmupFrames > 0) {
            --occWarmupFrames; // GPU 쿼리가 아직 초기화되지 않은 warmup 프레임은 읽지 않는다.
        } else {
            uint32_t base  = currentFrame * occQueryCount;
            uint32_t count = std::min(occQueryCount, (uint32_t)occResults.size());
            std::fill(occQueryBuf.begin(), occQueryBuf.end(), 0);
            VkResult queryResult = vkGetQueryPoolResults(
                device, occlusionQueryPool, base, count,
                occQueryBuf.size() * sizeof(uint64_t), occQueryBuf.data(),
                2 * sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
            (void)queryResult;
            for (uint32_t i = 0; i < count; ++i) {
                if (occQueryBuf[i * 2 + 1] == 0) continue;

                const uint64_t samples = occQueryBuf[i * 2];
                occResults[i] = samples;

                const bool eligible =
                    i < drawObjects.size() &&
                    !drawObjects[i].skipOcclusion &&
                    drawObjects[i].push.baseColor.w >= 0.999f;
                if (!eligible || samples > 0) {
                    occVisible[i] = 1u;
                    occZeroStreak[i] = 0u;
                    continue;
                }

                const int nextStreak = std::min<int>(
                    static_cast<int>(occZeroStreak[i]) + 1,
                    OCC_ZERO_FRAMES_TO_HIDE);
                occZeroStreak[i] = static_cast<uint8_t>(nextStreak);
                if (nextStreak >= OCC_ZERO_FRAMES_TO_HIDE)
                    occVisible[i] = 0u;
            }
        }
    }

    updateUniformBuffer(currentFrame);
    if (ghostMode) buildGizmoGeometry(currentFrame); // 커맨드 기록 전에 기즈모 버퍼를 먼저 업로드한다.
    recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

    VkSemaphore renderFinished = renderFinishedSems[imageIndex];
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &imageAvailableSems[currentFrame];
    si.pWaitDstStageMask    = &waitStage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &commandBuffers[currentFrame];
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &renderFinished;
    vkQueueSubmit(graphicsQueue, 1, &si, inFlightFences[currentFrame]);

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &renderFinished;
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &swapchain;
    pi.pImageIndices      = &imageIndex;
    result = vkQueuePresentKHR(presentQueue, &pi);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

// 스왑체인 재생성

// 디퍼드 렌더링 리소스
// ============================================================================
// [7] 오프스크린 리소스 (G-Buffer / 섀도맵)
// ============================================================================

// createDeferredResources — 디퍼드 렌더링용 G-Buffer 를 만든다.
// G-Buffer 구성 (스왑체인 해상도, 프레임 슬롯별 2벌):
//  0: RGBA8   albedo.rgb + specularStrength
//  1: RGBA16F normal(0..1 인코딩) + shininess/256
//  2: RGBA16F 월드 위치 + (1+발광 강도)  — w<0.5 면 배경 픽셀
//  3: RGBA16F reflectStrength
// 지오메트리 패스가 여기에 쓰고, 조명 패스가 샘플러로 읽는다.
// 창 크기가 바뀌면 recreateSwapChain() 에서 파괴 후 재생성된다.
void VulkanApp::createDeferredResources() {
    // G-Buffer 샘플러는 스왑체인 재생성 사이에도 재사용한다.
    if (gbufSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter    = VK_FILTER_NEAREST;
        si.minFilter    = VK_FILTER_NEAREST;
        si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(device, &si, nullptr, &gbufSampler) != VK_SUCCESS)
            throw std::runtime_error("vkCreateSampler (gbuf) failed");
    }

    // 현재 스왑체인 크기에 맞는 G-Buffer 포맷을 준비한다.
    VkFormat depthFmt = findDepthFormat();
    VkFormat fmts[GBUFFER_COLOR_ATTACHMENTS] = {
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R16G16B16A16_SFLOAT,
    };

    for (int f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f) {
        // G-Buffer 컬러 이미지를 프레임 슬롯별로 생성한다.
        for (int c = 0; c < GBUFFER_COLOR_ATTACHMENTS; ++c) {
            createImage(scExtent.width, scExtent.height, fmts[c],
                        VK_IMAGE_TILING_OPTIMAL,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        gbufImages[c][f], gbufMemories[c][f]);
            gbufViews[c][f] = createImageView(gbufImages[c][f], fmts[c],
                                              VK_IMAGE_ASPECT_COLOR_BIT);
        }

        // 지오메트리 패스 전용 깊이 이미지를 만든다.
        createImage(scExtent.width, scExtent.height, depthFmt,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    gbufDepthImages[f], gbufDepthMemories[f]);
        gbufDepthViews[f] = createImageView(gbufDepthImages[f], depthFmt,
                                            VK_IMAGE_ASPECT_DEPTH_BIT);

        // 컬러 attachment들과 깊이 1개를 묶어 G-Buffer 프레임버퍼를 만든다.
        VkImageView atts[GBUFFER_COLOR_ATTACHMENTS + 1] = {
            gbufViews[0][f], gbufViews[1][f], gbufViews[2][f], gbufViews[3][f], gbufDepthViews[f]
        };
        VkFramebufferCreateInfo fbCI{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fbCI.renderPass      = gbufRenderPass;
        fbCI.attachmentCount = GBUFFER_COLOR_ATTACHMENTS + 1;
        fbCI.pAttachments    = atts;
        fbCI.width           = scExtent.width;
        fbCI.height          = scExtent.height;
        fbCI.layers          = 1;
        if (vkCreateFramebuffer(device, &fbCI, nullptr, &gbufFramebuffers[f]) != VK_SUCCESS)
            throw std::runtime_error("vkCreateFramebuffer (G-Buffer) failed");
    }

    // 디퍼드 조명 패스용 디스크립터 풀과 세트를 만든다.
    if (deferredDescPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, deferredDescPool, nullptr);
        deferredDescPool = VK_NULL_HANDLE;
    }
    VkDescriptorPoolSize poolSz{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                GBUFFER_COLOR_ATTACHMENTS * MAX_FRAMES_IN_FLIGHT};
    VkDescriptorPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolCI.maxSets       = (uint32_t)MAX_FRAMES_IN_FLIGHT;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes    = &poolSz;
    if (vkCreateDescriptorPool(device, &poolCI, nullptr, &deferredDescPool) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDescriptorPool (deferred) failed");

    VkDescriptorSetLayout layouts[2] = {deferredDescSetLayout, deferredDescSetLayout};
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool     = deferredDescPool;
    allocInfo.descriptorSetCount = (uint32_t)MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts        = layouts;
    if (vkAllocateDescriptorSets(device, &allocInfo, deferredDescSets) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateDescriptorSets (deferred) failed");

    // 각 프레임 슬롯의 G-Buffer 이미지 뷰를 디스크립터에 연결한다.
    for (int f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f) {
        VkDescriptorImageInfo imgInfos[GBUFFER_COLOR_ATTACHMENTS] = {
            {gbufSampler, gbufViews[0][f], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {gbufSampler, gbufViews[1][f], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {gbufSampler, gbufViews[2][f], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {gbufSampler, gbufViews[3][f], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        };
        VkWriteDescriptorSet writes[GBUFFER_COLOR_ATTACHMENTS] = {};
        for (int i = 0; i < GBUFFER_COLOR_ATTACHMENTS; i++) {
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = deferredDescSets[f];
            writes[i].dstBinding      = (uint32_t)i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].pImageInfo      = &imgInfos[i];
        }
        vkUpdateDescriptorSets(device, GBUFFER_COLOR_ATTACHMENTS, writes, 0, nullptr);
    }
}

void VulkanApp::destroyDeferredResources() {
    if (deferredDescPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, deferredDescPool, nullptr);
        deferredDescPool = VK_NULL_HANDLE;
    }
    for (int f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f) {
        if (gbufFramebuffers[f] != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(device, gbufFramebuffers[f], nullptr);
            gbufFramebuffers[f] = VK_NULL_HANDLE;
        }
        for (int c = 0; c < GBUFFER_COLOR_ATTACHMENTS; ++c) {
            vkDestroyImageView(device, gbufViews[c][f],    nullptr);
            vkDestroyImage    (device, gbufImages[c][f],   nullptr);
            vkFreeMemory      (device, gbufMemories[c][f], nullptr);
            gbufViews[c][f]    = VK_NULL_HANDLE;
            gbufImages[c][f]   = VK_NULL_HANDLE;
            gbufMemories[c][f] = VK_NULL_HANDLE;
        }
        vkDestroyImageView(device, gbufDepthViews[f],    nullptr);
        vkDestroyImage    (device, gbufDepthImages[f],   nullptr);
        vkFreeMemory      (device, gbufDepthMemories[f], nullptr);
        gbufDepthViews[f]    = VK_NULL_HANDLE;
        gbufDepthImages[f]   = VK_NULL_HANDLE;
        gbufDepthMemories[f] = VK_NULL_HANDLE;
    }
}

// createShadowResources — 방향광(태양/달) 섀도맵 리소스를 만든다.
// 구성: 4096×4096 깊이 전용 이미지(D32) × 프레임 슬롯 2벌 + 전용 렌더패스/프레임버퍼.
// 원리: 매 프레임 라이트 시점(ortho)에서 씬 깊이를 먼저 굽고,
//       메인 패스의 픽셀 셰이더가 "이 픽셀이 라이트에서 보이는가"를 비교해
//       그림자를 판정한다 (shadowVisPCF — 16탭 Poisson 소프트 섀도).
// 샘플러는 clamp-to-border(흰색)라 섀도맵 밖은 "그림자 없음"으로 처리된다.
void VulkanApp::createShadowResources() {
    VkFormat depthFmt = findDepthFormat();

    // 경계 밖(섀도맵 밖)은 항상 밝게(깊이 1.0) 처리하도록 clamp-to-border 샘플러를 만든다.
    // 비교는 셰이더에서 직접 수행하므로 compareEnable은 끈다.
    {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter    = VK_FILTER_NEAREST;
        si.minFilter    = VK_FILTER_NEAREST;
        si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE; // 깊이 1.0 = 그림자 없음
        if (vkCreateSampler(device, &si, nullptr, &shadowSampler) != VK_SUCCESS)
            throw std::runtime_error("vkCreateSampler (shadow) failed");
    }

    // 깊이 전용 렌더패스: 컬러 attachment 없이 깊이만 기록하고 샘플링 가능 레이아웃으로 끝낸다.
    {
        VkAttachmentDescription depth{};
        depth.format         = depthFmt;
        depth.samples        = VK_SAMPLE_COUNT_1_BIT;
        depth.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp        = VK_ATTACHMENT_STORE_OP_STORE; // 메인 패스에서 샘플링하므로 보존한다.
        depth.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        depth.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkAttachmentReference depthRef{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount    = 0;
        sp.pDepthStencilAttachment = &depthRef;

        // 이전 프레임의 섀도맵 샘플링이 끝난 뒤 깊이 쓰기를 시작하고,
        // 깊이 쓰기가 끝난 뒤 메인 패스의 프래그먼트 샘플링이 시작되도록 동기화한다.
        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass    = 0;
        deps[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        deps[0].dstStageMask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].srcSubpass    = 0;
        deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask  = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkRenderPassCreateInfo rpCI{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rpCI.attachmentCount = 1;
        rpCI.pAttachments    = &depth;
        rpCI.subpassCount    = 1;
        rpCI.pSubpasses      = &sp;
        rpCI.dependencyCount = 2;
        rpCI.pDependencies   = deps;
        if (vkCreateRenderPass(device, &rpCI, nullptr, &shadowRenderPass) != VK_SUCCESS)
            throw std::runtime_error("vkCreateRenderPass (shadow) failed");
    }

    // 프레임 슬롯별 섀도맵 깊이 이미지·뷰·프레임버퍼를 만든다.
    for (int f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f) {
        createImage(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, depthFmt,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    shadowImages[f], shadowImageMemories[f]);
        shadowImageViews[f] = createImageView(shadowImages[f], depthFmt, VK_IMAGE_ASPECT_DEPTH_BIT);

        VkFramebufferCreateInfo fbCI{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fbCI.renderPass      = shadowRenderPass;
        fbCI.attachmentCount = 1;
        fbCI.pAttachments    = &shadowImageViews[f];
        fbCI.width           = SHADOW_MAP_SIZE;
        fbCI.height          = SHADOW_MAP_SIZE;
        fbCI.layers          = 1;
        if (vkCreateFramebuffer(device, &fbCI, nullptr, &shadowFramebuffers[f]) != VK_SUCCESS)
            throw std::runtime_error("vkCreateFramebuffer (shadow) failed");
    }
}

void VulkanApp::createShadowPipeline() {
    auto vertCode = readFile("shaders/spv/shadow.vert.spv");
    VkShaderModule vertM = createShaderModule(vertCode);

    VkPipelineShaderStageCreateInfo vertStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    vertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertM;
    vertStage.pName  = "main";

    // push constant = lightMVP (라이트 뷰·투영 × 모델)
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(glm::mat4);

    VkPipelineLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutCI.setLayoutCount         = 0;
    layoutCI.pushConstantRangeCount = 1;
    layoutCI.pPushConstantRanges    = &pcRange;
    if (vkCreatePipelineLayout(device, &layoutCI, nullptr, &shadowPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("vkCreatePipelineLayout (shadow) failed");

    auto binding = Vertex::getBindingDesc();
    auto attrs   = Vertex::getAttrDescs();
    VkPipelineVertexInputStateCreateInfo vertInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertInput.vertexBindingDescriptionCount   = 1;
    vertInput.pVertexBindingDescriptions      = &binding;
    vertInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
    vertInput.pVertexAttributeDescriptions    = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynState.dynamicStateCount = 2;
    dynState.pDynamicStates    = dynStates;

    // 섀도 액네(self-shadowing)를 줄이기 위해 경사 비례 깊이 바이어스를 적용한다.
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode             = VK_POLYGON_MODE_FILL;
    raster.cullMode                = VK_CULL_MODE_NONE; // 양면 모두 깊이를 남겨 누락을 막는다.
    raster.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth               = 1.0f;
    raster.depthBiasEnable         = VK_TRUE;
    raster.depthBiasConstantFactor = 1.25f;
    raster.depthBiasSlopeFactor    = 1.75f;

    VkPipelineMultisampleStateCreateInfo msaa{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthStencil.depthTestEnable  = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

    // 컬러 attachment가 없으므로 블렌드 상태도 비운다.
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 0;
    blend.pAttachments    = nullptr;

    // 깊이 전용 패스는 프래그먼트 셰이더가 필요 없다.
    VkGraphicsPipelineCreateInfo pipeCI{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeCI.stageCount          = 1;
    pipeCI.pStages             = &vertStage;
    pipeCI.pVertexInputState   = &vertInput;
    pipeCI.pInputAssemblyState = &inputAssembly;
    pipeCI.pViewportState      = &viewportState;
    pipeCI.pRasterizationState = &raster;
    pipeCI.pMultisampleState   = &msaa;
    pipeCI.pDepthStencilState  = &depthStencil;
    pipeCI.pColorBlendState    = &blend;
    pipeCI.pDynamicState       = &dynState;
    pipeCI.layout              = shadowPipelineLayout;
    pipeCI.renderPass          = shadowRenderPass;
    pipeCI.subpass             = 0;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &shadowPipeline) != VK_SUCCESS)
        throw std::runtime_error("vkCreateGraphicsPipelines (shadow) failed");

    vkDestroyShaderModule(device, vertM, nullptr);
}

// ============================================================================
// 블룸 포스트프로세스
// ============================================================================
// createBloomPipelines — 렌더패스 2종 + 파이프라인 3종을 만든다 (앱 시작 시 1회).
void VulkanApp::createBloomPipelines() {
    // 1) 하프 해상도 블룸 렌더패스: CLEAR 로 시작해 SHADER_READ 로 끝난다.
    {
        VkAttachmentDescription att{};
        att.format         = VK_FORMAT_B8G8R8A8_UNORM;
        att.samples        = VK_SAMPLE_COUNT_1_BIT;
        att.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        att.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        att.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1;
        sp.pColorAttachments    = &ref;
        // 이전 프레임의 샘플링과 이번 프레임의 쓰기 사이 해저드를 막는다.
        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass    = 0;
        deps[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[1].srcSubpass    = 0;
        deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        ci.attachmentCount = 1; ci.pAttachments = &att;
        ci.subpassCount = 1;    ci.pSubpasses  = &sp;
        ci.dependencyCount = 2; ci.pDependencies = deps;
        if (vkCreateRenderPass(device, &ci, nullptr, &bloomRenderPass) != VK_SUCCESS)
            throw std::runtime_error("bloom render pass failed");
    }
    // 2) 스왑체인 LOAD 렌더패스: 씬 결과를 유지한 채 합성/기즈모/HUD 를 얹는다.
    //    initialLayout=SHADER_READ (블룸이 샘플링한 뒤) -> finalLayout=PRESENT.
    {
        VkAttachmentDescription atts[2]{};
        atts[0].format         = scFormat;
        atts[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        atts[0].loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
        atts[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        atts[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        atts[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        atts[0].initialLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        atts[0].finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        atts[1] = atts[0];
        atts[1].format        = findDepthFormat();
        atts[1].loadOp        = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        atts[1].storeOp       = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        atts[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        atts[1].finalLayout   = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference cRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference dRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount    = 1;
        sp.pColorAttachments       = &cRef;
        sp.pDepthStencilAttachment = &dRef;
        VkSubpassDependency dep{};
        dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass    = 0;
        dep.srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dep.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        ci.attachmentCount = 2; ci.pAttachments = atts;
        ci.subpassCount = 1;    ci.pSubpasses  = &sp;
        ci.dependencyCount = 1; ci.pDependencies = &dep;
        if (vkCreateRenderPass(device, &ci, nullptr, &loadRenderPass) != VK_SUCCESS)
            throw std::runtime_error("load render pass failed");
    }
    // 3) 디스크립터/파이프라인 레이아웃 (샘플러 1개 + push 32B)
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = 1; lci.pBindings = &b;
        if (vkCreateDescriptorSetLayout(device, &lci, nullptr, &postDescSetLayout) != VK_SUCCESS)
            throw std::runtime_error("post desc layout failed");
        VkPushConstantRange pr{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
        VkPipelineLayoutCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pci.setLayoutCount = 1; pci.pSetLayouts = &postDescSetLayout;
        pci.pushConstantRangeCount = 1; pci.pPushConstantRanges = &pr;
        if (vkCreatePipelineLayout(device, &pci, nullptr, &postPipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("post pipeline layout failed");
    }
    // 4) 파이프라인 3종 (풀스크린 삼각형, 깊이 없음)
    {
        auto vertC = readFile("shaders/spv/post.vert.spv");
        VkShaderModule vertM = createShaderModule(vertC);
        auto makePipe = [&](const char* fragPath, VkRenderPass rp,
                            bool blendAdd, bool hasDepth) {
            auto fragC = readFile(fragPath);
            VkShaderModule fragM = createShaderModule(fragC);
            VkPipelineShaderStageCreateInfo st[2]{};
            st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT, vertM, "main", nullptr};
            st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, fragM, "main", nullptr};
            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vps.viewportCount = 1; vps.scissorCount = 1;
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;
            rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.f;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            VkPipelineColorBlendAttachmentState ba{};
            ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            if (blendAdd) { // 가산 합성: dst + src (블룸을 화면에 더한다)
                ba.blendEnable         = VK_TRUE;
                ba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
                ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
                ba.colorBlendOp        = VK_BLEND_OP_ADD;
                ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
                ba.alphaBlendOp        = VK_BLEND_OP_ADD;
            }
            VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cb.attachmentCount = 1; cb.pAttachments = &ba;
            VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dsi.dynamicStateCount = 2; dsi.pDynamicStates = dyn;
            VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            ci.stageCount = 2; ci.pStages = st;
            ci.pVertexInputState = &vi;   ci.pInputAssemblyState = &ia;
            ci.pViewportState = &vps;     ci.pRasterizationState = &rs;
            ci.pMultisampleState = &ms;   ci.pDepthStencilState = hasDepth ? &ds : nullptr;
            ci.pColorBlendState = &cb;    ci.pDynamicState = &dsi;
            ci.layout = postPipelineLayout; ci.renderPass = rp; ci.subpass = 0;
            VkPipeline pipe = VK_NULL_HANDLE;
            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe) != VK_SUCCESS)
                throw std::runtime_error("post pipeline failed");
            vkDestroyShaderModule(device, fragM, nullptr);
            return pipe;
        };
        brightPipeline         = makePipe("shaders/spv/bloom_bright.frag.spv", bloomRenderPass, false, false);
        blurPipeline           = makePipe("shaders/spv/bloom_blur.frag.spv",   bloomRenderPass, false, false);
        bloomCompositePipeline = makePipe("shaders/spv/bloom_composite.frag.spv", loadRenderPass, true, true);
        vkDestroyShaderModule(device, vertM, nullptr);
    }
}

// createBloomResources — 하프 해상도 이미지·프레임버퍼·디스크립터 (리사이즈마다)
void VulkanApp::createBloomResources() {
    bloomExtent = {std::max(1u, scExtent.width / 2), std::max(1u, scExtent.height / 2)};

    if (postSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = sci.addressModeV = sci.addressModeW =
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCreateSampler(device, &sci, nullptr, &postSampler);
    }

    for (int i = 0; i < 2; ++i) {
        createImage(bloomExtent.width, bloomExtent.height, VK_FORMAT_B8G8R8A8_UNORM,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    bloomImages[i], bloomMemories[i]);
        bloomViews[i] = createImageView(bloomImages[i], VK_FORMAT_B8G8R8A8_UNORM,
                                        VK_IMAGE_ASPECT_COLOR_BIT);
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = bloomRenderPass;
        fci.attachmentCount = 1; fci.pAttachments = &bloomViews[i];
        fci.width = bloomExtent.width; fci.height = bloomExtent.height; fci.layers = 1;
        if (vkCreateFramebuffer(device, &fci, nullptr, &bloomFramebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("bloom framebuffer failed");
    }

    // 디스크립터: 스왑체인 이미지별 밝기 추출 소스 + 블러 핑퐁 2개
    uint32_t nSets = (uint32_t)scImageViews.size() + 2;
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nSets};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = nSets; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(device, &pci, nullptr, &postDescPool) != VK_SUCCESS)
        throw std::runtime_error("post desc pool failed");

    auto allocSet = [&](VkImageView view) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = postDescPool;
        ai.descriptorSetCount = 1; ai.pSetLayouts = &postDescSetLayout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        vkAllocateDescriptorSets(device, &ai, &set);
        VkDescriptorImageInfo ii{postSampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set; w.dstBinding = 0; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &ii;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
        return set;
    };
    brightSets.clear();
    for (auto& iv : scImageViews) brightSets.push_back(allocSet(iv));
    blurSets[0] = allocSet(bloomViews[0]);
    blurSets[1] = allocSet(bloomViews[1]);
}

void VulkanApp::destroyBloomResources() {
    for (int i = 0; i < 2; ++i) {
        if (bloomFramebuffers[i]) { vkDestroyFramebuffer(device, bloomFramebuffers[i], nullptr); bloomFramebuffers[i] = VK_NULL_HANDLE; }
        if (bloomViews[i])        { vkDestroyImageView(device, bloomViews[i], nullptr);          bloomViews[i] = VK_NULL_HANDLE; }
        if (bloomImages[i])       { vkDestroyImage(device, bloomImages[i], nullptr);             bloomImages[i] = VK_NULL_HANDLE; }
        if (bloomMemories[i])     { vkFreeMemory(device, bloomMemories[i], nullptr);             bloomMemories[i] = VK_NULL_HANDLE; }
    }
    if (postDescPool) { vkDestroyDescriptorPool(device, postDescPool, nullptr); postDescPool = VK_NULL_HANDLE; }
    brightSets.clear();
    blurSets[0] = blurSets[1] = VK_NULL_HANDLE;
}

void VulkanApp::cleanupSwapChain() {
    destroyBloomResources();
    destroyDeferredResources();
    vkDestroyImageView(device, depthImageView, nullptr);
    vkDestroyImage(device, depthImage, nullptr);
    vkFreeMemory(device, depthImageMemory, nullptr);
    for (auto fb : framebuffers) vkDestroyFramebuffer(device, fb, nullptr);
    for (auto iv : scImageViews) vkDestroyImageView(device, iv, nullptr);
    vkDestroySwapchainKHR(device, swapchain, nullptr);
}

void VulkanApp::recreateSwapChain() {
    int w = 0, h = 0;
    while (w == 0 || h == 0) {
        glfwGetFramebufferSize(window, &w, &h);
        glfwWaitEvents();
    }
    vkDeviceWaitIdle(device);
    cleanupSwapChain();
    createSwapChain();
    createImageViews();
    recreatePresentSemaphores();
    createDepthResources();
    createFramebuffers();
    createDeferredResources();
    createBloomResources(); // 하프 해상도 블룸 타깃도 새 해상도로 재생성
}

// 리플레이와 정리
// 리소스 정리
// 리플레이 시스템

void VulkanApp::startRecording() {
    if (isReplaying) stopReplay();
    recordedFrames.clear();
    recordStartTime = static_cast<float>(glfwGetTime());
    isRecording     = true;
    std::cout << "[Replay] Recording started\n";
}

// 현재 맵 파일에서 확장자·디렉토리를 뗀 스템을 만든다 (리플레이 파일명 접두사).
std::string VulkanApp::currentMapStem() const {
    std::string name = currentMapFile;
    auto slash = name.find_last_of("/\\");
    if (slash != std::string::npos) name = name.substr(slash + 1);
    auto dot = name.rfind('.');
    if (dot != std::string::npos) name = name.substr(0, dot);
    for (char& c : name)
        if (c == ' ' || c == ',') c = '_'; // 파일명·CSV 안전화
    return name.empty() ? std::string("map") : name;
}

// 현재 맵 전용 리플레이 목록. 없으면 전체 목록으로 폴백(구버전 replay_NNN 호환).
std::vector<std::string> VulkanApp::replaysForCurrentMap(bool* usedFallback) const {
    auto files = findReplayFilesFor(replayDir, currentMapStem());
    if (usedFallback) *usedFallback = false;
    if (files.empty()) {
        files = findReplayFiles(replayDir);
        if (usedFallback && !files.empty()) *usedFallback = true;
    }
    return files;
}

void VulkanApp::stopRecording() {
    if (!isRecording) return;
    isRecording = false;
    if (recordedFrames.empty()) { std::cout << "[Replay] Nothing recorded\n"; return; }
    // 맵별로 리플레이를 분리 저장한다: replays/<맵이름>_NNN.replay
    std::string path = nextReplayPathFor(replayDir, currentMapStem());
    saveReplay(path, recordedFrames);
    lastSavedReplay = path;

    // Unity 등 다른 엔진과의 동일 경로 비교 실험용 CSV 를 함께 저장한다.
    std::string csvPath = path.substr(0, path.size() - 7) + ".csv"; // ".replay" -> ".csv"
    exportReplayCsv(csvPath, recordedFrames);
}

void VulkanApp::startReplay(const std::string& path) {
    if (isRecording) stopRecording();
    std::string target = path;
    if (target.empty()) {
        bool fallback = false;
        auto files = replaysForCurrentMap(&fallback);
        if (files.empty()) { std::cerr << "[Replay] No replay files in " << replayDir << "\n"; return; }
        if (fallback)
            std::cout << "[Replay] No replay for map '" << currentMapStem()
                      << "' — falling back to latest replay\n";
        target = files.back();
    }
    replayFrames.clear();
    if (!loadReplay(target, replayFrames) || replayFrames.empty()) return;
    replayFrameIdx  = 0;
    replayStartTime = static_cast<float>(glfwGetTime());
    isReplaying     = true;
    std::cout << "[Replay] Playback started: " << target << "\n";
}

void VulkanApp::stopReplay() {
    isReplaying = false;
    std::cout << "[Replay] Playback stopped\n";
}

// ============================================================================
// [10] 정리
// ============================================================================
// cleanup — 모든 Vulkan 리소스를 "생성의 역순"으로 해제한다.
// Vulkan 은 가비지 컬렉션이 없어서 vkDestroy*/vkFree* 를 빠짐없이 호출해야 하며,
// 사용 중인 리소스를 파괴하면 안 되므로 이 함수는 vkDeviceWaitIdle 이후에 불린다.
void VulkanApp::cleanup() {
    perfStats.cleanup();
    cleanupImGui();
    cleanupSwapChain();

    // 기즈모 버텍스 버퍼
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (i < (int)gizmoVBs.size()) {
            vkDestroyBuffer(device, gizmoVBs[i], nullptr);
            vkFreeMemory(device, gizmoVBMemories[i], nullptr);
        }
    }
    vkDestroyPipeline(device, gizmoPipeline, nullptr);

    // 오클루전 쿼리 풀
    if (occlusionQueryPool != VK_NULL_HANDLE)
        vkDestroyQueryPool(device, occlusionQueryPool, nullptr);

    // 인스턴스 SSBO와 디스크립터 풀
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (i < (int)instanceStagingBuffers.size()) {
            if (i < (int)instanceSSBOMapped.size() && instanceSSBOMapped[i])
                vkUnmapMemory(device, instanceStagingMemories[i]);
            vkDestroyBuffer(device, instanceStagingBuffers[i], nullptr);
            vkFreeMemory(device, instanceStagingMemories[i], nullptr);
        }
        if (i < (int)instanceSSBOs.size()) {
            vkDestroyBuffer(device, instanceSSBOs[i], nullptr);
            vkFreeMemory(device, instanceSSBOMemories[i], nullptr);
        }
    }
    if (instanceDescPool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device, instanceDescPool, nullptr);

    // 프레임별 UBO
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkDestroyBuffer(device, uniformBuffers[i], nullptr);
        vkFreeMemory(device, uniformBufferMemories[i], nullptr);
        vkDestroyBuffer(device, cullUniformBuffers[i], nullptr);
        vkFreeMemory(device, cullUniformBufferMemories[i], nullptr);
        if (i < (int)sceneLightUBOs.size()) {
            vkDestroyBuffer(device, sceneLightUBOs[i], nullptr);
            vkFreeMemory(device, sceneLightUBOMemories[i], nullptr);
        }
    }
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, instanceDescSetLayout, nullptr);

    // 씬 텍스처(씬 재로드마다 갱신)
    cleanupTextureResources();
    // 기본 1×1 흰색 텍스처와 샘플러(앱 생명주기 동안 유지)
    vkDestroyImageView(device, defaultTexView, nullptr);
    vkDestroyImage(device, defaultTexImage, nullptr);
    vkFreeMemory(device, defaultTexMemory, nullptr);
    vkDestroySampler(device, texSampler, nullptr);

    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);

    for (VkSemaphore sem : imageAvailableSems)
        vkDestroySemaphore(device, sem, nullptr);
    for (VkSemaphore sem : renderFinishedSems)
        vkDestroySemaphore(device, sem, nullptr);
    for (VkFence fence : inFlightFences)
        vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);

    // 렌더링 파이프라인
    vkDestroyPipeline(device, graphicsPipelineFlippedCull, nullptr);
    vkDestroyPipeline(device, graphicsPipeline,           nullptr);
    vkDestroyPipeline(device, graphicsPipelineNoCull,     nullptr);
    vkDestroyPipeline(device, graphicsPipelineAlpha,      nullptr);
    vkDestroyPipeline(device, graphicsPipelineQueryOnly, nullptr);
    vkDestroyPipeline(device, graphicsPipelineInst,       nullptr);
    vkDestroyPipeline(device, graphicsPipelineInstNoCull, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout,         nullptr);
    vkDestroyPipelineLayout(device, instancePipelineLayout, nullptr);
    // 디퍼드 렌더링 영구 리소스
    vkDestroyPipeline(device, gbufPipeline,          nullptr);
    vkDestroyPipeline(device, gbufPipelineNoCull,    nullptr);
    vkDestroyPipeline(device, deferredLightPipeline, nullptr);
    vkDestroyPipeline(device, skyPipeline, nullptr);
    vkDestroyPipelineLayout(device, deferredLightLayout, nullptr);
    // 블룸 포스트프로세스 영구 리소스
    vkDestroyPipeline(device, brightPipeline, nullptr);
    vkDestroyPipeline(device, blurPipeline, nullptr);
    vkDestroyPipeline(device, bloomCompositePipeline, nullptr);
    vkDestroyPipelineLayout(device, postPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, postDescSetLayout, nullptr);
    if (bloomRenderPass) vkDestroyRenderPass(device, bloomRenderPass, nullptr);
    if (loadRenderPass)  vkDestroyRenderPass(device, loadRenderPass, nullptr);
    if (postSampler)     vkDestroySampler(device, postSampler, nullptr);
    vkDestroyDescriptorSetLayout(device, deferredDescSetLayout, nullptr);
    if (gbufRenderPass != VK_NULL_HANDLE)
        vkDestroyRenderPass(device, gbufRenderPass, nullptr);
    if (gbufSampler    != VK_NULL_HANDLE)
        vkDestroySampler(device, gbufSampler, nullptr);

    // 방향광 섀도맵 리소스
    vkDestroyPipeline(device, shadowPipeline, nullptr);
    vkDestroyPipelineLayout(device, shadowPipelineLayout, nullptr);
    for (int f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f) {
        if (shadowFramebuffers[f] != VK_NULL_HANDLE)
            vkDestroyFramebuffer(device, shadowFramebuffers[f], nullptr);
        if (shadowImageViews[f] != VK_NULL_HANDLE)
            vkDestroyImageView(device, shadowImageViews[f], nullptr);
        if (shadowImages[f] != VK_NULL_HANDLE)
            vkDestroyImage(device, shadowImages[f], nullptr);
        if (shadowImageMemories[f] != VK_NULL_HANDLE)
            vkFreeMemory(device, shadowImageMemories[f], nullptr);
    }
    if (shadowRenderPass != VK_NULL_HANDLE)
        vkDestroyRenderPass(device, shadowRenderPass, nullptr);
    if (shadowSampler != VK_NULL_HANDLE)
        vkDestroySampler(device, shadowSampler, nullptr);

    vkDestroyRenderPass(device, renderPass, nullptr);
    vkDestroyDevice(device, nullptr);
    if (kEnableValidation) DestroyDebugMessenger(instance, debugMessenger);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
}

// ImGui 초기화와 정리
void VulkanApp::initImGui() {
    // ImGui가 사용할 descriptor pool을 만든다.
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    VkDescriptorPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolCI.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolCI.maxSets       = 1;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes    = &poolSize;
    if (vkCreateDescriptorPool(device, &poolCI, nullptr, &imguiDescPool) != VK_SUCCESS)
        throw std::runtime_error("ImGui descriptor pool creation failed");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr; // imgui.ini 저장을 비활성화한다.

    // ------------------------------------------------------------------
    // Material Design 다크 테마
    // 표면(surface) 색 위에 파랑 계열 primary 액센트, 라운드 코너, 얇은 보더.
    // ProgressBar 채움색은 ImGuiCol_PlotHistogram 을 쓴다는 점에 주의.
    // ------------------------------------------------------------------
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding    = 14.0f; // 카드 모서리
    style.FrameRounding     = 6.0f;  // 바/입력 위젯 모서리
    style.GrabRounding      = 6.0f;
    style.WindowBorderSize  = 1.0f;
    style.WindowPadding     = ImVec2(16.0f, 14.0f);
    style.ItemSpacing       = ImVec2(8.0f, 5.0f);
    style.FramePadding      = ImVec2(6.0f, 3.0f);
    style.Alpha             = 1.0f;

    ImVec4* c = style.Colors;
    c[ImGuiCol_WindowBg]      = ImVec4(0.082f, 0.086f, 0.110f, 1.0f); // 짙은 표면 (#15161C)
    c[ImGuiCol_Border]        = ImVec4(1.0f, 1.0f, 1.0f, 0.06f);      // 아주 옅은 외곽선
    c[ImGuiCol_Text]          = ImVec4(0.905f, 0.915f, 0.940f, 1.0f); // 고대비 텍스트
    c[ImGuiCol_TextDisabled]  = ImVec4(0.520f, 0.545f, 0.610f, 1.0f); // 보조 텍스트
    c[ImGuiCol_FrameBg]       = ImVec4(0.155f, 0.165f, 0.210f, 1.0f); // 바 트랙 배경
    c[ImGuiCol_PlotLines]     = ImVec4(0.392f, 0.710f, 0.965f, 1.0f); // 그래프 = primary
    c[ImGuiCol_PlotHistogram] = ImVec4(0.392f, 0.710f, 0.965f, 1.0f); // ProgressBar 채움
    c[ImGuiCol_Separator]     = ImVec4(1.0f, 1.0f, 1.0f, 0.07f);
    // 스크롤바: 얇고 은은하게 (패널 내용이 화면보다 길 때만 나타남)
    style.ScrollbarSize       = 8.0f;
    style.ScrollbarRounding   = 4.0f;
    c[ImGuiCol_ScrollbarBg]   = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    c[ImGuiCol_ScrollbarGrab] = ImVec4(1.0f, 1.0f, 1.0f, 0.15f);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(1.0f, 1.0f, 1.0f, 0.28f);
    c[ImGuiCol_ScrollbarGrabActive]  = ImVec4(0.392f, 0.710f, 0.965f, 0.6f);

    // Segoe UI 폰트 (Windows 기본 탑재). 없으면 내장 폰트로 자동 폴백.
    {
        const char* uiFont = "C:/Windows/Fonts/segoeui.ttf";
        const char* sbFont = "C:/Windows/Fonts/seguisb.ttf"; // Semibold (히어로 숫자용)
        if (std::filesystem::exists(uiFont)) {
            fontRegular = io.Fonts->AddFontFromFileTTF(uiFont, 17.0f);
            fontLarge   = io.Fonts->AddFontFromFileTTF(
                std::filesystem::exists(sbFont) ? sbFont : uiFont, 30.0f);
        }
    }

    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion     = VK_API_VERSION_1_2;
    initInfo.Instance       = instance;
    initInfo.PhysicalDevice = physDevice;
    initInfo.Device         = device;
    initInfo.QueueFamily    = findQueueFamilies(physDevice).graphics.value();
    initInfo.Queue          = graphicsQueue;
    initInfo.DescriptorPool = imguiDescPool;
    initInfo.MinImageCount  = 2;
    initInfo.ImageCount     = static_cast<uint32_t>(scImages.size());
    // ImGui 1.92 이상 규칙에 맞춰 렌더 패스와 MSAA 값을 지정한다.
    initInfo.PipelineInfoMain.RenderPass   = renderPass;
    initInfo.PipelineInfoMain.MSAASamples  = VK_SAMPLE_COUNT_1_BIT;
    ImGui_ImplVulkan_Init(&initInfo);
}

void VulkanApp::cleanupImGui() {
    vkDeviceWaitIdle(device);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    if (imguiDescPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, imguiDescPool, nullptr);
        imguiDescPool = VK_NULL_HANDLE;
    }
}

// 성능 오버레이
// 벤치마크 CSV 로거
void VulkanApp::startBenchmark() {
    benchmarkSamples.clear();
    benchmarkElapsed = 0.f;
    benchmarkActive  = true;
    printf("[Benchmark] Started (%.0f s)\n", benchmarkDuration);
}

void VulkanApp::finishBenchmark() {
    benchmarkActive = false;
    if (benchmarkSamples.empty()) return;

    int   n      = (int)benchmarkSamples.size();
    float sumFps = 0, sumFt = 0, sumCpu = 0, sumGpu = 0, sumRam = 0;
    float minFt  = 1e9f, maxFt = 0.f;
    int   sumDc  = 0,    sumCulled = 0;
    for (auto& s : benchmarkSamples) {
        sumFps    += s.fps;
        sumFt     += s.frameTimeMs;
        sumCpu    += s.cpuPercent;
        sumGpu    += s.gpuPercent;
        sumRam    += s.ramMB;
        sumDc     += s.drawCalls;
        sumCulled += s.culled;
        minFt = std::min(minFt, s.frameTimeMs);
        maxFt = std::max(maxFt, s.frameTimeMs);
    }
    float avgFt  = sumFt / n;
    float avgFps = (avgFt > 0.0f) ? 1000.0f / avgFt : 0.0f;

    std::filesystem::create_directories("results");

    auto       tp = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(tp);
    char ts[20]   = {};
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&t));

    // 최적화 플래그 문자열 (9비트)
    char flags[12] = {};
    std::snprintf(flags, sizeof(flags), "%d%d%d%d%d%d%d%d%d",
        (int)optFlags.frustumCulling,  (int)optFlags.lod,
        (int)optFlags.instancing,      (int)optFlags.backfaceCulling,
        (int)optFlags.depthSort,       (int)optFlags.occlusionCulling,
        (int)optFlags.viewDistCulling, (int)optFlags.smallCulling,
        (int)optFlags.deferredShading);

    std::string path = std::string("results/bench_") + flags
                     + "_stress" + std::to_string(stressLevel)
                     + "_cap" + std::to_string(currentFpsCap())
                     + "_" + ts + ".csv";

    std::ofstream csv(path);
    csv << "frame,fps,frametime_ms,cpu_pct,gpu_pct,ram_mb,draw_calls,culled\n";
    for (int i = 0; i < n; ++i) {
        auto& s = benchmarkSamples[i];
        csv << i << "," << s.fps << "," << s.frameTimeMs << ","
            << s.cpuPercent << "," << s.gpuPercent << "," << s.ramMB << ","
            << s.drawCalls  << "," << s.culled << "\n";
    }
    csv << "\n# summary: avg_fps,avg_ft_ms,min_ft_ms,max_ft_ms,avg_cpu_pct,avg_gpu_pct,avg_ram_mb,avg_dc,avg_culled\n";
    csv << "summary,"
        << avgFps      << "," << avgFt        << ","
        << minFt        << "," << maxFt        << ","
        << (sumCpu / n) << "," << (sumGpu / n) << ","
        << (sumRam / n) << ","
        << (sumDc  / n) << "," << (sumCulled / n) << "\n";
    csv << "# flags(FC,LOD,Inst,BFC,DS,OC,VDC,SC,Def): " << flags << "\n";
    csv << "# stress_level: " << stressLevel
        << "  objects: " << (int)drawObjects.size() << "\n";
    csv << "# fps_cap: " << currentFpsCap()
        << "  (0 = uncapped/max-FPS mode, >0 = fixed-frame mode)\n";

    printf("[Benchmark] Saved: %s  (avg FPS %.1f  avg ft %.2f ms)\n",
           path.c_str(), avgFps, avgFt);
}

// 스트레스 씬 복제
void VulkanApp::applyStress() {
    if (occlusionQueryPool != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
        vkDestroyQueryPool(device, occlusionQueryPool, nullptr);
        occlusionQueryPool = VK_NULL_HANDLE;
        occQueryCount      = 0;
    }

    if (stressLevel == 0) {
        drawObjects = baseDrawObjects;
    } else {
        const int   extra   = (1 << stressLevel) - 1; // 2x=1, 4x=3, 8x=7, 16x=15
        const float SPACING = 30.0f;
        drawObjects = baseDrawObjects;
        int row = 1, col = 0;
        for (int c = 0; c < extra; ++c) {
            glm::vec3 offset((float)col * SPACING, 0.f, (float)row * SPACING);
            for (const auto& obj : baseDrawObjects) {
                DrawObject copy      = obj;
                copy.push.model      = glm::translate(glm::mat4(1.f), offset) * obj.push.model;
                copy.boundCenter     = obj.boundCenter + offset;
                copy.skipOcclusion   = true;
                drawObjects.push_back(copy);
            }
            if (++col >= 4) { col = 0; ++row; }
        }
    }

    rebuildInstancingGroups();
    resetOcclusionState();
    if (!drawObjects.empty()) createOcclusionQueryPool();
    adaptiveSceneDirty = true; // 부하가 달라졌으니 적응형 측정 기록 무효화

    printf("[Stress] Level %d -> %dx  (%d objects)\n",
           stressLevel, 1 << stressLevel, (int)drawObjects.size());
}

// ============================================================================
// [8] 자동 벤치마크 (리플레이 기반)
// ============================================================================
// 실험 설계: baseline(전부 OFF) + 최적화 기법 1개씩 켠 실험 9개
//           + 적응형 컨트롤러(`) 실험 1개 = 총 11실험.
// 벤치마크는 전부 클래식 셰이더(현실적 셰이더 OFF)로만 측정한다.
// 각 실험을 두 FPS 조건(최대 프레임 / 60fps 고정)에서 AUTO_BENCH_RUNS(2)회씩,
// 같은 리플레이로 재생하며 측정한다 → 실험당 2조건 × 2반복 = 4런.
// 흐름: startAutoBenchmark → (실험×조건×반복) startAutoBenchRun → 리플레이 종료 감지
//       → onAutoBenchRunEnd(런 요약 집계) → ... → finishAutoBenchmark(CSV 저장)
// 리플레이는 현재 맵 전용 파일(<맵이름>_NNN.replay)을 자동 선택한다.
// B 키(startBenchmarkAllMaps)를 쓰면 모든 맵에 대해 이 과정을 순서대로 반복한다.
void VulkanApp::startAutoBenchmark() {
    // 현재 맵 전용 리플레이를 우선 사용한다 (맵마다 같은 movement 로 측정).
    bool fallback = false;
    auto files = replaysForCurrentMap(&fallback);
    if (files.empty()) {
        printf("[AutoBench] No replay found. Record a replay with [R] first.\n");
        return;
    }
    if (fallback)
        printf("[AutoBench] Warning: no replay for map '%s' — using latest replay (path may not match this map)\n",
               currentMapStem().c_str());
    autoBenchReplayPath = files.back(); // 이 맵의 가장 최근 리플레이

    // 적응형 최적화가 켜져 있으면 먼저 끄고 사용자 플래그로 되돌린다.
    // (컨트롤러가 실험별 OptFlags 를 중간에 뒤집지 못하도록)
    if (adaptive.enabled) {
        adaptive.reset(optFlags);
        adaptive.enabled = false;
        optFlags = adaptiveSavedFlags;
        printf("[Adaptive] OFF - auto benchmark started\n");
    }

    // 현재 상태 저장 & 불필요한 기능 비활성화 (순수 렌더 성능만 측정)
    autoBenchSavedFlags       = optFlags;
    autoBenchSavedGhost       = ghostMode;
    autoBenchSavedRealistic   = realisticShading; // 현실적 셰이더 상태 저장 (완료 시 복원)
    autoBenchSavedFpsCapIndex = fpsCapIndex;      // FPS 캡 상태 저장 (완료 시 복원)
    ghostMode        = false; // ghost 모드 OFF (기즈모·오버레이 컬링 시각화 제거)
    realisticShading = false; // 벤치마크는 항상 클래식 셰이더(현실적 셰이더 OFF)로만 측정

    // 10 실험 정의
    // 모든 실험은 baseline(all OFF) 기준으로 하나씩 최적화 기법을 추가
    OptFlags off{};
    off.frustumCulling = off.lod = off.instancing = off.backfaceCulling =
    off.depthSort = off.occlusionCulling = off.viewDistCulling =
    off.smallCulling = off.deferredShading = false;

    autoBenchExps.clear();
    auto addExp = [&](const std::string& name, OptFlags f, bool adaptiveExp = false) {
        AutoBenchExp e; e.name = name; e.flags = f; e.adaptiveExp = adaptiveExp;
        autoBenchExps.push_back(std::move(e));
    };

    OptFlags f;
    addExp("0.Baseline (all OFF)", off);

    f = off; f.frustumCulling   = true;  addExp("1.Frustum Culling",   f);
    f = off; f.lod               = true;  addExp("2.LOD",               f);
    f = off; f.instancing        = true;  addExp("3.GPU Instancing",    f);
    f = off; f.backfaceCulling   = true;  addExp("4.Backface Culling",  f);
    f = off; f.depthSort         = true;  addExp("5.Depth Sort",        f);
    f = off; f.occlusionCulling  = true;  addExp("6.Occlusion Culling", f);
    f = off; f.viewDistCulling   = true;  addExp("7.View Dist Cull",    f);
    f = off; f.smallCulling      = true;  addExp("8.Small Obj Cull",    f);
    f = off; f.deferredShading   = true;  addExp("9.Deferred Shading",  f);

    // 적응형 컨트롤러(`): baseline(전부 OFF)에서 시작해 리플레이 중 컨트롤러가
    // 직접 기법을 켜고 끈다. A/B 실측이 필요해 무제한(최대 프레임) 조건에서만
    // 실제로 동작하고, 60fps 고정 조건에서는 자동 일시정지되어 baseline 으로 측정된다.
    addExp("10.Adaptive (`)", off, /*adaptiveExp=*/true);

    autoBenchExpIdx  = 0;
    autoBenchCondIdx = 0;
    autoBenchRunIdx  = 0;
    autoBenchActive  = true;
    printf("[AutoBench] Starting: %d experiments x %d conditions(max,60fps) x %d runs  replay=%s\n",
           AUTO_BENCH_TOTAL, AUTOBENCH_CONDS, AUTO_BENCH_RUNS, autoBenchReplayPath.c_str());
    startAutoBenchRun();
}

void VulkanApp::startAutoBenchRun() {
    auto& exp = autoBenchExps[autoBenchExpIdx];
    optFlags = exp.flags;

    // 적응형 실험이면 컨트롤러를 켜서 리플레이 중 직접 최적화하게 한다.
    // 그 외 실험은 컨트롤러를 꺼서 고정 플래그로만 측정한다.
    autoBenchAdaptiveRun = exp.adaptiveExp;
    if (exp.adaptiveExp) {
        adaptive.reset(optFlags);   // 깨끗한 상태(플래그 유지)에서 시작
        adaptive.enabled   = true;
        adaptiveSceneDirty = true;  // 다음 update 에서 적용 가능 기법 재계산
    } else {
        adaptive.reset(optFlags);
        adaptive.enabled = false;
    }

    // 현재 조건에 맞는 FPS 캡을 적용한다 (0 = 최대 프레임, 60 = 60fps 고정).
    const int cap = AUTOBENCH_CAPS[autoBenchCondIdx];
    fpsCapIndex = 0; // 기본값(무제한)
    for (int i = 0; i < 4; ++i) if (FPS_CAPS[i] == cap) { fpsCapIndex = i; break; }
    exp.current.clear();
    autoBenchSkipFrames = 3;
    startReplay(autoBenchReplayPath);
    printf("[AutoBench] Exp %d/%d '%s'  cap=%s  run %d/%d\n",
           autoBenchExpIdx + 1, AUTO_BENCH_TOTAL, exp.name.c_str(),
           cap == 0 ? "max" : "60fps",
           autoBenchRunIdx + 1, AUTO_BENCH_RUNS);
}

void VulkanApp::onAutoBenchRunEnd() {
    auto& exp = autoBenchExps[autoBenchExpIdx];
    if (!exp.current.empty()) {
        int   n      = (int)exp.current.size();
        float sumFps = 0, sumFt = 0, sumCpu = 0, sumGpu = 0, sumRam = 0;
        float minFt  = 1e9f, maxFt = 0.f;
        int   sumDc  = 0, sumCulled = 0;
        for (auto& s : exp.current) {
            sumFps    += s.fps;
            sumFt     += s.frameTimeMs;
            sumCpu    += s.cpuPercent;
            sumGpu    += s.gpuPercent;
            sumRam    += s.ramMB;
            sumDc     += s.drawCalls;
            sumCulled += s.culled;
            minFt = std::min(minFt, s.frameTimeMs);
            maxFt = std::max(maxFt, s.frameTimeMs);
        }
        float avgFt  = sumFt / n;
        float avgFps = (avgFt > 0.0f) ? 1000.0f / avgFt : 0.0f;
        AutoBenchRunResult r{};
        r.fpsCap    = AUTOBENCH_CAPS[autoBenchCondIdx]; // 이 런의 FPS 조건
        r.avgFps    = avgFps;
        r.avgFtMs   = avgFt;
        r.minFtMs   = minFt;
        r.maxFtMs   = maxFt;
        r.avgCpu    = sumCpu / n;
        r.avgGpu    = sumGpu / n;
        r.avgRam    = sumRam / n;
        r.avgDc     = sumDc  / n;
        r.avgCulled = sumCulled / n;
        exp.runs.push_back(r);
        exp.current.clear();
    }

    // 진행 순서: 반복(run) -> FPS 조건(cond) -> 실험(exp)
    ++autoBenchRunIdx;
    if (autoBenchRunIdx < AUTO_BENCH_RUNS) {
        startAutoBenchRun();
        return;
    }
    // 이 조건의 반복이 끝났으면 다음 FPS 조건으로
    autoBenchRunIdx = 0;
    ++autoBenchCondIdx;
    if (autoBenchCondIdx < AUTOBENCH_CONDS) {
        startAutoBenchRun();
        return;
    }
    // 모든 조건이 끝났으면 다음 실험으로
    autoBenchCondIdx = 0;
    ++autoBenchExpIdx;
    if (autoBenchExpIdx < AUTO_BENCH_TOTAL) {
        startAutoBenchRun();
        return;
    }
    finishAutoBenchmark();
}

void VulkanApp::finishAutoBenchmark() {
    autoBenchActive      = false;
    autoBenchAdaptiveRun = false;
    // 적응형 실험이 마지막이므로 컨트롤러를 끄고 정리한다.
    adaptive.enabled = false;
    adaptive.reset(optFlags);
    optFlags         = autoBenchSavedFlags;
    ghostMode        = autoBenchSavedGhost;
    realisticShading = autoBenchSavedRealistic;   // 현실적 셰이더 상태 복원
    fpsCapIndex      = autoBenchSavedFpsCapIndex;  // FPS 캡 상태 복원

    std::filesystem::create_directories("results");

    auto       tp = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(tp);
    char ts[20]   = {};
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&t));

    // 맵 이름을 파일명에 넣어 맵별 결과를 구분한다 (전체 맵 벤치마크 시 특히 유용).
    std::string path = std::string("results/autobench_") + currentMapStem()
                     + "_" + ts + ".csv";
    std::ofstream csv(path);

    // 상세: 각 (실험 × FPS조건 × 반복) 런의 결과.
    // fps_cap: 0 = 최대 프레임(무제한), 60 = 60fps 고정
    csv << "# shading: classic (realistic shader OFF for all benchmarks)\n";
    csv << "experiment,fps_cap,run,avg_fps,avg_ft_ms,min_ft_ms,max_ft_ms,"
           "avg_cpu_pct,avg_gpu_pct,avg_ram_mb,avg_dc,avg_culled\n";

    for (auto& exp : autoBenchExps) {
        // 조건별로 반복 번호(run)를 1부터 다시 매겨 출력한다.
        for (int ci = 0; ci < AUTOBENCH_CONDS; ++ci) {
            int cap = AUTOBENCH_CAPS[ci];
            int runNo = 0;
            for (auto& res : exp.runs) {
                if (res.fpsCap != cap) continue;
                csv << exp.name << "," << cap << "," << (++runNo) << ","
                    << res.avgFps    << "," << res.avgFtMs   << ","
                    << res.minFtMs   << "," << res.maxFtMs   << ","
                    << res.avgCpu    << "," << res.avgGpu    << ","
                    << res.avgRam    << ","
                    << res.avgDc     << "," << res.avgCulled << "\n";
            }
        }
    }

    // FPS 조건별 요약: 각 조건 안에서 실험별 평균 + baseline 대비 개선률
    csv << "\n# --- 요약 (조건별, 반복 " << AUTO_BENCH_RUNS << "회 평균) ---\n";
    csv << "experiment,fps_cap,avg_fps,avg_ft_ms,fps_vs_baseline_pct\n";

    // 한 실험의 특정 조건 평균 fps/ft 를 구하는 헬퍼
    auto condAvg = [](const AutoBenchExp& e, int cap, float& fps, float& ft) {
        fps = ft = 0.f; int n = 0;
        for (auto& r : e.runs) if (r.fpsCap == cap) { fps += r.avgFps; ft += r.avgFtMs; ++n; }
        if (n > 0) { fps /= n; ft /= n; }
        return n;
    };

    float summaryBaseMaxFps = 0.f, summaryBaseMaxFt = 0.f; // 콘솔 출력용(최대프레임 baseline)
    for (int ci = 0; ci < AUTOBENCH_CONDS; ++ci) {
        int cap = AUTOBENCH_CAPS[ci];
        float baseFps = 0.f, baseFt = 0.f;
        if (!autoBenchExps.empty())
            condAvg(autoBenchExps[0], cap, baseFps, baseFt);
        if (cap == 0) { summaryBaseMaxFps = baseFps; summaryBaseMaxFt = baseFt; }
        for (auto& exp : autoBenchExps) {
            float fps = 0.f, ft = 0.f;
            if (condAvg(exp, cap, fps, ft) == 0) continue;
            float improvement = (baseFps > 0.f) ? (fps - baseFps) / baseFps * 100.f : 0.f;
            csv << exp.name << "," << cap << "," << fps << "," << ft << "," << improvement << "\n";
        }
    }

    csv << "# map: "    << currentMapStem() << "\n";
    csv << "# replay: " << autoBenchReplayPath << "\n";
    csv << "# conditions: max(uncapped) + 60fps,  runs_per_condition: " << AUTO_BENCH_RUNS << "\n";
    csv << "# stress_level: " << stressLevel
        << "  objects: " << (int)drawObjects.size() << "\n";

    printf("[AutoBench] Done! Saved: %s\n", path.c_str());
    printf("[AutoBench] Baseline (max-fps) avg FPS: %.1f  ft: %.2f ms\n",
           summaryBaseMaxFps, summaryBaseMaxFt);

    // 전체 맵 벤치마크 중이면 이번 맵을 마쳤으니 다음 맵으로 넘어간다.
    if (benchAllMapsActive) {
        ++benchAllMapIdx;
        benchAllMapsStartCurrent();
    }
}

// ---------------------------------------------------------------------------
// 전체 맵 벤치마크 (B 키)
// availableMaps 를 순서대로 로드하며 각 맵에서 자동 벤치마크를 돌린다.
// 리플레이가 없는 맵은 건너뛴다. 각 맵 결과는 개별 CSV(autobench_<맵>_*.csv)로 저장.
// ---------------------------------------------------------------------------
void VulkanApp::startBenchmarkAllMaps() {
    if (autoBenchActive || benchAllMapsActive) return;
    if (availableMaps.empty()) {
        printf("[BenchAll] No maps to benchmark.\n");
        return;
    }
    benchAllMapsActive    = true;
    benchAllMapIdx        = 0;
    benchAllMapsReturnMap = currentMapFile; // 완료 후 돌아올 맵 기억
    printf("[BenchAll] Benchmarking all %d maps (first -> last)...\n",
           (int)availableMaps.size());
    benchAllMapsStartCurrent();
}

void VulkanApp::benchAllMapsStartCurrent() {
    // 리플레이가 있는 맵을 찾을 때까지 앞으로 진행한다.
    while (benchAllMapIdx < (int)availableMaps.size()) {
        currentMapFile  = availableMaps[benchAllMapIdx];
        currentMapIndex = benchAllMapIdx;
        reloadScene(); // 실패 시 내부에서 이전 맵으로 되돌린다.
        if (optFlags.occlusionCulling) resetOcclusionState();

        if (replaysForCurrentMap().empty()) {
            printf("[BenchAll] Skip '%s' (no replay recorded)\n",
                   currentMapStem().c_str());
            ++benchAllMapIdx;
            continue;
        }

        printf("[BenchAll] Map %d/%d: %s\n",
               benchAllMapIdx + 1, (int)availableMaps.size(),
               currentMapStem().c_str());
        startAutoBenchmark();
        // startAutoBenchmark 가 어떤 이유로 시작하지 못했으면 다음 맵으로.
        if (!autoBenchActive) { ++benchAllMapIdx; continue; }
        return; // 정상 시작 -> 이 맵이 끝나면 finishAutoBenchmark 가 다음 맵을 호출
    }

    benchAllMapsActive = false;
    printf("[BenchAll] All maps done.\n");

    // 시작 시점의 맵으로 되돌린다 (벤치마크가 마지막 맵에 머무르지 않도록).
    if (!benchAllMapsReturnMap.empty() && currentMapFile != benchAllMapsReturnMap) {
        currentMapFile = benchAllMapsReturnMap;
        for (int i = 0; i < (int)availableMaps.size(); ++i)
            if (availableMaps[i] == currentMapFile) { currentMapIndex = i; break; }
        reloadScene();
        if (optFlags.occlusionCulling) resetOcclusionState();
    }
}

// M/B 벤치마크를 사용자가 중단할 때: 재생/상태를 원래대로 복원한다.
void VulkanApp::abortAutoBenchmark() {
    printf("[AutoBench] Aborted by user.\n");
    if (isReplaying) stopReplay();
    autoBenchActive      = false;
    autoBenchAdaptiveRun = false;
    benchAllMapsActive   = false;
    adaptive.enabled     = false;
    adaptive.reset(optFlags);
    optFlags           = autoBenchSavedFlags;
    ghostMode          = autoBenchSavedGhost;
    realisticShading   = autoBenchSavedRealistic;
    fpsCapIndex        = autoBenchSavedFpsCapIndex;
}

// ============================================================================
// [9] HUD (ImGui 좌측 패널)
// ============================================================================
// 표시 순서: FPS/프레임타임(+그래프) → CPU/RAM/GPU → 드로우콜 → 씬 로드 시간
// → 최적화 토글 상태(키 안내 포함) → 조명 → 환경(시간/안개/구름) → 벤치마크.
// 자동 벤치마크 중에는 진행률만 표시하는 최소 모드로 전환해 측정 오버헤드를 줄인다.
// ============================================================================
// 적응형 최적화 (` 키)
// ============================================================================
// 매 프레임: 씬 변경을 감지해 적용 가능 기법을 갱신하고, 컨트롤러에 프레임
// 시간을 공급한다. 컨트롤러가 오클루전 컬링을 방금 켰으면 쿼리 상태를 예열한다.
void VulkanApp::updateAdaptiveOptimizer(float dt) {
    if (!adaptive.enabled) return;

    // 씬 교체·스트레스 배율 변경 감지 → 측정 기록 무효화 + 적용 가능 기법 재계산
    if (adaptiveSceneDirty || drawObjects.size() != adaptiveLastObjCount) {
        adaptiveSceneDirty   = false;
        adaptiveLastObjCount = drawObjects.size();
        bool hasLod = false;
        for (const DrawObject& o : drawObjects)
            if (o.numLods > 0) { hasLod = true; break; }
        bool hasInst = false;
        for (const InstGroupDef& g : instGroupDefs)
            if ((int)g.members.size() >= MIN_INSTANCES_PER_DRAW) { hasInst = true; break; }
        adaptive.reset(optFlags);
        adaptive.setApplicability(hasLod, hasInst);
    }

    // FPS 캡(고정 프레임 모드)·벤치마크 중에는 프레임 시간 비교가 무의미 → 일시정지.
    // 단, 자동 벤치마크의 "적응형 실험"에서는 컨트롤러가 실제로 동작해야 하므로
    // (무제한 조건 한정) 일시정지하지 않는다. 60fps 고정 조건은 캡>0 이라 여전히 정지.
    const bool pauseNow = benchmarkActive || currentFpsCap() > 0 ||
                          (autoBenchActive && !autoBenchAdaptiveRun);

    const bool    occBefore = optFlags.occlusionCulling;
    const Camera& cam       = ghostMode ? observerCamera : camera;
    adaptive.update(optFlags, dt * 1000.f, pauseNow, cam.position, cam.yaw, cam.pitch);
    if (!occBefore && optFlags.occlusionCulling)
        resetOcclusionState(); // 컨트롤러가 방금 켠 오클루전 쿼리 예열
}

// 적응형 최적화 상태 오버레이 — 우상단 카드 (adaptive ON 일 때만).
// 지금 어떤 기법이 켜져 있는지, 무엇을 측정 중인지, 기법별 실측 이득(%)을 보여준다.
void VulkanApp::drawAdaptiveOverlay() {
    if (!adaptive.enabled || autoBenchActive) return;

    const ImVec4 OK_GREEN = {0.506f, 0.780f, 0.518f, 1.0f};
    const ImVec4 WARN_AMB = {1.000f, 0.835f, 0.310f, 1.0f};
    const ImVec4 ERR_RED  = {0.898f, 0.451f, 0.451f, 1.0f};
    const ImVec4 TEAL     = {0.302f, 0.816f, 0.882f, 1.0f};
    const ImVec4 TXT_HI   = {0.905f, 0.915f, 0.940f, 1.0f};
    const ImVec4 TXT_MED  = {0.640f, 0.665f, 0.730f, 1.0f};
    const ImVec4 TXT_DIM  = {0.470f, 0.495f, 0.560f, 1.0f};
    const float  W        = 258.0f;

    ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos({io.DisplaySize.x - W - 12.0f, 12.0f}, ImGuiCond_Always);
    ImGui::SetNextWindowSizeConstraints({W, 0}, {W, io.DisplaySize.y - 24.0f});
    ImGui::SetNextWindowSize({W, 0}, ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.90f);
    ImGuiWindowFlags wflags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav;
    if (!ImGui::Begin("##adaptive", nullptr, wflags)) { ImGui::End(); return; }
    ImDrawList* dl = ImGui::GetWindowDrawList();

    // 헤더: 액센트 바 + 제목 + 토글 키 표기
    {
        ImVec2 p = ImGui::GetCursorScreenPos();
        dl->AddRectFilled({p.x, p.y + 2}, {p.x + 3.0f, p.y + 16.0f},
                          ImGui::GetColorU32(TEAL), 1.5f);
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 10.0f);
        ImGui::TextColored(TEAL, "ADAPTIVE OPTIMIZER");
        ImGui::SameLine(W - 40.0f);
        ImGui::TextColored(TXT_DIM, "[ ` ]");
    }

    // 상태 줄: 일시정지 / 측정 중 <기법> / 대기(모니터링)
    if (adaptive.paused) {
        ImGui::TextColored(WARN_AMB, currentFpsCap() > 0
                           ? "Paused - FPS cap active"
                           : "Paused - benchmark running");
    } else if (adaptive.testTech >= 0) {
        const auto& t = adaptive.techs[adaptive.testTech];
        ImGui::TextColored(WARN_AMB, "Testing: %s", t.name);
        ImGui::TextColored(TXT_DIM, "window %c %d/%d   %d samples%s",
                           (adaptive.subWindow & 1) ? 'B' : 'A',
                           adaptive.subWindow + 1, 2 * AdaptiveOptimizer::ROUNDS,
                           adaptive.curWindowSamples(),
                           adaptive.phase == AdaptiveOptimizer::Phase::Warmup
                               ? "  (warmup)" : "");
    } else if (adaptive.sweepDone) {
        ImGui::TextColored(OK_GREEN, "Stable - retest on view change");
    } else if (adaptive.stillSec < AdaptiveOptimizer::STILL_SEC) {
        ImGui::TextColored(TXT_MED, "Waiting for camera to settle...");
    } else {
        ImGui::TextColored(TXT_MED, "Sweeping techniques...");
    }
    ImGui::Separator();

    // 기법별 행:  [상태 점] 이름 ..... 실측 이득%
    // 이득% = ON 이 OFF 대비 몇 % 빠른가. 초록 = 켜서 이득, 빨강 = 켜면 손해.
    for (int i = 0; i < AdaptiveOptimizer::NUM_TECH; ++i) {
        const auto& t       = adaptive.techs[i];
        bool        on      = optFlags.*(t.flag);
        bool        testing = (adaptive.testTech == i);

        ImVec2 rp  = ImGui::GetCursorScreenPos();
        ImVec4 dot = !t.applicable ? ImVec4{0.35f, 0.36f, 0.40f, 1.0f}
                   : on            ? OK_GREEN
                                   : ImVec4{0.45f, 0.47f, 0.52f, 1.0f};
        dl->AddCircleFilled({rp.x + 5.0f, rp.y + 8.0f}, 3.5f, ImGui::GetColorU32(dot));
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 15.0f);
        ImVec4 nameCol = testing       ? WARN_AMB
                       : !t.applicable ? TXT_DIM
                       : on            ? TXT_HI : TXT_MED;
        ImGui::TextColored(nameCol, "%s", t.name);

        char gtxt[32];
        ImVec4 gcol = TXT_DIM;
        if (!t.applicable) {
            std::snprintf(gtxt, sizeof(gtxt), "n/a");
        } else if (testing) {
            std::snprintf(gtxt, sizeof(gtxt), "...");
            gcol = WARN_AMB;
        } else if (!t.tested) {
            std::snprintf(gtxt, sizeof(gtxt), "--");
        } else {
            std::snprintf(gtxt, sizeof(gtxt), "%+.1f%%", t.fpsGainPct);
            gcol = (t.fpsGainPct >=  AdaptiveOptimizer::HYST_PCT) ? OK_GREEN
                 : (t.fpsGainPct <= -AdaptiveOptimizer::HYST_PCT) ? ERR_RED : TXT_MED;
        }
        ImVec2 gs = ImGui::CalcTextSize(gtxt);
        ImGui::SameLine(W - 16.0f - gs.x);
        ImGui::TextColored(gcol, "%s", gtxt);
    }

    ImGui::Separator();
    ImGui::TextColored(TXT_DIM, "gain%% = measured FPS of ON vs OFF");
    ImGui::TextColored(TXT_DIM, "keys 1-6/9/0 locked (7,8 stay manual)");
    ImGui::End();
}

void VulkanApp::drawStatsOverlay() {
    // ------------------------------------------------------------------
    // Material Design 팔레트 (다크 서피스 + 파랑 primary)
    // ------------------------------------------------------------------
    const ImVec4 PRIMARY   = {0.392f, 0.710f, 0.965f, 1.0f}; // Blue 300  (#64B5F6)
    const ImVec4 PRIM_DIM  = {0.392f, 0.710f, 0.965f, 0.35f};
    const ImVec4 OK_GREEN  = {0.506f, 0.780f, 0.518f, 1.0f}; // Green 300 (#81C784)
    const ImVec4 WARN_AMB  = {1.000f, 0.835f, 0.310f, 1.0f}; // Amber 300 (#FFD54F)
    const ImVec4 ERR_RED   = {0.898f, 0.451f, 0.451f, 1.0f}; // Red 300   (#E57373)
    const ImVec4 TEAL      = {0.302f, 0.816f, 0.882f, 1.0f}; // Cyan 300  (#4DD0E1)
    const ImVec4 TXT_HI    = {0.905f, 0.915f, 0.940f, 1.0f}; // 본문
    const ImVec4 TXT_MED   = {0.640f, 0.665f, 0.730f, 1.0f}; // 보조
    const ImVec4 TXT_DIM   = {0.470f, 0.495f, 0.560f, 1.0f}; // 힌트
    const float  PANEL_W   = 252.0f;

    // 오버레이 창을 좌상단에 고정 (Material 카드).
    // 내용이 화면보다 길면 잘리는 대신 카드 안에서 스크롤되도록 높이를 제한한다.
    ImGui::SetNextWindowPos({12.0f, 12.0f}, ImGuiCond_Always);
    ImGui::SetNextWindowSizeConstraints({PANEL_W, 0},
        {PANEL_W, ImGui::GetIO().DisplaySize.y - 24.0f});
    ImGui::SetNextWindowSize({PANEL_W, 0}, ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.90f);
    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav;

    if (!ImGui::Begin("##stats", nullptr, flags)) { ImGui::End(); return; }
    ImDrawList* dl = ImGui::GetWindowDrawList();

    // ---- 재사용 위젯 헬퍼 ------------------------------------------------

    // 섹션 헤더: 왼쪽 세로 액센트 바 + primary 색 제목 (Material overline 스타일)
    auto section = [&](const char* title) {
        ImGui::Dummy({0, 5});
        ImVec2 p = ImGui::GetCursorScreenPos();
        dl->AddRectFilled({p.x, p.y + 2}, {p.x + 3.0f, p.y + 16.0f},
                          ImGui::GetColorU32(PRIMARY), 1.5f);
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 10.0f);
        ImGui::TextColored(PRIMARY, "%s", title);
        ImGui::Dummy({0, 1});
    };

    // 키 칩: 둥근 알약 배경 위에 단축키 표기
    auto chip = [&](const char* txt) {
        ImVec2 sz = ImGui::CalcTextSize(txt);
        ImVec2 p  = ImGui::GetCursorScreenPos();
        float  padX = 7.0f, h = sz.y + 3.0f;
        float  w = std::max(sz.x + padX * 2.0f, h); // 한 글자도 원형 유지
        dl->AddRectFilled(p, {p.x + w, p.y + h},
                          ImGui::GetColorU32({0.180f, 0.195f, 0.260f, 1.0f}), h * 0.5f);
        dl->AddText({p.x + (w - sz.x) * 0.5f, p.y + 1.5f},
                    ImGui::GetColorU32({0.720f, 0.790f, 0.980f, 1.0f}), txt);
        ImGui::Dummy({w, h});
        ImGui::SameLine(0, 7);
    };

    // 머티리얼 스위치: 트랙(pill) + 노브(원). on 이면 primary, off 면 회색.
    auto switchAt = [&](float xRight, bool on) {
        const float w = 26.0f, h = 13.0f;
        ImGui::SameLine(xRight - w);
        ImVec2 p = ImGui::GetCursorScreenPos();
        float  cy = p.y + h * 0.5f + 2.0f;
        dl->AddRectFilled({p.x, p.y + 2}, {p.x + w, p.y + 2 + h},
                          ImGui::GetColorU32(on ? PRIM_DIM : ImVec4{0.24f, 0.25f, 0.30f, 1.0f}),
                          h * 0.5f);
        dl->AddCircleFilled({on ? p.x + w - h * 0.5f : p.x + h * 0.5f, cy},
                            h * 0.5f - 1.0f,
                            ImGui::GetColorU32(on ? PRIMARY : ImVec4{0.55f, 0.57f, 0.63f, 1.0f}));
        ImGui::Dummy({w, h + 3});
    };

    // 토글 행: [칩] 라벨 ......... (스위치)
    auto toggleRow = [&](const char* key, const char* label, bool on) {
        chip(key);
        ImGui::TextColored(on ? TXT_HI : TXT_MED, "%s", label);
        switchAt(PANEL_W - 18.0f, on);
    };

    // 사용률 바: 라벨 + 우측 정렬 수치 + 얇은 색상 바 (초록→호박→빨강)
    auto usageBar = [&](const char* label, float pct, const char* valTxt, bool known) {
        ImGui::TextColored(TXT_MED, "%s", label);
        ImVec2 vs = ImGui::CalcTextSize(valTxt);
        ImGui::SameLine(PANEL_W - 18.0f - vs.x);
        ImGui::TextColored(known ? TXT_HI : TXT_DIM, "%s", valTxt);
        ImVec4 barCol = (pct < 50.f) ? OK_GREEN : (pct < 80.f) ? WARN_AMB : ERR_RED;
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, known ? barCol : ImVec4{0.3f,0.3f,0.35f,1});
        ImGui::ProgressBar(known ? pct / 100.f : 0.f, {-1, 4.0f}, "");
        ImGui::PopStyleColor();
    };

    char buf[64];

    // ---- 자동 벤치마크 중: 진행 상황만 표시하는 최소 모드 -----------------
    if (autoBenchActive) {
        section(benchAllMapsActive ? "ALL-MAP BENCHMARK" : "AUTO-BENCHMARK");
        // 전체 맵 모드면 현재 맵 이름과 맵 진행도를 함께 표시한다.
        if (benchAllMapsActive) {
            ImGui::TextColored(TXT_HI, "Map %d/%d: %s",
                               benchAllMapIdx + 1, (int)availableMaps.size(),
                               currentMapStem().c_str());
        }
        int perExp    = AUTOBENCH_CONDS * AUTO_BENCH_RUNS; // 실험당 총 런 수
        int totalRuns = AUTO_BENCH_TOTAL * perExp;
        int doneRuns  = autoBenchExpIdx * perExp
                      + autoBenchCondIdx * AUTO_BENCH_RUNS + autoBenchRunIdx;
        std::snprintf(buf, sizeof(buf), "%d / %d", doneRuns, totalRuns);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, TEAL);
        ImGui::ProgressBar((float)doneRuns / totalRuns, {-1, 14.0f}, buf);
        ImGui::PopStyleColor();
        if (autoBenchExpIdx < (int)autoBenchExps.size()) {
            const auto& e   = autoBenchExps[autoBenchExpIdx];
            const int   cap = AUTOBENCH_CAPS[autoBenchCondIdx];
            ImGui::TextColored(TXT_HI, "%s", e.name.c_str());
            ImGui::TextColored(cap == 0 ? WARN_AMB : TEAL,
                               "cap: %s", cap == 0 ? "max frame" : "60 fps");
            // 적응형 실험이고 실제로 동작 중(무제한)이면 표시
            if (e.adaptiveExp)
                ImGui::TextColored(cap == 0 ? PRIMARY : TXT_DIM,
                                   "adaptive: %s", cap == 0 ? "live" : "paused (cap)");
            ImGui::TextColored(TXT_MED, "Run %d/%d", autoBenchRunIdx + 1, AUTO_BENCH_RUNS);
        }
        ImGui::TextColored((perfStats.fps >= 60.f) ? OK_GREEN : ERR_RED,
                           "FPS %.1f", perfStats.fps);
        chip(benchAllMapsActive ? "B" : "M"); ImGui::TextColored(TXT_DIM, "Abort");
        ImGui::End();
        return; // 나머지 UI 전부 스킵 -> 측정 오버헤드 최소화
    }

    // ---- FPS 히어로 영역 (모든 페이지에서 항상 표시) ----------------------
    {
        ImVec4 fpsCol = (perfStats.fps >= 60.f) ? OK_GREEN
                      : (perfStats.fps >= 30.f) ? WARN_AMB : ERR_RED;
        if (fontLarge) ImGui::PushFont(fontLarge);
        std::snprintf(buf, sizeof(buf), "%.0f", perfStats.fps);
        ImGui::TextColored(fpsCol, "%s", buf);
        ImVec2 numSz = ImGui::CalcTextSize(buf);
        if (fontLarge) ImGui::PopFont();
        ImGui::SameLine(0, 6);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + numSz.y - ImGui::GetTextLineHeight() - 3);
        ImGui::TextColored(TXT_MED, "FPS   %.2f ms", perfStats.frameTimeMs);
        if (currentFpsCap() > 0) {
            chip("I");
            ImGui::TextColored(TEAL, "Cap %d fps (fixed-frame)", currentFpsCap());
        }
    }

    // 프레임 시간 그래프 (링버퍼 -> 선형 언롤, 항상 표시)
    if (frameTimeHistCount > 0) {
        float hist[FRAME_HISTORY_SIZE];
        int n = frameTimeHistCount;
        int oldest = (frameTimeHistIdx - n + FRAME_HISTORY_SIZE) % FRAME_HISTORY_SIZE;
        for (int i = 0; i < n; ++i)
            hist[i] = frameTimeHistBuf[(oldest + i) % FRAME_HISTORY_SIZE];
        float ftMin = hist[0], ftMax = hist[0], ftSum = 0.f;
        for (int i = 0; i < n; ++i) {
            ftMin = std::min(ftMin, hist[i]);
            ftMax = std::max(ftMax, hist[i]);
            ftSum += hist[i];
        }
        std::snprintf(buf, sizeof(buf), "avg %.1f ms", ftSum / (float)n);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4{0.11f, 0.115f, 0.15f, 1.0f});
        ImGui::PlotLines("##ft", hist, n, 0, buf, 0.f,
                         std::max(ftMax * 1.2f, 33.f), {-1, 42});
        ImGui::PopStyleColor();
        ImGui::TextColored(TXT_DIM, "min %.2f   max %.2f ms", ftMin, ftMax);
    }

    // ---- 페이지 내비게이션 (Ctrl + 방향키) --------------------------------
    // 도트 인디케이터: 현재 페이지는 크고 파란 점, 나머지는 작은 회색 점.
    static const char* PAGE_NAMES[HUD_PAGE_COUNT] = {
        "System · Rendering", "Optimizations", "Lighting",
        "Environment", "Benchmark", "Modes · Replay",
    };
    {
        ImGui::Dummy({0, 5});
        const float gap = 16.0f;
        float totalW = gap * (HUD_PAGE_COUNT - 1);
        ImVec2 p = ImGui::GetCursorScreenPos();
        float x0 = p.x + (PANEL_W - 32.0f - totalW) * 0.5f;
        for (int i = 0; i < HUD_PAGE_COUNT; ++i)
            dl->AddCircleFilled({x0 + gap * i, p.y + 5.0f},
                                (i == hudPage) ? 3.8f : 2.4f,
                                ImGui::GetColorU32((i == hudPage) ? PRIMARY
                                                                  : ImVec4{1, 1, 1, 0.22f}));
        ImGui::Dummy({0, 11});
        // 페이지 제목(가운데) + 전환 힌트
        ImVec2 ts = ImGui::CalcTextSize(PAGE_NAMES[hudPage]);
        ImGui::SetCursorPosX((PANEL_W - ts.x) * 0.5f);
        ImGui::TextColored(TXT_HI, "%s", PAGE_NAMES[hudPage]);
        const char* hint = "Ctrl + < >  page";
        ImVec2 hs = ImGui::CalcTextSize(hint);
        ImGui::SetCursorPosX((PANEL_W - hs.x) * 0.5f);
        ImGui::TextColored(TXT_DIM, "%s", hint);
    }

    // ======================================================================
    // 페이지별 내용
    // ======================================================================
    switch (hudPage) {

    case 0: { // ---- System + Rendering ----------------------------------
        section("System");
        std::snprintf(buf, sizeof(buf), "%.0f %%", perfStats.cpuPercent);
        usageBar("CPU (system)", perfStats.cpuPercent, buf, true);
        std::snprintf(buf, sizeof(buf), "%.1f %%", perfStats.processCpuPercent);
        usageBar("CPU (process)", perfStats.processCpuPercent, buf, true);
        if (perfStats.gpuPercent >= 0.f) {
            std::snprintf(buf, sizeof(buf), "%.0f %%", perfStats.gpuPercent);
            usageBar("GPU", perfStats.gpuPercent, buf, true);
        } else {
            usageBar("GPU", 0.f, "N/A", false);
        }
        ImGui::TextColored(TXT_MED, "RAM");
        std::snprintf(buf, sizeof(buf), "%.1f MB", perfStats.ramMB);
        ImGui::SameLine(PANEL_W - 18.0f - ImGui::CalcTextSize(buf).x);
        ImGui::TextColored(TXT_HI, "%s", buf);

        section("Rendering");
        ImGui::TextColored(TXT_MED, "Draw calls");
        std::snprintf(buf, sizeof(buf), "%d / %d", renderedCount, (int)drawObjects.size());
        ImGui::SameLine(PANEL_W - 18.0f - ImGui::CalcTextSize(buf).x);
        ImGui::TextColored(TXT_HI, "%s", buf);
        {   // 컬링 비율 바 (culled / total)
            int total = (int)drawObjects.size();
            float cullFrac = total > 0 ? (float)culledCount / total : 0.f;
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, TEAL);
            ImGui::ProgressBar(cullFrac, {-1, 4.0f}, "");
            ImGui::PopStyleColor();
            ImGui::TextColored(TXT_DIM, "culled %d (%.0f%%)", culledCount, cullFrac * 100.f);
        }
        if (optFlags.instancing && instancedGroupCount > 0)
            ImGui::TextColored(TEAL, "instanced draws  %d", instancedGroupCount);
        if (lastLoadTiming.valid)
            ImGui::TextColored(TXT_DIM, "load %.0f ms (parse %.0f / up %.0f)",
                               lastLoadTiming.totalMs, lastLoadTiming.sceneParseMs,
                               lastLoadTiming.uploadMs);
        break;
    }

    case 1: { // ---- Optimizations (머티리얼 스위치) -----------------------
        section("Optimizations");
        toggleRow("1", "Frustum Culling",   optFlags.frustumCulling);
        toggleRow("2", "LOD",               optFlags.lod);
        toggleRow("3", "GPU Instancing",    optFlags.instancing);
        toggleRow("4", "Backface Culling",  optFlags.backfaceCulling);
        toggleRow("5", "Front-Back Sort",   optFlags.depthSort);
        toggleRow("6", "Occlusion Culling", optFlags.occlusionCulling);
        toggleRow("7", "View Dist Cull",    optFlags.viewDistCulling);
        toggleRow("8", "Small Obj Cull",    optFlags.smallCulling);
        toggleRow("9", "Deferred Shading",  optFlags.deferredShading);
        toggleRow("0", "All Optimizations", optFlags.frustumCulling && optFlags.lod &&
            optFlags.instancing && optFlags.backfaceCulling && optFlags.depthSort &&
            optFlags.occlusionCulling && optFlags.viewDistCulling &&
            optFlags.smallCulling && optFlags.deferredShading);
        break;
    }

    case 2: { // ---- Lighting --------------------------------------------
        section("Lighting");
        toggleRow("L", "Scene Lights", sceneLightsOn);
        toggleRow("N", "Ambient",      ambientOn);
        toggleRow("V", "Emissive",     emissiveOn);
        if (!sceneLights.empty()) {
            ImGui::TextColored(TXT_DIM, "GLTF lights (%d)", (int)sceneLights.size());
            for (int li = 0; li < (int)sceneLights.size() && li < MAX_SCENE_LIGHTS; ++li) {
                // 상태 점(도트) + 이름
                ImVec2 p = ImGui::GetCursorScreenPos();
                dl->AddCircleFilled({p.x + 7, p.y + ImGui::GetTextLineHeight() * 0.55f}, 3.0f,
                    ImGui::GetColorU32(sceneLights[li].enabled ? OK_GREEN
                                                               : ImVec4{0.35f, 0.36f, 0.42f, 1}));
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 18.0f);
                ImGui::TextColored(sceneLights[li].enabled ? TXT_MED : TXT_DIM,
                                   "%s", sceneLights[li].name.c_str());
            }
        }
        chip("Scroll");
        ImGui::TextColored(TXT_DIM, "rotate sun azimuth");
        break;
    }

    case 3: { // ---- Environment -----------------------------------------
        section("Environment");
        {
            int hh = (int)(timeOfDay * 24.f) % 24;
            int mm = (int)(timeOfDay * 24.f * 60.f) % 60;
            ImGui::TextColored(TXT_MED, "Time");
            std::snprintf(buf, sizeof(buf), "%02d:%02d %s", hh, mm,
                          isNightNow ? "night" : "day");
            ImGui::SameLine(PANEL_W - 18.0f - ImGui::CalcTextSize(buf).x);
            ImGui::TextColored(isNightNow ? ImVec4{0.62f, 0.68f, 0.98f, 1} : WARN_AMB,
                               "%s", buf);
        }
        toggleRow("H", "Realistic Shading", realisticShading);
        toggleRow("Y", "Day/Night Cycle",   dayNightCycle);
        toggleRow("X", "Distance Fog",      fogOn);
        chip("K");
        ImGui::TextColored(TXT_MED, "Clouds");
        std::snprintf(buf, sizeof(buf), "%.0f %%", cloudiness * 100.f);
        ImGui::SameLine(PANEL_W - 18.0f - ImGui::CalcTextSize(buf).x);
        ImGui::TextColored(TXT_HI, "%s", buf);
        chip("U"); chip("J");
        ImGui::TextColored(TXT_DIM, "time scrub");
        break;
    }

    case 4: { // ---- Benchmark -------------------------------------------
        section("Benchmark");
        chip("I");
        if (currentFpsCap() > 0)
            ImGui::TextColored(TEAL, "FPS cap %d (fixed-frame)", currentFpsCap());
        else
            ImGui::TextColored(TXT_MED, "FPS cap off (max-FPS)");

        if (benchmarkActive) {
            float prog = glm::clamp(benchmarkElapsed / benchmarkDuration, 0.f, 1.f);
            std::snprintf(buf, sizeof(buf), "%.1f / %.0f s", benchmarkElapsed, benchmarkDuration);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, TEAL);
            ImGui::ProgressBar(prog, {-1, 14.0f}, buf);
            ImGui::PopStyleColor();
            ImGui::TextColored(WARN_AMB, "%d samples", (int)benchmarkSamples.size());
            chip("M"); ImGui::TextColored(TXT_DIM, "stop & save CSV");
        } else {
            bool hasReplay = !replaysForCurrentMap().empty();
            chip("M");
            if (hasReplay) {
                ImGui::TextColored(TXT_HI, "Auto-bench  %d exp",
                                   AUTO_BENCH_TOTAL);
                ImGui::TextColored(TXT_DIM, "max+60fps x%d, this map", AUTO_BENCH_RUNS);
            } else {
                ImGui::TextColored(TXT_HI, "%.0fs benchmark", benchmarkDuration);
                ImGui::TextColored(TXT_DIM, "record replay for auto-bench");
            }
            // B 키: 전체 맵 순차 벤치마크
            chip("B");
            ImGui::TextColored(TXT_HI, "All maps");
            ImGui::TextColored(TXT_DIM, "first -> last, each map's replay");
        }
        {   // 스트레스 배율
            int mult = 1 << stressLevel;
            chip("T");
            ImGui::TextColored(TXT_MED, "Stress");
            std::snprintf(buf, sizeof(buf), "%dx  (%d objs)", mult, (int)drawObjects.size());
            ImGui::SameLine(PANEL_W - 18.0f - ImGui::CalcTextSize(buf).x);
            ImGui::TextColored(stressLevel == 0 ? TXT_HI
                               : (stressLevel <= 2 ? WARN_AMB : ERR_RED), "%s", buf);
        }
        break;
    }

    case 5: { // ---- Modes + Replay --------------------------------------
        section("Modes");
        toggleRow("G", "Ghost Mode", ghostMode);
        if (ghostMode)
            ImGui::TextColored(TXT_DIM, "arrows: move placed cam");
        toggleRow("C", "Cinematic Cam", camera.cinematic);
        chip("TAB");
        ImGui::TextColored(TXT_DIM, "next map");
        chip("F11");
        ImGui::TextColored(TXT_DIM, "fullscreen");

        section("Replay");
        if (isRecording) {
            float elapsed = static_cast<float>(glfwGetTime()) - recordStartTime;
            bool blink = (static_cast<int>(elapsed * 2.f) % 2) == 0; // 0.5초 간격 깜빡임
            ImVec2 p = ImGui::GetCursorScreenPos();
            if (blink)
                dl->AddCircleFilled({p.x + 7, p.y + ImGui::GetTextLineHeight() * 0.55f}, 4.5f,
                                    ImGui::GetColorU32(ERR_RED));
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 18.0f);
            ImGui::TextColored(ERR_RED, "REC  %.1fs  %d frames",
                               elapsed, (int)recordedFrames.size());
            chip("R"); ImGui::TextColored(TXT_DIM, "stop & save");
        } else if (isReplaying && !replayFrames.empty()) {
            float elapsed = static_cast<float>(glfwGetTime()) - replayStartTime;
            float total   = replayFrames.back().time;
            float prog    = (total > 0.f) ? glm::clamp(elapsed / total, 0.f, 1.f) : 1.f;
            ImGui::TextColored(PRIMARY, "REPLAY  %.1f / %.1fs", elapsed, total);
            ImGui::ProgressBar(prog, {-1, 4.0f}, "");
            chip("P"); ImGui::TextColored(TXT_DIM, "stop");
        } else {
            chip("R"); ImGui::TextColored(TXT_MED, "record path");
            chip("P");
            if (!replaysForCurrentMap().empty())
                ImGui::TextColored(TXT_MED, "play latest replay");
            else
                ImGui::TextColored(TXT_DIM, "no replay for this map");
        }
        break;
    }
    } // switch (hudPage)

    ImGui::End();
}
