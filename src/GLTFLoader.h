#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <glm/glm.hpp>

struct Vertex;
struct DrawObject;

// SceneLight: GLTF에서 추출한 조명
struct SceneLight {
    enum Type { Point = 0, Directional = 1, Spot = 2 };
    Type      type          = Point;
    glm::vec3 position      = {0.f, 0.f, 0.f}; // 월드 공간 (Point/Spot)
    glm::vec3 direction     = {0.f,-1.f, 0.f}; // 월드 공간 조명 방향
    glm::vec3 color         = {1.f, 1.f, 1.f};
    float     intensity     = 1.f;
    float     range         = 0.f; // 0 = 무제한
    float     innerConeAngle= 0.f; // Spot 내부 반각(rad)
    float     outerConeAngle= 0.785f; // Spot 외부 반각(rad)
    std::string name        = "Light";
    bool      enabled       = true; // 런타임 토글
    // 발광(emissive) 재질에서 자동 생성된 조명인지 여부.
    // true 면 V 키(emissive 토글)에 연동되어 함께 켜지고 꺼진다.
    bool      fromEmissive  = false;
};

// GLTFTextureData: CPU 메모리에 디코딩된 텍스처
// tinygltf + stb_image 가 GLB 내장 이미지를 RGBA8 로 자동 디코딩한다.
struct GLTFTextureData {
    int width  = 0;
    int height = 0;
    std::vector<uint8_t> pixels; // RGBA8 순서, 크기 = width * height * 4
};

// GLTFSceneDesc: GLTF 씬 메타데이터 (씬 이름·시작 카메라·조명)
struct GLTFSceneDesc {
    std::string name        = "GLTF Scene";
    glm::vec3   cameraPos   = {0.f, 5.f, -15.f};
    float       cameraYaw   = 90.f;
    float       cameraPitch = -10.f;
    std::vector<SceneLight> lights; // KHR_lights_punctual 에서 추출한 씬 조명
};

// loadGLTFScene
// .glb / .gltf 파일을 읽어 각 노드(Blender 오브젝트)를 별도의 DrawObject 로
// 변환한다. 오브젝트들은 합쳐지지 않으므로 Frustum Culling 등 최적화 기법을
// 오브젝트 단위로 적용할 수 있다.
// 인자:
// path      – .glb 또는 .gltf 파일 경로
// verts     – 글로벌 버텍스 배열 (데이터를 뒤에 추가함)
// inds      – 글로벌 인덱스 배열 (데이터를 뒤에 추가함)
// objects   – DrawObject 목록 (항목을 뒤에 추가함)
// sceneDesc – 씬 이름·카메라 초기 위치 등 메타데이터 반환
// textures  – RGBA8 디코딩된 텍스처 목록 (항목을 뒤에 추가함)
// 반환: 성공 true, 실패 false
bool loadGLTFScene(const std::string&          path,
                   std::vector<Vertex>&         verts,
                   std::vector<uint32_t>&       inds,
                   std::vector<DrawObject>&     objects,
                   GLTFSceneDesc&               sceneDesc,
                   std::vector<GLTFTextureData>& textures);
