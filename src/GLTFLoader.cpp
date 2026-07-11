// ============================================================================
// GLTFLoader.cpp — GLB/GLTF 맵 로더 (맵 형식은 이것 하나로 통일)
// ============================================================================
// loadGLTFScene() 의 처리 순서:
//  1. tinygltf 로 파일 파싱 (GLB 내장 텍스처는 stb_image 가 RGBA8 로 디코딩)
//  2. 임베드 안 된 외부 경로 이미지(C:/xxx.png 등)를 maps/texture/ 에서 파일명으로 복구
//  3. 동반 MTL(maps/mtl/N.mtl)의 map_Kd 로 텍스처 없는 재질에 외부 텍스처 자동 매칭
//  4. 씬 그래프 재귀 순회(traverseNode) → 노드마다 processPrimitive:
//     - 재질 해석: baseColor/알파/metallic·roughness(팩터+텍스처)/emissive(+strength)
//     - 버텍스 생성: 위치·법선·UV·버텍스컬러(COLOR_0)
//     - 감김 정렬: 삼각형 winding 을 법선 방향에 맞춰 통일
//     - LOD 자동 생성: 정점 클러스터링으로 저품질 메시 2단계
//     - ★ 프리미티브 캐시: 같은 메시를 쓰는 노드들이 지오메트리를 공유하고
//       자동으로 GPU 인스턴싱 그룹이 된다 (Blender 링크 복제 = 공짜 인스턴싱)
//  5. KHR_lights_punctual 조명 추출 (Point/Directional/Spot + 콘 각도)
//  6. 카메라 노드가 있으면 시작 위치로 사용
//
// tinygltf – 헤더 온리 라이브러리. 이 .cpp 파일 하나에서만 구현체를 정의한다.
// stb_image 로 GLB 내장 텍스처를 RGBA8 로 자동 디코딩한다.
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#include <tiny_gltf.h>

#include "GLTFLoader.h"
#include "VulkanApp.h" // 사용하는 렌더링 타입: Vertex, DrawObject, PushConstants

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <unordered_set>

// computeBoundSphereLocal
// 로컬 공간 AABB 와 월드 변환 행렬로 바운딩 구를 계산한다.
// VulkanApp::computeBoundSphere 와 동일한 알고리즘 (AABB 8 코너 변환).
static void computeBoundSphereLocal(DrawObject&      obj,
                                    const glm::vec3& bmin,
                                    const glm::vec3& bmax)
{
    const glm::mat4& M = obj.push.model;
    glm::vec3 corners[8];
    int k = 0;
    for (float x : {bmin.x, bmax.x})
        for (float y : {bmin.y, bmax.y})
            for (float z : {bmin.z, bmax.z})
                corners[k++] = glm::vec3(M * glm::vec4(x, y, z, 1.f));

    glm::vec3 center{};
    for (auto& c : corners) center += c;
    center /= 8.f;

    float radius = 0.f;
    for (auto& c : corners)
        radius = std::max(radius, glm::length(c - center));

    obj.boundCenter = center;
    obj.boundRadius = radius;
}
static float determinant3x3(const glm::mat4& m)
{
    const glm::vec3 c0(m[0]);
    const glm::vec3 c1(m[1]);
    const glm::vec3 c2(m[2]);
    return glm::dot(c0, glm::cross(c1, c2));
}

static void alignTriangleWindingToNormals(const std::vector<Vertex>& verts,
                                          std::vector<uint32_t>&     inds,
                                          uint32_t                   idxStart,
                                          uint32_t                   idxCount)
{
    for (uint32_t i = 0; i + 2 < idxCount; i += 3) {
        const Vertex& v0 = verts[inds[idxStart + i + 0]];
        const Vertex& v1 = verts[inds[idxStart + i + 1]];
        const Vertex& v2 = verts[inds[idxStart + i + 2]];

        glm::vec3 faceN = glm::cross(v1.pos - v0.pos, v2.pos - v0.pos);
        float faceLen2 = glm::dot(faceN, faceN);
        if (faceLen2 <= 1e-12f) continue;

        glm::vec3 avgN = v0.normal + v1.normal + v2.normal;
        float avgLen2 = glm::dot(avgN, avgN);
        if (avgLen2 <= 1e-12f) continue;

        if (glm::dot(faceN, avgN) < 0.f)
            std::swap(inds[idxStart + i + 1], inds[idxStart + i + 2]);
    }
}

static bool buildClusteredLod(std::vector<Vertex>&          verts,
                              std::vector<uint32_t>&        inds,
                              uint32_t                      vertBase,
                              uint32_t                      vertCount,
                              uint32_t                      idxStart,
                              uint32_t                      idxCount,
                              const glm::vec3&              bmin,
                              const glm::vec3&              bmax,
                              int                           maxCellsOnLongestAxis,
                              uint32_t&                     outStart,
                              uint32_t&                     outCount)
{
    struct Accum {
        glm::vec3 pos{};
        glm::vec3 normal{};
        glm::vec3 color{};
        glm::vec2 uv{};
        uint32_t  count = 0;
    };

    const glm::vec3 extent = bmax - bmin;
    const float maxExtent = std::max({extent.x, extent.y, extent.z});
    if (maxExtent <= 1e-6f || vertCount == 0 || idxCount < 3)
        return false;

    glm::ivec3 cells(1);
    for (int axis = 0; axis < 3; ++axis) {
        float e = extent[axis];
        cells[axis] = (e <= 1e-6f)
            ? 1
            : std::max(1, (int)std::ceil(maxCellsOnLongestAxis * e / maxExtent));
    }

    auto cellKey = [&](const glm::vec3& p) -> uint64_t {
        glm::ivec3 c(0);
        for (int axis = 0; axis < 3; ++axis) {
            if (extent[axis] > 1e-6f) {
                float t = (p[axis] - bmin[axis]) / extent[axis];
                c[axis] = glm::clamp((int)std::floor(t * cells[axis]), 0, cells[axis] - 1);
            }
        }
        return (uint64_t)c.x | ((uint64_t)c.y << 16) | ((uint64_t)c.z << 32);
    };

    std::unordered_map<uint64_t, uint32_t> clusterMap;
    std::vector<Accum> clusters;
    std::vector<uint32_t> remap(vertCount, UINT32_MAX);

    for (uint32_t i = 0; i < vertCount; ++i) {
        const Vertex& v = verts[vertBase + i];
        uint64_t key = cellKey(v.pos);
        auto it = clusterMap.find(key);
        uint32_t clusterIdx = 0;
        if (it == clusterMap.end()) {
            clusterIdx = static_cast<uint32_t>(clusters.size());
            clusterMap.emplace(key, clusterIdx);
            clusters.push_back({});
        } else {
            clusterIdx = it->second;
        }

        Accum& a = clusters[clusterIdx];
        a.pos    += v.pos;
        a.normal += v.normal;
        a.color  += v.color;
        a.uv     += v.uv;
        ++a.count;
        remap[i] = clusterIdx;
    }

    if (clusters.size() < 3 || clusters.size() >= vertCount)
        return false;

    std::vector<uint32_t> lodIndices;
    lodIndices.reserve(idxCount);
    std::unordered_set<uint64_t> seenTriangles;

    auto triKey = [](uint32_t a, uint32_t b, uint32_t c) -> uint64_t {
        uint32_t lo = std::min({a, b, c});
        uint32_t hi = std::max({a, b, c});
        uint32_t mid = a + b + c - lo - hi;
        return (uint64_t)lo | ((uint64_t)mid << 21) | ((uint64_t)hi << 42);
    };

    for (uint32_t i = 0; i + 2 < idxCount; i += 3) {
        uint32_t src[3] = {
            inds[idxStart + i + 0],
            inds[idxStart + i + 1],
            inds[idxStart + i + 2],
        };
        if (src[0] < vertBase || src[1] < vertBase || src[2] < vertBase)
            return false;
        src[0] -= vertBase;
        src[1] -= vertBase;
        src[2] -= vertBase;
        if (src[0] >= vertCount || src[1] >= vertCount || src[2] >= vertCount)
            return false;

        uint32_t a = remap[src[0]];
        uint32_t b = remap[src[1]];
        uint32_t c = remap[src[2]];
        if (a == b || b == c || c == a)
            continue;

        uint64_t key = triKey(a, b, c);
        if (!seenTriangles.insert(key).second)
            continue;

        lodIndices.push_back(a);
        lodIndices.push_back(b);
        lodIndices.push_back(c);
    }

    if (lodIndices.size() < 12 || lodIndices.size() >= idxCount)
        return false;

    const uint32_t lodVertStart = static_cast<uint32_t>(verts.size());
    for (const Accum& a : clusters) {
        const float inv = 1.0f / static_cast<float>(a.count);
        Vertex v{};
        v.pos    = a.pos * inv;
        v.normal = (glm::dot(a.normal, a.normal) > 1e-10f)
                 ? glm::normalize(a.normal)
                 : glm::vec3(0.f, 1.f, 0.f);
        v.color  = a.color * inv;
        v.uv     = a.uv * inv;
        verts.push_back(v);
    }

    outStart = static_cast<uint32_t>(inds.size());
    for (uint32_t idx : lodIndices)
        inds.push_back(lodVertStart + idx);
    outCount = static_cast<uint32_t>(inds.size()) - outStart;
    return true;
}

// nodeLocalTransform
// GLTF 노드의 로컬 변환 행렬을 반환한다.
// matrix 가 있으면 그대로 사용, 없으면 TRS 에서 조합.
static glm::mat4 nodeLocalTransform(const tinygltf::Node& node)
{
    if (!node.matrix.empty()) {
        // GLTF 행렬은 열 우선(column-major), double 타입
        glm::mat4 m;
        for (int col = 0; col < 4; ++col)
            for (int row = 0; row < 4; ++row)
                m[col][row] = static_cast<float>(node.matrix[col * 4 + row]);
        return m;
    }

    glm::mat4 T(1.f), R(1.f), S(1.f);

    if (!node.translation.empty())
        T = glm::translate(glm::mat4(1.f),
                           glm::vec3(static_cast<float>(node.translation[0]),
                                     static_cast<float>(node.translation[1]),
                                     static_cast<float>(node.translation[2])));

    if (!node.rotation.empty()) {
        // GLTF 쿼터니언 순서: [x, y, z, w]
        glm::quat q(static_cast<float>(node.rotation[3]), // w
                    static_cast<float>(node.rotation[0]), // x
                    static_cast<float>(node.rotation[1]), // y
                    static_cast<float>(node.rotation[2])); // z
        R = glm::mat4_cast(q);
    }

    if (!node.scale.empty())
        S = glm::scale(glm::mat4(1.f),
                       glm::vec3(static_cast<float>(node.scale[0]),
                                 static_cast<float>(node.scale[1]),
                                 static_cast<float>(node.scale[2])));

    return T * R * S; // 오른쪽부터 적용: Scale -> Rotate -> Translate
}

// CachedPrim: 같은 GLTF 메시·프리미티브를 여러 노드가 참조할 때
// 지오메트리(버텍스/인덱스/LOD)와 재질 파생값을 한 번만 만들어 공유한다.
// -> 메모리 절약 + 같은 메시를 쓰는 노드들이 자동으로 인스턴싱 그룹이 된다.
struct CachedPrim {
    DrawObject proto;              // 인덱스 범위·재질 푸시 상수·LOD 범위 (model 은 노드별로 덮어씀)
    glm::vec3  bmin{}, bmax{};     // 로컬 공간 AABB
    int        groupId      = -1;  // 두 번째 사용부터 할당되는 인스턴싱 그룹 ID
    int        firstObjIndex = -1; // 첫 사용 DrawObject 의 인덱스 (그룹 소급 지정용)
};

// 캐시된 프리미티브로부터 노드 하나의 DrawObject 를 만들어 추가한다.
static void emitCachedObject(CachedPrim&              c,
                             const glm::mat4&         world,
                             int&                     nextGroupId,
                             std::vector<DrawObject>& objects)
{
    DrawObject obj = c.proto;
    obj.push.model = world;
    for (int i = 0; i < obj.numLods; ++i)
        obj.lods[i].model = world;
    obj.reverseFrontFace = (determinant3x3(world) < 0.f);
    computeBoundSphereLocal(obj, c.bmin, c.bmax);

    // 같은 프리미티브를 두 번째로 쓰는 순간 인스턴싱 그룹을 만든다.
    int objIndex = (int)objects.size();
    if (c.firstObjIndex < 0) {
        c.firstObjIndex = objIndex;
    } else {
        if (c.groupId < 0) {
            c.groupId = nextGroupId++;
            objects[c.firstObjIndex].instanceGroupId = c.groupId;
        }
        obj.instanceGroupId = c.groupId;
    }
    objects.push_back(obj);
}

// processPrimitive
// GLTF 메시의 프리미티브 하나를 글로벌 verts/inds 에 추가하고
// 해당 DrawObject 를 objects 에 추가한다. 이미 캐시에 있으면 지오메트리를 재사용한다.
// texIndexMap: GLTF image index -> 로컬 텍스처 배열 인덱스 (없으면 -1)
static void processPrimitive(const tinygltf::Model&              model,
                              const tinygltf::Primitive&           prim,
                              uint64_t                             primKey,
                              const glm::mat4&                     worldTransform,
                              const std::unordered_map<int, int>&  texIndexMap,
                              const std::unordered_map<int, int>&  matTexOverride,
                              std::unordered_map<uint64_t, CachedPrim>& primCache,
                              int&                                 nextGroupId,
                              std::vector<Vertex>&                 verts,
                              std::vector<uint32_t>&               inds,
                              std::vector<DrawObject>&             objects)
{
    // 삼각형 프리미티브만 처리 (라인, 포인트 등은 렌더러가 지원하지 않음)
    if (prim.mode != TINYGLTF_MODE_TRIANGLES) return;

    // 이미 만들어 둔 프리미티브면 지오메트리 생성을 건너뛴다.
    auto cacheIt = primCache.find(primKey);
    if (cacheIt != primCache.end()) {
        emitCachedObject(cacheIt->second, worldTransform, nextGroupId, objects);
        return;
    }

    auto posIt = prim.attributes.find("POSITION");
    if (posIt == prim.attributes.end()) return; // 위치 없으면 건너뜀

    // POSITION 버퍼
    const tinygltf::Accessor&   posAcc  = model.accessors[posIt->second];
    const tinygltf::BufferView& posView = model.bufferViews[posAcc.bufferView];
    const uint8_t* posPtr = model.buffers[posView.buffer].data.data()
                          + posView.byteOffset + posAcc.byteOffset;
    int posStride = posAcc.ByteStride(posView);

    // NORMAL 버퍼 (선택적)
    const uint8_t* nrmPtr    = nullptr;
    int            nrmStride = 12;
    auto nrmIt = prim.attributes.find("NORMAL");
    if (nrmIt != prim.attributes.end()) {
        const tinygltf::Accessor&   nrmAcc  = model.accessors[nrmIt->second];
        const tinygltf::BufferView& nrmView = model.bufferViews[nrmAcc.bufferView];
        nrmPtr    = model.buffers[nrmView.buffer].data.data()
                  + nrmView.byteOffset + nrmAcc.byteOffset;
        nrmStride = nrmAcc.ByteStride(nrmView);
    }

    // TEXCOORD_0 버퍼 (선택적)
    const uint8_t* uvPtr    = nullptr;
    int            uvStride = 8; // 기본값: float2 (8 bytes)
    auto uvIt = prim.attributes.find("TEXCOORD_0");
    if (uvIt != prim.attributes.end()) {
        const tinygltf::Accessor&   uvAcc  = model.accessors[uvIt->second];
        const tinygltf::BufferView& uvView = model.bufferViews[uvAcc.bufferView];
        uvPtr    = model.buffers[uvView.buffer].data.data()
                 + uvView.byteOffset + uvAcc.byteOffset;
        uvStride = uvAcc.ByteStride(uvView);
    }

    // COLOR_0 버퍼 (선택적, Blender 버텍스 컬러 / 컬러 어트리뷰트)
    const uint8_t* colPtr      = nullptr;
    int            colStride   = 0;
    int            colCompType = 0;
    auto colIt = prim.attributes.find("COLOR_0");
    if (colIt != prim.attributes.end()) {
        const tinygltf::Accessor&   colAcc  = model.accessors[colIt->second];
        const tinygltf::BufferView& colView = model.bufferViews[colAcc.bufferView];
        colPtr      = model.buffers[colView.buffer].data.data()
                    + colView.byteOffset + colAcc.byteOffset;
        colStride   = colAcc.ByteStride(colView);
        colCompType = colAcc.componentType;
    }

    // 재질 베이스 컬러 + 텍스처 인덱스 + PBR 속성
    glm::vec3 matColor(0.8f);
    int       texIdx      = -1;
    int       mrTexIdx    = -1; // metallicRoughness 텍스처 (G=러프니스, B=메탈릭)
    int       emisTexIdx  = -1; // emissive 텍스처
    float     matAlpha    = 1.f; // alphaMode=BLEND 인 재질의 투명도
    bool      twoSided    = false;
    float     matShininess = 32.f;
    float     matSpecStr   = 0.3f;
    glm::vec4 matEmissive(0.f); // rgb=발광 색상(텍스처 있으면 곱셈 팩터), a=발광 강도

    if (prim.material >= 0) {
        const auto& material = model.materials[prim.material];
        const auto& pbr = material.pbrMetallicRoughness;
        twoSided = material.doubleSided;

        // 베이스 컬러 팩터 (텍스처 없을 때 색상으로 사용)
        if (pbr.baseColorFactor.size() >= 3)
            matColor = glm::vec3(static_cast<float>(pbr.baseColorFactor[0]),
                                 static_cast<float>(pbr.baseColorFactor[1]),
                                 static_cast<float>(pbr.baseColorFactor[2]));

        // Blender 알파 블렌드 재질: baseColorFactor 의 알파를 투명도로 사용
        if (material.alphaMode == "BLEND" && pbr.baseColorFactor.size() >= 4)
            matAlpha = static_cast<float>(pbr.baseColorFactor[3]);

        // 베이스 컬러 텍스처
        int baseColorTexIdx = pbr.baseColorTexture.index;
        if (baseColorTexIdx >= 0 && baseColorTexIdx < (int)model.textures.size()) {
            int imageIdx = model.textures[baseColorTexIdx].source;
            auto it = texIndexMap.find(imageIdx);
            if (it != texIndexMap.end())
                texIdx = it->second;
        }

        // metallicRoughness 텍스처 (Blender 러프니스/메탈릭 맵)
        int mrGltfIdx = pbr.metallicRoughnessTexture.index;
        if (mrGltfIdx >= 0 && mrGltfIdx < (int)model.textures.size()) {
            int imageIdx = model.textures[mrGltfIdx].source;
            auto it = texIndexMap.find(imageIdx);
            if (it != texIndexMap.end())
                mrTexIdx = it->second;
        }

        // emissive 텍스처 (Blender 발광 맵)
        if (material.emissiveTexture.index >= 0 &&
            material.emissiveTexture.index < (int)model.textures.size()) {
            int imageIdx = model.textures[material.emissiveTexture.index].source;
            auto it = texIndexMap.find(imageIdx);
            if (it != texIndexMap.end())
                emisTexIdx = it->second;
        }

        // baseColor 텍스처 없으면 emissiveTexture 를 albedo 로도 사용 (용암 등 발광 재질)
        if (texIdx < 0 && emisTexIdx >= 0)
            texIdx = emisTexIdx;

        // 텍스처 없는 재질 -> 외부 텍스처 자동 매칭 (재질 이름 기반)
        if (texIdx < 0 && prim.material >= 0) {
            auto ovr = matTexOverride.find(prim.material);
            if (ovr != matTexOverride.end())
                texIdx = ovr->second;
        }

        // 메탈릭 / 러프니스 -> specularStrength / shininess
        float metallic  = static_cast<float>(pbr.metallicFactor); // 0..1
        float roughness = static_cast<float>(pbr.roughnessFactor); // 0..1
        matSpecStr   = glm::mix(0.05f, 0.95f, metallic);
        matShininess = glm::mix(256.f, 4.f, roughness * roughness);

        // 발광 팩터 (KHR_materials_emissive_strength 확장 포함)
        // emissiveTexture 가 있으면 팩터는 텍스처와 곱해지는 계수로 쓰인다(셰이더에서 처리).
        glm::vec3 emissiveRGB(0.f);
        if (material.emissiveFactor.size() >= 3)
            emissiveRGB = glm::vec3(static_cast<float>(material.emissiveFactor[0]),
                                    static_cast<float>(material.emissiveFactor[1]),
                                    static_cast<float>(material.emissiveFactor[2]));
        float emissiveStrength = 1.f;
        auto esIt = material.extensions.find("KHR_materials_emissive_strength");
        if (esIt != material.extensions.end() && esIt->second.Has("emissiveStrength"))
            emissiveStrength = static_cast<float>(
                esIt->second.Get("emissiveStrength").GetNumberAsDouble());
        emissiveRGB *= emissiveStrength;
        // a(강도)는 텍스처가 없을 때만 상수 luminance 로 쓰인다 (디퍼드 G-Buffer 패킹용).
        float emissiveLum = (emisTexIdx >= 0)
            ? 0.f
            : std::max({emissiveRGB.x, emissiveRGB.y, emissiveRGB.z});
        matEmissive = glm::vec4(emissiveRGB, emissiveLum);
    }

    // 버텍스 생성
    uint32_t  vertBase = static_cast<uint32_t>(verts.size());
    glm::vec3 bboxMin( std::numeric_limits<float>::max());
    glm::vec3 bboxMax(-std::numeric_limits<float>::max());

    for (int vi = 0; vi < static_cast<int>(posAcc.count); ++vi) {
        const float* pf = reinterpret_cast<const float*>(posPtr + vi * posStride);
        Vertex vtx{};
        vtx.pos   = glm::vec3(pf[0], pf[1], pf[2]);
        vtx.color = matColor;

        // Blender 버텍스 컬러(COLOR_0)를 베이스 컬러에 곱한다.
        if (colPtr) {
            const uint8_t* cp = colPtr + vi * colStride;
            glm::vec3 c(1.f);
            switch (colCompType) {
            case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                const float* cf = reinterpret_cast<const float*>(cp);
                c = glm::vec3(cf[0], cf[1], cf[2]);
                break;
            }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                const uint16_t* cu = reinterpret_cast<const uint16_t*>(cp);
                c = glm::vec3(cu[0], cu[1], cu[2]) / 65535.f;
                break;
            }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                c = glm::vec3(cp[0], cp[1], cp[2]) / 255.f;
                break;
            }
            vtx.color = matColor * c;
        }

        if (nrmPtr) {
            const float* nf = reinterpret_cast<const float*>(nrmPtr + vi * nrmStride);
            vtx.normal = glm::vec3(nf[0], nf[1], nf[2]);
        } else {
            vtx.normal = glm::vec3(0.f, 1.f, 0.f);
        }

        if (uvPtr) {
            const float* uf = reinterpret_cast<const float*>(uvPtr + vi * uvStride);
            vtx.uv = glm::vec2(uf[0], uf[1]);
        } else {
            vtx.uv = glm::vec2(0.f, 0.f);
        }

        bboxMin = glm::min(bboxMin, vtx.pos);
        bboxMax = glm::max(bboxMax, vtx.pos);
        verts.push_back(vtx);
    }

    // 인덱스 생성
    uint32_t idxStart = static_cast<uint32_t>(inds.size());

    if (prim.indices >= 0) {
        const tinygltf::Accessor&   idxAcc  = model.accessors[prim.indices];
        const tinygltf::BufferView& idxView = model.bufferViews[idxAcc.bufferView];
        const uint8_t* idxPtr = model.buffers[idxView.buffer].data.data()
                              + idxView.byteOffset + idxAcc.byteOffset;
        int idxStride = idxAcc.ByteStride(idxView);

        for (int ii = 0; ii < static_cast<int>(idxAcc.count); ++ii) {
            const uint8_t* p = idxPtr + ii * idxStride;
            uint32_t idx = 0;
            switch (idxAcc.componentType) {
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                idx = *reinterpret_cast<const uint32_t*>(p); break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                idx = *reinterpret_cast<const uint16_t*>(p); break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                idx = *p; break;
            }
            inds.push_back(vertBase + idx);
        }
    } else {
        for (uint32_t i = 0; i < static_cast<uint32_t>(posAcc.count); ++i)
            inds.push_back(vertBase + i);
    }

    uint32_t idxCount = static_cast<uint32_t>(inds.size()) - idxStart;
    if (idxCount == 0) return;

    if (nrmPtr)
        alignTriangleWindingToNormals(verts, inds, idxStart, idxCount);

    // DrawObject 생성
    DrawObject obj{};
    obj.indexStart            = idxStart;
    obj.indexCount            = idxCount;
    obj.push.model            = worldTransform;
    obj.push.baseColor        = glm::vec4(1.f, 1.f, 1.f, matAlpha);
    obj.push.shininess        = matShininess;
    obj.push.specularStrength = matSpecStr;
    obj.push.reflectStrength  = 0.f;
    obj.push.textureIndex     = static_cast<float>(texIdx);
    obj.push.emissive         = matEmissive;
    obj.push.texIndices2      = glm::vec4(static_cast<float>(mrTexIdx),
                                          static_cast<float>(emisTexIdx), 0.f, 0.f);
    obj.textureIndex          = texIdx;
    obj.instanceGroupId       = -1;
    obj.twoSided              = twoSided;
    obj.reverseFrontFace      = (determinant3x3(worldTransform) < 0.f);

    computeBoundSphereLocal(obj, bboxMin, bboxMax);

    // LOD 자동 생성.
    // 삼각형을 삭제하지 않고 정점 클러스터링으로 새 LOD 메시를 만든다.
    // LOD1/LOD2는 격자 밀도를 다르게 해서 원본 표면의 연결성을 유지한다.
    const uint32_t triCount = idxCount / 3;
    if (triCount >= 48) {
        uint32_t lod1Start = 0, lod1Count = 0;
        uint32_t lod2Start = 0, lod2Count = 0;

        if (buildClusteredLod(verts, inds, vertBase, static_cast<uint32_t>(posAcc.count),
                              idxStart, idxCount, bboxMin, bboxMax,
                              16, lod1Start, lod1Count)) {
            obj.lods[0] = {lod1Start, lod1Count, worldTransform};
            obj.numLods = 1;
        }

        if (obj.numLods == 1 &&
            buildClusteredLod(verts, inds, vertBase, static_cast<uint32_t>(posAcc.count),
                              idxStart, idxCount, bboxMin, bboxMax,
                              8, lod2Start, lod2Count)) {
            obj.lods[1] = {lod2Start, lod2Count, worldTransform};
            obj.numLods = 2;
        }
    }

    // 캐시에 등록한 뒤 이 노드의 DrawObject 를 추가한다.
    CachedPrim c{};
    c.proto = obj;
    c.bmin  = bboxMin;
    c.bmax  = bboxMax;
    auto& stored = primCache.emplace(primKey, std::move(c)).first->second;
    emitCachedObject(stored, worldTransform, nextGroupId, objects);
}

// traverseNode
// GLTF 씬 그래프를 재귀 순회한다.
static void traverseNode(const tinygltf::Model&              model,
                          int                                  nodeIdx,
                          const glm::mat4&                     parentTransform,
                          const std::unordered_map<int, int>&  texIndexMap,
                          const std::unordered_map<int, int>&  matTexOverride,
                          std::unordered_map<uint64_t, CachedPrim>& primCache,
                          int&                                 nextGroupId,
                          std::vector<Vertex>&                 verts,
                          std::vector<uint32_t>&               inds,
                          std::vector<DrawObject>&             objects)
{
    const tinygltf::Node& node = model.nodes[nodeIdx];
    glm::mat4 worldTransform = parentTransform * nodeLocalTransform(node);

    if (node.mesh >= 0) {
        const tinygltf::Mesh& mesh = model.meshes[node.mesh];
        for (int pi = 0; pi < (int)mesh.primitives.size(); ++pi) {
            // 캐시 키 = (메시 인덱스, 프리미티브 인덱스)
            uint64_t key = (static_cast<uint64_t>(node.mesh) << 20) | static_cast<uint32_t>(pi);
            processPrimitive(model, mesh.primitives[pi], key, worldTransform,
                             texIndexMap, matTexOverride, primCache, nextGroupId,
                             verts, inds, objects);
        }
    }

    for (int child : node.children)
        traverseNode(model, child, worldTransform, texIndexMap, matTexOverride,
                     primCache, nextGroupId, verts, inds, objects);
}

// loadGLTFScene  – 공개 진입점
bool loadGLTFScene(const std::string&           path,
                   std::vector<Vertex>&          verts,
                   std::vector<uint32_t>&        inds,
                   std::vector<DrawObject>&      objects,
                   GLTFSceneDesc&                sceneDesc,
                   std::vector<GLTFTextureData>& textures)
{
    tinygltf::Model    model;
    tinygltf::TinyGLTF loader;
    std::string        err, warn;

    // 확장자로 바이너리(.glb) 와 텍스트(.gltf) 구분
    auto dotPos = path.find_last_of('.');
    std::string ext = (dotPos != std::string::npos) ? path.substr(dotPos + 1) : "";
    for (auto& c : ext) c = static_cast<char>(tolower(c));

    bool ok = (ext == "glb")
            ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
            : loader.LoadASCIIFromFile (&model, &err, &warn, path);

    if (!warn.empty()) std::cerr << "[GLTFLoader] Warning: " << warn << "\n";
    if (!ok) {
        std::cerr << "[GLTFLoader] Failed to load: " << path << "\n";
        if (!err.empty()) std::cerr << "  " << err << "\n";
        return false;
    }

    // 씬 이름
    if (!model.scenes.empty() && !model.scenes[0].name.empty())
        sceneDesc.name = model.scenes[0].name;
    else {
        auto slash = path.find_last_of("/\\");
        sceneDesc.name = (slash != std::string::npos) ? path.substr(slash + 1) : path;
    }

    // GLTF 카메라 노드에서 초기 위치 추출
    for (const auto& node : model.nodes) {
        if (node.camera >= 0 && !node.translation.empty()) {
            sceneDesc.cameraPos = glm::vec3(
                static_cast<float>(node.translation[0]),
                static_cast<float>(node.translation[1]),
                static_cast<float>(node.translation[2]));
            break;
        }
    }

    // 텍스처 디코딩
    // tinygltf + stb_image 가 GLB 내장 이미지를 RGBA8 로 자동 디코딩한다.
    // model.images[i].image 에 RGBA8 픽셀 데이터가 들어있다.
    std::unordered_map<int, int> texIndexMap; // GLTF image index -> 로컬 인덱스
    for (int i = 0; i < static_cast<int>(model.images.size()); ++i) {
        const auto& img = model.images[i];
        if (!img.image.empty() && img.width > 0 && img.height > 0) {
            texIndexMap[i] = static_cast<int>(textures.size());
            GLTFTextureData td;
            td.width  = img.width;
            td.height = img.height;
            td.pixels.assign(img.image.begin(), img.image.end());
            textures.push_back(std::move(td));
        }
    }

    if (!textures.empty())
        std::cout << "[GLTFLoader] Decoded " << textures.size() << " embedded texture(s)\n";

    // 외부 텍스처 자동 매칭
    // 1단계: 같은 폴더에 .mtl 파일이 있으면 파싱하여 재질별 map_Kd 파일명 수집
    // 2단계: 재질 이름 또는 MTL 매핑으로 텍스처 파일 탐색
    std::unordered_map<int, int> matTexOverride; // material index -> textures 배열 인덱스

    {
        namespace fs = std::filesystem;
        fs::path glbPath(path);
        fs::path mapDir = glbPath.parent_path();
        fs::path assetRoot = mapDir;
        std::string mapDirName = mapDir.filename().string();
        std::transform(mapDirName.begin(), mapDirName.end(), mapDirName.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (mapDirName == "map")
            assetRoot = mapDir.parent_path();

        std::vector<fs::path> imageSearchDirs = {
            mapDir,
            mapDir / "Textures",
            mapDir / "textures",
            mapDir / "texture",
            assetRoot / "texture",
            assetRoot / "textures",
            assetRoot / "Textures",
        };

        // GLB에 임베드되지 않은 이미지(절대 경로 URI 등) 복구:
        // Blender에서 이미지 모드를 'Keep original'로 내보내면 C:/xxx.png 같은
        // 원본 경로가 남는다. 파일명만 추출해 texture 폴더들에서 찾아 로드한다.
        for (int i = 0; i < static_cast<int>(model.images.size()); ++i) {
            if (texIndexMap.count(i)) continue; // 이미 디코딩된 이미지
            const auto& img = model.images[i];
            if (img.uri.empty()) continue;
            auto slash = img.uri.find_last_of("/\\");
            std::string fname = (slash != std::string::npos)
                ? img.uri.substr(slash + 1) : img.uri;
            for (const fs::path& dir : imageSearchDirs) {
                std::string tryPath = (dir / fname).string();
                int w = 0, h = 0, ch = 0;
                unsigned char* data = stbi_load(tryPath.c_str(), &w, &h, &ch, 4);
                if (!data) continue;
                texIndexMap[i] = static_cast<int>(textures.size());
                GLTFTextureData td;
                td.width  = w;
                td.height = h;
                td.pixels.assign(data, data + (size_t)w * h * 4);
                stbi_image_free(data);
                textures.push_back(std::move(td));
                std::cout << "[GLTFLoader] Recovered external image: "
                          << img.uri << " -> " << tryPath << "\n";
                break;
            }
        }

        // MTL 파일 파싱: 재질이름 -> map_Kd 파일명
        // Blender 내보내기 MTL의 절대 경로(C:/Lava.png)에서 파일명만 추출
        std::unordered_map<std::string, std::string> mtlMapKd; // matName -> 텍스처 파일명
        {
            std::vector<fs::path> mtlCandidates = {
                mapDir / (glbPath.stem().string() + ".mtl"),
                assetRoot / "mtl" / (glbPath.stem().string() + ".mtl"),
                assetRoot / "MTL" / (glbPath.stem().string() + ".mtl"),
            };
            fs::path mtlPath;
            std::ifstream mtlFile;
            for (const fs::path& candidate : mtlCandidates) {
                mtlFile.open(candidate);
                if (mtlFile.is_open()) {
                    mtlPath = candidate;
                    break;
                }
                mtlFile.clear();
            }
            if (mtlFile.is_open()) {
                std::string curMat;
                std::string line;
                while (std::getline(mtlFile, line)) {
                    if (line.rfind("newmtl ", 0) == 0)
                        curMat = line.substr(7);
                    else if (line.rfind("map_Kd ", 0) == 0 && !curMat.empty()) {
                        // 절대/상대 경로에서 파일명만 추출
                        std::string texPath = line.substr(7);
                        auto slash = texPath.find_last_of("/\\");
                        std::string fname = (slash != std::string::npos)
                            ? texPath.substr(slash + 1) : texPath;
                        mtlMapKd[curMat] = fname;
                    }
                }
                if (!mtlMapKd.empty())
                    std::cout << "[GLTFLoader] Parsed companion MTL: "
                              << mtlPath.string() << " (" << mtlMapKd.size() << " textures)\n";
            }
        }

        // GLTF 텍스처 참조가 실제 로드된 이미지로 해석되는지 확인한다.
        // (외부 경로 이미지가 없어서 디코딩 실패한 참조는 '없음'으로 취급)
        auto texResolves = [&](int gltfTexIdx) -> bool {
            if (gltfTexIdx < 0 || gltfTexIdx >= static_cast<int>(model.textures.size()))
                return false;
            return texIndexMap.count(model.textures[gltfTexIdx].source) > 0;
        };

        // 재질별 텍스처 매칭
        for (int mi = 0; mi < static_cast<int>(model.materials.size()); ++mi) {
            const auto& mat = model.materials[mi];
            // 이미 baseColor 또는 emissive 텍스처가 정상 로드된 재질은 건너뜀
            if (texResolves(mat.pbrMetallicRoughness.baseColorTexture.index)) continue;
            if (texResolves(mat.emissiveTexture.index)) continue;
            if (mat.name.empty()) continue;

            // Blender GLB 재질 이름은 공백, MTL 은 언더스코어를 쓰는 경우가 많다.
            std::string nameUnderscore = mat.name;
            std::replace(nameUnderscore.begin(), nameUnderscore.end(), ' ', '_');

            // 시도할 파일명 후보: MTL map_Kd 파일명 + 재질 이름
            std::vector<std::string> candidates;
            for (const std::string& key : {mat.name, nameUnderscore}) {
                auto mtlIt = mtlMapKd.find(key);
                if (mtlIt != mtlMapKd.end()) {
                    candidates.push_back(mtlIt->second); // MTL에서 가져온 파일명 (예: Lava.png)
                    break;
                }
            }
            // 재질 이름 + 확장자 (기존 방식)
            for (const char* ext : {".png", ".jpg", ".jpeg", ".bmp", ".tga"}) {
                candidates.push_back(mat.name + ext);
                if (nameUnderscore != mat.name)
                    candidates.push_back(nameUnderscore + ext);
            }

            bool found = false;
            std::vector<fs::path> textureDirs = {
                mapDir,
                mapDir / "Textures",
                mapDir / "textures",
                mapDir / "texture",
                assetRoot / "texture",
                assetRoot / "textures",
                assetRoot / "Textures",
            };
            for (const fs::path& texDir : textureDirs) {
                if (found) break;
                for (const auto& fname : candidates) {
                    fs::path tryFsPath(fname);
                    if (!tryFsPath.is_absolute())
                        tryFsPath = texDir / fname;
                    std::string tryPath = tryFsPath.string();
                    int w = 0, h = 0, ch = 0;
                    unsigned char* data = stbi_load(tryPath.c_str(), &w, &h, &ch, 4);
                    if (!data) continue;

                    int localIdx = static_cast<int>(textures.size());
                    GLTFTextureData td;
                    td.width  = w;
                    td.height = h;
                    td.pixels.assign(data, data + (size_t)w * h * 4);
                    stbi_image_free(data);
                    textures.push_back(std::move(td));
                    matTexOverride[mi] = localIdx;
                    std::cout << "[GLTFLoader] Auto-matched texture: "
                              << mat.name << " -> " << tryPath << "\n";
                    found = true;
                    break;
                }
            }
        }
    }

    if (!matTexOverride.empty())
        std::cout << "[GLTFLoader] Auto-matched " << matTexOverride.size()
                  << " external texture(s)\n";

    // 씬 그래프 순회
    int sceneIdx = (model.defaultScene >= 0) ? model.defaultScene : 0;
    if (sceneIdx >= static_cast<int>(model.scenes.size())) {
        std::cerr << "[GLTFLoader] No scenes found in: " << path << "\n";
        return false;
    }

    // 재질 요약 로그 (Blender 재질이 제대로 읽혔는지 확인용)
    for (int mi = 0; mi < static_cast<int>(model.materials.size()) && mi < 6; ++mi) {
        const auto& mat = model.materials[mi];
        const auto& bcf = mat.pbrMetallicRoughness.baseColorFactor;
        std::cout << "[GLTFLoader] Material " << mi << " '" << mat.name << "' baseColor=("
                  << (bcf.size() > 0 ? bcf[0] : -1) << ", "
                  << (bcf.size() > 1 ? bcf[1] : -1) << ", "
                  << (bcf.size() > 2 ? bcf[2] : -1) << ")"
                  << " rough=" << mat.pbrMetallicRoughness.roughnessFactor
                  << " metal=" << mat.pbrMetallicRoughness.metallicFactor << "\n";
    }

    size_t objsBefore = objects.size();
    std::unordered_map<uint64_t, CachedPrim> primCache; // 메시 공유·자동 인스턴싱용
    int nextGroupId = 0;
    for (int rootNode : model.scenes[sceneIdx].nodes)
        traverseNode(model, rootNode, glm::mat4(1.f), texIndexMap, matTexOverride,
                     primCache, nextGroupId, verts, inds, objects);

    std::cout << "[GLTFLoader] Loaded '" << sceneDesc.name << "' from " << path
              << "  (" << (objects.size() - objsBefore) << " objects, "
              << verts.size() << " vertices, "
              << nextGroupId << " instancing groups)\n";

    // KHR_lights_punctual 조명 추출
    // 1단계: 전역 조명 정의 목록 구성
    std::vector<SceneLight> lightDefs;
    auto glbLightIt = model.extensions.find("KHR_lights_punctual");
    if (glbLightIt != model.extensions.end() && glbLightIt->second.Has("lights")) {
        const auto& lightsArr = glbLightIt->second.Get("lights");
        int numDef = static_cast<int>(lightsArr.ArrayLen());
        lightDefs.resize(numDef);
        for (int li = 0; li < numDef; ++li) {
            const auto& l = lightsArr.Get(li);
            SceneLight& sl = lightDefs[li];
            if (l.Has("name") && l.Get("name").IsString())
                sl.name = l.Get("name").Get<std::string>();
            if (l.Has("type") && l.Get("type").IsString()) {
                std::string t = l.Get("type").Get<std::string>();
                if      (t == "directional") sl.type = SceneLight::Directional;
                else if (t == "spot")        sl.type = SceneLight::Spot;
                else                         sl.type = SceneLight::Point;
            }
            if (l.Has("color") && l.Get("color").IsArray()) {
                const auto& c = l.Get("color");
                if (static_cast<int>(c.ArrayLen()) >= 3)
                    sl.color = glm::vec3(static_cast<float>(c.Get(0).GetNumberAsDouble()),
                                         static_cast<float>(c.Get(1).GetNumberAsDouble()),
                                         static_cast<float>(c.Get(2).GetNumberAsDouble()));
            }
            if (l.Has("intensity"))
                sl.intensity = static_cast<float>(l.Get("intensity").GetNumberAsDouble());
            if (l.Has("range"))
                sl.range = static_cast<float>(l.Get("range").GetNumberAsDouble());
            if (l.Has("spot")) {
                const auto& spot = l.Get("spot");
                if (spot.Has("innerConeAngle"))
                    sl.innerConeAngle = static_cast<float>(
                        spot.Get("innerConeAngle").GetNumberAsDouble());
                if (spot.Has("outerConeAngle"))
                    sl.outerConeAngle = static_cast<float>(
                        spot.Get("outerConeAngle").GetNumberAsDouble());
            }
        }
    }

    // 2단계: 노드 순회 -> 조명 인스턴스 위치/방향 추출
    if (!lightDefs.empty()) {
        std::function<void(int, const glm::mat4&)> traverseLights;
        traverseLights = [&](int nodeIdx, const glm::mat4& parentTransform) {
            const tinygltf::Node& node = model.nodes[nodeIdx];
            glm::mat4 world = parentTransform * nodeLocalTransform(node);

            auto extIt = node.extensions.find("KHR_lights_punctual");
            if (extIt != node.extensions.end() && extIt->second.Has("light")) {
                int li = extIt->second.Get("light").GetNumberAsInt();
                if (li >= 0 && li < static_cast<int>(lightDefs.size())) {
                    SceneLight sl = lightDefs[li];
                    sl.position  = glm::vec3(world[3]);
                    // GLTF 조명 방향: 노드 로컬 -Z 축
                    sl.direction = glm::normalize(
                        glm::vec3(world * glm::vec4(0.f, 0.f, -1.f, 0.f)));
                    sceneDesc.lights.push_back(sl);
                }
            }
            for (int child : node.children)
                traverseLights(child, world);
        };
        for (int rootNode : model.scenes[sceneIdx].nodes)
            traverseLights(rootNode, glm::mat4(1.f));

        if (!sceneDesc.lights.empty())
            std::cout << "[GLTFLoader] Extracted " << sceneDesc.lights.size()
                      << " light(s) from KHR_lights_punctual\n";
    }

    // ------------------------------------------------------------------
    // 발광(emissive) 재질 -> 자동 포인트 라이트 생성
    // ------------------------------------------------------------------
    // emissive 는 표면만 밝게 할 뿐 주변을 비추지 못한다. 그래서 발광이 강한
    // 오브젝트를 골라 그 자리에 포인트 라이트를 자동으로 만들어 준다
    // (용암이 동굴을 붉게 비추고, 횃불 불꽃이 벽을 비추는 효과).
    // 남은 조명 슬롯(총 16 - 기존 GLTF 조명 수)만큼, 발광량×크기 점수가 큰
    // 순서대로 채운다. 이 조명들은 V 키(emissive 토글)에 연동된다.
    {
        const int MAX_TOTAL_LIGHTS = 16; // VulkanApp::MAX_SCENE_LIGHTS 와 동일
        int freeSlots = MAX_TOTAL_LIGHTS - (int)sceneDesc.lights.size();
        if (freeSlots > 0) {
            struct Cand { float score; size_t objIdx; };
            std::vector<Cand> cands;
            for (size_t i = objsBefore; i < objects.size(); ++i) {
                const DrawObject& o = objects[i];
                glm::vec3 e = glm::vec3(o.push.emissive);
                bool hasEmisTex = o.push.texIndices2.y >= 0.f;
                // 상수 발광은 a(luminance), 텍스처 발광은 팩터의 최대 성분으로 평가
                float lum = std::max(o.push.emissive.a,
                                     hasEmisTex ? std::max({e.x, e.y, e.z}) : 0.f);
                if (lum < 0.3f) continue; // 미미한 발광은 제외
                cands.push_back({lum * std::max(o.boundRadius, 0.1f), i});
            }
            std::sort(cands.begin(), cands.end(),
                      [](const Cand& a, const Cand& b) { return a.score > b.score; });

            int made = 0;
            for (const Cand& c : cands) {
                if (made >= freeSlots) break;
                const DrawObject& o = objects[c.objIdx];
                glm::vec3 e = glm::vec3(o.push.emissive);
                float maxc  = std::max({e.x, e.y, e.z, 0.05f});

                SceneLight sl;
                sl.type         = SceneLight::Point;
                sl.position     = o.boundCenter;
                sl.color        = (maxc > 0.f) ? e / maxc : glm::vec3(1.f); // 색조 유지
                // 강도: 발광량 × 오브젝트 크기에 비례, 과도한 값은 클램프
                float lum       = std::max(o.push.emissive.a, maxc);
                sl.intensity    = glm::clamp(lum * o.boundRadius * 6.f, 4.f, 60.f);
                sl.range        = glm::clamp(o.boundRadius * 10.f + 8.f, 10.f, 80.f);
                sl.name         = "Glow " + std::to_string(made + 1);
                sl.fromEmissive = true;
                sceneDesc.lights.push_back(sl);
                ++made;
            }
            if (made > 0)
                std::cout << "[GLTFLoader] Auto-generated " << made
                          << " glow light(s) from emissive materials\n";
        }
    }

    return true;
}
