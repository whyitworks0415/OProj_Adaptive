import sys

new_code = r"""void VulkanApp::buildScene() {
    SceneDesc desc;
    if (!loadScene(currentMapFile, desc)) {
        throw std::runtime_error("Failed to load scene: " + currentMapFile);
    }

    // 메시 로드
    std::vector<std::pair<std::string, MeshRange>> meshRanges;
    auto findMesh = [&](const std::string& id) -> const MeshRange* {
        for (auto& p : meshRanges) if (p.first == id) return &p.second;
        return nullptr;
    };
    auto findMeshDesc = [&](const std::string& id) -> const MeshDesc* {
        for (auto& m : desc.meshes) if (m.id == id) return &m;
        return nullptr;
    };

    for (auto& md : desc.meshes) {
        std::cout << "  Loading " << md.stlPath << " ...\n";
        MeshRange mr = loadSTL(md.stlPath, vertices, indices, md.color);
        meshRanges.push_back({md.id, mr});
    }

    // 바닥
    if (desc.hasFloor) {
        auto obj = addFloor(vertices, indices, desc.floor.size, desc.floor.divs);
        obj.push.model            = glm::mat4(1.f);
        obj.push.baseColor        = glm::vec4(1.f);
        obj.push.shininess        = 32.f;
        obj.push.specularStrength = 0.25f;
        obj.push.reflectStrength  = 0.f;
        obj.boundCenter           = {0.f, 0.f, 0.f};
        obj.boundRadius           = desc.floor.size * 1.2f;
        obj.skipOcclusion         = true;
        drawObjects.push_back(obj);
    }

    // 그리드 배치
    for (auto& gd : desc.grids) {
        const MeshRange* mr = findMesh(gd.meshId);
        const MeshDesc*  md = findMeshDesc(gd.meshId);
        if (!mr) {
            std::cerr << "  [buildScene] Unknown mesh '" << gd.meshId << "'\n";
            continue;
        }
        float shin = md ? md->shininess : 32.f;
        float spec = md ? md->specular  : 0.3f;
        float refl = md ? md->reflect   : 0.f;

        int half = gd.cells / 2;
        int C0 = 0, C1 = gd.cells - 1, R0 = 0, R1 = gd.cells - 1;
        if (gd.hasRange) { C0 = gd.ranC0; C1 = gd.ranC1; R0 = gd.ranR0; R1 = gd.ranR1; }

        int count = 0;
        for (int row = R0; row <= R1; ++row) {
            for (int col = C0; col <= C1; ++col) {
                if (gd.evenOnly   && (col % 2 != 0 || row % 2 != 0)) continue;
                if (gd.hasExclude && col >= gd.exC0 && col <= gd.exC1
                                  && row >= gd.exR0 && row <= gd.exR1) continue;
                if (gd.hasHollow  && col >= gd.holC0 && col <= gd.holC1
                                  && row >= gd.holR0 && row <= gd.holR1) continue;

                float x = (col - half) * gd.cellSize;
                float z = (row - half) * gd.cellSize;

                glm::mat4 model = makeSTLModel(*mr, gd.cellSize, {x, 0.f, z});
                auto obj = makeDrawObject(*mr, model, shin, spec, refl);
                obj.instanceGroupId = gd.instanceGroup;
                computeBoundSphere(obj, mr->bboxMin, mr->bboxMax);
                drawObjects.push_back(obj);
                ++count;
            }
        }
        std::cout << "  Grid '" << gd.meshId << "': " << count << " objects\n";
    }

    // 개별 오브젝트
    for (auto& od : desc.objects) {
        const MeshRange* mr = findMesh(od.meshId);
        const MeshDesc*  md = findMeshDesc(od.meshId);
        if (!mr) {
            std::cerr << "  [buildScene] Unknown mesh '" << od.meshId << "'\n";
            continue;
        }
        float shin = md ? md->shininess : 32.f;
        float spec = md ? md->specular  : 0.3f;
        float refl = md ? md->reflect   : 0.f;

        glm::mat4 model = makeSTLModel(*mr, od.scale, od.pos);
        auto obj = makeDrawObject(*mr, model, shin, spec, refl);
        obj.instanceGroupId = od.instanceGroup;
        computeBoundSphere(obj, mr->bboxMin, mr->bboxMax);
        drawObjects.push_back(obj);
    }

    // 오클루전 쿼리용 단위 박스 프록시
    occBBoxMesh = addBox(vertices, indices, 1.f, 1.f, 1.f, {0.f, 0.f, 0.f});

    // 오클루전 상태 초기화
    occResults.assign(drawObjects.size(), 1u);
    occQueryBuf.resize(drawObjects.size() * 2, 0);

    // 씬 파일의 카메라 설정 적용
    camera.position = desc.cameraPos;
    camera.yaw      = desc.cameraYaw;
    camera.pitch    = desc.cameraPitch;

    std::cout << "Scene '" << desc.name << "' built: "
              << drawObjects.size() << " draw objects, "
              << vertices.size()    << " vertices, "
              << indices.size() / 3 << " triangles\n";
}

void VulkanApp::reloadScene() {
    vkDeviceWaitIdle(device);

    // GPU 지오메트리 버퍼 해제
    vkDestroyBuffer(device, indexBuffer,  nullptr);
    vkFreeMemory(device, indexBufferMemory,  nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);
    indexBuffer        = VK_NULL_HANDLE;
    indexBufferMemory  = VK_NULL_HANDLE;
    vertexBuffer       = VK_NULL_HANDLE;
    vertexBufferMemory = VK_NULL_HANDLE;

    // 쿼리 풀 해제
    if (occlusionQueryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device, occlusionQueryPool, nullptr);
        occlusionQueryPool = VK_NULL_HANDLE;
        occQueryCount      = 0;
    }
    occWarmupFrames = 0;

    // CPU 씬 데이터 비우기
    vertices.clear();
    indices.clear();
    drawObjects.clear();
    occBBoxMesh = {};

    // 씬 재구성
    buildScene();
    createVertexBuffer();
    createIndexBuffer();
    createOcclusionQueryPool();

    std::cout << "Scene reloaded: " << currentMapFile << "\n";
}

"""

with open(sys.argv[1], 'rb') as f:
    raw = f.read()

start = raw.find(b'void VulkanApp::buildScene()')
end   = raw.find(b'\nvoid VulkanApp::run()', start)

raw = raw[:start] + new_code.encode('utf-8') + b'\n' + raw[end:]

with open(sys.argv[1], 'wb') as f:
    f.write(raw)

print('OK')
