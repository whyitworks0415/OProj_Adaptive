import sys

with open('C:/MyFolder/High3/OProj/src/VulkanApp.cpp', 'r', encoding='utf-8') as f:
    lines = f.readlines()

insert_idx = 1194  # copyBuffer 앞, 1194번째 줄 뒤에 삽입

impl_lines = [
    "// MeshCache::getOrLoad: 메시 파일을 캐시에서 찾거나 새로 파싱한다.\n",
    "MeshRange MeshCache::getOrLoad(const std::string& path, glm::vec3 color,\n",
    "                                std::vector<Vertex>& verts,\n",
    "                                std::vector<uint32_t>& inds)\n",
    "{\n",
    "    // 경로와 소수점 셋째 자리까지 반올림한 색상으로 캐시 키를 만든다.\n",
    "    char keyBuf[576];\n",
    "    snprintf(keyBuf, sizeof(keyBuf), \"%s|%.3f,%.3f,%.3f\",\n",
    "             path.c_str(), color.r, color.g, color.b);\n",
    "    std::string key(keyBuf);\n",
    "\n",
    "    MeshRange r{};\n",
    "    r.indexStart = static_cast<uint32_t>(inds.size());\n",
    "\n",
    "    auto it = cache_.find(key);\n",
    "    if (it != cache_.end()) {\n",
    "        // 캐시 적중: 저장된 지오메트리를 씬 배열 뒤에 복사한다.\n",
    "        const CachedMesh& cached = it->second;\n",
    "        uint32_t vBase = static_cast<uint32_t>(verts.size());\n",
    "        verts.insert(verts.end(), cached.verts.begin(), cached.verts.end());\n",
    "        for (auto idx : cached.inds)\n",
    "            inds.push_back(vBase + idx);\n",
    "        r.indexCount = static_cast<uint32_t>(cached.inds.size());\n",
    "        r.bboxMin    = cached.bboxMin;\n",
    "        r.bboxMax    = cached.bboxMax;\n",
    "        ++hits_;\n",
    "        return r;\n",
    "    }\n",
    "\n",
    "    // 캐시 미스: 디스크에서 메시를 파싱한다.\n",
    "    uint32_t vBase = static_cast<uint32_t>(verts.size());\n",
    "    uint32_t iBase = static_cast<uint32_t>(inds.size());\n",
    "    r = loadMesh(path, verts, inds, color);\n",
    "\n",
    "    // 파싱한 지오메트리를 0 기준 상대 인덱스로 캐시에 저장한다.\n",
    "    CachedMesh& entry = cache_[key];\n",
    "    entry.bboxMin = r.bboxMin;\n",
    "    entry.bboxMax = r.bboxMax;\n",
    "    entry.verts.assign(verts.begin() + vBase, verts.end());\n",
    "    for (uint32_t i = iBase; i < static_cast<uint32_t>(inds.size()); ++i)\n",
    "        entry.inds.push_back(inds[i] - vBase);\n",
    "\n",
    "    ++misses_;\n",
    "    return r;\n",
    "}\n",
    "\n",
]

lines[insert_idx:insert_idx] = impl_lines

with open('C:/MyFolder/High3/OProj/src/VulkanApp.cpp', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('Done. Total lines:', len(lines))
