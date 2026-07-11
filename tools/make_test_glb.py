"""
테스트용 city GLB 파일 생성 스크립트
건물 여러 채 + 도로를 배치하는 간단한 GLTF 씬을 만든다.
각 오브젝트가 별도 노드로 저장되므로 Frustum Culling 테스트에 적합하다.

실행: python make_test_glb.py
출력: maps/test_city.glb
"""

import struct, json, math, os

def pack_f3(x, y, z):
    return struct.pack('<fff', x, y, z)

def make_box(cx, cy, cz, sx, sy, sz, color):
    """
    중심 (cx,cy,cz), 크기 (sx,sy,sz) 인 박스의 vertex/index 데이터 반환.
    color: (r,g,b) float 0-1
    반환: (positions_bytes, normals_bytes, indices_bytes, vertex_count, index_count, bbox_min, bbox_max)
    """
    hx, hy, hz = sx/2, sy/2, sz/2

    # 각 면을 법선과 네 꼭짓점 순서로 정의한다.
    faces = [
        ( 0,  1,  0, [(-hx,hy,-hz),( hx,hy,-hz),( hx,hy, hz),(-hx,hy, hz)]),  # +Y 위쪽 면
        ( 0, -1,  0, [(-hx,-hy, hz),( hx,-hy, hz),( hx,-hy,-hz),(-hx,-hy,-hz)]),  # -Y 아래쪽 면
        ( 0,  0,  1, [(-hx,-hy, hz),( hx,-hy, hz),( hx, hy, hz),(-hx, hy, hz)]),  # +Z 앞쪽 면
        ( 0,  0, -1, [( hx,-hy,-hz),(-hx,-hy,-hz),(-hx, hy,-hz),( hx, hy,-hz)]),  # -Z 뒤쪽 면
        ( 1,  0,  0, [( hx,-hy, hz),( hx,-hy,-hz),( hx, hy,-hz),( hx, hy, hz)]),  # +X 오른쪽 면
        (-1,  0,  0, [(-hx,-hy,-hz),(-hx,-hy, hz),(-hx, hy, hz),(-hx, hy,-hz)]),  # -X 왼쪽 면
    ]

    positions = []
    normals   = []
    indices   = []
    vbase = 0
    for nx, ny, nz, verts in faces:
        for vx, vy, vz in verts:
            positions.append((cx+vx, cy+vy, cz+vz))
            normals.append((nx, ny, nz))
        indices += [vbase, vbase+1, vbase+2, vbase, vbase+2, vbase+3]
        vbase += 4

    pos_bytes = b''.join(struct.pack('<fff', *p) for p in positions)
    nrm_bytes = b''.join(struct.pack('<fff', *n) for n in normals)
    idx_bytes = b''.join(struct.pack('<H', i) for i in indices)

    all_x = [p[0] for p in positions]
    all_y = [p[1] for p in positions]
    all_z = [p[2] for p in positions]
    bbox_min = [min(all_x), min(all_y), min(all_z)]
    bbox_max = [max(all_x), max(all_y), max(all_z)]

    return pos_bytes, nrm_bytes, idx_bytes, len(positions), len(indices), bbox_min, bbox_max


def align4(n):
    return (n + 3) & ~3


def build_glb(objects):
    """
    objects: list of dict { name, cx,cy,cz, sx,sy,sz, color:(r,g,b) }
    각 오브젝트 → 별도 GLTF 노드 (합치지 않음)
    """
    buffers_bin   = b''
    accessors     = []
    buffer_views  = []
    meshes        = []
    nodes         = []
    materials     = []

    def add_buffer_view(data, target):
        nonlocal buffers_bin
        offset = len(buffers_bin)
        padded = data + b'\x00' * (align4(len(data)) - len(data))
        buffers_bin += padded
        idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target})
        return idx

    def add_accessor(bv_idx, count, component_type, atype, bmin=None, bmax=None):
        idx = len(accessors)
        acc = {"bufferView": bv_idx, "byteOffset": 0, "componentType": component_type,
               "count": count, "type": atype}
        if bmin: acc["min"] = bmin
        if bmax: acc["max"] = bmax
        accessors.append(acc)
        return idx

    for obj in objects:
        pos_b, nrm_b, idx_b, vc, ic, bmin, bmax = make_box(
            obj['cx'], obj['cy'], obj['cz'],
            obj['sx'], obj['sy'], obj['sz'],
            obj['color'])

        bv_pos = add_buffer_view(pos_b, 34962)  # GLTF ARRAY_BUFFER 대상
        bv_nrm = add_buffer_view(nrm_b, 34962)
        bv_idx = add_buffer_view(idx_b, 34963)  # GLTF ELEMENT_ARRAY_BUFFER 대상

        acc_pos = add_accessor(bv_pos, vc, 5126, "VEC3", bmin, bmax)  # GLTF FLOAT 컴포넌트
        acc_nrm = add_accessor(bv_nrm, vc, 5126, "VEC3")
        acc_idx = add_accessor(bv_idx, ic, 5123, "SCALAR")  # GLTF UNSIGNED_SHORT 컴포넌트

        mat_idx = len(materials)
        r, g, b = obj['color']
        materials.append({
            "name": obj['name'],
            "pbrMetallicRoughness": {
                "baseColorFactor": [r, g, b, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8
            }
        })

        mesh_idx = len(meshes)
        meshes.append({
            "name": obj['name'],
            "primitives": [{
                "attributes": {"POSITION": acc_pos, "NORMAL": acc_nrm},
                "indices": acc_idx,
                "material": mat_idx,
                "mode": 4
            }]
        })

        nodes.append({"name": obj['name'], "mesh": mesh_idx})

    gltf = {
        "asset": {"version": "2.0", "generator": "make_test_glb.py"},
        "scene": 0,
        "scenes": [{"name": "TestCity", "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(buffers_bin)}]
    }

    json_bytes = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    # JSON 청크는 공백(0x20)으로 4바이트 경계에 맞춘다.
    json_pad = (align4(len(json_bytes)) - len(json_bytes))
    json_chunk = json_bytes + b' ' * json_pad

    # BIN 청크는 0으로 패딩해 4바이트 경계에 맞춘다.
    bin_pad  = (align4(len(buffers_bin)) - len(buffers_bin))
    bin_chunk = buffers_bin + b'\x00' * bin_pad

    total_len = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    glb  = struct.pack('<III', 0x46546C67, 2, total_len)  # GLB 헤더: magic, version, length
    glb += struct.pack('<II', len(json_chunk), 0x4E4F534A) + json_chunk  # JSON 청크 헤더와 본문
    glb += struct.pack('<II', len(bin_chunk),  0x004E4942) + bin_chunk  # BIN 청크 헤더와 본문
    return glb


# 씬 정의: 건물 격자 + 도로
objects = []

# 5×5 격자로 건물 배치 (높이 랜덤화)
heights = [
    4, 6, 3, 7, 5,
    5, 8, 4, 6, 3,
    3, 5, 9, 4, 6,
    6, 4, 5, 8, 3,
    4, 7, 3, 5, 6,
]
colors = [
    (0.7, 0.5, 0.3),  # 벽돌색
    (0.6, 0.7, 0.8),  # 회청색
    (0.5, 0.6, 0.5),  # 녹회색
    (0.8, 0.7, 0.6),  # 크림색
    (0.6, 0.6, 0.7),  # 회보라
]

spacing = 12.0
building_w = 7.0

idx = 0
for row in range(5):
    for col in range(5):
        h = heights[idx]
        color = colors[idx % len(colors)]
        cx = (col - 2) * spacing
        cz = (row - 2) * spacing
        cy = h / 2.0
        objects.append({
            'name': f'Building_{row}_{col}',
            'cx': cx, 'cy': cy, 'cz': cz,
            'sx': building_w, 'sy': float(h), 'sz': building_w,
            'color': color
        })
        idx += 1

# 도로 (가로)
for row in range(6):
    z = (row - 2.5) * spacing
    objects.append({
        'name': f'Road_H_{row}',
        'cx': 0.0, 'cy': 0.05, 'cz': z,
        'sx': 72.0, 'sy': 0.1, 'sz': 4.0,
        'color': (0.25, 0.25, 0.25)
    })

# 도로 (세로)
for col in range(6):
    x = (col - 2.5) * spacing
    objects.append({
        'name': f'Road_V_{col}',
        'cx': x, 'cy': 0.05, 'cz': 0.0,
        'sx': 4.0, 'sy': 0.1, 'sz': 72.0,
        'color': (0.25, 0.25, 0.25)
    })

# 바닥
objects.append({
    'name': 'Ground',
    'cx': 0.0, 'cy': -0.05, 'cz': 0.0,
    'sx': 80.0, 'sy': 0.1, 'sz': 80.0,
    'color': (0.4, 0.5, 0.3)
})

os.makedirs('maps', exist_ok=True)
glb = build_glb(objects)
out_path = 'maps/test_city.glb'
with open(out_path, 'wb') as f:
    f.write(glb)

print(f"Generated {out_path}  ({len(glb)} bytes)")
print(f"  {len(objects)} objects: {sum(1 for o in objects if 'Building' in o['name'])} buildings, "
      f"{sum(1 for o in objects if 'Road' in o['name'])} roads, 1 ground")
