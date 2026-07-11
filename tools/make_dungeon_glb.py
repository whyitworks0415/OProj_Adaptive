"""
오클루전 컬링 시연용 던전 GLB 생성 스크립트 (low-poly 박스 버전)

뱀(snake) 모양으로 꺾이는 긴 던전: 방 7개 + 복도 6개 (~216m 경로).
- 벽·천장으로 완전히 막혀 있어 벽 뒤 방/소품이 오클루전 컬링 대상이 된다.
- 세그먼트 단위(5~8m)로 쪼갠 벽/바닥/천장 -> 컬링 granularity 확보.
- 횃불(emissive strength), 금괴(metallic), 수정(파랑 발광), 나무 상자 소품.
- KHR_lights_punctual 조명 8개 (포인트 7 + 방향광 1) — 렌더러 상한과 일치.
- 카메라 노드: 입구에서 던전 안쪽(+Z)을 바라보게 배치.

실행: python tools/make_dungeon_glb.py   (프로젝트 루트에서)
출력: maps/map/3.glb
"""

import struct, json, math, os, random

rng = random.Random(20260705)

# ---------------------------------------------------------------- GLB 빌더
def make_box(cx, cy, cz, sx, sy, sz):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    faces = [
        ( 0,  1,  0, [(-hx, hy,-hz),( hx, hy,-hz),( hx, hy, hz),(-hx, hy, hz)]),
        ( 0, -1,  0, [(-hx,-hy, hz),( hx,-hy, hz),( hx,-hy,-hz),(-hx,-hy,-hz)]),
        ( 0,  0,  1, [(-hx,-hy, hz),( hx,-hy, hz),( hx, hy, hz),(-hx, hy, hz)]),
        ( 0,  0, -1, [( hx,-hy,-hz),(-hx,-hy,-hz),(-hx, hy,-hz),( hx, hy,-hz)]),
        ( 1,  0,  0, [( hx,-hy, hz),( hx,-hy,-hz),( hx, hy,-hz),( hx, hy, hz)]),
        (-1,  0,  0, [(-hx,-hy,-hz),(-hx,-hy, hz),(-hx, hy, hz),(-hx, hy,-hz)]),
    ]
    positions, normals, indices = [], [], []
    vbase = 0
    for nx, ny, nz, verts in faces:
        for vx, vy, vz in verts:
            positions.append((cx + vx, cy + vy, cz + vz))
            normals.append((nx, ny, nz))
        indices += [vbase, vbase+1, vbase+2, vbase, vbase+2, vbase+3]
        vbase += 4
    pos_b = b''.join(struct.pack('<fff', *p) for p in positions)
    nrm_b = b''.join(struct.pack('<fff', *n) for n in normals)
    idx_b = b''.join(struct.pack('<H', i) for i in indices)
    xs = [p[0] for p in positions]; ys = [p[1] for p in positions]; zs = [p[2] for p in positions]
    return pos_b, nrm_b, idx_b, len(positions), len(indices), \
           [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]

def align4(n):
    return (n + 3) & ~3

objects = []   # {name, cx..sz, color, rough, metal, emissive, emissive_strength}
lights  = []   # {name, type, color, intensity, range, pos, dir}
camera_pos = [0.0, 1.7, 0.0]

def add_box_obj(name, cx, cy, cz, sx, sy, sz, color,
                rough=0.9, metal=0.0, emissive=None, emissive_strength=None):
    objects.append(dict(name=name, cx=cx, cy=cy, cz=cz, sx=sx, sy=sy, sz=sz,
                        color=color, rough=rough, metal=metal,
                        emissive=emissive, emissive_strength=emissive_strength))

def build_glb():
    buffers_bin  = b''
    accessors, buffer_views, meshes, nodes, materials, cameras = [], [], [], [], [], []

    def add_bv(data, target):
        nonlocal buffers_bin
        offset = len(buffers_bin)
        buffers_bin += data + b'\x00' * (align4(len(data)) - len(data))
        buffer_views.append({"buffer": 0, "byteOffset": offset,
                             "byteLength": len(data), "target": target})
        return len(buffer_views) - 1

    def add_acc(bv, count, ctype, atype, bmin=None, bmax=None):
        acc = {"bufferView": bv, "byteOffset": 0, "componentType": ctype,
               "count": count, "type": atype}
        if bmin: acc["min"] = bmin
        if bmax: acc["max"] = bmax
        accessors.append(acc)
        return len(accessors) - 1

    for o in objects:
        pos_b, nrm_b, idx_b, vc, ic, bmin, bmax = make_box(
            o['cx'], o['cy'], o['cz'], o['sx'], o['sy'], o['sz'])
        acc_pos = add_acc(add_bv(pos_b, 34962), vc, 5126, "VEC3", bmin, bmax)
        acc_nrm = add_acc(add_bv(nrm_b, 34962), vc, 5126, "VEC3")
        acc_idx = add_acc(add_bv(idx_b, 34963), ic, 5123, "SCALAR")

        r, g, b = o['color']
        mat = {"name": o['name'],
               "pbrMetallicRoughness": {
                   "baseColorFactor": [r, g, b, 1.0],
                   "metallicFactor": o['metal'],
                   "roughnessFactor": o['rough']}}
        if o['emissive'] is not None:
            mat["emissiveFactor"] = list(o['emissive'])
            if o['emissive_strength'] is not None:
                mat["extensions"] = {"KHR_materials_emissive_strength":
                                     {"emissiveStrength": o['emissive_strength']}}
        materials.append(mat)

        meshes.append({"name": o['name'],
                       "primitives": [{"attributes": {"POSITION": acc_pos, "NORMAL": acc_nrm},
                                       "indices": acc_idx,
                                       "material": len(materials) - 1, "mode": 4}]})
        nodes.append({"name": o['name'], "mesh": len(meshes) - 1})

    # 카메라 노드 (입구)
    cameras.append({"type": "perspective",
                    "perspective": {"yfov": 1.0472, "znear": 0.1, "aspectRatio": 1.7778}})
    nodes.append({"name": "EntranceCamera", "camera": 0, "translation": camera_pos})

    # KHR_lights_punctual
    light_defs = []
    for li, l in enumerate(lights):
        d = {"name": l['name'], "type": l['type'],
             "color": list(l['color']), "intensity": l['intensity']}
        if l['type'] != 'directional' and l.get('range'):
            d["range"] = l['range']
        light_defs.append(d)
        node = {"name": l['name'],
                "extensions": {"KHR_lights_punctual": {"light": li}}}
        if l['type'] == 'directional':
            # 방향광은 노드 -Z 축이 조명 방향. 회전으로 아래 45도를 만든다.
            ang = math.radians(-115.0) / 2.0
            node["rotation"] = [math.sin(ang), 0.0, 0.0, math.cos(ang)]
            node["translation"] = [0.0, 30.0, 0.0]
        else:
            node["translation"] = list(l['pos'])
        nodes.append(node)

    gltf = {
        "asset": {"version": "2.0", "generator": "make_dungeon_glb.py"},
        "extensionsUsed": ["KHR_lights_punctual", "KHR_materials_emissive_strength"],
        "extensions": {"KHR_lights_punctual": {"lights": light_defs}},
        "scene": 0,
        "scenes": [{"name": "Dungeon", "nodes": list(range(len(nodes)))}],
        "nodes": nodes, "meshes": meshes, "materials": materials,
        "cameras": cameras,
        "accessors": accessors, "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(buffers_bin)}],
    }
    json_bytes = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    json_chunk = json_bytes + b' ' * (align4(len(json_bytes)) - len(json_bytes))
    bin_chunk  = buffers_bin + b'\x00' * (align4(len(buffers_bin)) - len(buffers_bin))
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    glb  = struct.pack('<III', 0x46546C67, 2, total)
    glb += struct.pack('<II', len(json_chunk), 0x4E4F534A) + json_chunk
    glb += struct.pack('<II', len(bin_chunk),  0x004E4942) + bin_chunk
    return glb

# ---------------------------------------------------------------- 색상 팔레트
def jitter(c, amt=0.05):
    return tuple(max(0.0, min(1.0, v + rng.uniform(-amt, amt))) for v in c)

STONE_WALL  = (0.42, 0.38, 0.33)
STONE_FLOOR = (0.30, 0.28, 0.26)
STONE_CEIL  = (0.24, 0.22, 0.21)
PILLAR      = (0.48, 0.44, 0.38)
WOOD        = (0.42, 0.27, 0.12)
GOLD        = (1.00, 0.78, 0.25)
TORCH_WOOD  = (0.25, 0.15, 0.07)
FLAME       = (1.00, 0.55, 0.15)
CRYSTAL     = (0.35, 0.65, 1.00)

W  = 6.0    # 복도 폭
H  = 5.0    # 벽 높이
T  = 1.0    # 벽 두께
S  = 36.0   # 웨이포인트 간격
R  = 16.0   # 일반 방 크기
RB = 26.0   # 보스 방 크기

nid = [0]
def uid():
    nid[0] += 1
    return nid[0]

# ---------------------------------------------------------------- 부품 함수
def torch(cx, cy, cz):
    """벽걸이 횃불: 나무 자루 + 발광 불꽃"""
    add_box_obj(f"TorchStick_{uid()}", cx, cy, cz, 0.15, 0.7, 0.15, TORCH_WOOD, rough=0.95)
    add_box_obj(f"TorchFlame_{uid()}", cx, cy + 0.5, cz, 0.3, 0.35, 0.3, FLAME,
                rough=0.6, emissive=FLAME, emissive_strength=6.0)

def crate_cluster(cx, cz, n):
    for _ in range(n):
        s  = rng.uniform(0.8, 1.4)
        ox = rng.uniform(-1.6, 1.6)
        oz = rng.uniform(-1.6, 1.6)
        add_box_obj(f"Crate_{uid()}", cx + ox, s / 2, cz + oz, s, s, s,
                    jitter(WOOD), rough=0.85)

def gold_pile(cx, cz, n):
    for _ in range(n):
        s  = rng.uniform(0.3, 0.7)
        ox = rng.uniform(-1.8, 1.8)
        oz = rng.uniform(-1.8, 1.8)
        add_box_obj(f"Gold_{uid()}", cx + ox, s / 2, cz + oz, s, s, s,
                    jitter(GOLD, 0.03), rough=0.3, metal=1.0)

def crystal(cx, cz, height=1.8):
    add_box_obj(f"Crystal_{uid()}", cx, height / 2, cz, 0.5, height, 0.5, CRYSTAL,
                rough=0.2, emissive=CRYSTAL, emissive_strength=4.0)

def statue(cx, cz):
    """간단한 골렘 석상"""
    add_box_obj(f"StatueBase_{uid()}",  cx, 0.4, cz, 2.4, 0.8, 2.4, PILLAR, rough=0.9)
    add_box_obj(f"StatueBody_{uid()}",  cx, 1.9, cz, 1.4, 2.2, 1.0, jitter(PILLAR), rough=0.85)
    add_box_obj(f"StatueHead_{uid()}",  cx, 3.4, cz, 0.8, 0.8, 0.8, jitter(PILLAR), rough=0.85)
    add_box_obj(f"StatueArmL_{uid()}",  cx - 1.0, 1.9, cz, 0.5, 1.8, 0.6, jitter(PILLAR), rough=0.85)
    add_box_obj(f"StatueArmR_{uid()}",  cx + 1.0, 1.9, cz, 0.5, 1.8, 0.6, jitter(PILLAR), rough=0.85)

def wall_run(x0, z0, x1, z1, seg=6.0):
    """(x0,z0)->(x1,z1) 직선 벽을 seg 길이 조각으로 나눠 배치 (X 또는 Z 정렬)"""
    if abs(x1 - x0) > abs(z1 - z0):
        length, z = x1 - x0, z0
        n = max(1, int(round(abs(length) / seg)))
        step = length / n
        for i in range(n):
            cx = x0 + step * (i + 0.5)
            add_box_obj(f"Wall_{uid()}", cx, H / 2, z, abs(step), H, T,
                        jitter(STONE_WALL), rough=0.92)
    else:
        length, x = z1 - z0, x0
        n = max(1, int(round(abs(length) / seg)))
        step = length / n
        for i in range(n):
            cz = z0 + step * (i + 0.5)
            add_box_obj(f"Wall_{uid()}", x, H / 2, cz, T, H, abs(step),
                        jitter(STONE_WALL), rough=0.92)

def slab_grid(cx, cz, size, y, sy, color, tile=8.0):
    """바닥/천장을 타일로 쪼개 배치"""
    n = max(1, int(round(size / tile)))
    step = size / n
    for i in range(n):
        for j in range(n):
            add_box_obj(f"Slab_{uid()}",
                        cx - size / 2 + step * (i + 0.5), y,
                        cz - size / 2 + step * (j + 0.5),
                        step, sy, step, jitter(color, 0.03), rough=0.95)

def room(cx, cz, size, doors, boss=False):
    """방: 바닥/천장/벽(문 개구부) + 기둥 + 소품. doors: {'N','S','E','W'}"""
    half = size / 2
    slab_grid(cx, cz, size, -0.15, 0.3, STONE_FLOOR)
    slab_grid(cx, cz, size, H + 0.15, 0.3, STONE_CEIL)

    # 벽 4면 (문이 있는 면은 폭 W 개구부를 남긴다)
    for side in 'NSEW':
        if side == 'N':   z, horiz = cz + half, True
        elif side == 'S': z, horiz = cz - half, True
        elif side == 'E': x, horiz = cx + half, False
        else:             x, horiz = cx - half, False
        if horiz:
            if side in doors:
                wall_run(cx - half, z, cx - W / 2, z)
                wall_run(cx + W / 2, z, cx + half, z)
            else:
                wall_run(cx - half, z, cx + half, z)
        else:
            if side in doors:
                wall_run(x, cz - half, x, cz - W / 2)
                wall_run(x, cz + W / 2, x, cz + half)
            else:
                wall_run(x, cz - half, x, cz + half)

    # 기둥 4개
    q = size / 4
    for ox, oz in ((-q, -q), (q, -q), (-q, q), (q, q)):
        add_box_obj(f"Pillar_{uid()}", cx + ox, H / 2, cz + oz, 1.2, H, 1.2,
                    jitter(PILLAR), rough=0.9)

    # 코너 횃불 4개
    m = half - 1.5
    for ox, oz in ((-m, -m), (m, -m), (-m, m), (m, m)):
        torch(cx + ox, 2.8, cz + oz)

    # 소품
    if boss:
        statue(cx, cz + size / 4)
        gold_pile(cx - size / 4, cz, 14)
        gold_pile(cx + size / 4, cz + size / 5, 10)
        crystal(cx - size / 4, cz - size / 4, 2.4)
        crystal(cx + size / 4, cz - size / 4, 2.0)
    else:
        crate_cluster(cx - size / 4, cz - size / 4, rng.randint(2, 5))
        if rng.random() < 0.6:
            crate_cluster(cx + size / 4, cz + size / 4, rng.randint(1, 3))
        if rng.random() < 0.5:
            crystal(cx + size / 4, cz - size / 4)
        # 중앙 제단
        add_box_obj(f"Altar_{uid()}", cx, 0.5, cz, 2.0, 1.0, 1.2, jitter(PILLAR), rough=0.8)

def corridor(x0, z0, x1, z1):
    """방 가장자리 사이를 잇는 복도 (X 또는 Z 정렬). 바닥/천장/양측 벽 + 횃불"""
    along_z = abs(z1 - z0) > abs(x1 - x0)
    length  = abs(z1 - z0) if along_z else abs(x1 - x0)
    n = max(1, int(round(length / 5.0)))
    step = length / n
    sgn = 1 if ((z1 - z0) if along_z else (x1 - x0)) > 0 else -1
    for i in range(n):
        t = (i + 0.5) * step * sgn
        cx = x0 + (0 if along_z else t)
        cz = z0 + (t if along_z else 0)
        # 바닥·천장
        sx = W if along_z else step
        sz = step if along_z else W
        add_box_obj(f"CorFloor_{uid()}", cx, -0.15, cz, sx, 0.3, sz,
                    jitter(STONE_FLOOR, 0.03), rough=0.95)
        add_box_obj(f"CorCeil_{uid()}", cx, H + 0.15, cz, sx, 0.3, sz,
                    jitter(STONE_CEIL, 0.03), rough=0.95)
        # 양측 벽
        if along_z:
            add_box_obj(f"CorWall_{uid()}", cx - (W / 2 + T / 2), H / 2, cz, T, H, step,
                        jitter(STONE_WALL), rough=0.92)
            add_box_obj(f"CorWall_{uid()}", cx + (W / 2 + T / 2), H / 2, cz, T, H, step,
                        jitter(STONE_WALL), rough=0.92)
        else:
            add_box_obj(f"CorWall_{uid()}", cx, H / 2, cz - (W / 2 + T / 2), step, H, T,
                        jitter(STONE_WALL), rough=0.92)
            add_box_obj(f"CorWall_{uid()}", cx, H / 2, cz + (W / 2 + T / 2), step, H, T,
                        jitter(STONE_WALL), rough=0.92)
        # 횃불 (한 칸 걸러 좌우 교대)
        if i % 2 == 0:
            off = (W / 2 - 0.4) * (1 if i % 4 == 0 else -1)
            tx = cx + (off if along_z else 0)
            tz = cz + (0 if along_z else off)
            torch(tx, 2.8, tz)

# ---------------------------------------------------------------- 던전 레이아웃
# 뱀 모양 웨이포인트 (그리드 좌표 × S)
waypoints = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]
coords = [(gx * S, gz * S) for gx, gz in waypoints]

# 각 방의 문 방향 계산
def door_dir(a, b):
    ax, az = a; bx, bz = b
    if bz > az: return 'N'
    if bz < az: return 'S'
    if bx > ax: return 'E'
    return 'W'

for i, (cx, cz) in enumerate(coords):
    doors = set()
    if i > 0:
        doors.add(door_dir(coords[i], coords[i - 1]))
    if i < len(coords) - 1:
        doors.add(door_dir(coords[i], coords[i + 1]))
    if i == 0:
        doors.add('S')  # 입구
    is_boss = (i == len(coords) - 1)
    size = RB if is_boss else R
    room(cx, cz, size, doors, boss=is_boss)

# 복도 연결
for i in range(len(coords) - 1):
    (ax, az), (bx, bz) = coords[i], coords[i + 1]
    sb = RB if i + 1 == len(coords) - 1 else R
    d = door_dir(coords[i], coords[i + 1])
    if d == 'N':
        corridor(ax, az + R / 2, bx, bz - sb / 2)
    elif d == 'E':
        corridor(ax + R / 2, az, bx - sb / 2, bz)
    elif d == 'S':
        corridor(ax, az - R / 2, bx, bz + sb / 2)
    else:
        corridor(ax - R / 2, az, bx + sb / 2, bz)

# 입구 앞마당 (외부에서 던전으로 들어가는 짧은 통로)
corridor(0, -R / 2, 0, -R / 2 - 10)

# 카메라: 입구 통로에서 +Z(던전 안쪽)를 바라본다 (기본 yaw=90 -> +Z)
camera_pos = [0.0, 1.7, -R / 2 - 8.0]

# ---------------------------------------------------------------- 조명 (총 8개 = 렌더러 상한)
def add_point(name, x, y, z, color, intensity, rng_m=28.0):
    lights.append(dict(name=name, type='point', color=color,
                       intensity=intensity, range=rng_m, pos=[x, y, z]))

add_point("L_Entrance", coords[0][0], 3.5, coords[0][1], (1.0, 0.62, 0.3), 45)
add_point("L_Room1",    coords[1][0], 3.5, coords[1][1], (1.0, 0.60, 0.28), 40)
add_point("L_Room3",    coords[3][0], 3.5, coords[3][1], (1.0, 0.60, 0.28), 40)
add_point("L_Room5",    coords[5][0], 3.5, coords[5][1], (1.0, 0.60, 0.28), 40)
mid = ((coords[2][0] + coords[3][0]) / 2, (coords[2][1] + coords[3][1]) / 2)
add_point("L_Corridor", mid[0], 3.2, mid[1], (1.0, 0.55, 0.25), 30)
add_point("L_BossWarm", coords[-1][0], 4.0, coords[-1][1], (1.0, 0.65, 0.3), 60, 34)
add_point("L_BossBlue", coords[-1][0] - RB / 4, 3.0, coords[-1][1] - RB / 4, (0.4, 0.65, 1.0), 30)
lights.append(dict(name="L_Sun", type='directional', color=(1.0, 0.95, 0.85),
                   intensity=1.2, pos=None))

# ---------------------------------------------------------------- 출력
os.makedirs('maps/map', exist_ok=True)
glb = build_glb()
out = 'maps/map/3.glb'
with open(out, 'wb') as f:
    f.write(glb)

torches = sum(1 for o in objects if 'Flame' in o['name'])
print(f"Generated {out}  ({len(glb)/1024:.0f} KB)")
print(f"  objects: {len(objects)}  (torches {torches}, lights {len(lights)})")
print(f"  path: {len(coords)} rooms, {len(coords)-1} corridors, ~{int(S*(len(coords)-1))}m")
