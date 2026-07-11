"""
인스턴싱/LOD 시연용 소나무 숲 GLB 생성 스크립트

핵심: 나무/바위를 "하나의 메시"로 만들고 수백 개 노드가 같은 메시를 참조한다.
-> 렌더러의 GLTF 로더가 자동으로 인스턴싱 그룹을 만들어
   3번 키(GPU Instancing) 효과를 draw call 수치로 크게 보여준다.

구성: 노이즈 지형 1개 + 소나무 900그루(공유 메시 3종) + 바위 80개(공유 메시)
실행: python tools/make_forest_glb.py   (프로젝트 루트에서)
출력: maps/map/4.glb
"""

import struct, json, math, os, random

rng = random.Random(20260707)

# ---------------------------------------------------------------- 노이즈 (지형 높이)
def _h(ix, iz):
    n = (ix * 374761393 + iz * 668265263) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFF) / 65535.0

def vnoise2(x, z):
    ix, iz = math.floor(x), math.floor(z)
    fx, fz = x - ix, z - iz
    sx = fx * fx * (3 - 2 * fx)
    sz = fz * fz * (3 - 2 * fz)
    def lerp(a, b, t): return a + (b - a) * t
    return lerp(lerp(_h(ix, iz), _h(ix+1, iz), sx),
                lerp(_h(ix, iz+1), _h(ix+1, iz+1), sx), sz)

def terrain_h(x, z):
    f = 0.025
    return (vnoise2(x*f, z*f) + 0.5 * vnoise2(x*f*2.3 + 7, z*f*2.3)) / 1.5 * 6.0 - 2.0

# ---------------------------------------------------------------- 메시 빌더
class Mesh:
    def __init__(self):
        self.pos, self.nrm, self.idx = [], [], []
    def compute_normals(self):
        n = [[0.0, 0.0, 0.0] for _ in self.pos]
        p = self.pos
        for t in range(0, len(self.idx), 3):
            a, b, c = self.idx[t], self.idx[t+1], self.idx[t+2]
            ux, uy, uz = p[b][0]-p[a][0], p[b][1]-p[a][1], p[b][2]-p[a][2]
            vx, vy, vz = p[c][0]-p[a][0], p[c][1]-p[a][1], p[c][2]-p[a][2]
            fx, fy, fz = uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx
            for i in (a, b, c):
                n[i][0] += fx; n[i][1] += fy; n[i][2] += fz
        out = []
        for x, y, z in n:
            l = math.sqrt(x*x + y*y + z*z)
            out.append((x/l, y/l, z/l) if l > 1e-9 else (0.0, 1.0, 0.0))
        self.nrm = out

def lathe(m, profile, seg, y0=0.0):
    """profile [(y, r), ...] 회전체를 메시 m 에 추가 (CCW 바깥 감김)"""
    rings = []
    for (py, pr) in profile:
        base = len(m.pos)
        for s in range(seg):
            a = 2 * math.pi * s / seg
            m.pos.append((math.cos(a) * pr, y0 + py, math.sin(a) * pr))
        rings.append(base)
    for k in range(len(rings) - 1):
        b0, b1 = rings[k], rings[k+1]
        for s in range(seg):
            s2 = (s + 1) % seg
            m.idx += [b0+s, b1+s, b1+s2, b0+s, b1+s2, b0+s2]
    # 끝 캡 (반지름이 0 에 가까우면 생략)
    for ring_i, flip in ((0, True), (len(rings)-1, False)):
        if profile[ring_i][1] < 0.01: continue
        center = len(m.pos)
        m.pos.append((0.0, y0 + profile[ring_i][0], 0.0))
        b = rings[ring_i]
        for s in range(seg):
            s2 = (s + 1) % seg
            m.idx += ([center, b+s, b+s2] if flip else [center, b+s2, b+s])

def make_pine(height, trunk_r):
    """소나무 한 그루 = 몸통 + 원뿔 3단 (하나의 메시)"""
    m = Mesh()
    lathe(m, [(0, trunk_r), (height*0.35, trunk_r*0.7)], 8)               # 몸통
    tiers = [(height*0.25, height*0.55, height*0.30),
             (height*0.45, height*0.75, height*0.24),
             (height*0.65, height*1.00, height*0.17)]
    for y_lo, y_hi, r in tiers:
        lathe(m, [(y_lo, r), (y_hi, 0.001)], 10)
    m.compute_normals()
    return m

def make_rock(r):
    m = Mesh()
    lat, lon = 6, 8
    for j in range(lat + 1):
        th = math.pi * j / lat
        for i in range(lon):
            ph = 2 * math.pi * i / lon
            jit = 1.0 + (_h(i*7 + j*13, i + j) - 0.5) * 0.5
            m.pos.append((r * jit * math.sin(th) * math.cos(ph),
                          r * jit * 0.6 * math.cos(th),
                          r * jit * math.sin(th) * math.sin(ph)))
    for j in range(lat):
        for i in range(lon):
            i2 = (i + 1) % lon
            v0 = j*lon+i; v1 = j*lon+i2; v2 = (j+1)*lon+i; v3 = (j+1)*lon+i2
            m.idx += [v0, v3, v2, v0, v1, v3]
    m.compute_normals()
    return m

def make_terrain(size, cells):
    m = Mesh()
    step = size / cells
    for j in range(cells + 1):
        for i in range(cells + 1):
            x = -size/2 + i * step
            z = -size/2 + j * step
            m.pos.append((x, terrain_h(x, z), z))
    for j in range(cells):
        for i in range(cells):
            v0 = j*(cells+1)+i; v1 = v0+1; v2 = v0+cells+1; v3 = v2+1
            m.idx += [v0, v3, v1, v0, v2, v3]
    m.compute_normals()
    return m

# ---------------------------------------------------------------- 씬 구성
# meshes: 공유 메시 목록 / nodes: 메시 참조 + TRS
meshes = []   # {name, mesh, color, rough, metal}
nodes  = []   # {name, mesh:int, t:(x,y,z), ry:rad, s:float}

def add_mesh(name, mesh, color, rough=0.9, metal=0.0):
    meshes.append(dict(name=name, mesh=mesh, color=color, rough=rough, metal=metal))
    return len(meshes) - 1

def add_node(name, mesh_i, t, ry=0.0, s=1.0):
    nodes.append(dict(name=name, mesh=mesh_i, t=t, ry=ry, s=s))

SIZE = 240.0
mi_terrain = add_mesh("Terrain", make_terrain(SIZE, 110), (0.20, 0.34, 0.12), rough=0.95)
mi_pines = [add_mesh(f"Pine{k}", make_pine(h, tr), c, rough=0.9)
            for k, (h, tr, c) in enumerate([
                (7.0, 0.28, (0.05, 0.28, 0.08)),
                (9.5, 0.34, (0.07, 0.24, 0.06)),
                (5.5, 0.22, (0.10, 0.33, 0.09))])]
mi_rock = add_mesh("Rock", make_rock(1.0), (0.45, 0.43, 0.40), rough=0.95)

add_node("Terrain", mi_terrain, (0, 0, 0))

# 소나무 900그루 — 같은 메시를 공유하는 노드 (자동 인스턴싱 대상)
for i in range(900):
    x = rng.uniform(-SIZE/2 + 6, SIZE/2 - 6)
    z = rng.uniform(-SIZE/2 + 6, SIZE/2 - 6)
    if abs(x) < 5 and z < -SIZE/2 + 40:  # 카메라 진입로는 비워 둔다
        continue
    add_node(f"Pine_{i}", rng.choice(mi_pines),
             (x, terrain_h(x, z) - 0.15, z),
             ry=rng.uniform(0, 2*math.pi), s=rng.uniform(0.7, 1.35))

# 바위 80개
for i in range(80):
    x = rng.uniform(-SIZE/2 + 4, SIZE/2 - 4)
    z = rng.uniform(-SIZE/2 + 4, SIZE/2 - 4)
    add_node(f"Rock_{i}", mi_rock, (x, terrain_h(x, z), z),
             ry=rng.uniform(0, 2*math.pi), s=rng.uniform(0.5, 2.2))

camera_pos = [0.0, terrain_h(0, -SIZE/2 + 10) + 2.0, -SIZE/2 + 10]

# ---------------------------------------------------------------- GLB 출력
def align4(n): return (n + 3) & ~3

def build_glb():
    buffers_bin = bytearray()
    accessors, buffer_views, gltf_meshes, gltf_nodes, materials = [], [], [], [], []

    def add_bv(data, target):
        off = len(buffers_bin)
        buffers_bin.extend(data)
        buffers_bin.extend(b'\x00' * (align4(len(data)) - len(data)))
        buffer_views.append({"buffer": 0, "byteOffset": off, "byteLength": len(data), "target": target})
        return len(buffer_views) - 1

    def add_acc(bv, count, ctype, atype, bmin=None, bmax=None):
        acc = {"bufferView": bv, "byteOffset": 0, "componentType": ctype, "count": count, "type": atype}
        if bmin: acc["min"] = bmin
        if bmax: acc["max"] = bmax
        accessors.append(acc)
        return len(accessors) - 1

    for md in meshes:
        m = md['mesh']
        flat_p = [v for p in m.pos for v in p]
        flat_n = [v for nn in m.nrm for v in nn]
        idx_fmt, idx_ct = ('<%dI' % len(m.idx), 5125) if len(m.pos) > 65000 else ('<%dH' % len(m.idx), 5123)
        pos_b = struct.pack(f'<{len(flat_p)}f', *flat_p)
        nrm_b = struct.pack(f'<{len(flat_n)}f', *flat_n)
        idx_b = struct.pack(idx_fmt, *m.idx)
        xs = flat_p[0::3]; ys = flat_p[1::3]; zs = flat_p[2::3]
        acc_p = add_acc(add_bv(pos_b, 34962), len(m.pos), 5126, "VEC3",
                        [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)])
        acc_n = add_acc(add_bv(nrm_b, 34962), len(m.pos), 5126, "VEC3")
        acc_i = add_acc(add_bv(idx_b, 34963), len(m.idx), idx_ct, "SCALAR")
        r, g, b = md['color']
        materials.append({"name": md['name'],
                          "pbrMetallicRoughness": {"baseColorFactor": [r, g, b, 1.0],
                                                   "metallicFactor": md['metal'],
                                                   "roughnessFactor": md['rough']}})
        gltf_meshes.append({"name": md['name'],
                            "primitives": [{"attributes": {"POSITION": acc_p, "NORMAL": acc_n},
                                            "indices": acc_i, "material": len(materials)-1, "mode": 4}]})

    for nd in nodes:
        node = {"name": nd['name'], "mesh": nd['mesh'], "translation": list(nd['t'])}
        if nd['s'] != 1.0: node["scale"] = [nd['s']] * 3
        if nd['ry'] != 0.0:
            h = nd['ry'] / 2.0
            node["rotation"] = [0.0, math.sin(h), 0.0, math.cos(h)]
        gltf_nodes.append(node)

    # 카메라 + 방향광 (KHR_lights_punctual)
    gltf_nodes.append({"name": "Camera", "camera": 0, "translation": camera_pos})
    ang = math.radians(-115.0) / 2.0
    gltf_nodes.append({"name": "Sun", "translation": [0.0, 40.0, 0.0],
                       "rotation": [math.sin(ang), 0.0, 0.0, math.cos(ang)],
                       "extensions": {"KHR_lights_punctual": {"light": 0}}})

    gltf = {
        "asset": {"version": "2.0", "generator": "make_forest_glb.py"},
        "extensionsUsed": ["KHR_lights_punctual"],
        "extensions": {"KHR_lights_punctual": {"lights": [
            {"name": "Sun", "type": "directional", "color": [1.0, 0.96, 0.88], "intensity": 1.3}]}},
        "scene": 0,
        "scenes": [{"name": "PineForest", "nodes": list(range(len(gltf_nodes)))}],
        "nodes": gltf_nodes, "meshes": gltf_meshes, "materials": materials,
        "cameras": [{"type": "perspective", "perspective": {"yfov": 1.0472, "znear": 0.1}}],
        "accessors": accessors, "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(buffers_bin)}],
    }
    j = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    jc = j + b' ' * (align4(len(j)) - len(j))
    bc = bytes(buffers_bin) + b'\x00' * (align4(len(buffers_bin)) - len(buffers_bin))
    total = 12 + 8 + len(jc) + 8 + len(bc)
    return (struct.pack('<III', 0x46546C67, 2, total)
            + struct.pack('<II', len(jc), 0x4E4F534A) + jc
            + struct.pack('<II', len(bc), 0x004E4942) + bc)

os.makedirs('maps/map', exist_ok=True)
glb = build_glb()
out = 'maps/map/4.glb'
with open(out, 'wb') as f:
    f.write(glb)

tris = sum(len(m['mesh'].idx) for m in meshes) // 3
print(f"Generated {out}  ({len(glb)/1024:.0f} KB)")
print(f"  shared meshes: {len(meshes)}  nodes: {len(nodes)}  unique tris: {tris:,}")
