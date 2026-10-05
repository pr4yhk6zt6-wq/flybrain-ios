"""Every collision geom's lowest point against MuJoCo's own plane distance."""
import sys, numpy as np
sys.path.insert(0, 'tools')
import mujoco as mj
import fly_aba as fa

body = fa.FlyBody('build/fly_body.json')
TYPES = {mj.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid", mj.mjtGeom.mjGEOM_CAPSULE: "capsule",
         mj.mjtGeom.mjGEOM_SPHERE: "sphere", mj.mjtGeom.mjGEOM_BOX: "box",
         mj.mjtGeom.mjGEOM_CYLINDER: "cylinder", mj.mjtGeom.mjGEOM_MESH: "mesh"}
model = mj.MjModel.from_xml_path('data/flybody/flybody-main/flybody/fruitfly/assets/floor.xml')
mid = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g): g for g in range(model.ngeom)}
missing = [g["name"] for g in body.raw["collision"] if g["name"] not in mid]
print(f"{len(body.raw['collision'])} collision geoms, {len(missing)} unnamed in MuJoCo: {missing[:5]}")

def support(kind, sz, R):
    up = np.array([0.0, 0.0, 1.0])
    if kind == "sphere":
        return sz[0]
    if kind == "capsule":
        return sz[0] + sz[1] * abs(R[2, 2])
    if kind == "ellipsoid":
        return float(np.linalg.norm(np.asarray(sz[:3]) * R[2, :]))
    if kind == "cylinder":
        a = abs(R[2, 2]); return sz[0] * np.sqrt(max(0.0, 1 - a * a)) + sz[1] * a
    if kind == "box":
        return float(np.abs(R[2, :]) @ np.asarray(sz[:3]))
    return None

rows = []
for i, g in enumerate(body.raw["collision"]):
    if g["name"] not in mid:
        continue
    gid = mid[g["name"]]
    kind_mj = TYPES[int(model.geom_type[gid])]
    if kind_mj != g["type"]:
        print(f"  {g['name']}: mujoco says {kind_mj}, json says {g['type']}")
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    model.geom_contype[gid] = 1
    for p in range(model.ngeom):
        if int(model.geom_type[p]) == mj.mjtGeom.mjGEOM_PLANE:
            model.geom_conaffinity[p] = 1
    d = mj.MjData(model)
    d.qpos[0:3] = [0.0, 0.0, 0.0]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:] = body.stance_q
    lo, hi = body.floor_z - 0.5, body.floor_z + 0.5
    for _ in range(45):
        d.qpos[2] = 0.5 * (lo + hi)
        mj.mj_forward(model, d)
        if d.ncon > 0:
            lo = d.qpos[2]
        else:
            hi = d.qpos[2]
    mj_touch = 0.5 * (lo + hi)
    sim = fa.FlySim(body); sim.reset(); sim._kinematics()
    h = support(g["type"], g["size"], sim.grot[i])
    mine_touch = body.floor_z - (sim.gpos[i][2] - h)
    rows.append((abs(mine_touch - mj_touch), i, g["type"], g["body"], mine_touch, mj_touch))
rows.sort(reverse=True)
print(f"worst {min(8, len(rows))} of {len(rows)} geoms:")
for diff, i, kind, b, a, c in rows[:8]:
    print(f"   geom {i:3d} {kind:9s} {b:22s} mine {a:+.6f} mujoco {c:+.6f}  diff {diff*1e4:+8.2f} um")
ok = sum(1 for r in rows if r[0] <= 2e-5)
print(f"within 0.2 um: {ok} of {len(rows)}")
