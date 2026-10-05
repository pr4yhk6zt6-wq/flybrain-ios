"""Which damping discretisation does MuJoCo's Euler step actually use?

Two candidates that agree on one DOF and differ as soon as joints are coupled:
  A: fold dt*b into the diagonal of M, then solve  (what the solver does)
  B: solve with M, then divide each qacc by (1 + dt*b/M_ii)
"""
import sys, pathlib, json, tempfile, numpy as np
sys.path.insert(0, 'tools')
import mujoco as mj, fly_aba as fa, verify_aba as va

XML = """
<mujoco><compiler angle="radian"/><option timestep="1e-4" gravity="0 0 -981"/>
<worldbody>
  <body name="a" pos="0 0 0"><geom type="capsule" size="0.01 0.02"/>
    <joint name="ja" type="hinge" axis="0 1 0" damping="0.01" armature="0.0"
           stiffness="0.0" limited="false"/>
    <body name="b" pos="0.04 0 0"><geom type="capsule" size="0.01 0.02"/>
      <joint name="jb" type="hinge" axis="0 1 0" damping="0.01" armature="0.0"
             stiffness="0.0" limited="false"/>
    </body>
  </body>
</worldbody></mujoco>
"""
tmp = pathlib.Path(tempfile.mkdtemp()); path = tmp / "c.xml"; path.write_text(XML)
model = mj.MjModel.from_xml_path(str(path))
raw = json.loads(json.dumps(va._mjcf_asset(model, path)))
asset = tmp / "c.json"; asset.write_text(json.dumps(raw))
body = fa.FlyBody(asset)

rng = np.random.default_rng(5)
q = rng.uniform(-0.5, 0.5, body.nj); qd = np.array([3.0, -2.0])
d = mj.MjData(model)
for j, jj in enumerate(body.hinges):
    d.qpos[jj["qposadr"]] = q[j]; d.qvel[jj["dofadr"]] = qd[j]
mj.mj_forward(model, d)
M = np.zeros((model.nv, model.nv)); mj.mj_fullM(model, d, M)
r = np.array(d.qfrc_smooth)
b = model.dof_damping.copy()
dt = model.opt.timestep
A = np.linalg.solve(M + dt * np.diag(b), r)
B = np.linalg.solve(M, r) / (1 + dt * b / np.diag(M))

d2 = mj.MjData(model); d2.qpos, d2.qvel = d.qpos.copy(), d.qvel.copy()
mj.mj_step(model, d2)
acc = (d2.qvel - d.qvel) / dt
print(f"qfrc_smooth {r}  |qfrc| {np.abs(r).max():.4f}")
for tag, cand in (("A fold into M", A), ("B post-divide", B)):
    print(f"  {tag}: {cand}   |vs MuJoCo step| {np.abs(cand - acc).max():.3e}")

sim = fa.FlySim(body); sim.reset(q=q, qd=qd); sim.dt = dt
sim.step(dt, np.zeros(body.nj))
print(f"  solver:        {sim.qd}   |vs MuJoCo step| {np.abs(sim.qd - d2.qvel).max():.3e}")
