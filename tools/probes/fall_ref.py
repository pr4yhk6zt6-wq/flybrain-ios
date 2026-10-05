"""How reproducible is MuJoCo's own step from its own published M and qfrc?"""
import sys, numpy as np
sys.path.insert(0, 'tools')
import mujoco as mj, fly_aba as fa

body = fa.FlyBody('build/fly_body.json')
mjcf = 'data/flybody/flybody-main/flybody/fruitfly/assets/fruitfly.xml'
sim = fa.FlySim(body, floor_z=-1e6, limits=False)
sim.dt = body.dt_default
rng = np.random.default_rng(5)
sim.reset(root_z=0.0, q=rng.uniform(-0.3, 0.3, body.nj),
          qd=rng.uniform(-1, 1, body.nj),
          v=np.array([1.0, -0.5, 0.25]), omega=np.array([3.0, -2.0, 5.0]))
m, d = fa._mujoco_pair(body, sim, mjcf)
off = m.nv - body.nj
M = np.zeros((m.nv, m.nv)); mj.mj_fullM(m, d, M)
Ms = M[off:, off:]
r = np.array(d.qfrc_smooth)[off:]
dt = sim.dt
acc_exp = np.linalg.solve(Ms, r)
acc_imp = np.linalg.solve(Ms + dt * np.diag(m.dof_damping[off:]), r)
d2 = mj.MjData(m); d2.qpos, d2.qvel = d.qpos.copy(), d.qvel.copy()
mj.mj_step(m, d2)
acc_mj = (d2.qvel - d.qvel)[off:] / dt
names = body.joint_name
scale = np.abs(acc_mj).max()
print(f"cond(M_joints) {np.linalg.cond(Ms):.3e}, |qacc_mj|max {scale:.3e}")
for tag, cand in (("explicit (M, qfrc_smooth)", acc_exp),
                  ("implicit (M+dt*D, qfrc_smooth)", acc_imp)):
    d_ = np.abs(cand - acc_mj); i = int(np.argmax(d_))
    print(f"  MuJoCo step vs {tag:32s}: max abs {d_.max():.3e} at {names[i]:18s} "
          f"rel {d_.max()/abs(acc_mj[i]):.2e}   torque residual {np.abs(Ms@acc_mj-r).max():.3e} "
          f"vs |qfrc|max {np.abs(r).max():.3e}")
qdd = sim.step(dt, np.zeros(body.nj))
d_ = np.abs(qdd - acc_mj); i = int(np.argmax(d_))
print(f"  this solver vs MuJoCo's step:      max abs {d_.max():.3e} at {names[i]:18s} "
      f"rel {d_.max()/abs(acc_mj[i]):.2e}; median rel "
      f"{np.median(d_/np.maximum(np.abs(acc_mj),1e-300)):.2e}")
print(f"  |qdd|max {np.abs(qdd).max():.3e}; min |M_ii| {np.abs(np.diag(Ms)).min():.3e}; "
      f"max |M_ii| {np.abs(np.diag(Ms)).max():.3e}")
