"""One step of the fly in free fall: whose M, whose RHS, whose damping?"""
import sys, numpy as np
sys.path.insert(0, 'tools')
import mujoco as mj, fly_aba as fa

body = fa.FlyBody('build/fly_body.json')
mjcf = 'data/flybody/flybody-main/flybody/fruitfly/assets/fruitfly.xml'
model = mj.MjModel.from_xml_path(mjcf)
sim = fa.FlySim(body, floor_z=-1e6, limits=False)
sim.dt = float(model.opt.timestep)
rng = np.random.default_rng(12345)
q = rng.uniform(-0.3, 0.3, body.nj); qd = rng.uniform(-0.5, 0.5, body.nj)
sim.reset(root_z=0.5, q=q, qd=qd, v=[1.0, -0.5, 0.2], omega=[0.5, 1.0, -0.5])
m, d = fa._mujoco_pair(body, sim, mjcf)
dt = sim.dt
M = np.zeros((m.nv, m.nv)); mj.mj_fullM(m, d, M)
off = m.nv - body.nj
r = np.array(d.qfrc_smooth)[off:]
b = m.dof_damping[off:]
cand = np.linalg.solve(M[off:, off:] + dt * np.diag(b), r)
qdd_mine = sim.step(dt, np.zeros(body.nj))
d2 = mj.MjData(m); d2.qpos, d2.qvel = d.qpos.copy(), d.qvel.copy()
mj.mj_step(m, d2)
acc_mj = (d2.qvel - d.qvel)[off:] / dt
names = body.joint_name
err = np.abs(qdd_mine - cand)
print(f"|qdd_mine - cand(M,qfrc)|max {err.max():.3e} at {names[int(np.argmax(err))]}")
print(f"|qdd_mine - mj_step  |max {np.abs(qdd_mine - acc_mj).max():.3e} at "
      f"{names[int(np.argmax(np.abs(qdd_mine - acc_mj)))]}")
print(f"|cand - mj_step      |max {np.abs(cand - acc_mj).max():.3e}")
i = int(np.argmax(err))
print(f"worst joint {names[i]}: mine {qdd_mine[i]:+.6e} cand {cand[i]:+.6e} mj {acc_mj[i]:+.6e}")
print(f"  D_eff (mine) ~ {1/((qdd_mine[i] - 0) + 1e-30):.3e}; M_ii {M[off+i, off+i]:.3e} "
      f"b {b[i]:.3e} armature {body.armature[i]:.3e} damping {body.damping[i]:.3e} "
      f"stiffness {body.stiffness[i]:.3e} spring_ref {body.spring_ref[i]:+.4f} q {q[i]:+.4f}")
print(f"  qfrc_smooth[i] {r[i]:.6e}; explicit damping term {-b[i]*qd[i]:.3e}; "
      f"spring {-body.stiffness[i]*(q[i]-body.spring_ref[i]):.3e}")
