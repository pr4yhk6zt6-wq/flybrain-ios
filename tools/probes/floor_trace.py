"""Floor only, no muscles: what grows, and when."""
import sys, numpy as np
sys.path.insert(0, 'tools')
import fly_aba as fa

body = fa.FlyBody('build/fly_body.json')
sim = fa.FlySim(body); sim.reset(); sim.dt = body.dt_default
names = body.joint_name
for k in range(500):
    qdd = sim.step(sim.dt, np.zeros(body.nj))
    if k % 20 == 0 or not np.isfinite(qdd).all():
        i = int(np.argmax(np.abs(sim.qd)))
        f = sum(sim.foot_force.values())
        print(f"{k*sim.dt*1000:6.2f} ms |qd|max {np.abs(sim.qd).max():9.3e} at {names[i]:18s} "
              f"|qdd|max {np.abs(qdd).max():9.3e} contacts {sim.contacts:2d} normal {f:9.4f} "
              f"root_z {sim.root_pos[2]:+.5f} root_qd {np.linalg.norm(sim.vel):.3e} "
              f"omega {np.linalg.norm(sim.omega):.3e}")
        if np.abs(sim.qd).max() > 200:
            print("   -> runaway; q =", np.round(sim.q[i], 4), "D_eff ~",
                  body.armature[i], "stiffness", body.stiffness[i])
            break
