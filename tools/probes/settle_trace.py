import sys, numpy as np
sys.path.insert(0, 'tools')
import fly_aba as fa
body = fa.FlyBody('build/fly_body.json')
base = fa.hold_excitation(body)
names = body.joint_name
dt = body.dt_default
damp0 = body.damping.copy()
inertia = np.where(body.armature > 0, body.armature, 1e-6)
cap = np.where(body.max_torque > 0, body.max_torque, 1.0)
gp = 0.1 ** 2 * inertia / (dt ** 2 * cap); gd = inertia / (dt * cap)
sim = fa.FlySim(body); sim.reset(); sim.dt = dt
for k in range(2000):
    body.damping = damp0 + 0.05
    ramp = min(1.0, (k + 1) / 500)
    exc = np.clip(ramp * base + gp * (body.stance_q - sim.q) + gd * (0 - sim.qd), -1, 1)
    tau = fa.muscle_torque(body, sim.q, sim.qd, exc)
    qdd = sim.step(dt, tau)
    if k % 20 == 0 or not np.isfinite(qdd).all():
        i = int(np.argmax(np.abs(sim.qd)))
        print(f"{k:5d} {k*dt*1000:6.2f} ms |qd|max {np.abs(sim.qd).max():9.3e} at {names[i]:18s} "
              f"|qdd|max {np.abs(qdd).max():9.3e} |exc|max {np.abs(exc).max():5.2f} "
              f"contacts {sim.contacts:2d} support {sum(sim.foot_force.values()):9.3f} "
              f"com_z {sim.com()[2]:+.5f} root_z {sim.root_pos[2]:+.5f} "
              f"|q-stance| {np.abs(sim.q-body.stance_q).max():.4f}")
        if not np.isfinite(qdd).all():
            print("NaN; q =", sim.q[:8], "qd =", sim.qd[:8])
            break
