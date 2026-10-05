"""Standing, with servo gains derived from each joint's inertia and torque.

A joint's own armature sets how hard it can be pushed at this step size:
the servo has to be slower than omega*dt = 0.3 and no more damped than
dt*kd/D = 1, or the servo itself is the instability.
"""
import sys, numpy as np
sys.path.insert(0, 'tools')
import fly_aba as fa

body = fa.FlyBody('build/fly_body.json')
base = fa.hold_excitation(body)
names = body.joint_name
dt = body.dt_default
inertia = np.where(body.armature > 0, body.armature, 1e-6)
cap = np.where(body.max_torque > 0, body.max_torque, 1.0)

for omega_dt, kd_dt in ((0.3, 1.0), (0.1, 1.0), (0.3, 0.3), (0.03, 1.0)):
    gp = omega_dt ** 2 * inertia / (dt ** 2 * cap)
    gd = kd_dt * inertia / (dt * cap)
    sim = fa.FlySim(body); sim.reset(); sim.dt = dt
    ramp_steps = max(1, int(0.050 / dt))
    steps = int(1500 / 1000.0 / dt)
    worst = 0.0; ok = True; trace = []
    for k in range(steps):
        ramp = min(1.0, (k + 1) / ramp_steps)
        exc = np.clip(ramp * base + gp * (body.stance_q - sim.q) + gd * (0 - sim.qd), -1, 1)
        qdd = sim.step(dt, fa.muscle_torque(body, sim.q, sim.qd, exc))
        worst = max(worst, float(np.abs(sim.qd).max()))
        if k % 1000 == 0:
            trace.append((round(k*dt*1000), round(float(sim.com()[2]), 5),
                          round(sum(sim.foot_force.values()), 3),
                          int(sum(1 for v in sim.foot_force.values() if v > 0)),
                          round(float(np.abs(sim.q - body.stance_q).max()), 4)))
        if not np.isfinite(qdd).all() or worst > 300:
            ok = False; break
    com = sim.com()
    print(f"omega*dt {omega_dt:4.2f} kd*dt {kd_dt:4.2f}: {'stood' if ok else 'DIED %6.1f ms' % (k*dt*1000)}"
          f"  worst|qd| {worst:8.2e}  com_z {com[2]:+.5f}  drift {np.linalg.norm(com[:2])*10:6.3f} mm "
          f" support {sum(sim.foot_force.values()):.3f}")
    print("      ms, com_z, support, feet, |q-stance|max:", trace[:8])
