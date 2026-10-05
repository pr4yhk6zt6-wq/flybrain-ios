"""Contact damping against the explicit stability limit at the lightest limb."""
import sys, numpy as np
sys.path.insert(0, 'tools')
import fly_aba as fa
body = fa.FlyBody('build/fly_body.json')
base = fa.hold_excitation(body)
inertia = np.where(body.armature > 0, body.armature, 1e-6)
cap = np.where(body.max_torque > 0, body.max_torque, 1.0)

def trial(k, b, dt, ms=300.0):
    fa.CONTACT_STIFFNESS = k
    fa.CONTACT_DAMPING = b
    gp = 0.1 ** 2 * inertia / (dt ** 2 * cap); gd = 0.5 * inertia / (dt * cap)
    sim = fa.FlySim(body); sim.reset(); sim.dt = dt
    worst = 0.0; ok = True
    for i in range(int(ms / 1000.0 / dt)):
        ramp = min(1.0, (i + 1) / max(1, int(0.050 / dt)))
        exc = np.clip(ramp * base + gp * (body.stance_q - sim.q) + gd * (0 - sim.qd), -1, 1)
        qdd = sim.step(dt, fa.muscle_torque(body, sim.q, sim.qd, exc))
        worst = max(worst, float(np.abs(sim.qd).max()))
        if not np.isfinite(qdd).all():
            ok = False; break
    com = sim.com()
    print(f"k {k:6.1f} b {b:7.5f} dt {dt:.0e}: {'ran' if ok else 'DIED %6.1f ms' % (i*dt*1000)}  "
          f"worst|qd| {worst:9.3e}  com_z {com[2]:+.5f}  support {sum(sim.foot_force.values()):.3f}  "
          f"|q-stance| {np.abs(sim.q-body.stance_q).max():.4f}")
    return worst

import itertools
for dt, b in itertools.product((1e-4,), (0.5, 0.1, 0.02, 0.005, 0.0002, 0.0)):
    trial(150.0, b, dt, ms=100.0)
