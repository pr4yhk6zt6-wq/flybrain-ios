"""Same harness, three step sizes: is the kick an integration artifact or physics?"""
import sys, numpy as np
sys.path.insert(0, 'tools')
import fly_aba as fa
body = fa.FlyBody('build/fly_body.json')
base = fa.hold_excitation(body)
names = body.joint_name
inertia = np.where(body.armature > 0, body.armature, 1e-6)
cap = np.where(body.max_torque > 0, body.max_torque, 1.0)

def trial(tag, dt, ms, small_ms=0.0, small_dt=1e-5):
    gp = 0.1 ** 2 * inertia / (dt ** 2 * cap); gd = inertia / (dt * cap)
    gp_s = 0.1 ** 2 * inertia / (small_dt ** 2 * cap); gd_s = inertia / (small_dt * cap)
    sim = fa.FlySim(body); sim.reset(); sim.dt = dt
    worst = 0.0; ok = True
    steps_small = int(small_ms / 1000.0 / small_dt)
    steps = steps_small + int(ms / 1000.0 / dt)
    for k in range(steps):
        small = k < steps_small
        h = small_dt if small else dt
        g_p = gp_s if small else gp
        g_d = gd_s if small else gd
        ramp = min(1.0, (k + 1) / max(1, int(0.050 / h)))
        exc = np.clip(ramp * base + g_p * (body.stance_q - sim.q) + g_d * (0 - sim.qd), -1, 1)
        qdd = sim.step(h, fa.muscle_torque(body, sim.q, sim.qd, exc))
        if not small:
            worst = max(worst, float(np.abs(sim.qd).max()))
        if not np.isfinite(qdd).all():
            ok = False
            print(f"{tag}: NaN at {k*small_dt*1000 if small else (k-steps_small)*dt*1000:.2f} ms, "
                  f"|q-stance| {np.abs(sim.q-body.stance_q).max():.3f}")
            break
    com = sim.com()
    print(f"{tag}: {'ran' if ok else 'DIED'} worst|qd| {worst:9.3e} com_z {com[2]:+.5f} "
          f"support {sum(sim.foot_force.values()):.3f} |q-stance| {np.abs(sim.q-body.stance_q).max():.4f}")

trial("dt 1e-4, 20 ms", 1e-4, 20.0)
trial("dt 1e-5, 20 ms", 1e-5, 20.0)
trial("dt 3e-6, 20 ms", 3e-6, 20.0)
trial("1e-5 for 20 ms then 1e-4 for 500 ms", 1e-4, 500.0, small_ms=20.0)
