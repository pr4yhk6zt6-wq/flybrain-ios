"""Is it the floor, the muscles, or the pair?"""
import sys, numpy as np
sys.path.insert(0, 'tools')
import fly_aba as fa

body = fa.FlyBody('build/fly_body.json')
base = fa.hold_excitation(body)

def run(tag, floor, muscles, ms=1500.0, band=0.05):
    sim = fa.FlySim(body, floor_z=floor)
    sim.reset(); sim.dt = body.dt_default
    ramp_steps = max(1, int(0.050 / sim.dt))
    steps = int(ms / 1000.0 / sim.dt)
    worst = 0.0
    for k in range(steps):
        if muscles:
            ramp = min(1.0, (k + 1) / ramp_steps)
            exc = np.clip(ramp * base + (body.stance_q - sim.q) / band, -1, 1)
            tau = fa.muscle_torque(body, sim.q, sim.qd, exc)
        else:
            tau = np.zeros(body.nj)
        qdd = sim.step(sim.dt, tau)
        worst = max(worst, float(np.abs(sim.qd).max()))
        if not np.isfinite(qdd).all():
            break
    print(f"{tag:28s} {'ran' if np.isfinite(qdd).all() else 'NAN at %.1f ms' % (k*sim.dt*1000)}  "
          f"worst|qd| {worst:9.3e}  com {np.round(sim.com(), 4)}  contacts {sim.contacts}")
    return worst

run("floor + muscles", None, True)
run("floor, no muscles", None, False)
run("no floor + muscles", -1e6, True, ms=200.0)
run("no floor, no muscles", -1e6, False, ms=200.0)
