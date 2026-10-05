"""How much can the golden trace move if the arithmetic differs by one ulp?"""
import sys, json, math, numpy as np
sys.path.insert(0, 'tools')
import fly_aba as fa

body = fa.FlyBody('build/fly_body.json')
dt = body.dt_default
rng = np.random.default_rng(3)
q0 = rng.uniform(-0.3, 0.3, body.nj); qd0 = rng.uniform(-1, 1, body.nj)
amp = np.array(list(0.3 * np.sin(np.arange(body.nj) * 0.7)))

def run(eps=0.0, solve=None, steps=500):
    sim = fa.FlySim(body, floor_z=-1e6)
    sim.reset(root_z=0.0, q=q0, qd=qd0, v=[0.5, 0.0, -0.25], omega=[2.0, 1.0, -3.0])
    if solve is not None:
        sim._solve = solve
    for k in range(steps):
        tau = amp * math.cos(2 * math.pi * k * dt / 0.02) * (1 + eps)
        sim.step(dt, tau)
    return sim

def perturbed(M, b):
    """A different-but-equivalent solve: LU of a copy."""
    lu, piv = __import__("scipy.linalg", fromlist=["lu"]).lu(M)
    from scipy.linalg import lu_solve
    return lu_solve((lu, piv), b)

base = run()
for tag, other in (("torque +1 ulp", run(eps=2.2e-16)),
                   ("torque -1 ulp", run(eps=-2.2e-16))):
    print(f"  {tag:16s}: |dq|max {np.abs(other.q-base.q).max():.3e}  "
          f"|dqd|max {np.abs(other.qd-base.qd).max():.3e}  "
          f"|droot| {np.abs(other.root_pos-base.root_pos).max():.3e}")
try:
    scipy_run = run(solve=perturbed)
    print(f"  {'LU solve':16s}: |dq|max {np.abs(scipy_run.q-base.q).max():.3e}  "
          f"|dqd|max {np.abs(scipy_run.qd-base.qd).max():.3e}")
except ImportError:
    print("  (scipy missing, skipped the LU row)")
g = json.load(open('build/fly_golden.json'))
print(f"  golden expects |q|max {np.abs(g['expected_final']['q']).max():.3f}, "
      f"tolerance {g['tolerance']:.0e}, steps {g['steps']}")
