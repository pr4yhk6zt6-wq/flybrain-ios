"""Penalty floor in isolation: a sphere, a capsule and a box dropped on it."""
import sys, numpy as np, pathlib, json, tempfile
sys.path.insert(0, 'tools')
import mujoco as mj, fly_aba as fa, verify_aba as va

XML = """
<mujoco><compiler angle="radian"/><option timestep="1e-4" gravity="0 0 -981"/>
<worldbody>
  <geom name="floor" type="plane" size="5 5 .1" pos="0 0 -0.132"/>
  <body name="ball" pos="0 0 0.2">
    <freejoint/>
    <geom type="sphere" size="0.05" mass="0.001"/>
  </body>
</worldbody></mujoco>
"""
XML_CAP = XML.replace('<geom type="sphere" size="0.05" mass="0.001"/>',
                      '<geom type="capsule" size="0.02 0.04" mass="0.001" quat="0.7071 0 0.7071 0"/>')

for tag, xml in (("sphere", XML), ("tilted capsule", XML_CAP)):
    tmpdir = pathlib.Path(tempfile.mkdtemp())
    path = tmpdir / "m.xml"; path.write_text(xml)
    model = mj.MjModel.from_xml_path(str(path))
    raw = json.loads(json.dumps(va._mjcf_asset(model, path)))
    asset = tmpdir / "m.json"; asset.write_text(json.dumps(raw))
    body = fa.FlyBody(asset)
    sim = fa.FlySim(body, floor_z=-0.132)
    sim.reset(root_z=0.2, q=None)
    sim.q = np.zeros(body.nj)          # nothing to hinge
    sim._kinematics()
    sim.dt = float(model.opt.timestep)
    d = mj.MjData(model)
    d.qpos[2] = 0.2
    mj.mj_forward(model, d)
    print(f"== {tag}: free mass {body.body_mass.sum():.4f}, weight "
          f"{body.body_mass.sum()*981:.4f}")
    for k in range(6000):
        sim.step(sim.dt, np.zeros(body.nj))
        mj.mj_step(model, d)
        if k in (0, 100, 500, 1000, 3000, 5999):
            print(f"   {k*1e4:6.2f} ms  solver z {sim.root_pos[2]:+.6f} vel {sim.vel[2]:+.5f} "
                  f"contacts {sim.contacts} | mujoco z {d.qpos[2]:+.6f} vel {d.qvel[2]:+.5f} ncon {d.ncon}")
        if not np.isfinite(sim.root_pos).all():
            print("   solver diverged"); break
