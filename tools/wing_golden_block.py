"""Write the golden block into WingAero.swift, generated from the tool's own
output — the numbers the Swift port is pinned to are never typed by hand."""
import json
import pathlib
import re

golden = json.loads(pathlib.Path("/home/user/flybrain-ios/build/wing_golden.json").read_text())
g = golden["wing"]
law = golden["law"]

fmt = lambda v: f"{v!r}"


def line(*vals):
    return " ".join(f"{float(v):.15g}" for v in vals)


wing_rows = [line(1, g["area_cm2"], g["span_cm"], g["length_cm"], g["chord_cm"],
                  g["second_moment_cm4"], g["r2_hat_cm"], g["mass_ug"], g["com_cm"],
                  g["sin_theta"], golden["attitude_deg"], golden["weight_dyn"])]
wing_rows += [" ".join(f"{v:.15g}" for v in g[k]) for k in
              ("stroke_axis", "span_axis", "lift_dir", "hinge_cm")]
strip_rows = [line(r["omega_rad_s"], r["alpha_deg"], r["cl"], r["cd"],
                   r["lift_dyn"], r["drag_dyn"]) for r in golden["strip_force"]]
beat_rows = [line(b["activation_left"], b["activation_right"],
                  b["amplitude_left_deg"], b["amplitude_right_deg"],
                  b["lift_left_dyn"], b["lift_right_dyn"], b["flight_force_dyn"],
                  b["thrust_dyn"], b["vertical_dyn"], b["side_dyn"],
                  b["yaw_moment_dyn_cm"], b["roll_moment_dyn_cm"],
                  b["pitch_moment_dyn_cm"], b["drag_cost_dyn"]) for b in golden["beat"]]
station_rows = [line(s["r_cm"], s["chord_cm"], s["area_cm2"]) for s in g["stations"]]

block = f'''    // MARK: - the golden table
    //
    // Written by `python3 tools/wing_aero.py --golden build/wing_golden.json`
    // and pasted here by `tools/wing_golden_block.py`, so that these numbers are
    // the tool's own and not typed by hand. `WingAeroTests` parses them and
    // checks this port against them; CI compares the same rows against a fresh
    // run of the tool, so a change to either side fails the build.

    /// `area span length chord ∫r²dA r̂₂ mass com sinθ attitude weight`, then the
    /// stroke axis, the span axis, the lift direction and the hinge, each on its
    /// own line — the wing's measured geometry, in the body frame.
    static let goldenWing = """
{chr(10).join(wing_rows)}
"""

    /// The strips: `r chord area`, along the span.
    static let goldenStations = """
{chr(10).join(station_rows)}
"""

    /// The force law at a strip: `ω α CL CD lift drag`.
    static let goldenStrip = """
{chr(10).join(strip_rows)}
"""

    /// Whole beats, driven by each wing's own activation:
    /// `aL aR ampL ampR liftL liftR flight thrust vertical side yaw roll pitch drag`.
    static let goldenBeat = """
{chr(10).join(beat_rows)}
"""

    /// The law the beats were made with, and the two pools' measured rates.
    static let goldenLaw = """
{line(law["stroke_hz"], law["amplitude_full_deg"], law["alpha_mid_deg"], law["alpha_rot_deg"], law["rho_g_cm3"], law["body_mass_ug"])}
"""

'''

p = pathlib.Path("/home/user/flybrain-ios/FlyBrain/Sources/WingAero.swift")
t = p.read_text()
marker = "    // MARK: - the golden table"
if marker in t:
    t = t[:t.index(marker)]
# put the block inside the type, just before the closing brace of `struct WingAero`
idx = t.rindex("}\n")
t = t[:idx] + block + t[idx:]
p.write_text(t)
print(f"golden block written: {len(wing_rows)} wing lines, {len(station_rows)} stations, "
      f"{len(strip_rows)} strips, {len(beat_rows)} beats")
