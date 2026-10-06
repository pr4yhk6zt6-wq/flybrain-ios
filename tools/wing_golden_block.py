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
# Both wings' vectors, the right wing's measured rather than mirrored (see the
# tool: the meshes are mirrors only to ~2e-6).
wing_rows += [" ".join(f"{v:.15g}" for v in g[k]) for k in
              ("stroke_axis", "span_axis", "lift_dir", "hinge_cm")]
wing_rows += [" ".join(f"{v:.15g}" for v in g["right"][k]) for k in
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
station_rows_r = [line(s["r_cm"], s["chord_cm"], s["area_cm2"])
                  for s in g["right"]["stations"]]

block = f'''    // MARK: - the golden table
    //
    // Written by `python3 tools/wing_aero.py --golden build/wing_golden.json`
    // and pasted here by `tools/wing_golden_block.py`, so that these numbers are
    // the tool's own and not typed by hand. `WingAeroTests` parses them and
    // checks this port against them; CI compares the same rows against a fresh
    // run of the tool, so a change to either side fails the build.

    /// `area span length chord ∫r²dA r̂₂ mass com sinθ attitude weight`, then the
    /// stroke axis, the span axis, the lift direction and the hinge of the left
    /// wing and then of the right one — each on its own line. The wing's
    /// measured geometry, in the body frame; the right wing's rows are measured,
    /// not mirrored.
    static let goldenWing = """
{chr(10).join(wing_rows)}
"""

    /// The two wings' four vectors each, parsed out of the literal above, so
    /// `Params` can default to the tool's own numbers instead of to copies of
    /// them: order is stroke, span, lift, hinge, left wing then right.
    static var goldenWingVectors: [SIMD3<Double>] {{
        goldenWing.split(separator: "\\n").compactMap {{ line in
            let v = line.split(separator: " ").compactMap {{ Double($0) }}
            return v.count == 3 ? SIMD3(v[0], v[1], v[2]) : nil
        }}
    }}

    /// The strips: `r chord area`, along the span.
    static let goldenStations = """
{chr(10).join(station_rows)}
"""

    /// The same strips as the table the port integrates, parsed out of the
    /// literal above so the numbers exist in one copy. A `WingAero()` with no
    /// spec uses these; before this existed it had *no strips at all*, so every
    /// force came out zero and the tests said so ("the beat costs drag", 0.0).
    /// A port that silently makes no force is worse than one that crashes,
    /// because the fly then simply does not fly and nothing says why.
    static var defaultStations: [(r: Double, chord: Double, area: Double)] {{
        goldenStations.split(separator: "\\n").compactMap {{ line in
            let v = line.split(separator: " ").compactMap {{ Double($0) }}
            return v.count >= 3 ? (v[0], v[1], v[2]) : nil
        }}
    }}

    /// The right wing's strips. The two wings in the body model are *not*
    /// identical — their areas differ by 6.6e-05 — so each wing is integrated
    /// with its own table; sharing one put a 3e-05 error into every right-wing
    /// force in the port, which the golden table caught.
    static let goldenStationsRight = """
{chr(10).join(station_rows_r)}
"""

    /// Both tables as the port integrates them.
    static var defaultStations: [(r: Double, chord: Double, area: Double)] {{
        parseStations(goldenStations)
    }}
    static var defaultStationsRight: [(r: Double, chord: Double, area: Double)] {{
        parseStations(goldenStationsRight)
    }}
    private static func parseStations(_ text: String) -> [(r: Double, chord: Double, area: Double)] {{
        text.split(separator: "\\n").compactMap {{ line in
            let v = line.split(separator: " ").compactMap {{ Double($0) }}
            return v.count >= 3 ? (v[0], v[1], v[2]) : nil
        }}
    }}

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
# The block is a *member* of `struct WingAero`, so it goes after the last
# function and before the struct's closing brace. Splicing it in at the last
# "}" in the file is not that: if the file has no trailing newline, the last
# "}\n" is the *function's* closing brace, and the block lands inside the
# function — which is what the first version of this script did, and what the
# Swift compiler rejected with "static properties may only be declared on a
# type". So the code part is re-closed explicitly instead.
marker = "    // MARK: - the golden table"
i = t.index(marker) if marker in t else len(t)
code = t[:i].rstrip("\n")
if not code.rstrip().endswith("}"):
    code += "\n    }"          # re-close the function the block follows
code = code.rstrip("\n")
p.write_text(code + "\n\n" + block + "}\n")

# and check the shape rather than trusting it
out = p.read_text()
depth, in_string, i2 = 0, False, 0
while i2 < len(out):
    if out[i2:i2 + 3] == '"""':
        in_string = not in_string
        i2 += 3
        continue
    if not in_string:
        if out[i2] == "{":
            depth += 1
        elif out[i2] == "}":
            depth -= 1
    i2 += 1
if depth != 0 or in_string:
    raise SystemExit(f"unbalanced file: depth {depth}, in_string {in_string}")
# the block has to be inside a type (depth > 0) and after the last function
head = out[:out.index("static let goldenWing")]
depth_at_block = head.count("{") - head.count("}")
if depth_at_block != 1:
    raise SystemExit(f"the golden block is at brace depth {depth_at_block}, "
                     f"not inside exactly one type")
if out.index("static let goldenWing") < out.rindex("\n    func "):
    raise SystemExit("the golden block is not after the last function")
print("the golden block is inside the type, after the last function")
print(f"golden block written: {len(wing_rows)} wing lines, {len(station_rows)} stations, "
      f"{len(strip_rows)} strips, {len(beat_rows)} beats")
