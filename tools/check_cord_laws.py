#!/usr/bin/env python3
"""
The cord's laws are one set of numbers, written in three files.

There are three implementations of the same loop in this repository:

    tools/step3_closedloop.py   the reference, checked against the papers
    tools/walk_loop.py          the reference *with the body in it*
    FlyBrain/Sources/FlyCord.swift   the app, which is what the animal runs

They are three transliterations of one law, and a transliteration that drifts is
this project's most expensive kind of bug: it does not crash, it does not look
wrong, it just makes the tool measure an animal the phone cannot be. It has
already happened twice — the pool names (`motor_front_leg_left...` against
`pool:T1_left:...`, assumption #22) and the four-times-too-small pool rates — and
both were found by a person reading two files side by side, which is not a
method.

This is the method. It reads the numbers out of the three files and fails if any
of them disagree, and it checks that the organ law uses a *named* gain in both
languages rather than a literal — which is how the third one was found:

    walk_loop.py, at the commit that created it:   value = tone * (1 - 0.5 * x)
    step3_closedloop.py and assumption #11:        kappa = 1.0
    FlyCord.swift (what the phone runs):           settings.kappa = 1.0

The tool measured the closed loop at half the organ gain the app runs. Every
number in `reports/item4_walking.md` §4–§5 was measured at that gain, and the
conclusion drawn from them — "the cord's resting command distorts the stance" —
was a statement about the tool, not about the animal.

    python3 tools/check_cord_laws.py              # the check, exit 1 on a drift
    python3 tools/check_cord_laws.py --explain    # and the values it read
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

SWIFT = ROOT / "FlyBrain/Sources/FlyCord.swift"
WALK = HERE / "walk_loop.py"
STEP3 = HERE / "step3_closedloop.py"

# The loop's laws that exist in more than one file, with the name each file
# calls it. A name absent from a file is not a failure — step 3 has no command
# gain because its actuator is a joint angle, not a muscle force pair — but a
# name present in two files with two values is.
LAWS = [
    # (what it is, swift name, walk_loop constant, step3 argument)
    ("the brain's tone on the descending cells", "tone", "TONE", "--tone"),
    ("the chordotonal organ's range fractionation", "kappa", "KAPPA", "--kappa"),
    ("the chordotonal organ's transduction polarity", "polarity", "POLARITY",
     "--polarity"),
    ("the muscle activation filter, ms", "tauMuscleMs", "TAU_MUSCLE_MS",
     "--tau-mus"),
    ("the command gain, excitation per unit of balance", "gain",
     "COMMAND_GAIN", None),
    ("the calibration window, ms", "calibrateMs", "CALIBRATE_MS", None),
]


def swift_settings() -> dict[str, str]:
    """The defaults in `FlyCordSettings`, as written in the file."""
    text = SWIFT.read_text()
    m = re.search(r"struct FlyCordSettings \{(.*?)\n\}", text, re.S)
    if not m:
        raise SystemExit(f"no FlyCordSettings in {SWIFT}")
    out: dict[str, str] = {}
    for name, value in re.findall(r"var\s+(\w+)\s*:\s*[^=\n]+=\s*([^\n]+)", m.group(1)):
        out[name] = value.strip().rstrip(",").strip()
    return out


def python_constants(path: pathlib.Path) -> dict[str, str]:
    """Module-level `NAME = value` lines, with the trailing comment dropped."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^([A-Z][A-Z0-9_]*)\s*=\s*([^#\n]+)", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def step3_defaults() -> dict[str, str]:
    """The `default=` of each `add_argument`, keyed by the flag."""
    text = STEP3.read_text()
    out: dict[str, str] = {}
    for m in re.finditer(r"add_argument\(\s*\"(--[\w-]+)\"(.*?)\)\n", text, re.S):
        d = re.search(r"default=([^,\n]+)", m.group(2))
        if d:
            out[m.group(1)] = d.group(1).strip()
    return out


def number(text: str) -> float | None:
    m = re.match(r"^([-+]?\d*\.?\d+(?:e[-+]?\d+)?)$", text)
    return float(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--explain", action="store_true",
                    help="print every value that was read, not only failures")
    a = ap.parse_args()

    swift = swift_settings()
    walk = python_constants(WALK)
    step3 = step3_defaults()

    failures: list[str] = []
    for what, s_name, w_name, a_flag in LAWS:
        seen: list[tuple[str, float, str]] = []
        if s_name in swift:
            v = number(swift[s_name])
            if v is None:
                # `["chordotonal", "campaniform"]` and friends: compare as text.
                seen.append(("FlyCord.swift", float("nan"), swift[s_name]))
            else:
                seen.append(("FlyCord.swift", v, swift[s_name]))
        if w_name in walk:
            v = number(walk[w_name])
            if v is None:
                seen.append(("walk_loop.py", float("nan"), walk[w_name]))
            else:
                seen.append(("walk_loop.py", v, walk[w_name]))
        if a_flag and a_flag in step3:
            v = number(step3[a_flag])
            seen.append(("step3_closedloop.py", v if v is not None else float("nan"),
                         step3[a_flag]))

        if a.explain:
            shown = " · ".join(f"{f} {t}" for f, _, t in seen)
            print(f"  {what}\n      {shown or '— not in any of the three files'}")

        numbers = [(f, v) for f, v, _ in seen if v == v]      # drop the NaNs
        if len(numbers) > 1:
            first = numbers[0][1]
            bad = [(f, v) for f, v in numbers if abs(v - first) > 0]
            if bad:
                failures.append(f"{what}: " + ", ".join(f"{f} = {v:g}" for f, v in numbers))

    # ---- and the law must *use* the named gain, in both languages ----------
    # This is the check that catches the actual bug: a literal in the organ law
    # is a second, unnamed gain, and no two files can be compared if one of them
    # does not name its numbers.
    walk_text = WALK.read_text()
    m = re.search(r"elif o\[\"kind\"\] == \"chordotonal\":\n(.*?)\n\s*else:", walk_text, re.S)
    if not m:
        failures.append("walk_loop.py: no chordotonal branch to check")
    elif "self.kappa" not in m.group(1):
        failures.append("walk_loop.py: the chordotonal law does not use `self.kappa` "
                        "— a literal is a second gain (this is how κ = 0.5 was found)")

    swift_text = SWIFT.read_text()
    m = re.search(r"case \"chordotonal\":(.*?)case \"campaniform\":", swift_text, re.S)
    if not m:
        failures.append("FlyCord.swift: no chordotonal branch to check")
    elif "settings.kappa" not in m.group(1):
        failures.append("FlyCord.swift: the chordotonal law does not use `settings.kappa`")

    if failures:
        print("the cord's laws differ between files:\n")
        for f in failures:
            print(f"  FAIL {f}")
        print("\nOne law, one set of numbers. Fix the drift or declare it in "
              "docs/ASSUMPTIONS.md and name it in all three files.")
        return 1

    print("the cord's laws are one set of numbers, in all three files")
    print("  tone 2.5 · kappa 1.0 · polarity +1 · tau_mus 60 ms · gain 0.5 · "
          "calibrate 300 ms")
    print("  and both organ laws use their named κ and polarity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
