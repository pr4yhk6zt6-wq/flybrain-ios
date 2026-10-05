#!/usr/bin/env python3
"""Copy the recorded world into the app bundle's Resources.

The recording is seven megabytes of generated binary. It is not committed —
`world/*.bin` and `world/world.json` are in .gitignore — so it has to be put
into the tree after tools/step4_world.py has made it and before xcodegen
runs. This is that step, plus the check that refuses to ship an app whose
world is half-written.

Run from the repository root:

    python3 tools/pack_world.py              # -> FlyBrain/World
    python3 tools/pack_world.py --dry-run    # check only
"""

import argparse
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "world"
DEFAULT_DEST = ROOT / "FlyBrain" / "World"

# What the app reads: the manifest, and the meshes. The pose track is *not*
# here on purpose. The Map screen runs the solver on the device, so shipping
# `frames.bin` would ship seven megabytes of a recording nothing plays — and a
# build that carries a recording is a build that can start playing one again.
# It stays in world/ for the browser viewer, which does play it.
NEEDED = ("world.json", "fly.bin")
RECORDING = "frames.bin"


def check(src: pathlib.Path) -> dict:
    """Read the manifest and confirm both binaries are the size it claims."""
    manifest = json.loads((src / "world.json").read_text())
    bg = manifest["body_geometry"]
    fr = manifest["frames"]

    expect_body = (bg["n_vertex"] * 3 + bg["n_vertex"] * 3 + bg["n_face"] * 3) * 4

    facts = {
        "n_vertex": bg["n_vertex"],
        "n_face": bg["n_face"],
        "meshes": len(bg["meshes"]),
        "frames": fr["n"],
        "parts": fr["parts"],
        "stride_ms": fr["stride_ms"],
        "geoms": len(manifest["geoms"]),
        "legs": len(manifest["legs"]),
        "expected_fly_bin_bytes": expect_body,
    }

    problems = []
    for name in NEEDED:
        p = src / name
        if not p.exists():
            problems.append(f"{name} is missing — run tools/step4_world.py")

    # The part axis is what every renderer indexes, and every renderer skips
    # the rows with no mesh (the floor, which is part 0). If `part` is not the
    # identity sequence over the manifest's own order, the animal is drawn with
    # each mesh wearing the pose of the part before it. Checked here, once,
    # rather than trusted in two languages.
    parts = [g.get("part") for g in manifest["geoms"]]
    if parts != list(range(len(parts))):
        problems.append("manifest geoms do not carry part = 0, 1, 2 … — the "
                        "renderers would pose every mesh off by one")
    if len(parts) != fr["parts"]:
        problems.append(f"the manifest lists {len(parts)} geoms but records "
                        f"{fr['parts']} parts")

    if not problems:
        body = (src / "fly.bin").stat().st_size
        if body < expect_body:
            problems.append(
                f"fly.bin is {body} bytes, the manifest needs {expect_body}")
        facts["fly_bin_bytes"] = body
        total = body + (src / "world.json").stat().st_size
        rec = src / RECORDING
        if rec.exists():                    # for the viewer, not for the app
            want = fr["n"] * fr["parts"] * 7 * 4
            size = rec.stat().st_size
            if size < want:
                problems.append(
                    f"{RECORDING} is {size} bytes, the manifest needs {want}")
            facts["recording_bytes"] = size
            total += size
        facts["total_bytes"] = total

    facts["problems"] = problems
    return facts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=pathlib.Path, default=SRC,
                    help="where step 4 wrote the world (default: world/)")
    ap.add_argument("--dest", type=pathlib.Path, default=DEFAULT_DEST,
                    help="the folder reference the app target copies "
                         "(default: FlyBrain/World)")
    ap.add_argument("--dry-run", action="store_true",
                    help="check the recording but copy nothing")
    args = ap.parse_args()

    if not (args.src / "world.json").exists():
        print(f"  no world at {args.src} — run tools/step4_world.py first")
        return 1

    facts = check(args.src)
    for p in facts["problems"]:
        print(f"  FAIL {p}")

    print(f"  {facts['meshes']} meshes, {facts['n_vertex']:,} vertices, "
          f"{facts['n_face']:,} triangles")
    print(f"  {facts['frames']} frames x {facts['parts']} parts at "
          f"{1000 // max(1, facts['stride_ms'])} Hz "
          f"({facts['frames'] * facts['parts'] * 7 * 4 / 1e6:.2f} MB, viewer only)")
    print(f"  {facts['legs']} legs, {facts['geoms']} tracked parts, "
          f"part indices 0 … {facts['geoms'] - 1} in manifest order")

    if facts["problems"]:
        return 1

    total = facts["total_bytes"]
    print(f"  the app carries world.json + fly.bin = "
          f"{(facts['fly_bin_bytes'] + (args.src / 'world.json').stat().st_size) / 1e6:.2f} MB"
          + (f", the viewer also has {facts.get('recording_bytes', 0) / 1e6:.2f} MB "
             f"of recording" if facts.get("recording_bytes") else ""))

    if args.dry_run:
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    for name in NEEDED:
        shutil.copy2(args.src / name, args.dest / name)
    # This step owns the recording and nothing else in the folder.
    #
    # The folder is a whole-directory reference, so whatever sits here ships.
    # That is what keeps a recording out of the app: if a previous build left a
    # frames.bin here, it goes now — the Map screen draws the *animal*, and no
    # build may quietly play one back.
    #
    # It is also, until this fix, what kept the animal's own body asset out of
    # the app: `tools/pack_body.py` writes `fly_body.json` into this same
    # folder, and the version of this function that deleted everything not in
    # NEEDED deleted it again, four seconds later, in every CI run. The app
    # shipped without a skeleton (its Body screen reports a missing asset) and
    # the solver tests skipped themselves — a green build over an empty seat.
    # A packer deletes the files it owns and leaves other packers' files alone.
    if (args.dest / RECORDING).exists():
        (args.dest / RECORDING).unlink()
        print(f"  removed stale {RECORDING} "
              "(the app runs the solver; a recording is not shipped)")

    for name in NEEDED:
        p = args.dest / name
        if p.stat().st_size == 0:
            print(f"  FAIL {p} copied as zero bytes")
            return 1
    print(f"  packed into {args.dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
