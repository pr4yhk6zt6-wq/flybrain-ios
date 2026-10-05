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

# The three files the app reads, in FlyWorld.init order.
NEEDED = ("world.json", "fly.bin", "frames.bin")


def check(src: pathlib.Path) -> dict:
    """Read the manifest and confirm both binaries are the size it claims."""
    manifest = json.loads((src / "world.json").read_text())
    bg = manifest["body_geometry"]
    fr = manifest["frames"]

    expect_body = (bg["n_vertex"] * 3 + bg["n_vertex"] * 3 + bg["n_face"] * 3) * 4
    expect_frames = fr["n"] * fr["parts"] * 7 * 4

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
        "expected_frames_bin_bytes": expect_frames,
    }

    problems = []
    for name in NEEDED:
        p = src / name
        if not p.exists():
            problems.append(f"{name} is missing — run tools/step4_world.py")
    if not problems:
        body = (src / "fly.bin").stat().st_size
        frames = (src / "frames.bin").stat().st_size
        if body < expect_body:
            problems.append(
                f"fly.bin is {body} bytes, the manifest needs {expect_body}")
        if frames < expect_frames:
            problems.append(
                f"frames.bin is {frames} bytes, the manifest needs "
                f"{expect_frames}")
        facts["fly_bin_bytes"] = body
        facts["frames_bin_bytes"] = frames
        facts["total_bytes"] = (
            body + frames + (src / "world.json").stat().st_size)

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
          f"({facts['frames'] * facts['parts'] * 7 * 4 / 1e6:.2f} MB)")
    print(f"  {facts['legs']} legs, {facts['geoms']} tracked parts")

    if facts["problems"]:
        return 1

    total = facts["total_bytes"]
    print(f"  world.json + fly.bin + frames.bin = {total / 1e6:.2f} MB")

    if args.dry_run:
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    for name in NEEDED:
        shutil.copy2(args.src / name, args.dest / name)
    # A folder reference is copied whole, so anything stale in it would ship.
    kept = {p.name for p in args.dest.iterdir() if p.is_file()} - set(NEEDED)
    for name in sorted(kept):
        (args.dest / name).unlink()
        print(f"  removed stale {name}")

    for name in NEEDED:
        p = args.dest / name
        if p.stat().st_size == 0:
            print(f"  FAIL {p} copied as zero bytes")
            return 1
    print(f"  packed into {args.dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
