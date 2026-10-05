#!/usr/bin/env python3
"""
Copy the body asset into the app bundle's Resources.

`tools/build_body.py` writes the physics: the rigid-body tree, the joints, the
muscle table, the collision geoms and the welded meshes. The app needs the JSON
in its bundle (the meshes it draws come from the world packer, which packs the
same MJCF geoms). The file is generated and not committed, so this step puts it
into the tree after build_body.py has run and before xcodegen does — and
refuses to ship an app whose asset is half-written, because a truncated JSON
decodes to a fly with no legs.

Run from the repository root:

    python3 tools/pack_body.py              # -> FlyBrain/World/fly_body.json
    python3 tools/pack_body.py --dry-run    # check only

The golden trace goes along with it when it exists, so the test target can
replay the same run the Python solver was verified with.
"""

import argparse
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
DEST = ROOT / "FlyBrain" / "World"

# What the app reads. fly_body.json is required; the golden trace is only used
# by the tests, so a build without it still ships.
NEEDED = ("fly_body.json",)
OPTIONAL = ("fly_golden.json", "fly_meshes.bin")


def check(src: pathlib.Path) -> dict:
    """Read the asset and confirm it is the animal, not half of one."""
    body = json.loads(src.read_text())
    counts = body.get("counts", {})
    facts = {
        "path": str(src),
        "bytes": src.stat().st_size,
        "bodies": len(body.get("bodies", [])),
        "hinges": counts.get("hinges", sum(1 for j in body.get("joints", [])
                                           if j["kind"] == "hinge")),
        "collision_geoms": len(body.get("collision", [])),
        "visual_geoms": len(body.get("visual", [])),
        "muscles": len(body.get("muscles", {})),
        "floor_z": body.get("floor_z"),
        "timestep_s": body.get("timestep_s"),
        "stance_root_z": body.get("stance_root_z"),
    }
    if facts["bodies"] < 100:
        raise SystemExit(f"{src}: only {facts['bodies']} bodies — that is not "
                         "the fly; re-run tools/build_body.py")
    if facts["hinges"] < 100:
        raise SystemExit(f"{src}: only {facts['hinges']} hinges")
    if not body.get("muscles"):
        raise SystemExit(f"{src}: no muscle table — re-run tools/build_body.py")
    return facts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=BUILD, type=pathlib.Path)
    ap.add_argument("--dest", default=DEST, type=pathlib.Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    asset = args.src / "fly_body.json"
    if not asset.exists():
        print(f"missing {asset} — run tools/build_body.py first", file=sys.stderr)
        return 2
    facts = check(asset)

    copied = []
    if not args.dry_run:
        args.dest.mkdir(parents=True, exist_ok=True)
    for name in NEEDED + OPTIONAL:
        src = args.src / name
        if not src.exists():
            if name in NEEDED:
                print(f"missing {src}", file=sys.stderr)
                return 2
            continue
        if not args.dry_run:
            shutil.copy2(src, args.dest / name)
        copied.append((name, src.stat().st_size))

    print(f"body asset: {facts['bodies']} bodies, {facts['hinges']} hinges, "
          f"{facts['collision_geoms']} collision geoms, "
          f"{facts['muscles']} muscles, dt {facts['timestep_s']} s, "
          f"floor z {facts['floor_z']}")
    for name, size in copied:
        print(f"  -> FlyBrain/World/{name} ({size/1e6:.2f} MB)")
    if args.dry_run:
        print("dry run: nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
