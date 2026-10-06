#!/usr/bin/env python3
"""
The .ipa, checked for the things that make an install hang.

A build that passes every scientific gate in this repository can still be a build
nobody can put on a phone, and the failure looks like nothing: the icon appears
greyed with *Installing…* under it and never finishes. Three of the ways that
happens are structural, and all three are cheap to check:

  1. **The wrong slice.** `xcodebuild -sdk iphoneos` and `-sdk iphonesimulator`
     write to the same derived-data path if the packaging step is careless, and
     a simulator binary (platform 7) or an x86_64 one installs nowhere. The
     Mach-O's own `LC_BUILD_VERSION` says which platform it was built for, so
     this reads it rather than trusting the folder it came from.
  2. **The binary is not executable inside the archive.** `Payload/App.app/App`
     has to carry mode 0755 in the zip. A packer that rebuilds the payload with
     a tool that drops permissions produces an .ipa that re-signs fine and then
     hangs at installation.
  3. **The same megabyte, twice.** `World/fly.bin` and `World/fly_meshes.bin`
     were byte-identical 6.3 MB files in every shipped build — two tools each
     writing "the meshes" — so `installd` copied 6.3 MB twice for no reason at
     all. Nothing failed; the .ipa was just a third bigger than it needed to be,
     which on a phone installing over a slow link is the difference between a
     progress bar and a hang. Every entry is hashed here and a duplicate above
     `DUP_MIN_BYTES` is a failure.

It also checks the plist keys `installd` reads (`CFBundleExecutable`,
`CFBundleSupportedPlatforms`, `MinimumOSVersion`, `UIDeviceFamily`), the data
files the app cannot open without, and that no recording is inside.

    python3 tools/audit_ipa.py build/FlyBrain-unsigned.ipa
    python3 tools/audit_ipa.py build/FlyBrain-unsigned.ipa --expect-version 0.7

Exit 0 if the archive is installable-shaped, 1 otherwise. CI runs it on the
built product, next to the "must contain its data" step.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import plistlib
import struct
import sys
import zipfile

# The largest size at which two identical entries are allowed to be a
# coincidence (an icon repeated in two folders, a PkgInfo). Anything bigger that
# appears twice is a packing mistake, not a coincidence.
DUP_MIN_BYTES = 64 * 1024

# Uncompressed budget. The app is a 20 MB connectome, a 6 MB body and a 4 MB
# icon; anything much past this means something is being shipped twice or a
# build product tag-along got in.
SIZE_BUDGET = 40 * 1024 * 1024

REQUIRED_KEYS = (
    "CFBundleExecutable", "CFBundleIdentifier", "CFBundleShortVersionString",
    "CFBundleVersion", "CFBundleSupportedPlatforms", "MinimumOSVersion",
    "UIDeviceFamily", "CFBundlePackageType",
)

# Files the app opens, under either of the names this repository has used.
REQUIRED_FILES = (
    ("the connectome", ("flybanc.bin",)),
    ("the connectome's metadata", ("flybanc_meta.json",)),
    ("the anatomy report", ("fly_anatomy.json",)),
    ("the body asset", ("World/fly_body.json", "fly_body.json")),
    ("the world manifest", ("World/world.json", "world.json")),
    # The mesh blob has two names in this repository's history: `fly.bin` is
    # what FlyWorld.swift asks for, `fly_meshes.bin` is what build_body.py
    # writes. Exactly one of them may ship (see the duplicate check).
    ("the mesh blob", ("World/fly.bin", "World/fly_meshes.bin", "fly.bin",
                       "fly_meshes.bin")),
)

PLATFORM_NAMES = {1: "macOS", 2: "iOS", 3: "tvOS", 4: "watchOS", 5: "bridgeOS",
                  6: "macCatalyst", 7: "iOS Simulator", 8: "tvOS Simulator",
                  9: "watchOS Simulator", 10: "DriverKit", 11: "visionOS",
                  12: "visionOS Simulator"}


def mach_o_facts(blob: bytes) -> dict:
    """What the main binary says it is: architecture and build platform."""
    facts: dict = {}
    if len(blob) < 32:
        return {"error": "too short to be a Mach-O"}
    magic = struct.unpack_from("<I", blob, 0)[0]
    if magic in (0xCAFEBABE, 0xCAFEBABF):       # universal / fat
        n = struct.unpack_from(">I", blob, 4)[0]
        facts["fat_slices"] = []
        for i in range(n):
            cputype, _sub, off, size = struct.unpack_from(">IIII", blob, 8 + i * 20)
            facts["fat_slices"].append(
                {"cpu": "arm64" if (cputype & 0x01000000) and cputype & 0xFFFFFF == 12
                        else "x86_64" if cputype & 0xFFFFFF == 7 else hex(cputype),
                 "bytes": size})
        return facts
    if magic not in (0xFEEDFACF, 0xFEEDFACE):
        return {"error": f"not a Mach-O (magic {magic:#x})"}
    cputype, _cpusub, _ft, ncmds, _sizeofcmds, flags = struct.unpack_from(
        "<iiIIII", blob, 4)
    cpu = cputype & 0xFFFFFF
    facts["arch"] = ("arm64" if (cputype & 0x01000000) and cpu == 12
                     else "x86_64" if cpu == 7
                     else "arm" if cpu == 12
                     else hex(cputype))
    facts["flags"] = flags
    # Walk the load commands for LC_BUILD_VERSION (0x32) / LC_VERSION_MIN_IPHONEOS.
    off = 32 if magic == 0xFEEDFACF else 28
    for _ in range(ncmds):
        if off + 8 > len(blob):
            break
        cmd, cmdsize = struct.unpack_from("<II", blob, off)
        if cmd == 0x32 and cmdsize >= 24:       # LC_BUILD_VERSION
            platform, minos, sdk, _ntools = struct.unpack_from("<IIII", blob, off + 8)
            facts["platform"] = platform
            facts["platform_name"] = PLATFORM_NAMES.get(platform, str(platform))
            facts["min_os"] = f"{minos >> 16}.{(minos >> 8) & 0xFF}"
            facts["sdk"] = f"{sdk >> 16}.{(sdk >> 8) & 0xFF}"
        elif cmd == 0x25:                       # LC_VERSION_MIN_IPHONEOS
            _v, sdk = struct.unpack_from("<II", blob, off + 8)
            facts["platform"] = 2
            facts["platform_name"] = "iOS (version-min command)"
            facts["sdk"] = f"{sdk >> 16}.{(sdk >> 8) & 0xFF}"
        if cmdsize < 8:
            break
        off += cmdsize
    return facts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ipa", type=pathlib.Path)
    ap.add_argument("--expect-version", default=None,
                    help="the CFBundleShortVersionString the build should carry "
                         "(the number that is printed on the phone's screen)")
    args = ap.parse_args()

    if not args.ipa.exists():
        print(f"  no .ipa at {args.ipa}", file=sys.stderr)
        return 2

    problems: list[str] = []
    z = zipfile.ZipFile(args.ipa)
    infos = z.infolist()
    names = [i.filename for i in infos]

    # ---- the shape of the archive ------------------------------------------
    apps = sorted({parts[1] for n in names
                   if n.startswith("Payload/")
                   for parts in [n.split("/")]
                   if len(parts) > 1 and parts[1]})
    if len(apps) != 1:
        problems.append(f"Payload/ holds {len(apps)} app bundles: {apps}")
        return _report(problems, args, None, None, None)
    app = apps[0]
    root = f"Payload/{app}"
    stray = [n for n in names if not n.startswith("Payload/")]
    if stray:
        problems.append(f"files outside Payload/: {stray[:4]}")

    files = {n[len(root) + 1:]: i for n, i in
             ((i.filename, i) for i in infos) if n.startswith(root + "/")}

    # ---- the binary ---------------------------------------------------------
    exe_name = None
    plist_info = None
    for key in ("Info.plist",):
        if key not in files:
            problems.append(f"{key} is not in the bundle")
    else:
        pl = plistlib.loads(z.read(files["Info.plist"].filename))
        plist_info = pl
        exe_name = pl.get("CFBundleExecutable")

    mach = None
    if exe_name and exe_name in files:
        info = files[exe_name]
        mode = (info.external_attr >> 16) & 0o777
        if not (mode & 0o111):
            problems.append(f"{exe_name} carries mode {mode:o} in the archive — "
                            "it must be executable (0755) or the install hangs")
        blob = z.read(info.filename)
        mach = mach_o_facts(blob)
        if "arch" in mach and mach["arch"] != "arm64":
            problems.append(f"the binary is {mach['arch']}, not arm64 — a "
                            "simulator or Intel slice installs nowhere")
        if mach.get("platform") is not None and mach["platform"] != 2:
            problems.append("the binary was built for "
                            f"{mach.get('platform_name')}, not iOS (device) — "
                            "the packaging step picked up the wrong build")
    else:
        problems.append(f"CFBundleExecutable {exe_name!r} is not in the bundle")

    # ---- the plist installd reads ------------------------------------------
    if plist_info is not None:
        for k in REQUIRED_KEYS:
            if not plist_info.get(k):
                problems.append(f"Info.plist has no {k}")
        platforms = plist_info.get("CFBundleSupportedPlatforms") or []
        if platforms and "iPhoneOS" not in platforms:
            problems.append(f"CFBundleSupportedPlatforms is {platforms}")
        if args.expect_version and \
                plist_info.get("CFBundleShortVersionString") != args.expect_version:
            problems.append(
                f"CFBundleShortVersionString is "
                f"{plist_info.get('CFBundleShortVersionString')!r}, and the "
                f"build this was asked to check is {args.expect_version!r} — "
                "the version on the phone's screen would lie")

    # ---- the data the app opens --------------------------------------------
    for what, candidates in REQUIRED_FILES:
        if not any(c in files for c in candidates):
            problems.append(f"{what} is missing (looked for "
                            f"{' / '.join(candidates)})")
    blob_names = [n for n in files if n in ("World/fly.bin", "World/fly_meshes.bin",
                                            "fly.bin", "fly_meshes.bin")]
    if len(blob_names) > 1:
        problems.append("the mesh blob ships under both names "
                        f"({', '.join(sorted(blob_names))}) — one of them is "
                        "the same megabytes a second time")
    if any(n.endswith("frames.bin") for n in names):
        problems.append("frames.bin is inside the .ipa — this build could play "
                        "a recording back")

    # ---- the same megabytes twice ------------------------------------------
    digests: dict[str, list[str]] = {}
    uncompressed = 0
    for i in infos:
        uncompressed += i.file_size
        if i.file_size < DUP_MIN_BYTES or i.is_dir():
            continue
        h = hashlib.sha256(z.read(i.filename)).hexdigest()
        digests.setdefault(h, []).append(f"{i.filename} ({i.file_size/1e6:.1f} MB)")
    for h, group in digests.items():
        if len(group) > 1:
            problems.append("two entries are byte-identical: " + " == ".join(group)
                            + " — the phone copies both")

    facts = {
        "app": app, "files": len(files), "uncompressed": uncompressed,
        "compressed": args.ipa.stat().st_size, "mach": mach,
        "version": (plist_info or {}).get("CFBundleShortVersionString"),
        "build": (plist_info or {}).get("CFBundleVersion"),
        "bundle_id": (plist_info or {}).get("CFBundleIdentifier"),
        "min_os": (plist_info or {}).get("MinimumOSVersion"),
    }
    if uncompressed > SIZE_BUDGET:
        problems.append(f"the bundle is {uncompressed/1e6:.1f} MB uncompressed, "
                        f"over the {SIZE_BUDGET/1e6:.0f} MB budget")
    return _report(problems, args, facts, files, mach)


def _report(problems, args, facts, files, mach) -> int:
    if facts:
        m = facts["mach"] or {}
        print(f"{args.ipa}: {facts['files']} entries, "
              f"{facts['uncompressed']/1e6:.1f} MB uncompressed, "
              f"{facts['compressed']/1e6:.1f} MB to download")
        print(f"  {facts['bundle_id']} v{facts['version']} ({facts['build']}), "
              f"min iOS {facts['min_os']}")
        print(f"  binary: {m.get('arch', '?')} · platform "
              f"{m.get('platform_name', '?')} · sdk {m.get('sdk', '?')}")
        if files:
            blob = [n for n in files if n.startswith("World/fly")
                    or n.startswith("fly_")]
            biggest = sorted(((i.file_size, n) for n, i in files.items()
                              if not i.is_dir()), reverse=True)[:3]
            print("  biggest: " + ", ".join(f"{n} {s/1e6:.1f} MB"
                                            for s, n in biggest))
            del blob
    if problems:
        print("\nthe .ipa is not installable-shaped:\n")
        for p in problems:
            print(f"  FAIL {p}")
        return 1
    print("\ninstallable-shaped: one app, an arm64 device binary that is "
          "executable in the archive,\nthe plist keys installd reads, the data "
          "the app opens, no recording, and no\nmegabyte shipped twice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
