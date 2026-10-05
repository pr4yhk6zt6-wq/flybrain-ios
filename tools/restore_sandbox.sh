#!/usr/bin/env bash
#
# Put a reset sandbox back the way this project needs it.
#
#     bash tools/restore_sandbox.sh            # everything that is missing
#     bash tools/restore_sandbox.sh --git      # only the repository
#     bash tools/restore_sandbox.sh --swift    # only the Linux toolchain
#     bash tools/restore_sandbox.sh --build    # only build/ (from an .ipa)
#
# The workspace this project lives in is snapshotted, but not everything in it:
# `.git` comes back empty (so `git status` says "not a repository"), `build/` is
# not captured at all (it is one of the excluded directory names), `data/flybody`
# comes back partial, and `~/.cache` — which is where the Swift toolchain lives —
# is excluded too. Every one of those has cost a session's worth of time at least
# once, so they are one script now, and the script is idempotent: each step
# checks for its own marker and says "skip" rather than downloading 200 MB again.
#
# The assets, and where each one comes from:
#   data/banc/**       tools/download_banc.sh (public BANC v888, ~86 MB)
#   data/flybody/**    tools/download_flybody.sh (FlyBody, ~194 MB)
#   build/*.json|bin   extracted from the newest .ipa in builds/, which is the
#                      exact set of files the app ships (flybanc.bin,
#                      flybanc_meta.json, fly_anatomy.json, World/*)
#   ~/.cache/swift     the Swift 6 toolchain, for tools/run_tests_linux.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

TOKEN="${FLYBRAIN_TOKEN:-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}"
REMOTE="https://${TOKEN}@github.com/pr4yhk6zt6-wq/flybrain-ios.git"
SWIFT_VERSION="${SWIFT_VERSION:-6.0.3}"
SWIFT_HOME="${SWIFT_HOME:-$HOME/.cache/swift}"

DO_GIT=0 DO_ASSETS=0 DO_BUILD=0 DO_SWIFT=0
if [ $# -eq 0 ]; then DO_GIT=1 DO_ASSETS=1 DO_BUILD=1 DO_SWIFT=1; fi
for a in "$@"; do
  case "$a" in
    --git) DO_GIT=1;;
    --assets) DO_ASSETS=1;;
    --build) DO_BUILD=1;;
    --swift) DO_SWIFT=1;;
    *) echo "unknown option $a" >&2; exit 2;;
  esac
done

# ---- the repository --------------------------------------------------------
# A fresh `git init` has no main, and `git push` from it fails with "src refspec
# main does not match any" until the branch is renamed. There is no global
# identity in the sandbox, so it is set per-repo, and it is not captured in the
# snapshot (`.git/config` is on the exclusion list) — hence every reset needs it.
if [ "$DO_GIT" = 1 ]; then
  if [ -d .git ] && git rev-parse --verify -q main >/dev/null; then
    echo "git:      $REPO already has main at $(git log --oneline -1)"
  else
    echo "git:      recovering from $REMOTE"
    [ -d .git ] || git init -q
    git remote get-url origin >/dev/null 2>&1 || git remote add origin "$REMOTE"
    git fetch -q origin main
    git reset -q --hard origin/main
    git config user.name  flybrain
    git config user.email flybrain@local
    git branch -M main
    echo "git:      at $(git log --oneline -1)"
  fi
fi

# ---- the published assets --------------------------------------------------
if [ "$DO_ASSETS" = 1 ]; then
  bash tools/download_banc.sh data/banc
  bash tools/download_flybody.sh data/flybody
fi

# ---- build/, from the .ipa the app actually ships --------------------------
# The .ipa's own contents are the authority: flybanc.bin is the packed connectome
# tools/build_banc.py made, fly_anatomy.json is step 1's, and World/* is what
# tools/pack_world.py and tools/pack_body.py put in the bundle. Extracting them
# is faster and more honest than re-deriving them, and tools/audit_meshes.py and
# tools/pool_probe.py only need the World/ half.
if [ "$DO_BUILD" = 1 ]; then
  if [ -s build/flybanc.bin ] && [ -s build/fly_body.json ]; then
    echo "build:    already populated"
  else
    ipa="$(ls -1t ../builds/*.ipa 2>/dev/null | head -1 || true)"
    if [ -z "$ipa" ]; then
      echo "build:    no .ipa in ../builds — skipping (CI's artifact is the source)" >&2
    else
      echo "build:    from $(basename "$ipa")"
      mkdir -p build
      rm -rf /tmp/restore_ipa && mkdir -p /tmp/restore_ipa
      unzip -q -o "$ipa" -d /tmp/restore_ipa
      app=/tmp/restore_ipa/Payload/FlyBrain.app
      cp "$app/flybanc.bin" "$app/flybanc_meta.json" "$app/fly_anatomy.json" build/
      cp "$app/World/fly_body.json" "$app/World/fly_golden.json" \
         "$app/World/world.json"    "$app/World/fly_meshes.bin" build/
      ls -lh build/
    fi
  fi
fi

# ---- the Swift toolchain ---------------------------------------------------
# Debian trixie: Swift's official Ubuntu 24.04 build is the one that runs here.
if [ "$DO_SWIFT" = 1 ]; then
  if [ -x "$SWIFT_HOME/usr/bin/swiftc" ]; then
    echo "swift:    $("$SWIFT_HOME/usr/bin/swiftc" --version | head -1)"
  else
    echo "swift:    installing $SWIFT_VERSION into $SWIFT_HOME (~700 MB)"
    mkdir -p "$SWIFT_HOME"
    url="https://download.swift.org/swift-${SWIFT_VERSION}-release/ubuntu2404/swift-${SWIFT_VERSION}-RELEASE/swift-${SWIFT_VERSION}-RELEASE-ubuntu24.04.tar.gz"
    curl -sSL --retry 3 -o /tmp/swift.tar.gz "$url"
    tar -xzf /tmp/swift.tar.gz -C "$SWIFT_HOME" --strip-components=1
    rm -f /tmp/swift.tar.gz
    "$SWIFT_HOME/usr/bin/swiftc" --version | head -1
  fi
fi

# ---- the Python side -------------------------------------------------------
python3 - <<'PY'
import importlib.util
missing = [m for m in ("numpy", "scipy") if importlib.util.find_spec(m) is None]
if missing:
    print("python:   missing", ", ".join(missing), "— pip install numpy scipy")
else:
    import numpy, scipy
    print(f"python:   numpy {numpy.__version__}, scipy {scipy.__version__}")
try:
    import mujoco
    print(f"python:   mujoco {mujoco.__version__}")
except ImportError:
    print("python:   mujoco missing — pip install mujoco "
          "(needed by tools/audit_meshes.py --xml and tools/judge_port.py)")
PY

echo
echo "next:"
echo "  SWIFTC=$SWIFT_HOME/usr/bin/swiftc bash tools/run_tests_linux.sh"
echo "  python3 tools/pool_probe.py --ms 200 --only device"
