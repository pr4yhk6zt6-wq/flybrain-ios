#!/usr/bin/env bash
#
# Run the test suite that does not need a GPU, without Xcode and without CI.
#
#     bash tools/run_tests_linux.sh
#
# Why this exists: a forty-five minute CI cycle to be told that a *test file*
# does not compile is not a working loop, and `swiftc -parse` does not
# type-check, so it does not catch it either. This compiles the same sources
# and the same test files against the toolchain's own XCTest (Linux Swift ships
# one) and runs them in under a second. CI stays the authority — the app, the
# GPU kernels and the .ipa are only ever built there — but this is what says
# whether a change is worth pushing.
#
# Two things have to be stubbed, because they live in files that import Metal
# and QuartzCore: `SimulationEngine` and the monotonic clock. The test files'
# `@testable import FlyBrain` is stripped, which puts them in the same module as
# the sources — the same code, linked differently, which is exactly why this can
# type-check a test file that CI would refuse to compile.
#
# Assets: build/fly_body.json, build/fly_golden.json and build/fly_anatomy.json
# (from tools/build_body.py, tools/fly_aba.py and tools/step1_anatomy.py). With
# an asset missing, the tests that need it skip — and skipping is visible here
# for the same reason tools/check_tests.py exists: a suite of skips is not a
# pass.
#
# The toolchain: set SWIFTC if it is not on PATH.
#     SWIFTC=$HOME/.cache/swift/usr/bin/swiftc bash tools/run_tests_linux.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWIFTC="${SWIFTC:-swiftc}"
OUT="${OUT:-${TMPDIR:-/tmp}/flybrain-linux-tests}"
SWIFT_LIB_DIR="$("$SWIFTC" -print-target-info 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["paths"]["runtimeLibraryPaths"][0])' \
    2>/dev/null || echo /usr/lib/swift/linux)"
cd "$REPO"

for f in build/fly_body.json build/fly_golden.json build/fly_anatomy.json; do
    if [ ! -f "$f" ]; then
        echo "missing $f — run tools/build_body.py, tools/fly_aba.py and" \
             "tools/step1_anatomy.py first" >&2
        exit 2
    fi
done

rm -rf "$OUT/src" "$OUT/fakes"
mkdir -p "$OUT/src" "$OUT/fakes" "$OUT/World"

# ---- the shims -------------------------------------------------------------
cat > "$OUT/fakes/QuartzCore.swift" <<'EOF'
// A stand-in for QuartzCore: the app uses a monotonic clock and CoreFoundation's
// time interval type from it, which Foundation can supply.
import Foundation

public typealias CFTimeInterval = Double

public func CACurrentMediaTime() -> CFTimeInterval {
    var ts = timespec()
    clock_gettime(CLOCK_MONOTONIC, &ts)
    return Double(ts.tv_sec) + Double(ts.tv_nsec) * 1e-9
}
EOF
"$SWIFTC" -emit-module -emit-library -module-name QuartzCore \
    "$OUT/fakes/QuartzCore.swift" -o "$OUT/fakes/libQuartzCore.so" \
    -emit-module-path "$OUT/fakes/QuartzCore.swiftmodule"

# ---- the test files, import stripped ---------------------------------------
for f in FlyBrainTests/*.swift; do
    sed -e 's/^@testable import FlyBrain$//' "$f" > "$OUT/src/$(basename "$f")"
done

# ---- the runner ------------------------------------------------------------
# XCTMain needs the list of tests, and the list is read out of the files.
python3 - "$OUT/src/main.swift" <<'PY'
import pathlib, re, sys
tests = {}
for f in sorted(pathlib.Path("FlyBrainTests").glob("*.swift")):
    pattern = r"func (test\w+)\([^)]*\)\s*(throws)?"
    for m in re.finditer(pattern, f.read_text()):
        tests.setdefault(f.stem, []).append((m.group(1), bool(m.group(2))))
lines = ["import XCTest", ""]
for cls, names in tests.items():
    lines += [f"extension {cls} {{",
              "    static var allTestsLocal: "
              f"[(String, ({cls}) -> () throws -> Void)] {{",
              "        ["]
    for name, throws in sorted(names):
        body = f"try c.{name}()" if throws else f"c.{name}()"
        lines.append(f'            ("{name}", {{ (c: {cls}) in {{ {body} }} }}),')
    lines += ["        ]", "    }", "}", ""]
lines.append("XCTMain([")
for cls in tests:
    lines.append(f"    testCase({cls}.allTestsLocal),")
lines.append("])")
pathlib.Path(sys.argv[1]).write_text("\n".join(lines) + "\n")
total = sum(len(v) for v in tests.values())
print(f"runner: {total} tests — " + ", ".join(f"{k} {len(v)}" for k, v in tests.items()))
PY

cat > "$OUT/src/ZZShim.swift" <<'EOF'
import Foundation

// The one thing FlyCord.swift needs from the engine, which is a Metal file.
final class SimulationEngine {
    var groupNames: [String] { [] }
    func groupRate(_ name: String) -> Float { 0 }
    func setGroupDrive(_ name: String, _ value: Float?) {}
}
EOF

# ---- compile and run -------------------------------------------------------
# Bundle.main on Linux is the executable's directory, so the assets go beside it.
cp -f build/fly_body.json build/fly_golden.json build/fly_anatomy.json "$OUT/"
cp -f build/fly_body.json build/fly_golden.json build/fly_anatomy.json "$OUT/World/"

"$SWIFTC" -O -enable-testing \
    -I "$SWIFT_LIB_DIR" -L "$SWIFT_LIB_DIR" \
    -I "$OUT/fakes" -L "$OUT/fakes" -lQuartzCore \
    FlyBrain/Sources/FlyDynamics.swift \
    FlyBrain/Sources/FlyCord.swift \
    FlyBrain/Sources/FlyLiveBody.swift \
    "$OUT"/src/*.swift \
    -o "$OUT/tests" -lXCTest

LD_LIBRARY_PATH="$OUT/fakes:$SWIFT_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$OUT/tests" "$@"
