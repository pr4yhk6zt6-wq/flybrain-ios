#!/usr/bin/env python3
"""
Read the test log and refuse to call a suite of skips a pass.

`xcodebuild test` exits 0 when every test that ran passed — and a test whose
fixture is not in the bundle does not run, it calls `XCTSkip`, because a
developer who has not built the 41 MB body model locally should still be able to
compile and test the rest. That is the right default and it is a trap in CI:
the whole body-solver suite (`FlyDynamicsTests` — the golden trace against
MuJoCo, the animal standing on muscle tone) was skipped in every green run,
because `tools/pack_world.py` had deleted `fly_body.json` out of the bundle
four seconds after `tools/pack_body.py` wrote it there.

CI builds every asset before it builds the app, so in CI a skip is a failure.
This reads the log and says so, by name.

    xcodebuild ... test 2>&1 | tee build/tests.log
    python3 tools/check_tests.py build/tests.log
"""

from __future__ import annotations
import pathlib
import re
import sys

# Tests whose absence is never acceptable: they are what the numbers in the
# reports are pinned to.
REQUIRED = (
    # the solver, against the MuJoCo-verified Python
    "testGoldenTraceIsReproduced",
    "testTheAnimalStandsWithMuscleToneOnly",
    "testTheMuscleModelIsForceBasedAndBraked",
    # the closed loop
    "testEveryPoolIsPlacedOnTheJointTheAssetNames",
    "testTheConnectomeIsAskedForTheNamesTheAssetCarries",
    "testAMissingGroupIsReportedRatherThanReadAsSilence",
    "testTheDescendingNeuronsCarryTheBrainsToneIntoTheCord",
    "testTheJointListStartsWithTheFreeJoint",
    "testTheLoopIsBalancedAroundTheStanceItMeasured",
    "testTheCordHoldsTheStanceItIsMeasuring",
    "testTheStandingLoadIsTheWindowAndNotOneSample",
    "testTheLoopReachesBothDirectionsFromTheStance",
    "testSilenceGivesTheStanceBackRatherThanADrift",
    "testTheChordotonalOrganReportsTheJointItSpans",
    "testReversingThePublishedPolarityReversesTheOrgan",
    "testTheCampaniformOrganReportsTheLoadTheLegCarries",
    # the cord owns what it reads from (uploads/IMG_2714.png: a weak
    # source that ARC freed, and a screen full of 0.0 Hz)
    "testTheCordOwnsTheRateSourceItReadsFrom",
    # the pose the meshes are drawn at (uploads/IMG_2713.png)
    "testAQuaternionFromAMatrixIsTheRotationItself",
    "testEveryPartIsDrawnWhereTheBodyCarriesIt",
    # the readout path, on a GPU (the phone's `0/42 pools firing` line)
    "testTheGroupTalliesSeeThePoolsFiring",
    # the animal
    "testTheAnimalHasThePublishedAnatomy",
    "testTheAnimalIsTheRightSizeAndMass",
    "testEveryArticulatedDegreeOfFreedomIsAnatomicallyLimitedOrFree",
    "testTheAdhesionModelIsPresent",
)

SKIPPED = re.compile(r"Test Case '-\[(?P<suite>[\w.]+) (?P<name>\w+)\]' skipped")
PASSED = re.compile(r"Test Case '-\[(?P<suite>[\w.]+) (?P<name>\w+)\]' passed")
# xcodebuild writes the skip count *before* the failure count:
#   Executed 7 tests, with 3 tests skipped and 0 failures (0 unexpected)
#   Executed 4 tests, with 0 failures (0 unexpected)
SUMMARY = re.compile(r"Executed (?P<n>\d+) tests?, with "
                     r"(?:(?P<skip>\d+) tests? skipped and )?"
                     r"(?P<fail>\d+) failures? \((?P<unexpected>\d+) "
                     r"unexpected\)")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: check_tests.py <xcodebuild test log>")
        return 2
    path = pathlib.Path(sys.argv[1])
    if not path.exists():
        print(f"  FAIL {path} does not exist — did the test step run?")
        return 1
    text = path.read_text(errors="replace")

    passed = {m.group("name") for m in PASSED.finditer(text)}
    skipped = {m.group("name"): m.group("suite") for m in SKIPPED.finditer(text)}
    # xcodebuild prints a summary per suite and then one for the run; take the
    # widest, so the number quoted is the whole run and not one suite of it.
    summaries = [m.groupdict() for m in SUMMARY.finditer(text)]
    if not summaries:
        print("  FAIL the log carries no test summary at all — xcodebuild did "
              "not get as far as running tests")
        return 1
    last = max(summaries, key=lambda d: int(d["n"]))
    total, failed = int(last["n"]), int(last["fail"])
    skipped_n = int(last["skip"] or 0)

    print(f"  {total} tests, {failed} failures, {skipped_n} skipped")

    problems = []
    if failed:
        problems.append(f"{failed} test(s) failed")
    for name, suite in sorted(skipped.items()):
        problems.append(f"{suite}.{name} was skipped — its fixture is not in "
                        f"the bundle, and CI builds every fixture")
    for name in REQUIRED:
        if name not in passed:
            problems.append(f"{name} did not run" if name not in skipped
                            else f"{name} was skipped")

    for p in problems:
        print(f"  FAIL {p}")
    if problems:
        return 1
    print("  every required test ran, and nothing was skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
