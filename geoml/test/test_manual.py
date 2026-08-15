"""The manual's code runs.

Every fenced Python block of every chapter, executed in order through
`docs/manual/run_blocks.py` -- the same tool used while writing them, so
what this checks is exactly what the manual promises: the code in it works
against the package as it stands, and the figures beside it are what that
code produced.

**Release-only, deliberately.** A full pass trains real models and takes
about twenty minutes, which is not a price to pay on every push. The file
is on the structural job's ignore list, so it runs with the full suite --
release tags and a manual dispatch -- and nowhere else. It also writes into
`docs/manual/figures/`, which is the one test here that touches the working
tree: the figures are tracked, so a chapter whose numbers moved shows up as
a diff rather than as a silent pass.
"""
import os
import pathlib
import subprocess
import sys

import pytest

MANUAL = pathlib.Path(__file__).resolve().parents[2] / "docs" / "manual"


def test_every_chapter_runs():
    chapters = sorted(MANUAL.glob("[0-9][0-9]-*.md"))
    if not chapters:
        pytest.skip("the manual is not in this checkout")
    assert len(chapters) == 17, "expected seventeen chapters, found %d" % len(
        chapters)

    # the runner reports a failure per chapter and exits non-zero; its
    # output is the useful part, so it is captured and shown on failure
    finished = subprocess.run(
        [sys.executable, str(MANUAL / "run_blocks.py")]
        + [str(path) for path in chapters],
        cwd=str(MANUAL.parent.parent),
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True, text=True)

    assert finished.returncode == 0, (
        "the manual did not run clean:\n%s" % finished.stdout[-4000:])
    assert "ALL GREEN" in finished.stdout
