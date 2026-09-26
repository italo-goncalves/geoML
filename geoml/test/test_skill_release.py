"""What the package skill shows runs, and its copy of the manual is current.

Release-only, as `test_manual.py` is: the Python blocks of the skill's
`SKILL.md` train a small model, and they are run here in one namespace, in
order, as a reader would run them. The manual's chapters are the skill's
worked examples, copied by `plugins/geoml/sync.py`; a copy that differs
from what the script would write now means the release forgot to run it.
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SKILL = ROOT / "plugins" / "geoml" / "skills" / "geoml" / "SKILL.md"


def test_the_skill_s_copy_of_the_manual_is_current():
    finished = subprocess.run(
        [sys.executable, str(ROOT / "plugins" / "geoml" / "sync.py"),
         "--check"], capture_output=True, text=True)
    assert finished.returncode == 0, finished.stdout


def test_every_python_block_in_the_skill_runs():
    blocks = re.findall(r"```python\n(.*?)```",
                        SKILL.read_text(encoding="utf-8"), re.DOTALL)
    assert blocks
    namespace = {"__name__": "__skill__"}
    for block in blocks:
        exec(compile(block, str(SKILL), "exec"), namespace)
