# geoML - machine learning models for geospatial data
# Copyright (C) 2026  Ítalo Gomes Gonçalves
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR a PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Copy the manual into the package skill.

    python plugins/geoml/sync.py [--check]

The skill's worked examples are the manual's chapters, which
`test_manual.py` runs at every release, so they are true of the version the
skill ships with. They are copied as Markdown into
`skills/geoml/references/manual/`, their figure links pointed at the
documentation site: the figures are for people, several megabytes of them,
and an agent reads the text and the code. `--check` copies nothing and
fails if the copy differs from what a sync would write, which is how the
release test knows this step was run.
"""
import argparse
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
MANUAL = HERE.parents[1] / "docs" / "manual"
TARGET = HERE / "skills" / "geoml" / "references" / "manual"
# where Sphinx publishes a page's images: one folder, by file name
FIGURES = "https://italo-goncalves.github.io/geoML/_images/"


def rewritten(text):
    """A chapter with its figure links pointed at the site."""
    return re.sub(r"\]\(figures/([^)]+)\)",
                  lambda m: "](%s%s)" % (FIGURES, m.group(1)), text)


def copies():
    """`{file name: text}` for every file the manual's copy holds."""
    sources = sorted(MANUAL.glob("[0-9][0-9]-*.md")) + [MANUAL / "README.md"]
    return {path.name: rewritten(path.read_text(encoding="utf-8"))
            for path in sources}


def stale():
    """The names of the files a sync would write, add or remove."""
    wanted = copies()
    present = {path.name for path in TARGET.glob("*.md")}
    differ = [name for name, text in wanted.items()
              if name not in present
              or (TARGET / name).read_text(encoding="utf-8") != text]
    return sorted(differ + sorted(present - set(wanted)))


def sync():
    TARGET.mkdir(parents=True, exist_ok=True)
    wanted = copies()
    for path in TARGET.glob("*.md"):
        if path.name not in wanted:
            path.unlink()
    for name, text in wanted.items():
        (TARGET / name).write_text(text, encoding="utf-8", newline="\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python plugins/geoml/sync.py",
        description="Copy the manual into the package skill.")
    parser.add_argument("--check", action="store_true",
                        help="copy nothing; fail if the copy is out of date")
    if parser.parse_args(argv).check:
        differ = stale()
        if differ:
            print("the skill's copy of the manual is out of date: %s; run "
                  "python plugins/geoml/sync.py" % ", ".join(differ))
            return 1
        return 0
    sync()
    print("copied %d files into %s" % (len(copies()), TARGET))
    return 0


if __name__ == "__main__":
    sys.exit(main())
