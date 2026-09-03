"""Executes every fenced Python block of the given chapters, in order.

The manual's promise is that its code runs. Blocks are cumulative within a
chapter (each executes in the same namespace), figures land in `figures/`
beside the chapters, and a chapter with a suspicious block count refuses --
which is what caught a quoting bug in the first version of this very tool.

Usage, from the repository root:

    python docs/manual/run_blocks.py docs/manual/*.md
"""
import os
import re
import sys
import time
import urllib.request

# the manual documents the tree it sits in. Python puts a script's own
# folder on the path, not the working directory, so the repository root goes
# there too and the chapters run against this checkout whether or not the
# package is installed -- locally it is not; CI installs it editable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

FENCE = re.compile("```python\n(.*?)```", re.S)

failures = 0
for path in sys.argv[1:]:
    if path.endswith("README.md"):
        continue
    with open(path, encoding="utf-8") as file:
        text = file.read()

    # a chapter may need data the reader downloads themselves; it declares
    # the environment variable naming the location, and is skipped without it
    needs = re.search(r"<!-- requires-env: (\w+) -->", text)
    if needs and not os.environ.get(needs.group(1)):
        print("%s: skipped (%s not set)"
              % (os.path.basename(path), needs.group(1)), flush=True)
        continue

    # a chapter may fetch its data from the web. Skipped rather than failed
    # when there is no connection, so an offline run still proves the rest
    # of the manual -- the chapter caches its download, so only the first
    # run needs the network at all
    if "<!-- requires-network -->" in text:
        cached = os.path.join(os.path.dirname(os.path.abspath(path)), "data")
        if not os.path.isdir(cached):
            try:
                urllib.request.urlopen("https://drive.google.com", timeout=10)
            except Exception:
                print("%s: skipped (no network, nothing cached)"
                      % os.path.basename(path), flush=True)
                continue

    blocks = FENCE.findall(text)
    if len(blocks) == 0:
        continue
    if len(blocks) > 20:
        raise SystemExit("%s: suspicious block count %d" % (path, len(blocks)))

    print("%s: %d block(s)" % (os.path.basename(path), len(blocks)),
          flush=True)
    here = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(path)))
    started = time.monotonic()
    space = {}
    try:
        for i, block in enumerate(blocks):
            exec(compile(block, "%s[%d]" % (path, i), "exec"), space)
    except Exception as error:
        failures += 1
        print("  FAILED at block %d: %r" % (i, error), flush=True)
    else:
        print("  ran in %.1f s" % (time.monotonic() - started), flush=True)
    finally:
        os.chdir(here)
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except ImportError:
            pass

print("ALL GREEN" if failures == 0 else "%d CHAPTER(S) FAILED" % failures,
      flush=True)
raise SystemExit(1 if failures else 0)
