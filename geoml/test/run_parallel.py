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
"""The full suite, one process per test file, several at a time.

    python geoml/test/run_parallel.py [-j JOBS] [--cpus N] [--no-manual]
                                      [FILE ...]

As one process the suite peaks at 23 GB, every file's TensorFlow graphs,
matplotlib figures and pyvista scenes piling up in it, and takes over half
an hour. One process per file bounds the memory at the heaviest file's peak,
as CI has done since 0.6.11, and running several at a time bounds the wall
time at the slowest one. The manual runs one process per chapter, so its
slowest chapter rather than all seventeen is on the critical path. **The
longest units start first**, by the time each took last on this machine
(kept in `~/.cache/geoml/test-durations.json`, written after every run) or,
for a unit never timed here, by the 0.9.0 release run's: chapter 17 alone
took 51 minutes at 0.9.0, and started late it was the whole tail.

**Half the machine, and no more.** The first version gave each process a
quarter of the CPUs in threads and froze the desktop: the worker pools some
tests fork (mesh sets, mesh distance queries) sized themselves to the whole
machine, every core was busy, and every process took the GPU the display
runs on. The runner takes half the physical cores by default, pins each
process to cores of its own -- which every thread and forked worker
inherits, and which geoML's pools read -- lowers the tests' priority, and
hides the GPU. **A process gets physical cores**, each with its hyperthread
siblings: counted in logical CPUs, four of them were two cores on a
hyperthreaded machine, and chapter 17 ran on two cores at 0.9.0.

Output goes to one log per unit; a failed unit's tail is printed, and the
exit code is non-zero if any failed. The sunspot test downloads from a
third-party server and is left out, as CI leaves it out.
"""
import argparse
import concurrent.futures
import json
import os
import pathlib
import queue
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANUAL = ROOT / "docs" / "manual"

DESELECT = ["geoml/test/test_real_cases.py::test_sunspot_deep_model"]
# seconds each unit took in the 0.9.0 release run (four physical cores
# shared by two units), the order for units this machine has not timed; a
# unit in neither starts after these, larger files first
MEASURED = {
    "manual/17-case-study-quartz-vein": 3077,
    "test_batched_experts.py": 1198,
    "manual/16-case-study-jura": 758,
    "test_expected_kernel.py": 381,
    "manual/05-latent-networks": 306,
    "manual/15-case-study-walker-lake": 254,
    "manual/13-validation": 253,
    "test_faults.py": 251,
    "test_real_cases.py": 233,
    "test_catalogue.py": 160,
    "manual/04-warpings-likelihoods-and-the-gaussian": 157,
    "test_vgp.py": 153,
    "manual/11-building-and-training": 151,
    "test_experts.py": 147,
    "test_blockset.py": 140,
    "test_leaves.py": 112,
    "manual/03-inducing-points-and-the-elbo": 109,
    "manual/07-simulation": 103,
    "test_predict_node.py": 96,
    "test_cross_validation.py": 86,
    "test_cnf.py": 81,
    "test_gaussian_mixture.py": 73,
    "test_model_persistence.py": 61,
    "test_block_prediction.py": 60,
}
# the times of the last run here, which take precedence
DURATIONS = pathlib.Path.home() / ".cache" / "geoml" / "test-durations.json"
# a process's peak, with room: the heaviest file measured 6.1 GB
GB_PER_JOB = 8
# the physical cores a process is pinned to
CPUS_PER_JOB = 4


def _available_gb():
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return 16.0


def _machine_cpus():
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return list(range(os.cpu_count() or 2))


def _physical_cores(cpus):
    """The logical CPUs `cpus` grouped by physical core, each group a core
    and its hyperthread siblings, in the order of their first CPU. Where the
    topology cannot be read every CPU is a core of its own."""
    cores = {}
    for cpu in cpus:
        topology = pathlib.Path("/sys/devices/system/cpu/cpu%d/topology" % cpu)
        try:
            key = (int((topology / "physical_package_id").read_text()),
                   int((topology / "core_id").read_text()))
        except (OSError, ValueError):
            key = ("cpu", cpu)
        cores.setdefault(key, []).append(cpu)
    return sorted(cores.values(), key=min)


def _recorded():
    try:
        return json.loads(DURATIONS.read_text())
    except (OSError, ValueError):
        return {}


def _record(results):
    """Keep each unit's time for the next run's order."""
    durations = _recorded()
    durations.update({name: round(seconds) for seconds, name, _, _ in results})
    try:
        DURATIONS.parent.mkdir(parents=True, exist_ok=True)
        DURATIONS.write_text(json.dumps(durations, indent=1, sort_keys=True))
    except OSError:
        pass


def _default_jobs(budget):
    return max(1, min(budget // CPUS_PER_JOB,
                      int(_available_gb() // GB_PER_JOB)))


def _pinned(cores):
    """What the child runs before the test process starts: its own cores,
    which every thread and forked worker inherits, and a lower priority."""
    def start():
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cores)
        if hasattr(os, "nice"):
            os.nice(10)
    return start


def _units(files, manual):
    """`(name, pytest arguments)` per process, the slowest first: by the
    time recorded here, else the 0.9.0 run's, else larger files first."""
    units = []
    if manual:
        chapters = MANUAL.glob("[0-9][0-9]-*.md")
        units += [("manual/" + c.stem,
                   ["geoml/test/test_manual.py::test_the_chapter_runs[%s]"
                    % c.stem], c.stat().st_size) for c in chapters]
        units.append(("manual/count", [
            "geoml/test/test_manual.py::test_the_manual_has_every_chapter"],
            0))
    units += [(f.name, [str(f.relative_to(ROOT))], f.stat().st_size)
              for f in files if f.name != "test_manual.py"]
    known = dict(MEASURED, **_recorded())
    units.sort(key=lambda u: (u[0] not in known, -known.get(u[0], 0), -u[2]))
    return [(name, args) for name, args, _ in units]


def _run(unit, slots, logs):
    name, args = unit
    slot = slots.get()
    try:
        return _run_on(name, args, slot, logs)
    finally:
        slots.put(slot)


def _run_on(name, args, slot, logs):
    # the slot's logical CPUs, and as many threads as it has physical cores
    cores, threads = slot
    log = logs / (name.replace("/", "_") + ".log")
    # No GPU: TensorFlow takes nearly all of a card's memory in every
    # process that sees it, and several at once on the card that drives the
    # display is the other half of how the first version froze the desktop.
    # CI has none, and the suite is written for the CPU.
    env = dict(os.environ, MPLBACKEND="Agg", TF_CPP_MIN_LOG_LEVEL="3",
               CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS=str(threads),
               TF_NUM_INTRAOP_THREADS=str(threads),
               TF_NUM_INTEROP_THREADS="2",
               PYTHONPATH=os.pathsep.join(
                   [str(ROOT)] + [p for p in [os.environ.get("PYTHONPATH")]
                                  if p]))
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               *args] + [a for d in DESELECT for a in ("--deselect", d)]
    start = time.monotonic()
    with open(log, "w") as out:
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=out,
                              stderr=subprocess.STDOUT,
                              preexec_fn=_pinned(cores)
                              if os.name == "posix" else None).returncode
    # 5 is pytest's "nothing collected", which a deselection can leave
    return name, code in (0, 5), time.monotonic() - start, log


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python geoml/test/run_parallel.py",
        description="Run the test suite one process per file, several at "
                    "a time.")
    parser.add_argument("files", nargs="*",
                        help="test files to run; all of them by default")
    parser.add_argument("--cpus", type=int, default=None,
                        help="how many physical cores the whole run may use, "
                             "each with its hyperthread siblings; half the "
                             "machine's by default")
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="processes at a time, sharing those cores; by "
                             "default one per %d of them, and no more than "
                             "%d GB of free memory each"
                             % (CPUS_PER_JOB, GB_PER_JOB))
    parser.add_argument("--no-manual", action="store_true",
                        help="leave the manual's chapters out")
    parser.add_argument("--logs", default=None,
                        help="where to write each unit's output")
    options = parser.parse_args(argv)

    files = [pathlib.Path(f).resolve() for f in options.files] \
        or sorted(HERE.glob("test_*.py"))
    manual = not options.no_manual and any(
        f.name == "test_manual.py" for f in files)
    units = _units(files, manual)
    machine = _physical_cores(_machine_cpus())
    budget = max(1, min(len(machine), options.cpus or len(machine) // 2))
    jobs = min(options.jobs or _default_jobs(budget), budget)
    per_job = budget // jobs
    slots = queue.Queue()
    for j in range(jobs):
        cores = machine[j * per_job:(j + 1) * per_job]
        slots.put(({cpu for core in cores for cpu in core}, per_job))
    logs = pathlib.Path(options.logs or tempfile.mkdtemp(prefix="geoml-tests-"))
    logs.mkdir(parents=True, exist_ok=True)
    print("%d units, %d at a time on %d of %d physical cores, %d each; logs "
          "in %s" % (len(units), jobs, jobs * per_job, len(machine), per_job,
                     logs), flush=True)

    start = time.monotonic()
    results = []
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        running = [pool.submit(_run, unit, slots, logs) for unit in units]
        for future in concurrent.futures.as_completed(running):
            name, passed, seconds, log = future.result()
            results.append((seconds, name, passed, log))
            print("%s %6.0f s  %s" % ("pass" if passed else "FAIL", seconds,
                                      name), flush=True)

    _record(results)
    failed = [r for r in results if not r[2]]
    for _, name, _, log in failed:
        print("\n==== %s ====\n%s" % (name, log.read_text()[-4000:]))
    print("\n%d units in %.0f s of wall time, %.0f s of work; %d failed"
          % (len(results), time.monotonic() - start,
             sum(r[0] for r in results), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
