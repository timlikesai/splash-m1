#!/usr/bin/env python3
"""Run a fixed set of Apple7/8 benchmarks and record what they measured.

Every result line the harness writes carries the machine, the build and the
measurement, so a candidate can be compared against a baseline recorded on
another day by another person. It records; it never decides: see compare.py
for the diff, and docs/apple78-perf.md for what the numbers are for.

  ./run.py --label baseline                     # every sweep in sweeps.txt
  ./run.py --label candidate --only q4k          # a subset, by name regex
  ./run.py --label baseline --resume             # finish an interrupted run
  ./run.py --label probe --sweeps /dev/null ...  # see --help for the rest

Results append to results/<host>-<label>.jsonl, one JSON object per measurement.
Keep the GPU to yourself while this runs: a benchmark measured beside another
GPU workload measures nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# "Apple M2 Max (GPU family 8, 30 cores): q4k 17408x5120 none, 1 weight copies
# of 85.0 MB, median ms of 25 rounds:" (dev/benchmarks/gguf_projection_benchmark.mm)
HEADING = re.compile(
    r"^(?P<device>.+?) \(GPU family (?P<family>\d+), (?P<cores>\d+) cores\): "
    r"(?P<shape>.+?), (?P<copies>\d+) weight copies of (?P<megabytes>[\d.]+) MB, "
    r"median ms of (?P<rounds>\d+) rounds:\s*$"
)
# " | L1 register S4* 0.1234"
CANDIDATE = re.compile(r"^(?P<label>.+?)(?P<chosen>\*)?\s+(?P<ms>\d+\.\d+)$")


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def identity() -> dict:
    """What was measured against: the build and the machine, as far as we can tell."""
    return {
        "git": git("rev-parse", "--short", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", "runtime", "dev", "server")),
        "host": os.uname().nodename,
        "macos": os.uname().release,
    }


def read_sweeps(path: Path) -> list[tuple[str, str]]:
    """`name<TAB>command` lines, with # comments. Command runs from the repository root."""
    sweeps = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, _, command = line.partition("\t")
        if not command.strip():
            raise SystemExit(f"{path}: line for {name!r} has no tab-separated command")
        sweeps.append((name.strip(), command.strip()))
    return sweeps


def parse(output: str) -> list[dict]:
    """Every `label median` the benchmark printed, with its case and heading."""
    records, context = [], {}
    for line in output.splitlines():
        if heading := HEADING.match(line.strip()):
            context = heading.groupdict()
            continue
        if "|" not in line:
            continue
        case, _, rest = line.partition("|")
        for part in rest.split("|"):
            # One record per candidate, each keeping its case and its heading.
            if not (candidate := CANDIDATE.match(part.strip())):
                continue
            record = dict(context)
            record["case"] = case.strip()
            record["label"] = candidate["label"]
            record["chosen"] = candidate["chosen"] == "*"
            record["ms"] = float(candidate["ms"])
            records.append(record)
    return records


def done_sweeps(path: Path) -> set[str]:
    if not path.exists():
        return set()
    names = set()
    for line in path.read_text().splitlines():
        try:
            names.add(json.loads(line)["sweep"])
        except (json.JSONDecodeError, KeyError):
            continue
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Apple7/8 benchmark sweeps.")
    parser.add_argument(
        "--label", required=True, help="what this run is, e.g. baseline or span-chain"
    )
    parser.add_argument("--sweeps", type=Path, default=HERE / "sweeps.txt")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--only", help="run sweeps whose name matches this regex")
    parser.add_argument("--skip", help="skip sweeps whose name matches this regex")
    parser.add_argument(
        "--resume", action="store_true", help="skip sweeps already in the output file"
    )
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per sweep")
    parser.add_argument(
        "--build",
        action="store_true",
        help="make each sweep's binary before running it: a stale one "
        "measures the previous policy and reports it as the current",
    )
    parser.add_argument("--dry-run", action="store_true", help="print what would run")
    args = parser.parse_args()

    sweeps = read_sweeps(args.sweeps)
    if args.only:
        sweeps = [s for s in sweeps if re.search(args.only, s[0])]
    if args.skip:
        sweeps = [s for s in sweeps if not re.search(args.skip, s[0])]
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"{os.uname().nodename}-{args.label}.jsonl"
    if args.resume:
        finished = done_sweeps(destination)
        sweeps = [s for s in sweeps if s[0] not in finished]

    if args.dry_run:
        for name, command in sweeps:
            print(f"{name}\t{command}")
        print(f"# {len(sweeps)} sweeps -> {destination}")
        return 0

    build = identity()
    print(
        f"# {build['branch']} {build['git']}{' dirty' if build['dirty'] else ''} -> {destination}"
    )
    failures = 0
    with destination.open("a") as sink:
        for index, (name, command) in enumerate(sweeps, 1):
            started = time.time()
            print(f"[{index}/{len(sweeps)}] {name} ...", flush=True)
            if args.build:
                # The sweeps name their binaries under build/, which are exactly
                # the make targets that rebuild them. Without this, a stale
                # binary measures the previous policy and reports it as now.
                target = command.split()[0]
                if target.startswith("build/"):
                    made = subprocess.run(
                        ["make", "-j4", target],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                    )
                    if made.returncode:
                        print(
                            f"    BUILD FAILED: {made.stderr.strip().splitlines()[-1:]}"
                        )
                        failures += 1
                        continue
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                )
            except subprocess.TimeoutExpired:
                print(f"    TIMEOUT after {args.timeout}s")
                failures += 1
                continue
            took = time.time() - started
            records = parse(result.stdout)
            if not records:
                print(
                    f"    FAILED to measure anything ({result.returncode}): "
                    f"{(result.stderr or result.stdout).strip().splitlines()[-1:]}"
                )
                failures += 1
                continue
            for record in records:
                record.update(
                    build,
                    sweep=name,
                    seconds=round(took, 1),
                    stamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                sink.write(json.dumps(record) + "\n")
            sink.flush()
            best = min(
                (r for r in records if r["chosen"]), key=lambda r: r["ms"], default=None
            )
            chosen = (
                f"chosen {best['case']} {best['label']} {best['ms']:.4f} ms"
                if best
                else "no chosen row"
            )
            print(f"    {len(records)} rows in {took:.0f}s; {chosen}")
    print(
        f"# {'FAILED: ' + str(failures) + ' sweep(s)' if failures else 'all sweeps recorded'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
