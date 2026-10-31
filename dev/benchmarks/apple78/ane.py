#!/usr/bin/env python3
"""Measure what the Neural Engine is given on an Apple7/8 machine, and whether it earns it.

The prefill FFN can run part of a chunk on the ANE. The runtime calibrates that
share against the machine at startup, which is the right thing to do and the
wrong thing to trust: the calibration was fitted on newer hardware, and the
share it picks is a claim about this GPU's bandwidth versus this ANE's.

    # One model, four shares, ~10 minutes each on an M2 Max at 27B:
    dev/benchmarks/apple78/ane.py sweep --root "$MODEL_ROOT" --share 0 --share 0.25 \
        --share 0.5 --calibrated --dir /tmp/ane

    # What they cost:
    dev/benchmarks/apple78/ane.py compare /tmp/ane/*.jsonl

`compare` keys rows on scenario and prompt length and takes the median of their
samples, because the scenarios repeat each prompt and the spread between two
samples of the same prompt (0.03% on a 95 s prefill) is the noise floor. The
first file named is the reference.

Nothing else should touch the GPU while this runs, and the machine should be
the one the numbers are for: `sweep` refuses when another process holds Metal.
"""
from __future__ import annotations

import argparse
import json
import re

import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def rows(path: Path) -> dict[tuple[str, int], list[float]]:
    """(scenario, prompt tokens) -> wall milliseconds, one per sample."""
    out: dict[tuple[str, int], list[float]] = {}
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") != "complete" or "wall_ms" not in record:
                continue
            if record.get("kind") != "request":
                continue
            out.setdefault((record["scenario"], record["prompt_tokens"]), []).append(
                record["wall_ms"])
    return out


def label(path: Path) -> str:
    """A file's share from its name: .../ane-0.25.jsonl is 0.25."""
    found = re.findall(r"calibrated|\d+(?:\.\d+)?", path.stem)
    return found[-1] if found else path.stem

def compare(paths: list[Path], tol: float) -> int:
    runs = [(label(p), rows(p)) for p in paths]
    if not runs:
        print("nothing to compare", file=sys.stderr)
        return 2
    keys = sorted(set.intersection(*(set(r[1]) for r in runs)))
    if not keys:
        print("the runs share no scenario and prompt length", file=sys.stderr)
        return 2
    base_name, _ = runs[0]
    print(f"# medians of {len(keys)} (scenario, tokens), reference {base_name}")
    header = f"{'scenario':<20} {'tokens':>7}" + "".join(f" {name:>13}" for name, _ in runs)
    print(header + "   best share vs reference")
    wins = {name: [] for name, _ in runs}
    for key in keys:
        medians = [statistics.median(runs[i][1][key]) for i in range(len(runs))]
        cells = "".join(f" {value / 1000:>12.2f}s" for value in medians)
        notes = []
        for (name, _), value in zip(runs, medians):
            delta = (value / medians[0] - 1) * 100
            wins[name].append(delta)
            if abs(delta) >= tol and name != base_name:
                notes.append(f"{name} {delta:+.1f}%")
        print(f"{key[0]:<20} {key[1]:>7}{cells}   {'; '.join(notes)}")
    print(f"\n{'':<28}" + "".join(f" {name:>13}" for name, _ in runs) + "   geomean vs reference")
    for name, values in wins.items():
        gm = 100 * (statistics.geometric_mean([v / 100 + 1 for v in values]) - 1) if values else 0
        print(f"{'median over scenarios':<28} {name:>13}"
              f"   {gm:+.2f}%   ({len(values)} rows)")
    return 0


def busy() -> str:
    """A process that would contend for the GPU, or an empty string."""
    try:
        out = subprocess.run(["pgrep", "-fl", "backend-benchmark|serve|splash"],
                             capture_output=True, text=True).stdout
    except OSError:
        return ""
    return "\n".join(l for l in out.splitlines()
                     if "ane.py" not in l and str(ROOT) in l and "pgrep" not in l)


def sweep(args) -> int:
    root = Path(args.root)
    if not (root / "model.json").exists():
        sys.exit(f"{root} is not a model root (no model.json)")
    if busy() and not args.force:
        sys.exit("another process holds the GPU; measurements would be its noise too "
                 "(--force to run anyway)")
    args.dir.mkdir(parents=True, exist_ok=True)
    shares = ([str(s) for s in args.share] if args.share else []) + \
             (["calibrated"] if args.calibrated else [])
    if not shares:
        sys.exit("name at least one --share, or --calibrated")
    for share in shares:
        command = [str(ROOT / "build/engine-tests/backend-benchmark"),
                   str(ROOT / args.library), str(root), "--samples", str(args.samples),
                   "--scenario", args.scenario, "--max-context", str(args.max_context),
                   "--progress", str(args.dir / f"ane-{share}.jsonl")]
        if share != "calibrated":
            command += ["--ane-ffn-share", share]
        print(f"=== {share}", flush=True)
        started = time.time()
        with (args.dir / f"ane-{share}.log").open("w") as sink:
            code = subprocess.run(command,
                                  cwd=ROOT, stdout=sink, stderr=subprocess.STDOUT).returncode
        print(f"    rc={code} in {time.time() - started:.0f}s", flush=True)
        if code:
            print(f"    see {args.dir / f'ane-{share}.log'}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    where = parser.add_subparsers(dest="mode", required=True)

    one = where.add_parser("compare", help="print the medians of runs already taken")
    one.add_argument("paths", nargs="+", type=Path)
    one.add_argument("--tol", type=float, default=1.0, help="percent that is a change")
    one.set_defaults(function=compare)

    two = where.add_parser("sweep", help="run the shares, one at a time")
    two.add_argument("--root", required=True, help="the model's root")
    two.add_argument("--share", type=float, action="append", default=[],
                     help="a prefill FFN share for the GPU to give the ANE (repeat); "
                          "0 is the GPU alone")
    two.add_argument("--calibrated", action="store_true", help="also run the runtime's own share")
    two.add_argument("--scenario", default="short,partial")
    two.add_argument("--samples", type=int, default=2)
    two.add_argument("--max-context", type=int, default=16384,
                     help="bounded, because the default is what memory holds and that "
                          "means a 30-minute sample on a 27B")
    two.add_argument("--library", default="build/splash.metallib")
    two.add_argument("--dir", type=Path, default=Path("/tmp/ane"))
    two.add_argument("--force", action="store_true")
    two.set_defaults(function=sweep)

    args = parser.parse_args()
    if args.mode == "compare":
        return compare(args.paths, args.tol)
    return sweep(args)


if __name__ == "__main__":
    sys.exit(main())
