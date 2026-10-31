#!/usr/bin/env python3
"""Diff two Apple7/8 harness runs: candidate against baseline.

  ./compare.py results/<host>-baseline.jsonl results/<host>-candidate.jsonl
  ./compare.py --chosen A.jsonl B.jsonl        # only what the policy picks
  ./compare.py --tol 3 A.jsonl B.jsonl         # what counts as a regression

Rows join on shape, case and candidate label, so a sweep renamed between runs
still compares. Lower milliseconds is faster: a positive delta is a
regression. The summary is the geometric mean of the ratios, because these
measurements multiply through a decode step rather than adding.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[tuple[str, str, str], dict]:
    rows: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    with path.open() as source:
        for line in source:
            if line.strip():
                record = json.loads(line)
                rows[(record["shape"], record["case"], record["label"])].append(record)
    # A resumed file holds each measurement once per run; keep the last.
    return {key: values[-1] for key, values in rows.items()}

def resolve(name: str) -> Path:
    """A run's path, or just its label: `baseline` finds this host's baseline."""
    path = Path(name)
    if path.exists():
        return path
    host = os.uname().nodename
    results = Path(__file__).resolve().parent / "results"
    for candidate in (results / f"{host}-{name}.jsonl", results / f"{name}.jsonl"):
        if candidate.exists():
            return candidate
    sys.exit(f"no run '{name}' under {results} (looked for {host}-{name}.jsonl)")

def compare_policy(base: dict, new: dict, args) -> int:
    """Per shape and case, the row each run's policy chose, and how they differ."""
    def chosen(rows):
        out = {}
        for (shape, case, _label), record in rows.items():
            if record.get("chosen"):
                out[(shape, case)] = record
        return out

    old, cur = chosen(base), chosen(new)
    keys = sorted(set(old) & set(cur))
    moved = [(shape, case, old[(shape, case)], cur[(shape, case)])
             for (shape, case) in keys
             if abs(cur[(shape, case)]["ms"] / old[(shape, case)]["ms"] - 1) * 100 >= args.tol]
    ratios = [cur[k]["ms"] / old[k]["ms"] for k in keys]
    print(f"# policy chosen per case: {len(keys)} comparable, "
          f"{len(moved)} moved {args.tol}% or more")
    if args.all or moved:
        for shape, case, before, after in sorted(
                moved, key=lambda row: row[3]["ms"] / row[2]["ms"]):
            print(f"  {(shape + ' ' + case)[:38]:<40} {before['label']:<16} {before['ms']:>8.4f}"
                  f" -> {after['label']:<16} {after['ms']:>8.4f}  "
                  f"{100 * (after['ms'] / before['ms'] - 1):+6.1f}%")
    # Cases where the policy changed its tile at all, win or lose, are the ones
    # a reader needs to see; the list above only shows those over the tolerance.
    switched = [(s, c) for (s, c) in keys if old[(s, c)]["label"] != cur[(s, c)]["label"]]
    print(f"  {len(switched)} case(s) chose a different tile or split")
    for s, c in sorted(switched):
        gain = 100 * (cur[(s, c)]["ms"] / old[(s, c)]["ms"] - 1)
        print(f"    {(s + ' ' + c)[:44]:<46} {old[(s, c)]['label']} -> {cur[(s, c)]['label']}"
              f" {gain:+.1f}%")
    if ratios:
        gm = math.exp(sum(map(math.log, ratios)) / len(ratios))
        print(f"# geometric mean of the policy's chosen time: {(gm - 1) * 100:+.2f}%")
    return 0


def devices(rows) -> str:
    """What machine recorded these, for the warning when the two runs differ."""
    rows = list(rows)
    if not rows:
        return "none"
    first = rows[0]
    return f"{first['device']} (family {first['family']}, {first['cores']} cores)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Diff two Apple7/8 harness runs.")
    parser.add_argument("baseline", help="a run's path, or just its label")
    parser.add_argument("candidate", help="a run's path, or just its label")
    parser.add_argument("--policy", action="store_true",
                        help="compare what each run's policy chose per shape and case: "
                             "after a policy change the chosen tile differs, so rows "
                             "cannot be joined by label")
    parser.add_argument("--chosen", action="store_true",
                        help="only rows the policy chose in either run")
    parser.add_argument("--tol", type=float, default=2.0, help="percent that counts as a change")
    parser.add_argument("--limit", type=int, default=25, help="rows to print, by largest change")
    parser.add_argument("--all", action="store_true", help="print every comparable row")
    args = parser.parse_args()

    base, new = load(resolve(args.baseline)), load(resolve(args.candidate))
    if args.policy:
        return compare_policy(base, new, args)
    if not (base and new):
        print("nothing to compare: one of the files holds no rows", file=sys.stderr)
        return 2
    if devices(base.values()) != devices(new.values()):
        print(f"warning: different machines\n  baseline  {devices(base.values())}"
              f"\n  candidate {devices(new.values())}", file=sys.stderr)
    print(f"baseline  {args.baseline.name}: {base[list(base)[0]]['branch']} "
          f"{base[list(base)[0]]['git']}{' dirty' if base[list(base)[0]]['dirty'] else ''}")
    print(f"candidate {args.candidate.name}: {new[list(new)[0]]['branch']} "
          f"{new[list(new)[0]]['git']}{' dirty' if new[list(new)[0]]['dirty'] else ''}")

    shared = sorted(set(base) & set(new))
    rows = []
    for key in shared:
        before, after = base[key], new[key]
        if args.chosen and not (before["chosen"] or after["chosen"]):
            continue
        ratio = after["ms"] / before["ms"]
        rows.append((ratio, key, before, after))
    rows.sort(key=lambda row: abs(math.log(row[0])), reverse=True)

    if not rows:
        print("no rows in common", file=sys.stderr)
        return 2
    shown = rows if args.all else rows[:args.limit]
    print(f"\n{'shape / case / label':<58} {'base ms':>9} {'new ms':>9} {'delta':>8}")
    for ratio, (shape, case, label), before, after in shown:
        name = f"{shape} {case} {label}"
        marker = "*" if after["chosen"] else " "
        print(f"{name[:56]:<58}{marker}{before['ms']:>8.4f} {after['ms']:>9.4f} "
              f"{(ratio - 1) * 100:>+7.1f}%")

    geomean = math.exp(sum(math.log(r) for r, *_ in rows) / len(rows))
    worse = [r for r in rows if (r[0] - 1) * 100 > args.tol]
    better = [r for r in rows if (r[0] - 1) * 100 < -args.tol]
    same = len(rows) - len(worse) - len(better)
    print(f"\n{len(rows)} comparable {'row' if len(rows) == 1 else 'rows'}: "
          f"{len(better)} faster, {len(worse)} slower, "
          f"{same} within {args.tol:.1f}%")
    print(f"geometric mean {(geomean - 1) * 100:+.2f}% "
          f"({'candidate faster' if geomean < 1 else 'candidate slower'})")
    if len(set(base) - set(new)) or len(set(new) - set(base)):
        print(f"note: {len(set(base) - set(new))} baseline rows and "
              f"{len(set(new) - set(base))} candidate rows had no counterpart")
    return 0


if __name__ == "__main__":
    sys.exit(main())
