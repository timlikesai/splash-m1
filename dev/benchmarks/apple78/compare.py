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


def devices(rows) -> str:
    """What machine recorded these, for the warning when the two runs differ."""
    rows = list(rows)
    if not rows:
        return "none"
    first = rows[0]
    return f"{first['device']} (family {first['family']}, {first['cores']} cores)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Diff two Apple7/8 harness runs.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--chosen", action="store_true",
                        help="only rows the policy chose in either run")
    parser.add_argument("--tol", type=float, default=2.0, help="percent that counts as a change")
    parser.add_argument("--limit", type=int, default=25, help="rows to print, by largest change")
    parser.add_argument("--all", action="store_true", help="print every comparable row")
    args = parser.parse_args()

    base, new = load(args.baseline), load(args.candidate)
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
