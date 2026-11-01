#!/usr/bin/env python3
"""Where this branch touches upstream's lines, so a rebase knows where to look.

The port's work is ~95% new files and new blocks, which git merges without
complaint. What costs hours at each upstream release is the other 5%: a line of
upstream's that this branch edited -- an `if` given another condition, a helper
given another argument -- because those collide with every upstream change to
the same line.

  dev/tools/apple78-rebase-audit.py                      # summary vs incoai/main
  dev/tools/apple78-rebase-audit.py --show runtime/ops/LinearGguf.cpp
  dev/tools/apple78-rebase-audit.py --upstream a56988e7  # against a specific release

`--show` prints, for one file, each upstream line the branch modified, with the
replacement, in the order they appear: the list to work through when a merge
conflicts. A modified line whose replacement is textually close to the original
tends to merge anyway; the ones that matter are where upstream's expression was
rewritten rather than extended, which `--show` marks.
"""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys

DEFAULT_UPSTREAM = "incoai/main"
# Fork-only files: whole-file additions, which never conflict.
FORK_ONLY = re.compile(r"apple7|_sgf|_mma|apple78", re.I)


def diff(upstream: str) -> list[str]:
    return subprocess.run(
        [
            "git",
            "diff",
            "--no-color",
            f"{upstream}...HEAD",
            "--",
            "runtime",
            "dev",
            "server",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()


def classify(lines: list[str]) -> dict[str, dict]:
    """Per file: added lines, and upstream lines the branch replaced."""
    out: dict[str, dict] = {}
    path = None
    hunk: list[str] = []
    current: dict[str, list[str]] = {}  # path -> the branch's own lines, read once

    def still_there(old: str, where: str) -> bool:
        """Whether the line the diff removed is elsewhere in the branch's file.

        A comment or a statement that moved is not a conflict: git will resolve
        it, and the next release's diff will show the same place. A line that is
        genuinely gone is the one to think about twice.
        """
        if where not in current:
            shown = subprocess.run(
                ["git", "show", f"HEAD:{where}"], capture_output=True, text=True
            )
            current[where] = shown.stdout.splitlines() if shown.returncode == 0 else []
        return old in current[where]

    def flush() -> None:
        if path is None:
            return
        record = out.setdefault(path, {"added": 0, "removed": 0, "pairs": []})
        # Pair each removed run with the added run that follows it in the hunk:
        # that is the edit, as opposed to a deletion or an insertion.
        removed = [row[1:] for row in hunk if row.startswith("-")]
        added = [row[1:] for row in hunk if row.startswith("+")]
        record["added"] += len(added)
        record["removed"] += len(removed)
        for old in removed:
            best, ratio = None, 0.0
            for new in added:
                score = difflib.SequenceMatcher(None, old, new).ratio()
                if score > ratio:
                    best, ratio = new, score
            record["pairs"].append((old, best, ratio, still_there(old, path)))

    for line in lines:
        if line.startswith("+++ b/"):
            flush()
            hunk = []
            path = line[6:]
            out.setdefault(path, {"added": 0, "removed": 0, "pairs": []})
        elif line.startswith("@@"):
            flush()
            hunk = []
        elif line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            hunk.append(line)
    flush()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    parser.add_argument(
        "--show", metavar="PATH", help="print the modified upstream lines of one file"
    )
    parser.add_argument(
        "--min", type=int, default=1, help="hide files below this many modified lines"
    )
    args = parser.parse_args()

    if subprocess.run(
        ["git", "rev-parse", "--verify", "-q", args.upstream], capture_output=True
    ).returncode:
        sys.exit(f"no ref '{args.upstream}'; fetch it first (git fetch incoai main)")

    records = classify(diff(args.upstream))
    if args.show:
        record = records.get(args.show)
        if not record:
            print(
                f"{args.show}: the branch does not modify any of upstream's lines there"
            )
            return 0
        print(
            f"# {args.show}: {len(record['pairs'])} upstream line(s) modified, "
            f"{record['added']} line(s) added"
        )
        for old, new, ratio, moved in record["pairs"]:
            kind = "moved" if moved else ("edited" if ratio >= 0.6 else "rewritten")
            print(f"\n  [{kind}] upstream said:\n    {old.strip()}")
            if new is not None and not moved:
                print(f"  the branch says:\n    {new.strip()}")
        return 0

    def weight(record: dict) -> tuple[int, int]:
        """(lines that will conflict, lines that merely moved) — the first matters."""
        real = sum(1 for _o, _n, _r, moved in record["pairs"] if not moved)
        return real, len(record["pairs"]) - real

    files = [
        (weight(r)[0], weight(r)[1], r["added"], path)
        for path, r in records.items()
        if weight(r)[0] >= args.min
    ]
    files.sort(reverse=True)
    whole = [p for _c, _m, _a, p in files if FORK_ONLY.search(p)]
    print(f"# vs {args.upstream}: where the branch edits upstream's own lines")
    print(f"{'edits':>6} {'moved':>6} {'added':>7}  file")
    for conflicts, moved, added, path in files:
        note = "  (fork-only file)" if FORK_ONLY.search(path) else ""
        print(f"{conflicts:>6} {moved:>6} {added:>7}  {path}{note}")
    touched = sum(1 for _c, _m, _a, p in files if not FORK_ONLY.search(p))
    print(
        f"\n{touched} upstream file(s) carry edits; {len(whole)} files are the branch's own."
    )
    print(
        "Work through a conflicting file with --show, which lists each modified line."
    )
    print(
        "Rule for new work: insert a block rather than extend a condition, and put logic in a"
    )
    print(
        "fork-only file when it does not need to sit inside an upstream function (docs/apple78-port.md)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
