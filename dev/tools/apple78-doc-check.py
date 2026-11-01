#!/usr/bin/env python3
"""Check that the identifiers and paths the Apple7/8 docs cite actually exist.

This branch's docs are written from measurements and read later as facts. Three
inventions have already been caught in them -- a constant that never existed
(`kIsolatedPrefillMilliseconds`), a header under a directory that is not in the
This branch's docs are written from measurements and read later as facts. Four
inventions have been caught in them: a constant that never existed
(`kIsolatedPrefillMilliseconds`), a header under a directory that is not in the
tree (`dev/tuning/LinearNumerics.hpp`), a tool that never existed
(`tune_kernels`), and a claim about register pressure that was backwards. Each
was of exactly the kind that only shows up when someone greps for it -- so grep
for it automatically.

    dev/tools/apple78-doc-check.py [docs/apple78-port.md docs/apple78-perf.md]

Every code span in the docs is one of: a path in the repository, an identifier
defined somewhere in runtime/, dev/ or server/, a kernel-name fragment that
appears in the Metal sources, a command-line flag the engine or a tool parses,
or prose. Anything that is none of those is printed, because a doc that cites a
name the code does not have sends the next reader somewhere they cannot go.

It is deliberately loose about prose and strict about names: a false positive
costs a minute, a missed invention costs an evening.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
# A code span is a name to check if it looks like one of these.
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{3,}$")
PATHISH = re.compile(r"^[A-Za-z0-9_.\-/]+\.[A-Za-z]{1,6}$|^[a-z_]+/[a-z_/.-]+$")
FLAG = re.compile(r"^--?[a-z][a-z0-9-]*$")
PROSE = re.compile(r"[ ()]|^[-\d.]+$|^[A-Z][a-z]+ [A-Z]")


def grep(text: str) -> bool:
    """Whether `text` appears anywhere in the tracked sources."""
    found = subprocess.run(
        ["git", "grep", "-q", "--", text, "--", "runtime", "dev", "server", "install"],
        cwd=ROOT,
        capture_output=True,
    )
    return found.returncode == 0


def exists_anywhere(name: str) -> bool:
    """Whether a file of this name is tracked, at any depth.

    Docs cite `LinearGguf.cpp`, not its path; matching the full path only would
    call every such citation an error.
    """
    listed = subprocess.run(
        ["git", "ls-files", "--", f"*/{name}", name],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return bool(listed.stdout.strip())


def check_token(token: str) -> str | None:
    """None when the token is accounted for, else why it is suspicious."""
    token = token.strip()
    if not token or (PROSE.search(token) and not PATHISH.match(token)):
        return None
    if re.fullmatch(r"[0-9a-f]{7,40}", token):  # a commit: nothing in the tree names it
        return None
    if "." in token and IDENTIFIER.match(token.split(".")[-1]):  # `p.windowTokens`
        return check_token(token.split(".")[-1])
    if PATHISH.match(token):  # repo path, or path:suffix noise
        candidate = token.split(":")[0]
        if (ROOT / candidate).exists() or exists_anywhere(candidate.split("/")[-1]):
            return None
        return "no such path in the repository"
    if FLAG.match(token):
        stripped = token.lstrip("-")
        return None if grep(stripped) else "nothing parses this flag"
    if IDENTIFIER.match(token):
        if grep(token):
            return None
        if token == token.lower() and "_" in token:  # a composed kernel name
            # Most kernel names never appear as a literal: LinearGguf.cpp builds
            # `gguf_decode_mma_` + format + "_m" + rows + "_" + epilogue from
            # parts, and the Metal side makes them from a macro. So walk the
            # name back one segment at a time and accept it when some leading
            # part is in the tree. This cannot catch a wrong *suffix* -- a row
            # count or epilogue letter that no instance carries -- and does not
            # try: the failure it exists for is a name nobody ever wrote.
            parts = token.split("_")
            for end in range(len(parts) - 1, 2, -1):
                if grep("_".join(parts[:end])):
                    return None
        return "no such identifier or kernel base in runtime/, dev/ or server/"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "docs",
        nargs="*",
        type=pathlib.Path,
        default=[ROOT / "docs" / "apple78-port.md", ROOT / "docs" / "apple78-perf.md"],
    )
    args = parser.parse_args()

    spans = re.compile(r"`([^`\n]{2,90})`")
    problems = 0
    for doc in args.docs:
        text = (doc if doc.is_absolute() else ROOT / doc).read_text()
        seen: set[str] = set()
        for line, line_text in enumerate(text.splitlines(), 1):
            for token in spans.findall(line_text):
                if token in seen:
                    continue
                seen.add(token)
                why = check_token(token)
                if why:
                    problems += 1
                    print(f"{doc.name}:{line}: {why}\n    {token}")
    if problems:
        print(
            f"\n{problems} citation(s) to look at. Either fix the doc or add the thing it names."
        )
        return 1
    print(
        "every code span in the docs resolves to a path, identifier, kernel name or flag"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
