#!/usr/bin/env python3
"""Verify the file:line citations in docs/algorithm-explained.md.

The walkthrough quotes real code and captions each excerpt with the file
and line it came from. Those citations rot in two different ways, and only
one of them is visible: a commit that inserts lines above a quoted block
shifts the number, and a commit that edits the block itself makes the
quoted text wrong while the number may still look plausible.

This checks both. For every caption, the first non-blank line of the
fenced block above it must equal the cited line of the cited file. When it
does not, and that line is found exactly once elsewhere in the file, the
correct number is reported (and rewritten under --fix). When the quoted
text is nowhere in the file, the excerpt itself is stale and a human has
to decide what the document should now say -- so that case is always
reported and never auto-fixed.

Standard library only; not part of the build. Run it by hand, or from a
pre-commit hook, after touching anything the walkthrough quotes.
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MD = ROOT / "docs" / "algorithm-explained.md"
CAPTION = re.compile(r"^\*(?P<path>[\w./]+\.(?:cpp|hpp|sh|py)):(?P<line>\d+)\s+—\s+`(?P<sym>[^`]+)`")

_cache = {}


def source_lines(path):
    if path not in _cache:
        full = ROOT / path
        _cache[path] = full.read_text().splitlines() if full.exists() else None
    return _cache[path]


def excerpt_above(lines, index):
    """The fenced block immediately above a caption, as a list of lines."""
    i = index - 1
    while i >= 0 and not lines[i].strip():
        i -= 1
    if i < 0 or not lines[i].startswith("```"):
        return None
    start = i - 1
    while start >= 0 and not lines[start].startswith("```"):
        start -= 1
    return lines[start + 1:i]


def first_quoted_line(excerpt):
    for line in excerpt:
        if line.strip():
            return line
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true",
                        help="rewrite line numbers that moved (never rewrites stale quoted text)")
    args = parser.parse_args()

    lines = MD.read_text().splitlines()
    shifted, stale, missing, ok = [], [], [], 0

    for index, line in enumerate(lines):
        m = CAPTION.match(line)
        if not m:
            continue
        path, cited, sym = m["path"], int(m["line"]), m["sym"]
        src = source_lines(path)
        if src is None:
            missing.append(f"{MD.name}:{index + 1}: `{sym}` cites {path}, which does not exist")
            continue
        excerpt = excerpt_above(lines, index)
        quoted = first_quoted_line(excerpt) if excerpt else None
        if quoted is None:
            missing.append(f"{MD.name}:{index + 1}: caption for `{sym}` has no fenced block above it")
            continue

        actual = src[cited - 1] if 0 < cited <= len(src) else None
        if actual is not None and actual.rstrip() == quoted.rstrip():
            ok += 1
            continue

        hits = [n for n, l in enumerate(src, 1) if l.rstrip() == quoted.rstrip()]
        if len(hits) == 1:
            shifted.append((index, path, cited, hits[0], sym))
        else:
            stale.append(f"{MD.name}:{index + 1}: `{sym}` quotes a line no longer in {path} "
                         f"(cites :{cited}) — the excerpt itself needs updating:\n"
                         f"      quoted: {quoted.strip()[:90]}")

    if args.fix and shifted:
        for index, path, cited, correct, _ in shifted:
            lines[index] = lines[index].replace(f"{path}:{cited}", f"{path}:{correct}", 1)
        MD.write_text("\n".join(lines) + "\n")
        print(f"rewrote {len(shifted)} shifted line number(s)")
    else:
        for index, path, cited, correct, sym in shifted:
            print(f"{MD.name}:{index + 1}: `{sym}` cites {path}:{cited}, moved to {correct}")

    for problem in stale + missing:
        print(problem)

    remaining = (0 if args.fix else len(shifted)) + len(stale) + len(missing)
    print(f"\n{ok} citation(s) verified, {len(shifted)} shifted, "
          f"{len(stale)} with stale quoted text, {len(missing)} unresolvable")
    return 1 if remaining else 0


if __name__ == "__main__":
    sys.exit(main())
