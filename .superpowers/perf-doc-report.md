# Perf-pass documentation update — report

## Status: DONE

Commit: `81dcf50` — "Update docs for the perf pass and checkpoint resume"
(branch `perf-allocator-and-board`)

## Job 1 — stale citations

`python3 tools/check_citations.py` final output:

```
21 citation(s) verified, 0 shifted, 0 with stale quoted text, 0 unresolvable
```

Exit code 0.

The checker only compares the *first* line of each quoted block against
the cited source line, so it caught six stale excerpts but missed three
more whose first line happened to still match verbatim (a function
signature or a `for` loop header shared by the old and new code) while the
body underneath had actually changed. I fixed all nine so the walkthrough
is honest end to end, not just checker-clean:

Checker-flagged (the required six):
1. `MuZeroNetwork::initialInference` (network.cpp) — quoted the new
   `Workspace&`-taking overload verbatim; it's now the only overload that
   actually shows the h-then-f computation (the 1-arg overload is a thin
   wrapper), so there was no "clean" alternative to prefer.
2. `MuZeroNetwork::recurrentInference` (network.cpp) — same reasoning,
   quoted the `Workspace&` overload; updated the prose naming `f`'s pieces
   since the old `predictionHidden` helper it referenced is no longer
   called from here.
3. `MCTS::run` expansion (mcts.cpp) — quoted the arena-index version
   as-is; this is exactly the case where "the buffered form IS the point,"
   so no simplification.
4. `MCTS::selectChild` (mcts.cpp) — same arena-index treatment; added one
   sentence clarifying `-1` now plays `nullptr`'s old role.
5. `MuZeroNetwork::trainStep` reverse pass, prediction head (network.cpp)
   — quoted the `TrainScratch`-based code; this section is specifically
   about the reused scratch buffer, so the buffered form is the point here
   too.
6. `MuZeroNetwork::trainStep` half gradient (network.cpp) — same, quoted
   the current `ws.*`-based block (its explanatory comment is also shorter
   in the real source now; quoted verbatim).

Also fixed (matched the checker's first line but were stale underneath):
7. `MCTS::run` root setup (mcts.cpp) — was still `Node root; ... make_unique<Node>()`;
   replaced with the actual arena-based root construction and added a short
   paragraph on what the arena is.
8. `minMaxNormalize` (mlp.cpp) — the quoted function is now a 2-line
   allocating wrapper; switched the excerpt to `minMaxNormalizeInto`, the
   function that actually does the computation, per the "prefer the
   clearest accurate code" guidance (it reads almost identically to the
   old body).
9. `MuZeroNetwork::trainStep` forward unroll (network.cpp) — was still
   quoting the old `dynamicsInput[k] = makeDynamicsInput(...)` style;
   replaced with the current `ws.*`/`*Into` sequence.

I also refreshed the six now-stale `file:line` pointers in the section 11
concept-map table that reference the same relocated functions (these are
plain table cells, not checker-covered captions, so they were quietly
wrong too).

## Job 2 — new section

Added `### A performance pass, and how it was checked` at the end of
section 09 (right after the "vs AlphaZero" callout, before the section 10
separator) — a subsection, not a new top-level numbered section, per "keep
it proportionate." Three short paragraphs: what profiling found and the
four fixes (with concrete before/after numbers), how the byte-identical-
checkpoint discipline was actually checked (fixed seeds 777/4242, full
pipeline, byte-for-byte checkpoint diff), and the `kLinesThrough` table bug
that discipline's two-seed spot-check couldn't see but
`test_cached_outcome_matches_a_full_rescan_everywhere` (549946-node
exhaustive walk) did.

## Job 3 — README

- Documented `train`'s new `[resume-from]` fourth argument: weights-only,
  empty replay buffer on resume, fails fast on a missing/corrupt
  checkpoint, reproducible-but-not-identical framing.
- Added `tools/check_citations.py` to the project-layout tools line (the
  only place tools are enumerated in the README).
- Ran `ctest --test-dir build --output-on-failure`; verbatim summary:
  `100% tests passed out of 10` / `Total Test time (real) =   0.47 sec`.
  This matches the README's existing "(10 tests)" claim, so no change was
  needed there. No other timing/count claims exist in the README.
- Verified every relative link in README.md resolves (`docs/*.md`,
  `docs/algorithm-explained.html`, `docs/superpowers/...`, and
  `../alphazero-tictactoe`) — all present.

## Job 4 — HTML regeneration

Ran `python3 tools/render_doc.py` (wrote 1890 lines). `git diff` on
`docs/algorithm-explained.html` shows only the sections corresponding to
the markdown edits: the nine rewritten excerpts/captions/prose, the new
performance subsection, and the concept-map table line-number fixes.
Nothing else in the HTML changed.

## Verification

- `python3 tools/check_citations.py` → `21 citation(s) verified, 0
  shifted, 0 with stale quoted text, 0 unresolvable` (exit 0).
- `ctest --test-dir build --output-on-failure` → `100% tests passed out of
  10`, `Total Test time (real) = 0.47 sec`.
- All README relative links resolve.
- HTML diff corresponds exactly to the markdown diff (verified by reading
  the full `git diff` of both files side by side).
- `git status` after staging showed only `README.md`,
  `docs/algorithm-explained.md`, `docs/algorithm-explained.html` — no
  code/test/CMake file was touched.
- Git author was already `Mark Castelluccio <markacastelluccio@gmail.com>`
  globally; no change needed. Commit carries the
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` trailer.

## Concerns (genuine code issues found, not fixed — per instructions)

1. **Dead code from the buffering refactor.** `MuZeroNetwork::representationHidden`,
   `dynamicsHidden`, `predictionHidden` (the `HiddenTrace`-returning private
   helpers) and `MuZeroNetwork::makeDynamicsInput` (the static allocating
   one-hot builder) are declared, defined, and now called from nowhere in
   `src/`, `include/`, `tests/`, `apps/`, or `tools/` — `initialInference`/
   `recurrentInference`/`trainStep` all moved to the `*Into`/`fillDynamicsInput`
   forms. They still compile silently (non-static member functions don't
   trigger `-Wunused-function`). Worth a follow-up cleanup pass, or removal,
   in the code (not touched here, per the documentation-only constraint).
2. I extended job 1 beyond the checker's literal six findings (see the
   three extra fixes above) because leaving them would have meant three
   more inaccurate excerpts sitting right next to the ones I was asked to
   fix, undermining the same walkthrough. Flagging this explicitly in case
   it's considered out of scope.
