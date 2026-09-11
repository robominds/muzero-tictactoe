#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
CHECKPOINT="$(mktemp -t mz_integration_XXXXXX.bin)"
trap 'rm -f "$CHECKPOINT"' EXIT

echo "== train (3 tiny iterations) =="
"$BUILD_DIR/train" 3 "$CHECKPOINT"
test -s "$CHECKPOINT"

echo "== resume from the checkpoint just written =="
RESUMED="$(mktemp -t mz_integration_resume_XXXXXX.bin)"
trap 'rm -f "$CHECKPOINT" "$RESUMED"' EXIT
RESUME_OUTPUT="$("$BUILD_DIR/train" 2 "$RESUMED" 7 "$CHECKPOINT")"
echo "$RESUME_OUTPUT"
echo "$RESUME_OUTPUT" | grep -q "^resumed from " || { echo "train did not report resuming" >&2; exit 1; }
test -s "$RESUMED"

echo "== resuming from a missing checkpoint fails, rather than training from noise =="
if "$BUILD_DIR/train" 1 "$RESUMED" 7 /nonexistent/checkpoint.bin >/dev/null 2>&1; then
  echo "train accepted a missing resume checkpoint" >&2
  exit 1
fi

echo "== evaluate =="
"$BUILD_DIR/evaluate" "$CHECKPOINT" 3

echo "== latent_probe =="
"$BUILD_DIR/latent_probe" "$CHECKPOINT"

echo "== diag_eval =="
"$BUILD_DIR/diag_eval" "$CHECKPOINT"

echo "== play_cli (scripted game) =="
PLAY_OUTPUT="$(printf '0\n1\n2\n3\n4\n5\n6\n7\n8\n' | "$BUILD_DIR/play_cli" "$CHECKPOINT")"
echo "$PLAY_OUTPUT"
echo "$PLAY_OUTPUT" | grep -qE '^(Draw\.|You win!|Agent wins\.)$' || { echo "play_cli did not finish a game" >&2; exit 1; }

echo "integration smoke test passed"
