#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
CHECKPOINT="$(mktemp -t mz_integration_XXXXXX.bin)"
trap 'rm -f "$CHECKPOINT"' EXIT

echo "== train (3 tiny iterations) =="
"$BUILD_DIR/train" 3 "$CHECKPOINT"
test -s "$CHECKPOINT"

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
