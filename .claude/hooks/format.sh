#!/usr/bin/env bash
# format.sh – PostToolUse(Edit|Write|MultiEdit).
# Auto-format edited Python files. Never blocks Claude (always exit 0):
# the file is already written; this is cleanup, not a gate.
set -uo pipefail

INPUT=$(cat)
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
[[ -z "$FILE" || ! -f "$FILE" ]] && exit 0

# Resolve the repo root the way .githooks/pre-push does; outside a repo → no-op.
ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
RUFF="$ROOT/.venv/bin/ruff"  # uv's in-repo venv — create it with `uv sync`
[[ -x "$RUFF" ]] || exit 0   # no formatter installed yet → no-op, not an error

case "$FILE" in
  *.py)
    "$RUFF" format "$FILE" >/dev/null 2>&1 || true
    "$RUFF" check --fix "$FILE" >/dev/null 2>&1 || true
    ;;
esac

exit 0
