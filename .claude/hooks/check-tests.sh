#!/usr/bin/env bash
# check-tests.sh – Stop hook. Runs the test suite when Claude tries to end its turn.
# Exit 2 + reason on stderr forces Claude to keep working (fix the failure).
# The stop_hook_active guard prevents an infinite block loop.
#
# PHILOSOPHY NOTE: this enforces "tests green by end of turn", which is in mild
# tension with red-green TDD (you may deliberately want to END a turn on a red gate).
# It auto-skips until BOTH a runner and a real test file exist, so it stays dormant
# through early Stage 1. Disable it if you want to checkpoint on a red gate.
set -uo pipefail

INPUT=$(cat)

# Already blocked once and Claude is looping → let it stop.
if [[ "$(echo "$INPUT" | jq -r '.stop_hook_active')" == "true" ]]; then
  exit 0
fi

PYTEST="$HOME/.venvs/rexhunter/bin/pytest"  # venv lives outside iCloud — see scripts/setup-venv.sh

# No runner yet, or no tests yet → nothing to gate on. This is a SKIP, not a "red".
[[ -x "$PYTEST" ]] || exit 0
ls tests/test_*.py >/dev/null 2>&1 || exit 0

if ! "$PYTEST" -q >/tmp/rex-pytest.log 2>&1; then
  echo "Tests are red – do not end the turn yet. Failing output:" >&2
  tail -n 25 /tmp/rex-pytest.log >&2
  exit 2
fi

exit 0
