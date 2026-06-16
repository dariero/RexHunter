#!/usr/bin/env bash
# setup-venv.sh — create RexHunter's virtualenv OUTSIDE iCloud and link it in.
#
# WHY: this repo lives under ~/Documents, which iCloud syncs. iCloud evicts/round-trips
# .venv binary payloads and leaves the .dist-info metadata behind — producing a venv that
# every metadata-level check (uv pip check, uv sync, uv pip list) reports healthy while
# imports fail with ModuleNotFoundError. The fix is to keep the real venv off the synced
# path: it lives at ~/.venvs/rexhunter, and ./.venv is a symlink to it, so `uv run`,
# pyright, and the git hooks resolve it transparently. (Mirrors the ~/.venvs/<project>
# convention used in the sibling RagaliQ project.)
#
# Idempotent: re-run any time to heal a vanished symlink or a real .venv that crept back in.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
EXT="$HOME/.venvs/rexhunter"

mkdir -p "$HOME/.venvs"
[ -x "$EXT/bin/python" ] || uv venv "$EXT" --python 3.14

# If a real .venv directory crept back (e.g. a bare `uv sync` run before this script),
# remove it so the symlink can take its place. find -delete, never `rm -rf`.
if [ -e "$ROOT/.venv" ] && [ ! -L "$ROOT/.venv" ]; then
  find "$ROOT/.venv" -delete
fi

ln -sfn "$EXT" "$ROOT/.venv"
uv sync

echo "✅ venv ready — ./.venv -> $(readlink "$ROOT/.venv")"
