#!/usr/bin/env bash
# guard-bash.sh – PreToolUse(Bash) guard.
# Exit 2 blocks the command; stderr is shown to Claude as the reason.
# Covers what CLAUDE.md prose cannot enforce 100% of the time.
# Patterns are POSIX ERE (no \b / \s) so they work under macOS BSD grep.
set -uo pipefail

INPUT=$(cat)
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
[[ -z "$CMD" ]] && exit 0

# --- Irreversible / destructive -------------------------------------------
# recursive rm (any flag order/spelling), DROP TABLE, hard reset, force push.
# Plain single-file `rm -f foo` is intentionally allowed.
if echo "$CMD" | grep -qiE 'rm[[:space:]]+(-[a-z]*r|--recursive)|drop[[:space:]]+table|git[[:space:]]+reset[[:space:]]+--hard|git[[:space:]]+push[[:space:]].*(--force|-f([[:space:]]|$))'; then
  echo "BLOCKED: destructive/irreversible command (recursive rm, DROP TABLE, hard reset, or force-push). RexHunter rule – irreversible actions need explicit human approval. Ask first." >&2
  exit 2
fi

# --- Protect the trajectory log (single source of truth, invariant 1) -----
# Block deleting or shell-redirecting onto any *.db path.
if echo "$CMD" | grep -qiE 'rm[[:space:]][^|;&]*\.db([^a-z0-9]|$)|>[[:space:]]*[^|;&]*\.db([^a-z0-9]|$)'; then
  echo "BLOCKED: do not delete or overwrite a SQLite .db – the trajectory log is the source of truth (invariant 1). Use a fresh tmp path for tests." >&2
  exit 2
fi

# --- ToS-prohibited boards (accepted-limit, never crossed) ----------------
if echo "$CMD" | grep -qiE 'linkedin\.com|seek\.com|indeed\.com'; then
  echo "BLOCKED: live adapters hit public ATS APIs ONLY (Greenhouse / Lever). Scraping ToS-prohibited boards is a hard accepted-limit in the ADR." >&2
  exit 2
fi

# --- No direct push to main -----------------------------------------------
if echo "$CMD" | grep -qiE 'git[[:space:]]+push' && echo "$CMD" | grep -qiE '(^|[^a-z])(main|master)([^a-z]|$)'; then
  echo "BLOCKED: no direct push to main/master. Work ships through a branch + PR." >&2
  exit 2
fi

exit 0
