#!/usr/bin/env bash
# protect-files.sh – PreToolUse(Edit|Write|MultiEdit) guard. Exit 2 blocks the write.
set -uo pipefail

INPUT=$(cat)
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
[[ -z "$FILE" ]] && exit 0

# Never let the agent hand-edit the trajectory log or any committed secret.
case "$FILE" in
  *.db|*.db-wal|*.db-shm)
    echo "BLOCKED: $FILE is the SQLite trajectory log – written only through db.py append paths, never edited as a file (invariant 1)." >&2
    exit 2
    ;;
  *.env|*.env.*|*.pem|*credentials*|*secret*)
    echo "BLOCKED: $FILE looks like a secret. Do not create or edit credential files." >&2
    exit 2
    ;;
esac

exit 0
