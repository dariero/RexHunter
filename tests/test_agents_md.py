"""AGENTS.md is the working contract; docs/build-log.md holds its history.

CLAUDE.md is a symlink to AGENTS.md, so every Claude Code session in this repo loads all of it.
Each slice and wave used to append its entry there: at c3054e0 the entries and the per-slice list
were 1012 of the file's 1192 lines. These tests keep that history in docs/build-log.md.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "AGENTS.md"
BUILD_LOG = ROOT / "docs" / "build-log.md"
ENTRY_MARK = "**▶"  # opens every log entry since P3's


def test_agents_md_holds_no_build_log_entry() -> None:
    lines = AGENTS.read_text(encoding="utf-8").splitlines()
    entries = [n for n, line in enumerate(lines, start=1) if line.startswith(ENTRY_MARK)]
    assert entries == []


def test_agents_md_points_at_the_build_log() -> None:
    assert BUILD_LOG.is_file()
    assert "docs/build-log.md" in AGENTS.read_text(encoding="utf-8")
