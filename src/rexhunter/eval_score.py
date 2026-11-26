"""P2.5: eval tier 2's score of an existing log, read-only. Spends nothing, needs no key.

    PYTHONPATH=src uv run python -m rexhunter.eval_score PATH

Each run of a gym board in the log at PATH is scored against the gym's planted marks
(`board.score`), and the table is printed: a row per run, then the micro totals over the runs
expected to complete, then the log's whole spend. The score is printed and never stored
(invariant 5): the log is opened read-only, and the file is left as it was.
"""

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

import aiosqlite

from rexhunter import cost, db
from rexhunter.board.catalogue import EVAL_BOARDS, EVAL_CATALOGUE, GYM_BOARDS, GYM_CATALOGUE
from rexhunter.board.score import (
    COMPLETED,
    EXPECTED_OUTCOMES,
    Outcome,
    RunScore,
    Tally,
    planted_marks,
    score_log,
    wave_total,
)

MARKS = planted_marks(GYM_BOARDS | EVAL_BOARDS, GYM_CATALOGUE | EVAL_CATALOGUE)
HEADER = (
    "territory",
    "outcome",
    "as expected",
    "prey",
    "read p/d",
    "caught p/d",
    "unread catch",
    "off board",
    "calls",
    "cost",
    "precision",
    "recall",
)


def _described(outcome: Outcome) -> str:
    name, reason = outcome
    return name if reason is None else f"{name}: {reason}"


# The expected outcomes come from the score's own table, so the legend cannot drift from it.
_EXPECTED = ", ".join(f"{t} {_described(o)}" for t, o in EXPECTED_OUTCOMES.items())
LEGEND = (
    "prey: on the board; read and caught: prey/decoys; unread catch: catch references capture "
    f"refused; as expected: {_EXPECTED}, every other board {_described(COMPLETED)}; precision is "
    "undefined (\N{EN DASH}) on an empty catch."
)


def read_only_uri(path: Path) -> str:
    # A log its writers have closed has no -wal or -shm: open it immutable, so the read leaves
    # nothing beside it. With either there, a writer may be live, or may have died with frames in
    # the WAL: read through the WAL and its locks, so no committed event is missed.
    uri = f"{path.resolve().as_uri()}?mode=ro"
    sidecars = (path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm"))
    return uri if any(p.exists() for p in sidecars) else f"{uri}&immutable=1"


def _ratio(value: float | None) -> str:
    return "\N{EN DASH}" if value is None else f"{value:.2f}"


def _outcome(score: RunScore) -> str:
    if score.outcome is None:
        return "live"
    return _described((score.outcome, score.abort_reason))


def _cells(territory: str, outcome: str, met: str, tally: Tally, off_board: str) -> list[str]:
    return [
        territory,
        outcome,
        met,
        str(tally.prey_on_board),
        f"{tally.prey_read}/{tally.decoys_read}",
        f"{tally.prey_caught}/{tally.decoys_caught}",
        str(tally.capture_errors),
        off_board,
        str(tally.calls),
        f"${tally.cost_usd:.4f}",
        _ratio(tally.precision),
        _ratio(tally.recall),
    ]


def render(scores: Sequence[RunScore], spend_usd: float) -> str:
    """The score table: a row per run in start order, the wave's micro totals, the spend."""
    expected = [s for s in scores if s.expected[0] == "completed"]
    rows = [list(HEADER)]
    rows += [
        _cells(
            s.territory, _outcome(s), "yes" if s.outcome_met else "no", s.tally, str(s.off_board)
        )
        for s in scores
    ]
    rows.append(
        _cells("wave", f"{len(expected)} runs expected to complete", "", wave_total(scores), "")
    )
    widths = [max(len(row[i]) for row in rows) for i in range(len(HEADER))]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in rows
    ]
    lines.insert(len(lines) - 1, "")
    return "\n".join([*lines, "", LEGEND, f"spend: ${spend_usd:.4f}, the whole log"])


async def score_text(path: Path) -> str:
    """The table for the log at `path`, an existing file, read through a read-only connection."""
    conn = await aiosqlite.connect(read_only_uri(path), uri=True)
    try:
        scores = await score_log(conn, MARKS)
        spend = cost.fold_cost(await db.read_usage(conn))
    finally:
        await conn.close()
    return render(scores, spend)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m rexhunter.eval_score",
        description="Print eval tier 2's score of a hunt log, read-only.",
        allow_abbrev=False,
    )
    parser.add_argument("path", type=Path, help="the log (a RexHunter SQLite file)")
    path: Path = parser.parse_args().path
    if not path.is_file():
        raise SystemExit(f"no log at {path}: eval_score reads an existing log and creates none.")
    print(asyncio.run(score_text(path)))


if __name__ == "__main__":
    main()
