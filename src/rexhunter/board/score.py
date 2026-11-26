"""Eval tier 2 (`P2.5`): a hunt's catch scored against the gym's planted marks, one run at a time.

The score is a projection of the run's own log (invariants 2 and 5): folded from its typed events
whenever it is asked for, and never stored in a table or a file. A posting counts as read when the
run holds a successful `read_posting` result for it, and as caught when the run holds a
`PreyCapturedEvent` for it; a posting the run only saw in a list counts as neither. Each posting is
counted once, by its `(board, posting_id)`, under the mark the board's author planted on it
(`planted_marks`). Only the run's own board is scored; a posting of another board is counted apart
(`off_board`).

Precision is the share of the catch that is prey, and undefined (None) on an empty catch; recall is
the share of the board's prey that was caught. The slice sets no pass threshold: the score reports.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import aiosqlite

from rexhunter import cost, db
from rexhunter.board import Board
from rexhunter.board.gym import GymBoard
from rexhunter.events import (
    BrainExchangeEvent,
    ErrorEvent,
    PostingRef,
    PreyCapturedEvent,
    ToolResultEvent,
    TrajectoryEvent,
)

type Mark = Literal["prey", "decoy"]
type PostingKey = tuple[str, str]  # (board, posting_id), the reference a catch names a posting by
type Outcome = tuple[str, str | None]  # (runs.outcome, runs.abort_reason)

COMPLETED: Outcome = ("completed", None)
# gym-drift's list has a renamed field, which fails the board boundary: a hunt there should stop at
# that fatal tool failure. Every other board can be hunted to a catch.
EXPECTED_OUTCOMES: dict[str, Outcome] = {"gym-drift": ("aborted", "tool failure")}


def expected_outcome(territory: str) -> Outcome:
    return EXPECTED_OUTCOMES.get(territory, COMPLETED)


def planted_marks(
    boards: Mapping[str, Board], catalogue: Mapping[str, GymBoard]
) -> dict[PostingKey, Mark]:
    """Each posting's planted mark, under the reference a hunt names it by."""
    return {
        (name, posting.posting_id): posting.planted
        for name, board in boards.items()
        for posting in catalogue[board.token].postings
    }


def _share(part: int, whole: int) -> float | None:
    # Nothing to share out is undefined, never 0 or 1: an empty catch is neither clean nor dirty.
    return None if whole == 0 else part / whole


@dataclass(frozen=True)
class Tally:
    prey_on_board: int
    prey_read: int
    decoys_read: int
    prey_caught: int
    decoys_caught: int
    capture_errors: int  # `<capture>` ErrorEvents: catch references the run never read
    calls: int  # brain attempts that sent a request (BrainExchangeEvents)
    cost_usd: float

    @property
    def precision(self) -> float | None:
        return _share(self.prey_caught, self.prey_caught + self.decoys_caught)

    @property
    def recall(self) -> float | None:
        return _share(self.prey_caught, self.prey_on_board)


def total(tallies: Iterable[Tally]) -> Tally:
    """The micro totals: every count summed, so precision and recall weigh each posting once."""
    items = list(tallies)
    return Tally(
        prey_on_board=sum(t.prey_on_board for t in items),
        prey_read=sum(t.prey_read for t in items),
        decoys_read=sum(t.decoys_read for t in items),
        prey_caught=sum(t.prey_caught for t in items),
        decoys_caught=sum(t.decoys_caught for t in items),
        capture_errors=sum(t.capture_errors for t in items),
        calls=sum(t.calls for t in items),
        cost_usd=sum((t.cost_usd for t in items), 0.0),
    )


@dataclass(frozen=True)
class RunScore:
    run_id: str
    territory: str
    outcome: str | None  # None while the run is live
    abort_reason: str | None
    expected: Outcome
    off_board: int  # distinct postings read or caught on a board other than the run's
    tally: Tally

    @property
    def outcome_met(self) -> bool:
        return (self.outcome, self.abort_reason) == self.expected


def _mark(marks: Mapping[PostingKey, Mark], key: PostingKey) -> Mark:
    # A posting the run read or caught must come from a marked board: a score that skipped it would
    # look cleaner than the run was.
    try:
        return marks[key]
    except KeyError:
        raise ValueError(f"no planted mark for {key[0]} {key[1]}") from None


def _key(ref: PostingRef) -> PostingKey:
    return (ref.board, ref.posting_id)


def score_run(
    events: Sequence[TrajectoryEvent],
    marks: Mapping[PostingKey, Mark],
    *,
    run_id: str,
    territory: str,
    outcome: str | None,
    abort_reason: str | None,
) -> RunScore:
    """One run's score, folded from its events: the same events give the same score."""
    read = {
        _key(PostingRef.model_validate_json(e.raw_request))
        for e in events
        if isinstance(e, ToolResultEvent) and e.tool == "read_posting"
    }
    caught = {(e.board, e.posting_id) for e in events if isinstance(e, PreyCapturedEvent)}
    seen = read | caught
    own = {key for key in seen if key[0] == territory}

    def count(keys: set[PostingKey], mark: Mark) -> int:
        return sum(1 for key in keys & own if _mark(marks, key) == mark)

    tally = Tally(
        prey_on_board=sum(1 for k, m in marks.items() if k[0] == territory and m == "prey"),
        prey_read=count(read, "prey"),
        decoys_read=count(read, "decoy"),
        prey_caught=count(caught, "prey"),
        decoys_caught=count(caught, "decoy"),
        capture_errors=sum(
            1 for e in events if isinstance(e, ErrorEvent) and e.tool == "<capture>"
        ),
        calls=sum(1 for e in events if isinstance(e, BrainExchangeEvent)),
        cost_usd=cost.fold_cost(events),
    )
    return RunScore(
        run_id=run_id,
        territory=territory,
        outcome=outcome,
        abort_reason=abort_reason,
        expected=expected_outcome(territory),
        off_board=len(seen - own),
        tally=tally,
    )


def wave_total(scores: Sequence[RunScore]) -> Tally:
    """The wave's micro totals over the runs expected to complete, so gym-drift's planned stop
    neither helps nor hurts them."""
    return total(score.tally for score in scores if score.expected == COMPLETED)


async def score_log(conn: aiosqlite.Connection, marks: Mapping[PostingKey, Mark]) -> list[RunScore]:
    """Each run of a marked board in the log, scored, in the order the runs started. It only reads:
    each run's events cross the typed boundary (`db.read_events`, invariant 3) and nothing is
    written, so a read-only connection serves as well as any other."""
    boards = {board for board, _posting_id in marks}
    rows = await conn.execute_fetchall(
        "SELECT id, territory, outcome, abort_reason FROM runs ORDER BY started_at, id"
    )
    scores: list[RunScore] = []
    for run_id, territory, outcome, abort_reason in rows:
        if str(territory) not in boards:
            continue
        scores.append(
            score_run(
                await db.read_events(conn, str(run_id)),
                marks,
                run_id=str(run_id),
                territory=str(territory),
                outcome=None if outcome is None else str(outcome),
                abort_reason=None if abort_reason is None else str(abort_reason),
            )
        )
    return scores
