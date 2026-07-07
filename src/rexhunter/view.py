"""The projection reducer (ADR invariant 2 — the pillar the whole system exists to serve).

Everything the retro-game frontend shows is DERIVED from the event log, never authoritative on its
own (invariant 2). This module is that derivation: a PURE, TOTAL fold of trajectory rows into a
``ViewState`` a dumb renderer draws. The live renderer and the ghost replayer are ONE reducer fed by
two cursors — the global ``id`` cursor (live feed) and the per-run ``seq`` cursor (ghost replay).
Within a run, single-writer monotonic append (inv 7) makes them the same sequence, so one reducer
serves both.

Purity is the point (mirrors cost.py's "below the loop" rule, cost.py:11-13): this module imports
ONLY ``events`` (the union) and ``cost`` (the spend fold) — no db/hub/server/loop/verdicts, no I/O.
That is what makes it Hypothesis-testable in isolation, and it is asserted by the gate
(tests/test_view.py::test_view_imports_only_events_and_cost).

Three laws hold, each the view-layer image of a proven P3 streaming guarantee:
  * **incremental == batch** — ``fold_from(fold(prefix), tail) == fold(prefix ++ tail)`` (the
    catch-up + live-splice guarantee: a reconnecting viewer folds only the missed tail).
  * **idempotent replay** — a row whose global ``id`` is at or below the high-water mark is a no-op
    (the monotonic-id dedup: a reconnect double-delivers nothing).
  * **determinism** — same (log, clock) → same ViewState, no ``now()``/random inside.

The clock (invariant 5, "derive don't store"): day/night is folded from an INJECTED ``datetime``,
never a wall-clock read — the live renderer passes the current instant, the ghost replayer passes
the scrubbed event's time. The fold is clock-free (so the two laws above are clock-independent);
the clock enters only in ``finalize``.

Scope: the TRAJECTORY tier only. Run outcome / liveness (``runs.outcome``) and a run's territory
are NOT events — terminal decisions close a run through the ``runs`` table with no trajectory event
(ADR Pillar 2 reconciliation) — so the pure fold NEVER sets them: ``RunView.territory``/``outcome``
and ``ViewState.territories`` default empty here and are overlaid by the assembler from ``runs``
(the runs ⊕ tier — invariant 2's "tables transactionally maintained alongside it" clause). Verdict
STATUS is the ``pen_events`` tier (verdicts.fold), composed OUTSIDE this pure reducer (invariant 7 —
a verdict is not a trajectory event of the closed run); the pen here is the capture base only.
"""

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import NamedTuple, assert_never

from rexhunter import cost, events
from rexhunter.events import TrajectoryEvent

# Day runs [06:00, 18:00); everything else is night. A pure function of the injected clock's hour.
_DAY_START_HOUR = 6
_NIGHT_START_HOUR = 18

# The capture-time prey status the trajectory tier knows (a PreyCapturedEvent implies it). The real
# current status is the ⊕ pen_events tier, overlaid by the assembler; re-declared here (not imported
# from verdicts) so view.py stays pure — it mirrors verdicts.AWAITING / the prey.status projection.
_AWAITING_VERDICT = "awaiting_verdict"


class LogRow(NamedTuple):
    """One stored event plus the columns the payload union does not carry (grep ``created_at``
    events.py → empty). ``id`` is the global stream cursor (the high-water dedup key), ``seq`` the
    per-run replay cursor, ``run_id`` the bucket, ``created_at`` the ghost clock source. The reducer
    needs the columns; the frozen tuple pairs them with the typed event for the fold."""

    id: int
    seq: int
    run_id: str
    created_at: str
    event: TrajectoryEvent


class Phase(Enum):
    """Day/night terrarium phase — derived from the injected clock (invariant 5), never stored."""

    DAY = "day"
    NIGHT = "night"


@dataclass(frozen=True)
class PreyCard:
    """One penned posting. ``status``/``reason``/``provenance`` are the capture-time base: the pure
    tier sets ``awaiting_verdict`` (a PreyCapturedEvent implies it), and the assembler overlays the
    real ⊕ pen_events status OUTSIDE this reducer (invariant 7 — a verdict is not a trajectory event
    of the closed run)."""

    prey_id: str
    territory: str
    posting: str
    status: str = _AWAITING_VERDICT
    reason: str | None = None
    provenance: str | None = None


@dataclass(frozen=True)
class RunView:
    """The trajectory-derivable state of one run, plus the runs ⊕ overlay base.
    ``territory``/``outcome`` are ``runs``-table facts with no trajectory event (terminal decisions
    emit none — the settled ADR Pillar 2/4 reconciliation): the pure fold NEVER sets them (defaults
    only); the assembler overlays them from ``runs``, like the pen's verdict status. ``outcome``
    None = still live (or not overlaid). ``turns`` is the stamina numerator: turns Rex ACTED —
    one per ToolCallEvent (an unknown-tool/invalid-args iteration appends only an ErrorEvent and
    costs no turn). ``current_tool`` is the tool of the last ToolCallEvent not yet closed by a
    result (or a matching error) — paired by run-scoped position, not ``tool_use_id`` (which is
    "" on the stub, loop.py:198,233)."""

    run_id: str
    current_tool: str | None
    thinking: str
    spent_usd: float
    prey_count: int
    error_count: int
    turns: int = 0
    territory: str | None = None
    outcome: str | None = None


@dataclass(frozen=True)
class TerritoryView:
    """One territory's scene state: its latest run (by ``started_at``), a ``runs``-table derivation
    mirroring the /snapshot territory dict's SHAPE (server.snapshot_state — its JSON key is
    ``last_outcome``; the name is deliberately not copied). The pure tier always emits the empty
    ``ViewState.territories`` default; the assembler fills it (the runs ⊕ tier).
    ``latest_outcome`` None = that latest run is still live."""

    territory: str
    latest_outcome: str | None
    last_started_at: str


@dataclass(frozen=True)
class ViewState:
    """The immutable snapshot a dumb renderer draws. A value type (frozen) so equality is structural
    and deterministic — the basis of the determinism law. ``territories`` is the runs ⊕ tier's
    per-territory face — always () from the pure fold, filled by the assembler."""

    high_water: int
    runs: tuple[RunView, ...]
    pen: tuple[PreyCard, ...]
    spent_usd: float
    phase: Phase
    territories: tuple[TerritoryView, ...] = ()


@dataclass
class _RunAcc:
    """Mutable per-run fold state. Holds only primitives + the run's UsageEvents, so an Accumulator
    deep-copies cheaply (fold_from) and compares by value (the incremental==batch law)."""

    open_tool: str | None = None
    thinking: str = ""
    prey_count: int = 0
    error_count: int = 0
    turns: int = 0
    usage: list[events.UsageEvent] = field(default_factory=list[events.UsageEvent])


@dataclass
class Accumulator:
    """The resumable fold state. ``high_water`` is the global-id dedup key; ``runs`` buckets per-run
    state in first-seen order; ``pen`` collects captures. Value equality (dataclass ``__eq__``) is
    what the incremental==batch law asserts on."""

    high_water: int = 0
    runs: dict[str, _RunAcc] = field(default_factory=dict[str, _RunAcc])
    pen: list[PreyCard] = field(default_factory=list[PreyCard])


def _apply(acc: Accumulator, row: LogRow) -> None:
    """Fold one already-deduped row into the accumulator (mutating). Total over the whole union."""
    run = acc.runs.get(row.run_id)
    if run is None:
        run = _RunAcc()
        acc.runs[row.run_id] = run
    event = row.event
    match event:
        case events.ToolCallEvent():
            run.open_tool = event.tool  # a call opens; the last unclosed one is `current_tool`
            run.turns += 1  # one dispatch = one turn Rex acted (the stamina numerator)
        case events.ToolResultEvent():
            run.open_tool = None  # a result closes the outstanding call (loop runs one tool/iter)
        case events.ErrorEvent():
            run.error_count += 1
            if event.tool == run.open_tool:
                run.open_tool = None  # a tool's own error closes it; a <brain>/<loop> error doesn't
        case events.PreyCapturedEvent():
            run.prey_count += 1
            # raw_posting is inv-6 bytes and MAY be non-utf8 — decode defensively to stay total.
            acc.pen.append(
                PreyCard(
                    prey_id=event.prey_id,
                    territory=event.territory,
                    posting=event.raw_posting.decode(errors="replace"),
                )
            )
        case events.ThinkingDelta():
            run.thinking += event.text  # the live consciousness buffer (this event's real consumer)
        case events.UsageEvent():
            run.usage.append(event)  # spend is folded at finalize via cost.fold_cost (invariant 5)
        case events.SniffEvent():
            pass  # no producer in src/ (P2.1 vestige); folds to nothing, but stays total over it
        case _:
            assert_never(event)


def step(acc: Accumulator, row: LogRow) -> None:
    """Fold one row, idempotent on the GLOBAL ``id`` (never ``seq``): a row at or below the
    high-water mark is a replay already folded in — a no-op. This is the view-layer twin of the P3
    splice dropping ``id <= high-water`` (server.py:162), and it is what makes replay idempotent."""
    if row.id <= acc.high_water:
        return
    acc.high_water = row.id
    _apply(acc, row)


def fold(rows: Sequence[LogRow]) -> Accumulator:
    """Fold a row sequence into a fresh accumulator (clock-free)."""
    acc = Accumulator()
    for row in rows:
        step(acc, row)
    return acc


def fold_from(acc: Accumulator, rows: Sequence[LogRow]) -> Accumulator:
    """Continue folding ``rows`` onto a COPY of ``acc`` (side-effect free), returning the new
    accumulator. ``fold_from(fold(prefix), tail) == fold(prefix ++ tail)`` — the incremental==batch
    law, the view-layer image of P3 catch-up + live-splice == the whole live stream."""
    resumed = copy.deepcopy(acc)
    for row in rows:
        step(resumed, row)
    return resumed


def _phase(clock: datetime) -> Phase:
    return Phase.DAY if _DAY_START_HOUR <= clock.hour < _NIGHT_START_HOUR else Phase.NIGHT


def finalize(acc: Accumulator, clock: datetime) -> ViewState:
    """Render the accumulator into the immutable ViewState, applying day/night ONCE from the
    injected clock (invariant 5). Spend is ``cost.fold_cost`` over the usage sub-log — a fold, not a
    stored counter. Collections build in deterministic first-seen order (the determinism law)."""
    runs = tuple(
        RunView(
            run_id=run_id,
            current_tool=run.open_tool,
            thinking=run.thinking,
            spent_usd=cost.fold_cost(run.usage),
            prey_count=run.prey_count,
            error_count=run.error_count,
            turns=run.turns,
        )
        for run_id, run in acc.runs.items()
    )
    all_usage = [usage for run in acc.runs.values() for usage in run.usage]
    return ViewState(
        high_water=acc.high_water,
        runs=runs,
        pen=tuple(acc.pen),
        spent_usd=cost.fold_cost(all_usage),
        phase=_phase(clock),
    )


def project(rows: Sequence[LogRow], clock: datetime) -> ViewState:
    """The whole projection: ``finalize(fold(rows), clock)``. Pure and total — same (rows, clock)
    always yields the same ViewState (the determinism law). Feed it the global-id cursor for the
    live feed or a run's seq cursor for the ghost replay; within a run they are one sequence."""
    return finalize(fold(rows), clock)
