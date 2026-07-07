"""The ViewState assembler (ADR invariant 2) — composes the pure trajectory-tier projection with the
pen-events ⊕ tier into the full ViewState the frontend renders.

This is the one layer that spans tiers. It reads the log (``read_log_rows``), runs the PURE reducer
(``view.project`` — trajectory tier), overlays each run's territory/outcome and the per-territory
tier from the ``runs`` table (the runs ⊕ tier — terminal decisions emit no trajectory event, the
settled ADR Pillar 2/4 reconciliation, so these are facts of a table transactionally maintained
alongside the log, invariant 2), then overlays each prey's verdict status by folding its
``pen_events`` (``verdicts.fold`` — the pen ⊕ tier, OUTSIDE the pure reducer because a verdict is
not a trajectory event of the closed run, invariant 7). ``view.py`` stays pure (never imports
verdicts or db); ``db.py`` stays Pillar 1. The logs are truth; the maintained ``prey.status`` is a
convenience the assembler agrees with (proven by the gate), never one it depends on. Both ⊕ tiers
apply on BOTH cursors — the ghost cursor thus stamps a run's CURRENT outcome, a documented deferral
(pinned by test) to revisit when ghost replay gets a UI.

``read_log_rows`` is the two-cursor adapter (inv 2): no ``run_id`` → the global ``id`` cursor (the
live feed, all runs); a ``run_id`` → the per-run ``seq`` cursor (the ghost replay). Within a run
they are one sequence (single-writer append, inv 7).
"""

import dataclasses
from datetime import datetime

import aiosqlite

from rexhunter import events, verdicts, view
from rexhunter.view import LogRow, TerritoryView, ViewState

# The columns the payload union does not carry (the reducer needs id/seq/run_id/created_at alongside
# the decoded event). One SELECT, two orderings — the two cursors below.
_LOG_ROWS_SELECT = "SELECT id, seq, run_id, created_at, payload FROM trajectory_events"


async def read_log_rows(conn: aiosqlite.Connection, *, run_id: str | None = None) -> list[LogRow]:
    """Read the trajectory log into typed ``LogRow``s via one of the two cursors (invariant 2).

    ``run_id is None`` → all runs, ``ORDER BY id`` (the live/catch-up global cursor, cf.
    server.catch_up). A ``run_id`` → that run only, ``ORDER BY seq`` (the ghost-replay cursor, cf.
    db.read_events). Each payload crosses the validation boundary (invariant 3) via decode_event.
    """
    if run_id is None:
        sql, params = f"{_LOG_ROWS_SELECT} ORDER BY id", ()
    else:
        sql, params = f"{_LOG_ROWS_SELECT} WHERE run_id = ? ORDER BY seq", (run_id,)
    async with conn.execute(sql, params) as cursor:
        rows = await cursor.fetchall()
    return [
        LogRow(
            id=int(row[0]),
            seq=int(row[1]),
            run_id=str(row[2]),
            created_at=str(row[3]),
            event=events.decode_event(str(row[4])),
        )
        for row in rows
    ]


async def build_viewstate(
    conn: aiosqlite.Connection,
    clock: datetime,
    *,
    run_id: str | None = None,
    daemon_spend_ceiling_usd: float | None = None,
    schedule: tuple[str, ...] = (),
) -> ViewState:
    """Assemble the full ViewState: project the log (trajectory tier), overlay each run's
    territory/outcome/ceilings + the territory tier from ``runs`` (the runs ⊕ tier), then overlay
    each prey's verdict status folded from its ``pen_events`` (the pen ⊕ tier). ``clock``,
    ``daemon_spend_ceiling_usd`` and ``schedule`` are injected (invariant 5's never-stored
    category) and pass through to ``view.project`` uninterpreted; ``run_id`` selects the ghost
    cursor for a single-run replay (default: the live all-runs feed)."""
    state = view.project(
        await read_log_rows(conn, run_id=run_id),
        clock,
        daemon_spend_ceiling_usd=daemon_spend_ceiling_usd,
        schedule=schedule,
    )

    # The runs ⊕ tier: territory/outcome/ceilings are runs-table facts (no trajectory event
    # carries them; the ceilings are recorded write-once at start_run — 4a). `.get` keeps the
    # overlay total over a RunView with no runs row (can't happen live — the FK forbids it — but
    # a defensive default beats a KeyError, cf. verdicts.fold's guard).
    async with conn.execute(
        "SELECT id, territory, outcome, cost_ceiling_usd, max_iterations FROM runs"
    ) as cursor:
        run_rows = await cursor.fetchall()
    facts = {
        str(r[0]): (
            str(r[1]),
            None if r[2] is None else str(r[2]),
            None if r[3] is None else float(r[3]),
            None if r[4] is None else int(r[4]),
        )
        for r in run_rows
    }
    runs: list[view.RunView] = []
    for rv in state.runs:
        territory, outcome, ceiling_usd, max_iters = facts.get(
            rv.run_id, (rv.territory, rv.outcome, rv.cost_ceiling_usd, rv.max_iterations)
        )
        runs.append(
            dataclasses.replace(
                rv,
                territory=territory,
                outcome=outcome,
                cost_ceiling_usd=ceiling_usd,
                max_iterations=max_iters,
            )
        )

    # The territory tier: the schedule/runs-seen UNION (4b). The pure tier emitted the base from
    # the injected schedule; each runs-derived tile REPLACES its dormant twin or JOINS the set
    # (union — a retired territory keeps its history), then the whole set sorts by name (the
    # determinism the ORDER BY alone used to provide). The bare `outcome` column pairs with the
    # row achieving MAX(started_at) — SQLite's documented bare-column-with-MAX behaviour, the
    # same /snapshot already relies on (server.snapshot_state).
    async with conn.execute(
        "SELECT territory, outcome, MAX(started_at) FROM runs GROUP BY territory ORDER BY territory"
    ) as cursor:
        latest = await cursor.fetchall()
    merged = {tile.territory: tile for tile in state.territories}  # the dormant base
    for r in latest:
        merged[str(r[0])] = TerritoryView(
            territory=str(r[0]),
            latest_outcome=None if r[1] is None else str(r[1]),
            last_started_at=str(r[2]),
        )
    territories = tuple(sorted(merged.values(), key=lambda tile: tile.territory))
    state = dataclasses.replace(state, runs=tuple(runs), territories=territories)

    if not state.pen:
        return state
    pen: list[view.PreyCard] = []
    for card in state.pen:
        pen_events = await verdicts.read_pen_events(conn, card.prey_id)
        status, reason, provenance = verdicts.fold(pen_events)
        pen.append(dataclasses.replace(card, status=status, reason=reason, provenance=provenance))
    return dataclasses.replace(state, pen=tuple(pen))
