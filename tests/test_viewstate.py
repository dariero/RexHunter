"""Slice A · the ViewState assembler.

``build_viewstate`` composes the PURE trajectory-tier projection (``view.project``) with the
pen-events ⊕ tier (``verdicts.fold``) against the REAL log: a stub hunt's tool stream + capture
project into the run card + prey pen, and a human verdict overlays the prey's status. The assembler
FOLDS ``pen_events`` (invariant 2, rebuildable from the log) and the gate proves it agrees with the
maintained ``prey.status`` projection that ``/snapshot`` reads (server.py:194) — no drift.

``read_log_rows`` is the two-cursor adapter: the global ``id`` cursor (live feed, all runs) and the
per-run ``seq`` cursor (ghost replay) — the same two cursors invariant 2 names, now over SQLite.
"""

from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
import pytest

from rexhunter import db, events, verdicts, view, viewstate
from rexhunter.events import Verdict

pytestmark = pytest.mark.anyio

_CLOCK = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)  # noon -> DAY; injected, never a wall clock


async def _seed_stub_hunt(conn: aiosqlite.Connection) -> tuple[str, str]:
    """Persist one stub hunt's trajectory — a sniff tool_call -> tool_result, then a capture — the
    default daemon's emitted set (loop.py:195,229; verdicts.py:72). Returns (run_id, prey_id)."""
    run_id = await db.start_run(conn, territory="mock-gym")
    await db.append_event(conn, run_id, events.ToolCallEvent(tool="sniff", raw_request=b"{}"))
    await db.append_event(
        conn,
        run_id,
        events.ToolResultEvent(tool="sniff", raw_request=b"{}", raw_response=b"posting:mock-gym"),
    )
    prey_id = await verdicts.capture_prey(
        conn, run_id, territory="mock-gym", posting="posting:mock-gym"
    )
    return run_id, prey_id


async def test_build_viewstate_projects_the_penned_prey_awaiting(tmp_path: Path) -> None:
    """The whole path: real log -> read_log_rows -> project -> ⊕ overlay. With no verdict yet, the
    prey shows the capture-time ``awaiting_verdict`` status, and the run card reflects the closed
    tool stream."""
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id, prey_id = await _seed_stub_hunt(conn)
        state = await viewstate.build_viewstate(conn, _CLOCK)
        assert state.high_water == 3  # tool_call(1), tool_result(2), prey_captured(3)
        assert len(state.pen) == 1
        card = state.pen[0]
        assert card.prey_id == prey_id
        assert card.territory == "mock-gym"
        assert card.posting == "posting:mock-gym"
        assert card.status == "awaiting_verdict"
        assert card.reason is None
        (run,) = state.runs
        assert run.run_id == run_id
        assert run.current_tool is None  # the tool_result closed the sniff call
        assert run.prey_count == 1
    finally:
        await conn.close()


async def test_build_viewstate_overlays_a_feast_verdict(tmp_path: Path) -> None:
    """The ⊕ tier moves the prey off the awaiting base: a FEAST verdict -> status ``feasted``."""
    conn = await db.connect(tmp_path / "rex.db")
    try:
        _, prey_id = await _seed_stub_hunt(conn)
        assert await verdicts.submit_verdict(conn, prey_id, Verdict.FEAST)
        state = await viewstate.build_viewstate(conn, _CLOCK)
        assert state.pen[0].status == "feasted"
    finally:
        await conn.close()


async def test_assembler_status_agrees_with_the_maintained_prey_row(tmp_path: Path) -> None:
    """Consistency lock: the folded ⊕ status/reason (assembler, from pen_events) == the maintained
    ``prey.status``/``reason`` projection that ``/snapshot`` reads. fold reproduces the row — the
    assembler (log-derived) and the maintained table never disagree."""
    conn = await db.connect(tmp_path / "rex.db")
    try:
        _, prey_id = await _seed_stub_hunt(conn)
        assert await verdicts.submit_verdict(
            conn, prey_id, Verdict.RELEASE, reason="not an AI-eng role"
        )
        state = await viewstate.build_viewstate(conn, _CLOCK)
        card = next(c for c in state.pen if c.prey_id == prey_id)
        async with conn.execute("SELECT status, reason FROM prey WHERE id = ?", (prey_id,)) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert card.status == str(row[0]) == "released"
        assert card.reason == row[1] == "not an AI-eng role"
    finally:
        await conn.close()


async def test_two_cursors_agree_on_the_real_log(tmp_path: Path) -> None:
    """Invariant 2 "one renderer, two cursors" on REAL persisted bytes: the same run read via the
    global ``id`` cursor (all runs, ORDER BY id) and via the per-run ``seq`` cursor (ghost, ORDER BY
    seq) projects to the identical ViewState — single-writer monotonic append (inv 7) makes them one
    sequence. A contract lock on the adapter, not a discriminating property."""
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id, _ = await _seed_stub_hunt(conn)
        by_id = await viewstate.read_log_rows(conn)  # global/live cursor
        by_seq = await viewstate.read_log_rows(conn, run_id=run_id)  # per-run ghost cursor
        assert view.project(by_id, _CLOCK) == view.project(by_seq, _CLOCK)
    finally:
        await conn.close()


async def test_build_viewstate_on_an_empty_log_is_empty(tmp_path: Path) -> None:
    """Totality at the assembler boundary: an empty log projects to an empty ViewState, no query on
    a missing prey/pen."""
    conn = await db.connect(tmp_path / "rex.db")
    try:
        state = await viewstate.build_viewstate(conn, _CLOCK)
        assert state.high_water == 0
        assert state.pen == ()
        assert state.runs == ()
    finally:
        await conn.close()


# ── Step 1 · the runs ⊕ overlay tier: run outcome/territory + territories, from `runs` ───────────


async def test_overlay_fills_run_territory_and_outcome(tmp_path: Path) -> None:
    """The runs ⊕ tier: a closed run's RunView carries its territory and outcome, overlaid from the
    `runs` table (invariant 2's "tables transactionally maintained alongside" clause) — NO
    trajectory event carries either (terminal decisions emit no event, the settled ADR Pillar 2/4
    reconciliation), so this is assembler work, never the pure fold's."""
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id, _ = await _seed_stub_hunt(conn)
        await db.finish_run(conn, run_id, outcome="completed")
        state = await viewstate.build_viewstate(conn, _CLOCK)
        (run,) = state.runs
        assert run.run_id == run_id
        assert run.territory == "mock-gym"
        assert run.outcome == "completed"
    finally:
        await conn.close()


async def test_open_run_overlays_none_outcome(tmp_path: Path) -> None:
    """A run never closed is still live: the overlay stamps outcome None (runs.outcome IS NULL),
    while territory — not liveness-dependent — is filled."""
    conn = await db.connect(tmp_path / "rex.db")
    try:
        await _seed_stub_hunt(conn)
        state = await viewstate.build_viewstate(conn, _CLOCK)
        (run,) = state.runs
        assert run.outcome is None
        assert run.territory == "mock-gym"
    finally:
        await conn.close()


async def test_territories_tier_reflects_latest_run_per_territory(tmp_path: Path) -> None:
    """ViewState.territories shows each territory's LATEST run (MAX(started_at)) — the scene-tile
    source, the same GROUP BY /snapshot derives (server.snapshot_state). started_at is pinned by
    explicit UPDATEs so the MAX winner is deterministic, never a wall-clock microsecond race. The
    three runs carry no trajectory events: the tier is runs-table-driven by construction."""
    conn = await db.connect(tmp_path / "rex.db")
    try:
        old = await db.start_run(conn, territory="mock-gym")
        new = await db.start_run(conn, territory="mock-gym")
        other = await db.start_run(conn, territory="greenhouse")
        await db.finish_run(conn, old, outcome="completed")
        await db.finish_run(conn, new, outcome="needs_help")
        await db.finish_run(conn, other, outcome="completed")
        for run_id, started_at in [
            (old, "2026-07-01T00:00:00+00:00"),
            (new, "2026-07-02T00:00:00+00:00"),
            (other, "2026-07-03T00:00:00+00:00"),
        ]:
            await conn.execute("UPDATE runs SET started_at = ? WHERE id = ?", (started_at, run_id))
        await conn.commit()
        state = await viewstate.build_viewstate(conn, _CLOCK)
        assert [(t.territory, t.latest_outcome, t.last_started_at) for t in state.territories] == [
            ("greenhouse", "completed", "2026-07-03T00:00:00+00:00"),
            ("mock-gym", "needs_help", "2026-07-02T00:00:00+00:00"),
        ]
    finally:
        await conn.close()


# ── Step 4a · budget ceilings: recorded at start, served via the runs ⊕ overlay ──────────────────


async def test_overlay_fills_run_budget_ceilings(tmp_path: Path) -> None:
    """The per-run denominators (Step 5's HP/stamina bars): ceilings recorded on the runs row at
    start_run are overlaid onto the RunView by the same runs ⊕ tier that fills
    territory/outcome — a RECORDED input fact, not injected config (a ghost replays under the
    caps it actually ran under)."""
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await db.start_run(
            conn, territory="mock-gym", cost_ceiling_usd=0.05, max_iterations=3
        )
        await db.append_event(conn, run_id, events.ToolCallEvent(tool="sniff", raw_request=b"{}"))
        state = await viewstate.build_viewstate(conn, _CLOCK)
        (run,) = state.runs
        assert run.cost_ceiling_usd == 0.05
        assert run.max_iterations == 3
    finally:
        await conn.close()


async def test_pre_4a_run_overlays_none_ceilings(tmp_path: Path) -> None:
    """A run started without ceilings (NULL columns — every pre-4a row's shape) overlays None on
    both: backward-compatible and total. Step 5 draws no bar for such a run."""
    conn = await db.connect(tmp_path / "rex.db")
    try:
        await _seed_stub_hunt(conn)  # plain start_run: NULL ceiling columns
        state = await viewstate.build_viewstate(conn, _CLOCK)
        (run,) = state.runs
        assert run.cost_ceiling_usd is None
        assert run.max_iterations is None
    finally:
        await conn.close()


async def test_recorded_ceiling_is_immutable_across_reads(tmp_path: Path) -> None:
    """Write-once/no-drift (inv 5): the ceilings are an input fact recorded at start — like
    territory/started_at — never re-derived or re-written. Pinned against the ONE post-start
    UPDATE a runs row ever receives (finish_run): closing the run moves outcome, not the caps."""
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await db.start_run(
            conn, territory="mock-gym", cost_ceiling_usd=0.2, max_iterations=7
        )
        await db.append_event(conn, run_id, events.ToolCallEvent(tool="sniff", raw_request=b"{}"))
        (before,) = (await viewstate.build_viewstate(conn, _CLOCK)).runs
        await db.finish_run(conn, run_id, outcome="completed")
        (after,) = (await viewstate.build_viewstate(conn, _CLOCK)).runs
        assert (before.cost_ceiling_usd, before.max_iterations) == (0.2, 7)
        assert (after.cost_ceiling_usd, after.max_iterations) == (0.2, 7)
        assert after.outcome == "completed"  # finish_run wrote outcome; the ceilings didn't move
    finally:
        await conn.close()


async def test_connect_migrates_a_pre_4a_runs_table(tmp_path: Path) -> None:
    """The upgrade pin (DoD #1's spirit: an upgrade never strands an existing log). CREATE TABLE
    IF NOT EXISTS skips an existing `runs`, so a pre-4a database file lacks the ceiling COLUMNS
    entirely — db.connect must ALTER them in (idempotent, nullable, O(1)) or the next boot's
    start_run dies on 'no such column' while every fresh-DB gate stays green."""
    db_path = tmp_path / "rex.db"
    old = await aiosqlite.connect(db_path)
    try:
        await old.execute(
            "CREATE TABLE runs (id TEXT PRIMARY KEY, territory TEXT NOT NULL,"
            " started_at TEXT NOT NULL, ended_at TEXT, outcome TEXT, abort_reason TEXT)"
        )
        await old.execute(
            "INSERT INTO runs (id, territory, started_at)"
            " VALUES ('old-run', 'mock-gym', '2026-06-01T00:00:00+00:00')"
        )
        await old.commit()
    finally:
        await old.close()

    conn = await db.connect(db_path)  # the reopen that must migrate
    try:
        run_id = await db.start_run(
            conn, territory="mock-gym", cost_ceiling_usd=0.05, max_iterations=3
        )
        await db.append_event(conn, run_id, events.ToolCallEvent(tool="sniff", raw_request=b"{}"))
        await db.append_event(
            conn, "old-run", events.ToolCallEvent(tool="sniff", raw_request=b"{}")
        )
        state = await viewstate.build_viewstate(conn, _CLOCK)
        new_view = next(rv for rv in state.runs if rv.run_id == run_id)
        old_view = next(rv for rv in state.runs if rv.run_id == "old-run")
        assert (new_view.cost_ceiling_usd, new_view.max_iterations) == (0.05, 3)
        assert (old_view.cost_ceiling_usd, old_view.max_iterations) == (None, None)
    finally:
        await conn.close()


# ── Step 4b · injected daemon ceiling + schedule: the territory union ─────────────────────────────


async def test_build_viewstate_threads_daemon_ceiling_and_schedule(tmp_path: Path) -> None:
    """The assembler passes the injected values through uninterpreted: the ceiling lands on
    ViewState, and a scheduled-but-never-hunted territory finally EXISTS — a dormant tile
    (latest_outcome None AND last_started_at None; an open run would carry a started_at)."""
    conn = await db.connect(tmp_path / "rex.db")
    try:
        state = await viewstate.build_viewstate(
            conn, _CLOCK, daemon_spend_ceiling_usd=0.75, schedule=("never-hunted",)
        )
        assert state.daemon_spend_ceiling_usd == 0.75
        (tile,) = state.territories
        assert tile.territory == "never-hunted"
        assert tile.latest_outcome is None
        assert tile.last_started_at is None  # dormant: never hunted, not an open run
    finally:
        await conn.close()


async def test_scheduled_and_hunted_territory_keeps_its_outcome(tmp_path: Path) -> None:
    """Union with overlay-wins: a territory in BOTH schedule and runs shows its run-derived
    outcome (the dormant base never clobbers a real tile); its scheduled sibling stays dormant."""
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await db.start_run(conn, territory="mock-gym")
        await db.finish_run(conn, run_id, outcome="completed")
        state = await viewstate.build_viewstate(conn, _CLOCK, schedule=("greenhouse", "mock-gym"))
        summary = [
            (t.territory, t.latest_outcome, t.last_started_at is None) for t in state.territories
        ]
        assert summary == [
            ("greenhouse", None, True),  # dormant: scheduled, never hunted
            ("mock-gym", "completed", False),  # the runs overlay won over the dormant base
        ]
    finally:
        await conn.close()


async def test_hunted_but_unscheduled_territory_still_shows(tmp_path: Path) -> None:
    """Union, not intersection: a retired territory (runs exist, no longer scheduled) keeps its
    history on the board alongside the scheduled dormant one."""
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await db.start_run(conn, territory="mock-gym")
        await db.finish_run(conn, run_id, outcome="completed")
        state = await viewstate.build_viewstate(conn, _CLOCK, schedule=("greenhouse",))
        assert [(t.territory, t.latest_outcome) for t in state.territories] == [
            ("greenhouse", None),
            ("mock-gym", "completed"),
        ]
    finally:
        await conn.close()


async def test_ghost_cursor_stamps_current_outcome_is_a_documented_deferral(
    tmp_path: Path,
) -> None:
    """PINS a documented deferral, not an ideal: the ghost (per-run seq) cursor overlays the run's
    CURRENT outcome — a replay of a completed run shows "completed" at every scrub position,
    because outcome lives on `runs` (one row, no history), not in the trajectory. Revisit when
    ghost replay gets a UI (outcome-as-of-position needs a position-aware overlay)."""
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id, _ = await _seed_stub_hunt(conn)
        await db.finish_run(conn, run_id, outcome="completed")
        state = await viewstate.build_viewstate(conn, _CLOCK, run_id=run_id)
        (run,) = state.runs
        assert run.outcome == "completed"
    finally:
        await conn.close()
