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
