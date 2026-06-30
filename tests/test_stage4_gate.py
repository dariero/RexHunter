"""Stage 4 gate — ADR definition-of-done #4 (the prey-pen clause).

The slice's reason to exist: **crash-equivalence**. A process restart with prey in the pen must
be indistinguishable from a quiet moment — verdicts still work, nothing pending is lost, and a
verdict caught mid-restart is never double-applied. Proven through the REAL ASGI lifespan (the
`lifespan_context` path P2.3-wiring established) — booting the daemon, draining it, and booting
it AGAIN over the same database file is the restart — never a harness mock.

Park-and-persist (DoD #4's other clause) is deferred to a later increment; see the ADR.
"""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import aiosqlite
import pytest

from rexhunter import db, server, stub, verdicts
from rexhunter.events import Verdict
from rexhunter.loop import Brain, Decision, HuntComplete
from rexhunter.tools import ToolRegistry
from rexhunter.verdicts import Drafter

pytestmark = pytest.mark.anyio

BrainFactory = Callable[[Sequence[Decision]], Brain]
Config = tuple[Mapping[str, float], Callable[[str], Brain], ToolRegistry, int, Drafter]


async def _status(conn: aiosqlite.Connection, prey_id: str) -> str:
    async with conn.execute("SELECT status FROM prey WHERE id = ?", (prey_id,)) as cur:
        row = await cur.fetchone()
    assert row is not None
    return str(row[0])


async def _count(conn: aiosqlite.Connection, table: str, prey_id: str) -> int:
    async with conn.execute(f"SELECT COUNT(*) FROM {table} WHERE prey_id = ?", (prey_id,)) as cur:
        row = await cur.fetchone()
    assert row is not None
    return int(row[0])


async def _poll(conn: aiosqlite.Connection, sql: str, params: tuple[object, ...]) -> object:
    """Poll a single-value query until it is non-null (bounded so the suite can never hang)."""
    for _ in range(500):
        async with conn.execute(sql, params) as cur:
            row = await cur.fetchone()
        if row is not None and row[0] is not None:
            return row[0]
        await asyncio.sleep(0.01)
    raise AssertionError(f"condition never held: {sql!r}")


async def test_prey_survives_a_real_restart_and_verdicts_stay_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scripted_brain: BrainFactory
) -> None:
    db_path = tmp_path / "rex.db"

    def brain_for(territory: str) -> Brain:
        # the hunt completes immediately, capturing one posting — no tool, no spend
        return scripted_brain([HuntComplete(catch=[f"posting:{territory}"])])

    async def drafter(_conn: aiosqlite.Connection, prey_id: str) -> str:
        return f"draft for {prey_id}"  # stub; no LLM

    def cfg() -> Config:
        # one territory, a long interval so it hunts exactly ONCE per boot (deterministic pen)
        return {"gate": 3600.0}, brain_for, ToolRegistry(), 4, drafter

    monkeypatch.setattr(server, "DB_PATH", str(db_path))
    monkeypatch.setattr(stub, "daemon_config", cfg)

    # ── Boot #1: the daemon hunts once and pens one posting, now durable on disk. ───────────
    async with server.app.router.lifespan_context(server.app):
        reader = await db.connect(db_path)
        try:
            prey_id = str(
                await _poll(
                    reader,
                    "SELECT id FROM prey WHERE status = 'awaiting_verdict' ORDER BY captured_at"
                    " LIMIT 1",
                    (),
                )
            )
        finally:
            await reader.close()
    # exiting ran the REAL shutdown; the completed hunt's run is 'completed', nothing dangling.

    # ── RESTART: boot the daemon AGAIN over the same file — the real boot path (crash-sweep). ─
    async with server.app.router.lifespan_context(server.app):
        conn = await db.connect(db_path)
        try:
            # the pen survived the restart, untouched: still awaiting a verdict.
            assert await _status(conn, prey_id) == "awaiting_verdict"

            # verdicts still work after a restart.
            assert await verdicts.submit_verdict(conn, prey_id, Verdict.FEAST) is True
            # a double-click is a harmless no-op (idempotency within the process).
            assert await verdicts.submit_verdict(conn, prey_id, Verdict.FEAST) is False
            assert await _status(conn, prey_id) == "feasted"
            assert await _count(conn, "pen_events", prey_id) == 1  # one verdict event, not two

            # the running worker drains the enqueued draft_pitch job to a draft (Rex drafts).
            await _poll(
                conn, "SELECT result FROM jobs WHERE prey_id = ? AND status = 'done'", (prey_id,)
            )
        finally:
            await conn.close()

    # ── RESTART AGAIN: a verdict replayed across the restart must NOT double-apply. ──────────
    conn = await db.connect(db_path)
    try:
        assert await verdicts.submit_verdict(conn, prey_id, Verdict.FEAST) is False  # still a no-op
        assert await _status(conn, prey_id) == "feasted"
        assert await _count(conn, "pen_events", prey_id) == 1  # one verdict event, ever
        assert await _count(conn, "jobs", prey_id) == 1  # one job, ever — not re-enqueued
    finally:
        await conn.close()


async def test_boot_crash_sweep_marks_the_run_but_leaves_prey_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scripted_brain: BrainFactory
) -> None:
    # The crash-between-capture-and-finish_run miniature: a hunt pens a posting, then the process
    # dies BEFORE the run closes. At reboot the dangling RUN is marked 'crashed' — but the prey,
    # already committed, survives independent of the run's label (mark_crashed_runs only touches
    # runs WHERE outcome IS NULL). Crash-equivalence in the small.
    db_path = tmp_path / "rex.db"

    seed = await db.connect(db_path)
    run_id = await db.start_run(seed, territory="gate")
    prey_id = await verdicts.capture_prey(seed, run_id, territory="gate", posting="posting:gate")
    await seed.close()  # the process "dies" — the run never reached finish_run

    def brain_for(_territory: str) -> Brain:
        return scripted_brain([HuntComplete()])  # unused: the schedule below is empty

    async def drafter(_conn: aiosqlite.Connection, _prey_id: str) -> str:
        return "draft"

    def cfg() -> Config:
        return {}, brain_for, ToolRegistry(), 4, drafter  # no territories: no new hunts, clean pen

    monkeypatch.setattr(server, "DB_PATH", str(db_path))
    monkeypatch.setattr(stub, "daemon_config", cfg)

    async with server.app.router.lifespan_context(server.app):
        conn = await db.connect(db_path)
        try:
            async with conn.execute("SELECT outcome FROM runs WHERE id = ?", (run_id,)) as cur:
                row = await cur.fetchone()
            assert row is not None and row[0] == "crashed"  # the dangling run was swept

            assert await _status(conn, prey_id) == "awaiting_verdict"  # the prey survived untouched
            # and it still takes a verdict after the reboot
            assert (
                await verdicts.submit_verdict(conn, prey_id, Verdict.RELEASE, reason="stale")
                is True
            )
            assert await _status(conn, prey_id) == "released"
        finally:
            await conn.close()
