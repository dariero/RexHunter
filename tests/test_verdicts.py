"""P4 · prey pen + verdict machine.

Unit 1 (here): capture on completion. A hunt that finishes with a catch writes prey rows with
status='awaiting_verdict'; the capture is a run-scoped trajectory event (invariant 7) carrying
the raw posting bytes (invariant 6), and the event + row land in ONE transaction (atomicity).
Later units grow this file with the verdict state machine and the enqueued follow-up job.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import aiosqlite
import pytest

from rexhunter import db, verdicts
from rexhunter.events import PreyCapturedEvent
from rexhunter.loop import Brain, Decision, HuntComplete, ToolCallDecision, run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

BrainFactory = Callable[[Sequence[Decision]], Brain]

PreyRow = tuple[str, str, str, str, str, str | None]


async def pen_rows(conn: aiosqlite.Connection) -> list[PreyRow]:
    async with conn.execute(
        "SELECT id, run_id, territory, posting, status, decided_at FROM prey ORDER BY captured_at"
    ) as cur:
        return [
            (str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]), r[5])
            for r in await cur.fetchall()
        ]


async def test_completed_hunt_captures_prey_as_awaiting_verdict(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    reg = ToolRegistry()

    @reg.tool
    async def sniff(prey: str) -> str:
        return f"posting:{prey}"

    conn = await db.connect(tmp_path / "rex.db")
    try:
        brain = scripted_brain(
            [
                ToolCallDecision(tool=sniff.__name__, args={"prey": "acme"}),
                HuntComplete(catch=["posting:acme", "posting:globex"]),
            ]
        )
        run_id = await run_hunt(conn, territory="mock-gym", brain=brain, registry=reg)

        rows = await pen_rows(conn)
        assert len(rows) == 2
        assert all(r[1] == run_id for r in rows)  # run provenance: which hunt caught it
        assert all(r[2] == "mock-gym" for r in rows)  # territory
        assert {r[3] for r in rows} == {"posting:acme", "posting:globex"}
        assert all(r[4] == "awaiting_verdict" for r in rows)  # the pending status
        assert all(r[5] is None for r in rows)  # not yet decided

        # capture is a run-scoped trajectory event (invariant 7) carrying raw posting bytes (inv 6)
        events = await db.read_events(conn, run_id)
        captures = [e for e in events if isinstance(e, PreyCapturedEvent)]
        assert len(captures) == 2
        assert {c.raw_posting for c in captures} == {b"posting:acme", b"posting:globex"}
        assert {c.prey_id for c in captures} == {r[0] for r in rows}  # event ↔ row pairing
    finally:
        await conn.close()


async def test_completion_without_catch_pens_nothing(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    reg = ToolRegistry()
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await run_hunt(
            conn, territory="t", brain=scripted_brain([HuntComplete()]), registry=reg
        )
        assert await pen_rows(conn) == []
        events = await db.read_events(conn, run_id)
        assert [e for e in events if isinstance(e, PreyCapturedEvent)] == []
    finally:
        await conn.close()


async def test_capture_prey_is_atomic(tmp_path: Path) -> None:
    # The PreyCapturedEvent and the prey row are ONE transaction: if the row INSERT fails, the
    # event must not durably land either. Proven on a FRESH connection (an uncommitted write is
    # invisible to a second reader), so this is real durability, not same-txn read-your-write.
    conn = await db.connect(tmp_path / "rex.db")
    run_id = await db.start_run(conn, territory="t")

    real_execute = conn.execute

    def fail_on_prey_insert(sql: str, *args: object, **kwargs: object) -> object:
        if "INSERT INTO prey" in sql:
            raise RuntimeError("disk full mid-capture")
        return real_execute(sql, *args, **kwargs)  # type: ignore[arg-type]

    conn.execute = fail_on_prey_insert  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await verdicts.capture_prey(conn, run_id, territory="t", posting="posting:acme")
    conn.execute = real_execute  # type: ignore[method-assign]
    await conn.rollback()  # discard the broken transaction
    await conn.close()

    fresh = await db.connect(tmp_path / "rex.db")  # the "second reader" / a restart
    try:
        events = await db.read_events(fresh, run_id)
        assert [e for e in events if isinstance(e, PreyCapturedEvent)] == []  # event rolled back
        async with fresh.execute("SELECT COUNT(*) FROM prey") as cur:
            row = await cur.fetchone()
        assert row is not None and row[0] == 0  # row rolled back
    finally:
        await fresh.close()
