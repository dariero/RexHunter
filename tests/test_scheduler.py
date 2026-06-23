"""P2.3 · scheduler — bounded concurrent orchestration (Unit 1), failure isolation +
cancellation (Unit 3), per-territory deadline timing (Unit 4).

Unit 1 here: the bounded task group runs N single-hunt loops, each its OWN connection, never
more than `max_concurrent` at once. The cap assertion is the trap the plan calls out — it
passes vacuously if hunts finish too fast to overlap. So every hunt's tool BLOCKS on a shared
gate: the test forces exactly `cap` hunts to pile up in-flight, proves the (cap+1)th has not
started, then releases.
"""

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path

import aiosqlite
import pytest

from rexhunter import db
from rexhunter.events import ErrorEvent, ToolCallEvent, ToolResultEvent
from rexhunter.loop import Brain, Decision, HuntComplete, ToolCallDecision, run_hunt
from rexhunter.scheduler import run_hunts
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

BrainFactory = Callable[[Sequence[Decision]], Brain]


async def outcome_of(conn: aiosqlite.Connection, run_id: str) -> str:
    async with conn.execute("SELECT outcome FROM runs WHERE id = ?", (run_id,)) as cur:
        row = await cur.fetchone()
    assert row is not None and row[0] is not None
    return str(row[0])


async def abort_reason_of(conn: aiosqlite.Connection, run_id: str) -> str | None:
    async with conn.execute("SELECT abort_reason FROM runs WHERE id = ?", (run_id,)) as cur:
        row = await cur.fetchone()
    assert row is not None
    return None if row[0] is None else str(row[0])


async def sole_run_id(conn: aiosqlite.Connection) -> str:
    # run_hunt creates the run internally and only returns its id on completion; a cancelled
    # hunt never returns, so the test recovers the id from the (single) run row.
    async with conn.execute("SELECT id FROM runs") as cur:
        rows = list(await cur.fetchall())
    assert len(rows) == 1
    return str(rows[0][0])


async def test_bounded_group_never_exceeds_cap(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    cap = 2
    territories = [f"t{i}" for i in range(5)]  # 5 hunts, cap 2 -> 3 must queue and wait
    db_path = tmp_path / "rex.db"

    in_flight = 0
    peak = 0
    reached_cap = asyncio.Event()
    release = asyncio.Event()
    reg = ToolRegistry()

    @reg.tool
    async def blocker() -> str:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        if in_flight == cap:
            reached_cap.set()
        try:
            await release.wait()  # hold the hunt in-flight until the test lets go
            return "ok"
        finally:
            in_flight -= 1

    def brain_for(_territory: str) -> Brain:
        return scripted_brain([ToolCallDecision(tool=blocker.__name__, args={}), HuntComplete()])

    task = asyncio.create_task(
        run_hunts(db_path, territories, brain_for=brain_for, registry=reg, max_concurrent=cap)
    )

    await asyncio.wait_for(reached_cap.wait(), timeout=2)  # cap hunts reached the tool
    for _ in range(20):  # give any (cap+1)th task a chance to wrongly start
        await asyncio.sleep(0)
    assert in_flight == cap  # the surplus hunts are blocked on the semaphore, not running
    assert peak == cap  # the bound was never exceeded

    release.set()  # let them all finish
    run_ids = await asyncio.wait_for(task, timeout=5)
    assert peak == cap  # still never exceeded across the whole batch

    assert len(run_ids) == len(territories)
    assert all(rid is not None for rid in run_ids)
    reader = await db.connect(db_path)
    try:
        for rid in run_ids:
            assert rid is not None
            assert await outcome_of(reader, rid) == "completed"
    finally:
        await reader.close()


# ── Unit 3a · failure isolation between siblings ─────────────────────────────


async def test_one_hunt_failure_does_not_corrupt_siblings(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    # Mixed fates in ONE bounded group: a raise, a hang-past-timeout, an unknown tool, two
    # clean completions. The group must NOT raise; failures are isolated to their own runs with
    # typed events; successes complete with UNCORRUPTED logs (no sibling's events leak in).
    db_path = tmp_path / "rex.db"
    reg = ToolRegistry()

    @reg.tool
    async def boom(target: str) -> None:
        raise ValueError(f"cannot reach {target}")

    @reg.tool
    async def hang() -> None:
        await asyncio.sleep(10)  # cancellable; the per-tool deadline fires in ms

    @reg.tool
    async def good(board: str) -> str:
        return f"posting@{board}"

    territories = ["raises", "hangs", "unknown", "ok1", "ok2"]

    def brain_for(territory: str) -> Brain:
        if territory == "raises":
            return scripted_brain([ToolCallDecision(tool=boom.__name__, args={"target": "acme"})])
        if territory == "hangs":
            return scripted_brain([ToolCallDecision(tool=hang.__name__, args={})])
        if territory == "unknown":
            return scripted_brain([ToolCallDecision(tool="ghost", args={"q": "AI"})])
        return scripted_brain(
            [ToolCallDecision(tool=good.__name__, args={"board": territory}), HuntComplete()]
        )

    run_ids = await run_hunts(
        db_path,
        territories,
        brain_for=brain_for,
        registry=reg,
        max_concurrent=5,  # all at once: failures and successes interleave
        tool_timeout_s=0.05,
        retry_budget=0,
    )
    assert all(rid is not None for rid in run_ids)
    by_territory = dict(zip(territories, run_ids, strict=True))

    reader = await db.connect(db_path)
    try:
        # the three failure modes each abort their OWN run, cleanly
        for territory in ("raises", "hangs", "unknown"):
            rid = by_territory[territory]
            assert rid is not None
            assert await outcome_of(reader, rid) == "aborted"

        # the raise was recorded as that run's own typed, fatal ErrorEvent
        raises_events = await db.read_events(reader, by_territory["raises"] or "")
        raise_errors = [e for e in raises_events if isinstance(e, ErrorEvent)]
        assert len(raise_errors) == 1 and "ValueError" in raise_errors[0].error

        # the successes are intact: exactly call+result, nothing leaked from a failing sibling
        for territory in ("ok1", "ok2"):
            rid = by_territory[territory]
            assert rid is not None
            assert await outcome_of(reader, rid) == "completed"
            events = await db.read_events(reader, rid)
            assert [type(e).__name__ for e in events] == ["ToolCallEvent", "ToolResultEvent"]
            assert isinstance(events[1], ToolResultEvent)
            assert events[1].raw_response == f'"posting@{territory}"'.encode()
            assert not [e for e in events if isinstance(e, ErrorEvent)]
            # and exactly one ToolCallEvent — no double-dispatch from concurrency
            assert len([e for e in events if isinstance(e, ToolCallEvent)]) == 1
    finally:
        await reader.close()


# ── Unit 3b · graceful shutdown closes a cancelled run ───────────────────────


async def test_cancelled_hunt_marks_its_run_aborted(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    # A cancelled in-flight hunt closes its OWN run as aborted/"daemon shutdown" — NOT left
    # outcome IS NULL for the boot crash-sweep, which would mislabel a graceful stop as 'crashed'.
    reg = ToolRegistry()
    started = asyncio.Event()

    @reg.tool
    async def blocker() -> None:
        started.set()
        await asyncio.Event().wait()  # hang until cancelled

    conn = await db.connect(tmp_path / "rex.db")
    try:
        brain = scripted_brain([ToolCallDecision(tool=blocker.__name__, args={})])
        task = asyncio.create_task(run_hunt(conn, territory="gate", brain=brain, registry=reg))
        await asyncio.wait_for(started.wait(), timeout=2)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        run_id = await sole_run_id(conn)
        assert await outcome_of(conn, run_id) == "aborted"
        assert await abort_reason_of(conn, run_id) == "daemon shutdown"
    finally:
        await conn.close()


async def test_cancelled_cleanup_survives_a_second_cancel(
    tmp_path: Path, scripted_brain: BrainFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ADVERSARIAL: a SECOND cancel lands while the cleanup write is in flight (aggressive
    # shutdown, double-cancel from a group). The mark-aborted write must still complete. This is
    # RED against a naive `await mark; raise` (the second cancel interrupts the write, leaving
    # the run NULL) and green only when the cleanup write is shielded to completion. A single
    # clean cancel would pass by luck — this won't.
    reg = ToolRegistry()
    started = asyncio.Event()

    @reg.tool
    async def blocker() -> None:
        started.set()
        await asyncio.Event().wait()

    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    real_finish = db.finish_run

    async def gated_finish(
        conn: aiosqlite.Connection,
        run_id: str,
        *,
        outcome: str,
        abort_reason: str | None = None,
    ) -> None:
        cleanup_started.set()
        await release_cleanup.wait()  # hold the cleanup write open — the window for cancel #2
        await real_finish(conn, run_id, outcome=outcome, abort_reason=abort_reason)

    monkeypatch.setattr(db, "finish_run", gated_finish)

    conn = await db.connect(tmp_path / "rex.db")
    try:
        brain = scripted_brain([ToolCallDecision(tool=blocker.__name__, args={})])
        task = asyncio.create_task(run_hunt(conn, territory="gate", brain=brain, registry=reg))
        await asyncio.wait_for(started.wait(), timeout=2)

        task.cancel()  # cancel #1 -> enter the cleanup path
        await asyncio.wait_for(cleanup_started.wait(), timeout=2)
        task.cancel()  # cancel #2 -> lands while the cleanup write is suspended
        release_cleanup.set()  # let the (shielded) write proceed

        with pytest.raises(asyncio.CancelledError):
            await task

        run_id = await sole_run_id(conn)
        assert await outcome_of(conn, run_id) == "aborted"  # survived the double cancel
        assert await abort_reason_of(conn, run_id) == "daemon shutdown"
    finally:
        await conn.close()
