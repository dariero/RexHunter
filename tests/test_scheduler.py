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
from rexhunter.loop import Brain, Decision, HuntComplete, ToolCallDecision
from rexhunter.scheduler import run_hunts
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

BrainFactory = Callable[[Sequence[Decision]], Brain]


async def outcome_of(conn: aiosqlite.Connection, run_id: str) -> str:
    async with conn.execute("SELECT outcome FROM runs WHERE id = ?", (run_id,)) as cur:
        row = await cur.fetchone()
    assert row is not None and row[0] is not None
    return str(row[0])


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
