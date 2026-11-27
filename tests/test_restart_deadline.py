"""A start's first deadlines, derived from the log (`P5.1`): each territory's first hunt waits out
what is left of its interval since its newest run began, by the scheduler's clock, or comes at
once when it has no run or that run began an interval or more ago. Nothing is stored (invariant
5): the deadline is folded from `runs.started_at` and the clock. The scheduler's clock and sleep
are injected, each run's start is stamped through the clock the log stamps with (`db.utcnow`),
and every brain is a local function.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rexhunter import db, scheduler
from rexhunter.loop import Brain, Context, Decision, HuntComplete, ThinkingSink
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)
HOUR = 3600.0


async def _plant(db_path: Path, territory: str, at: datetime) -> None:
    """A closed run of `territory`, its `started_at` stamped `at`."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(db, "utcnow", lambda: at.isoformat())
        conn = await db.connect(db_path)
        try:
            run_id = await db.start_run(conn, territory=territory)
            await db.finish_run(conn, run_id, outcome="completed")
        finally:
            await conn.close()


async def _hunts(db_path: Path) -> dict[str, int]:
    conn = await db.connect(db_path)
    try:
        rows = await conn.execute_fetchall("SELECT territory, COUNT(*) FROM runs GROUP BY 1")
    finally:
        await conn.close()
    return {str(row[0]): int(row[1]) for row in rows}


def _done(_territory: str) -> Brain:
    async def brain(_context: Context, _sink: ThinkingSink) -> Decision:
        return HuntComplete()

    return brain


async def _waits_of_a_start(db_path: Path, schedule: dict[str, float], count: int) -> list[float]:
    """Start the scheduler at NOW and return the first `count` sleeps it asks for; each sleep holds
    its territory for good, so a territory hunts at most once before its first one."""
    waits: list[float] = []
    asked = asyncio.Event()

    async def sleep(delay: float) -> None:
        waits.append(delay)
        if len(waits) == count:
            asked.set()
        await asyncio.Event().wait()

    seam: Callable[[float], Awaitable[None]] = sleep
    task = asyncio.create_task(
        scheduler.run_scheduler(
            db_path,
            schedule,
            brain_for=_done,
            registry=ToolRegistry(),
            max_concurrent=len(schedule),
            daemon_spend_ceiling_usd=None,
            now=lambda: NOW,
            sleep=seam,
        )
    )
    try:
        await asyncio.wait_for(asked.wait(), timeout=5)
    finally:
        task.cancel()
        await asyncio.wait({task}, timeout=10)
    return waits


def test_the_first_wait_is_what_is_left_of_the_interval() -> None:
    assert scheduler.first_wait(HOUR, None, NOW) == 0.0
    assert scheduler.first_wait(HOUR, NOW - timedelta(seconds=600), NOW) == 3000.0
    assert scheduler.first_wait(HOUR, NOW - timedelta(seconds=HOUR), NOW) == 0.0
    assert scheduler.first_wait(HOUR, NOW - timedelta(hours=5), NOW) == 0.0
    # A start stamped after the clock (a clock set back) holds a territory one interval at most.
    assert scheduler.first_wait(HOUR, NOW + timedelta(hours=5), NOW) == HOUR


async def test_a_restart_waits_out_the_interval(tmp_path: Path) -> None:
    # "recent" began 10 minutes ago, after an older run, so it waits the rest of its hour and does
    # not hunt; "fresh" has no run, hunts at once, then sleeps its interval.
    db_path = tmp_path / "rex.db"
    await _plant(db_path, "recent", NOW - timedelta(hours=3))
    await _plant(db_path, "recent", NOW - timedelta(seconds=600))
    waits = await _waits_of_a_start(db_path, {"recent": HOUR, "fresh": HOUR}, count=2)
    assert sorted(waits) == [3000.0, HOUR]
    assert await _hunts(db_path) == {"recent": 2, "fresh": 1}


async def test_a_territory_past_its_interval_hunts_at_once(tmp_path: Path) -> None:
    # One began exactly an interval ago and one five hours ago: both hunt at once, then sleep.
    db_path = tmp_path / "rex.db"
    await _plant(db_path, "edge", NOW - timedelta(seconds=HOUR))
    await _plant(db_path, "stale", NOW - timedelta(hours=5))
    waits = await _waits_of_a_start(db_path, {"edge": HOUR, "stale": HOUR}, count=2)
    assert waits == [HOUR, HOUR]
    assert await _hunts(db_path) == {"edge": 2, "stale": 2}
