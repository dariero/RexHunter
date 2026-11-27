"""The daemon's spend ceiling over a rolling day (`P5.1`).

`run_scheduler`, the daemon, folds only the UsageEvents stored in the 24 h before its clock reads,
at the launch gate and before every brain attempt, so spend a day old no longer holds it;
`run_hunts`, the wave harness, still folds the whole log; and the board's daemon spend folds the
window the gates fold. Each spend is stored at a chosen instant through the clock the log stamps
with (`db.utcnow`), every brain is a local function, and nothing leaves the process.
"""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import aiosqlite
import pytest

from rexhunter import db, eval_wave, scheduler, server, viewstate
from rexhunter.events import TrajectoryEvent, UsageEvent
from rexhunter.loop import Brain, Context, Decision, HuntComplete, ThinkingSink
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)
DAY = timedelta(hours=24)  # the window's length, written out so a change to it shows here
EDGE = NOW - DAY  # the window's first instant, inside it
OUTSIDE = EDGE - timedelta(microseconds=1)  # the instant before it
CEILING_USD = 0.01
MTOK = 1_000_000  # Sonnet 5.5 input tokens worth $2.00
TERRITORY = "mock-gym"


async def _plant(db_path: Path, at: datetime, input_tokens: int) -> None:
    """One Sonnet 5.5 UsageEvent under a throwaway run, stored with `at` as its `created_at`."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(db, "utcnow", lambda: at.isoformat())
        conn = await db.connect(db_path)
        try:
            run_id = await db.start_run(conn, territory="seed")
            await db.append_event(
                conn,
                run_id,
                UsageEvent(model="claude-sonnet-5-5", input_tokens=input_tokens, output_tokens=0),
            )
        finally:
            await conn.close()


def _brain_for(called: asyncio.Event) -> Callable[[str], Brain]:
    def brain_for(_territory: str) -> Brain:
        async def brain(_context: Context, _sink: ThinkingSink) -> Decision:
            called.set()
            return HuntComplete()

        return brain

    return brain_for


async def _hunts(db_path: Path) -> int:
    conn = await db.connect(db_path)
    try:
        rows = list(
            await conn.execute_fetchall(
                "SELECT COUNT(*) FROM runs WHERE territory = ?", (TERRITORY,)
            )
        )
    finally:
        await conn.close()
    return int(rows[0][0])


async def _run_the_daemon(
    db_path: Path, called: asyncio.Event, monkeypatch: pytest.MonkeyPatch
) -> list[float]:
    """Run `run_scheduler`, its clock at NOW, until its launch gate's second check, by which time
    its first hunt has ended or been refused; return the spend the gate saw at each check."""
    checked: list[float] = []
    twice = asyncio.Event()
    real = scheduler.daemon_spend_usd

    async def gate(conn: aiosqlite.Connection, *, since: datetime | None = None) -> float:
        spend = await real(conn, since=since)
        checked.append(spend)
        if len(checked) == 2:
            twice.set()
        return spend

    monkeypatch.setattr(scheduler, "daemon_spend_usd", gate)
    task = asyncio.create_task(
        scheduler.run_scheduler(
            db_path,
            {TERRITORY: 0.01},
            brain_for=_brain_for(called),
            registry=ToolRegistry(),
            max_concurrent=1,
            retry_budget=0,
            max_iterations=1,
            daemon_spend_ceiling_usd=CEILING_USD,
            now=lambda: NOW,
        )
    )
    try:
        await asyncio.wait_for(twice.wait(), timeout=5)
    finally:
        task.cancel()
        await asyncio.wait({task}, timeout=10)
    return checked


@pytest.mark.parametrize(
    ("at", "launched"), [(OUTSIDE, True), (EDGE, False)], ids=["a-day-old", "at-the-edge"]
)
async def test_spend_older_than_a_day_does_not_hold_the_daemon(
    at: datetime, launched: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # $2.00 against a $0.01 ceiling holds the daemon's launch gate while it is inside the day, and
    # only then; a launched hunt's brain runs, so its per-attempt gate let it through too.
    db_path = tmp_path / "rex.db"
    await _plant(db_path, at, MTOK)
    called = asyncio.Event()
    checked = await _run_the_daemon(db_path, called, monkeypatch)
    assert (checked[0], called.is_set(), bool(await _hunts(db_path))) == (
        0.0 if launched else 2.0,
        launched,
        launched,
    )


async def test_the_per_attempt_gate_uses_the_same_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every spend read the daemon makes starts at the same instant, its clock less a day: the first
    # hunt's launch check, its one brain attempt (`loop._call_brain`), and the next launch check.
    db_path = tmp_path / "rex.db"
    await _plant(db_path, OUTSIDE, MTOK)
    windows: list[datetime | None] = []
    real = db.read_usage

    async def read_usage(
        conn: aiosqlite.Connection, *, since: datetime | None = None
    ) -> list[TrajectoryEvent]:
        windows.append(since)
        return await real(conn, since=since)

    monkeypatch.setattr(db, "read_usage", read_usage)
    called = asyncio.Event()
    await _run_the_daemon(db_path, called, monkeypatch)
    assert (called.is_set(), windows[:3]) == (True, [EDGE, EDGE, EDGE])


async def test_the_wave_harness_still_folds_the_whole_log(tmp_path: Path) -> None:
    # The wave hunts into a fresh log, so its ceiling bounds the whole log: $2.00 stored two days
    # ago still refuses every territory at the wave's $0.75.
    path = tmp_path / "wave.db"
    await _plant(path, datetime.now(UTC) - 2 * DAY, MTOK)
    called = asyncio.Event()
    run_ids = await eval_wave.hunt_wave(path, _brain_for(called), ToolRegistry())
    assert (list(run_ids), called.is_set()) == ([None] * len(eval_wave.TERRITORIES), False)


async def test_the_view_shows_the_gates_window(tmp_path: Path) -> None:
    # The board's daemon spend at NOW folds what the gates fold at NOW: of $2.00 just outside the
    # day, $0.20 at its edge and $0.02 an hour old, $0.22. Each run's own spend stays all of it.
    db_path = tmp_path / "rex.db"
    for at, tokens in (
        (OUTSIDE, MTOK),
        (EDGE, MTOK // 10),
        (NOW - timedelta(hours=1), MTOK // 100),
    ):
        await _plant(db_path, at, tokens)
    conn = await db.connect(db_path)
    try:
        state = await viewstate.build_viewstate(
            conn, NOW, daemon_spend_ceiling_usd=CEILING_USD, daemon_spend_window=DAY
        )
        gate = await scheduler.daemon_spend_usd(conn, since=NOW - DAY)
        whole = await scheduler.daemon_spend_usd(conn)
    finally:
        await conn.close()
    runs = sorted(run.spent_usd for run in state.runs)
    assert (state.spent_usd, gate, whole, runs) == (
        pytest.approx(0.22),
        pytest.approx(0.22),
        pytest.approx(2.22),
        pytest.approx([0.02, 0.2, 2.0]),
    )


async def test_a_window_start_in_another_offset_reads_the_same_spend(tmp_path: Path) -> None:
    # The gate and the view compare the stored UTC strings, so an aware instant in another offset
    # names the same window: the $0.20 at its edge, not the $2.00 just before it.
    db_path = tmp_path / "rex.db"
    await _plant(db_path, OUTSIDE, MTOK)
    await _plant(db_path, EDGE, MTOK // 10)
    aedt = timezone(timedelta(hours=11))
    conn = await db.connect(db_path)
    try:
        gate = await scheduler.daemon_spend_usd(conn, since=EDGE.astimezone(aedt))
        state = await viewstate.build_viewstate(conn, NOW.astimezone(aedt), daemon_spend_window=DAY)
    finally:
        await conn.close()
    assert (gate, state.spent_usd) == (pytest.approx(0.2), pytest.approx(0.2))


async def test_the_board_shows_the_daemons_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `/viewstate` folds the scheduler's window at the wall clock: the HUD shows the $0.02 an hour
    # old, not the $2.00 two days old.
    db_path = tmp_path / "rex.db"
    monkeypatch.setattr(server, "DB_PATH", str(db_path))
    monkeypatch.delenv("DAEMON_SPEND_CEILING_USD", raising=False)
    wall = datetime.now(UTC)
    await _plant(db_path, wall - 2 * DAY, MTOK)
    await _plant(db_path, wall - timedelta(hours=1), MTOK // 100)
    body = bytes((await server.get_viewstate()).body).decode()
    assert '<span class="spend">$0.0200</span>' in body


async def test_a_naive_window_start_is_refused(tmp_path: Path) -> None:
    # `astimezone` reads a naive instant as the machine's local time, which would move the window
    # by the local offset.
    conn = await db.connect(tmp_path / "rex.db")
    try:
        with pytest.raises(ValueError, match="aware"):
            await db.read_usage(conn, since=NOW.replace(tzinfo=None))
    finally:
        await conn.close()
