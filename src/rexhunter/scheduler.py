"""The hunt scheduler (ADR pillar 2, `P2.3`): bounded concurrent orchestration over the
single-hunt loop built in `P2.2`, plus per-territory deadline timing.

Two concerns under one slice. `run_hunts` is concern (a): a **bounded** task group running N
single-hunt loops at once — the concrete enforcement of invariant 7 (single writer per run).
`run_scheduler` is concern (b): each territory hunts on its own interval, forever, until the
daemon shuts down. Both share `_bounded_hunt`: each hunt gets its OWN aiosqlite connection (the
model the ADR's "SQLite write-lock serialises writers" presupposes — a single shared connection
would serialise every write onto one worker thread and never exercise the write lock that makes
invariant 7 load-bearing), opened only after a `Semaphore` slot is acquired (so the cap also
caps open connections, and territories past the cap WAIT for a slot, never rejected).

Failure isolation: `run_hunt` is a total backstop (DoD #2) — it returns normally on every
`Exception`, so the `TaskGroup` never sees a child raise, so its cancel-all-siblings behaviour
never triggers. `_bounded_hunt` adds defence in depth: it catches `Exception` (never
`CancelledError`, which must still propagate so daemon shutdown tears the group down on
purpose). One hunt's failure marks only its own run; siblings run on untouched by it. Spend is
shared: with a daemon ceiling set, one hunt's spend can stop a sibling at its next brain attempt
(W3.2-C1).

Timing reads the monotonic event-loop clock (`asyncio.sleep`), never wall-clock, and persists
NO "next fire" anywhere (invariant 5, derive don't store): a missed tick is simply the next
loop iteration, never a stored schedule that could drift from reality. One exception since
`P5.1`: a start derives each territory's first deadline from the log and the wall clock, as what
`first_wait` leaves of the interval since its newest run's `started_at`, by `now`, so a restart
does not hunt every territory again at once; every later deadline is the monotonic clock's.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite

from rexhunter import cost, db
from rexhunter.loop import COST_CEILING_USD, Brain, NotifyFn, run_hunt
from rexhunter.tools import ToolRegistry

logger = logging.getLogger("rexhunter")

# The id-scoped daemon spend ceiling (USD): the seq-scoped per-run `COST_CEILING_USD`'s analogue.
# The per-run ceiling folds ONE run's UsageEvents (seq-window); this bounds spend across a day of
# scheduled hunts by folding EVERY run's UsageEvents along the global `id` cursor, consulted before
# each launch and, inside run_hunt, before every brain attempt (W3.2-C1). Stub/scripted brains
# carry no usage (fold → $0), so — like COST_CEILING_USD — this never bites them; the gated live
# daemon overrides it low. Tiny Arms for money (inv 4's spirit): the daemon goes past its budget by
# at most the one brain call each hunt has in flight, of the calls the log counts (W3.2-C1; with
# the launch gate alone, one hunt could spend to its own ceiling). The two gates fold only the
# UsageEvents stored in the last `DAEMON_SPEND_WINDOW` (`P5.1`), so the ceiling bounds a rolling day
# and older spend does not hold the daemon; `run_hunts` (the wave harness, which hunts into a fresh
# log) folds the whole log.
DAEMON_SPEND_CEILING_USD = 1.0
DAEMON_SPEND_WINDOW = timedelta(hours=24)


def _wall_clock() -> datetime:
    return datetime.now(UTC)


def first_wait(interval: float, last_start: datetime | None, now: datetime) -> float:
    """Seconds before a territory's first hunt after a start: what is left of its interval since
    its newest run began, or none when it has no run or that run began an interval or more ago.
    Never more than the interval, so a start stamped after `now` cannot hold a territory longer."""
    if last_start is None:
        return 0.0
    elapsed = (now - last_start).total_seconds()
    return min(interval, max(0.0, interval - elapsed))


async def daemon_spend_usd(conn: aiosqlite.Connection, *, since: datetime | None = None) -> float:
    """Total USD spent across ALL runs (id-scoped), folded from the whole log's `UsageEvent`s
    (invariant 5 — a projection of the log, never a second counter). The SAME `cost.fold_cost` the
    per-run breaker uses, over the id-window (`db.read_usage`) instead of one run's seq-window;
    `since` narrows it to the UsageEvents stored at or after that instant (`P5.1`)."""
    return cost.fold_cost(await db.read_usage(conn, since=since))


async def _bounded_hunt(
    db_path: str | Path,
    sem: asyncio.Semaphore,
    territory: str,
    *,
    brain_for: Callable[[str], Brain],
    registry: ToolRegistry,
    tool_timeout_s: float,
    retry_budget: int,
    max_iterations: int,
    cost_ceiling_usd: float = COST_CEILING_USD,
    daemon_spend_ceiling_usd: float | None = None,
    daemon_spend_since: Callable[[], datetime] | None = None,
    wrap_up: bool = False,
    publish: db.PublishFn | None = None,
    notify: NotifyFn | None = None,
) -> str | None:
    """Run one hunt under the concurrency cap, on its own connection, fully isolated.

    Returns the run_id, `None` if the daemon spend ceiling refused this launch (over budget — a
    clean pause, not a crash), or `None` if the hunt escaped `run_hunt`'s backstop (a bug — logged
    loudly, contained so it cannot cancel siblings; under DoD #2 it never happens).
    `daemon_spend_since` gives the window's first instant at each check (`run_scheduler`'s rolling
    day, `P5.1`); `None` folds the whole log.
    """
    async with sem:  # surplus hunts wait here for a slot (bound + connection cap)
        conn = await db.connect(db_path)  # one writer connection per run (invariant 7)
        try:
            # The id-scoped budget gate (Tiny Arms for money, inv 4): fold EVERY run's spend from
            # the log (inv 5) BEFORE starting this run. Over the ceiling → refuse to launch (no
            # run_hunt, no start_run, no spend). Inside run_hunt the same fold runs again before
            # every brain attempt (W3.2-C1), so an in-flight hunt stops at the ceiling too, after
            # at most the one call it has in flight; the per-run ceiling is checked there as well.
            # Both gates get the same `daemon_spend_since`, so they fold the same window.
            since = None if daemon_spend_since is None else daemon_spend_since()
            if (
                daemon_spend_ceiling_usd is not None
                and await daemon_spend_usd(conn, since=since) >= daemon_spend_ceiling_usd
            ):
                logger.info(
                    "daemon spend ceiling $%.4f reached; refusing to launch a hunt for %r",
                    daemon_spend_ceiling_usd,
                    territory,
                )
                return None
            return await run_hunt(
                conn,
                territory=territory,
                brain=brain_for(territory),
                registry=registry,
                tool_timeout_s=tool_timeout_s,
                retry_budget=retry_budget,
                max_iterations=max_iterations,
                cost_ceiling_usd=cost_ceiling_usd,
                daemon_spend_ceiling_usd=daemon_spend_ceiling_usd,
                daemon_spend_since=daemon_spend_since,
                wrap_up=wrap_up,
                publish=publish,
                notify=notify,
            )
        except Exception as exc:  # defence in depth over run_hunt's own backstop — never expected
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                # Being cancelled (daemon shutdown): never swallow, or the territory loop would
                # relaunch a hunt after its own cancel (R2-RV1).
                logger.exception("hunt for %r failed while being cancelled", territory)
                raise asyncio.CancelledError from exc
            logger.exception("hunt for %r escaped run_hunt's backstop", territory)
            return None
        finally:
            await conn.close()


async def run_hunts(
    db_path: str | Path,
    territories: Sequence[str],
    *,
    brain_for: Callable[[str], Brain],
    registry: ToolRegistry,
    max_concurrent: int,
    tool_timeout_s: float = 30.0,
    retry_budget: int = 2,
    max_iterations: int = 50,
    cost_ceiling_usd: float = COST_CEILING_USD,
    daemon_spend_ceiling_usd: float | None = None,
    wrap_up: bool = False,
    publish: db.PublishFn | None = None,
    notify: NotifyFn | None = None,
) -> list[str | None]:
    """Run one hunt per territory in a bounded task group; return their run_ids in order (`None` for
    a territory the daemon budget refused).

    `brain_for(territory)` yields a FRESH brain per hunt (each hunt owns its decision stream). The
    per-run `cost_ceiling_usd` guards each hunt; `daemon_spend_ceiling_usd` (default `None` =
    ungated, the daemon bound being `run_scheduler`'s concern) applies the id-scoped gate when set,
    over the whole log: the wave harness hunts into a fresh log, so its ceiling bounds the wave.
    `wrap_up` reaches every hunt's `run_hunt` (`P5.1`).
    """
    boot = await db.connect(db_path)  # bootstrap schema + WAL once, before the per-hunt opens
    await boot.close()

    sem = asyncio.Semaphore(max_concurrent)
    run_ids: list[str | None] = [None] * len(territories)

    async def _store(index: int, territory: str) -> None:
        run_ids[index] = await _bounded_hunt(
            db_path,
            sem,
            territory,
            brain_for=brain_for,
            registry=registry,
            tool_timeout_s=tool_timeout_s,
            retry_budget=retry_budget,
            max_iterations=max_iterations,
            cost_ceiling_usd=cost_ceiling_usd,
            daemon_spend_ceiling_usd=daemon_spend_ceiling_usd,
            wrap_up=wrap_up,
            publish=publish,
            notify=notify,
        )

    async with asyncio.TaskGroup() as tg:
        for i, territory in enumerate(territories):
            tg.create_task(_store(i, territory))

    return run_ids


async def run_scheduler(
    db_path: str | Path,
    schedule: Mapping[str, float],
    *,
    brain_for: Callable[[str], Brain],
    registry: ToolRegistry,
    max_concurrent: int,
    tool_timeout_s: float = 30.0,
    retry_budget: int = 2,
    max_iterations: int = 50,
    cost_ceiling_usd: float = COST_CEILING_USD,
    daemon_spend_ceiling_usd: float | None = DAEMON_SPEND_CEILING_USD,
    publish: db.PublishFn | None = None,
    notify: NotifyFn | None = None,
    now: Callable[[], datetime] = _wall_clock,
    wrap_up: bool = False,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Run forever: each territory in `schedule` (territory -> interval seconds) hunts on its own
    deadline, all bounded by `max_concurrent`. Cancellation (daemon shutdown) tears down every
    per-territory loop and any in-flight hunt; each in-flight run closes itself (run_hunt's
    shielded cancel path). The deadline is DERIVED from the monotonic clock via `sleep`
    (`asyncio.sleep`) and never stored (invariant 5); the first one after a start is what
    `first_wait` leaves of the interval since the territory's newest run began, by `now` (`P5.1`).
    `publish` (P3) feeds each committed event to the broadcast hub
    post-commit; `None` keeps the loop a pure log-writer (the prototype endpoint still polls).
    `notify` (frontend Step 2a) mirrors it for the id-less run-finished pulse — the board's live
    tick when a run closes; `None` skips the pulse.

    Budget guards: `cost_ceiling_usd` bounds ONE hunt (per-run, seq-scoped);
    `daemon_spend_ceiling_usd` bounds the daemon's spend across ALL hunts in the last
    `DAEMON_SPEND_WINDOW`, as `now` reads it at each check (id-scoped, folded from the log;
    `P5.1`). Over the daemon ceiling, `_bounded_hunt` refuses to launch (the territory loop keeps
    ticking, a clean pause, and never crashes), and an in-flight hunt stops at its next
    brain attempt, "daemon spend ceiling" ("cost ceiling" when its own ceiling is crossed too,
    checked first; W3.2-C1). Set the daemon ceiling `None` to disarm. `wrap_up` reaches every
    hunt's `run_hunt` (`P5.1`)."""
    boot = await db.connect(db_path)
    try:
        # Each territory's newest run start, read once (`P5.1`), so a restart waits out the rest
        # of an interval a run has already begun instead of hunting every territory at once.
        last_starts = {
            territory: await db.latest_run_start(boot, territory) for territory in schedule
        }
    finally:
        await boot.close()

    sem = asyncio.Semaphore(max_concurrent)

    def daemon_spend_since() -> datetime:
        return now() - DAEMON_SPEND_WINDOW

    async def territory_loop(territory: str, interval: float) -> None:
        wait = first_wait(interval, last_starts[territory], now())
        if wait > 0:
            await sleep(wait)
        while True:
            await _bounded_hunt(
                db_path,
                sem,
                territory,
                brain_for=brain_for,
                registry=registry,
                tool_timeout_s=tool_timeout_s,
                retry_budget=retry_budget,
                max_iterations=max_iterations,
                cost_ceiling_usd=cost_ceiling_usd,
                daemon_spend_ceiling_usd=daemon_spend_ceiling_usd,
                daemon_spend_since=daemon_spend_since,
                wrap_up=wrap_up,
                publish=publish,
                notify=notify,
            )
            await sleep(interval)  # monotonic deadline; derived, never persisted

    async with asyncio.TaskGroup() as tg:
        for territory, interval in schedule.items():
            tg.create_task(territory_loop(territory, interval))
