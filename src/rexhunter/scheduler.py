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
purpose). One hunt's failure marks only its own run; siblings run to completion untouched.

Timing reads the monotonic event-loop clock (`asyncio.sleep`), never wall-clock, and persists
NO "next fire" anywhere (invariant 5, derive don't store): a missed tick is simply the next
loop iteration, never a stored schedule that could drift from reality.
"""

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import aiosqlite

from rexhunter import cost, db
from rexhunter.loop import COST_CEILING_USD, Brain, run_hunt
from rexhunter.tools import ToolRegistry

logger = logging.getLogger("rexhunter")

# The id-scoped daemon spend ceiling (USD): the seq-scoped per-run `COST_CEILING_USD`'s analogue.
# The per-run ceiling folds ONE run's UsageEvents (seq-window); this bounds spend across a day of
# scheduled hunts by folding EVERY run's UsageEvents along the global `id` cursor, consulted before
# each launch. Stub/scripted brains carry no usage (fold → $0), so — like COST_CEILING_USD — this
# never bites them; the gated live daemon overrides it low. Tiny Arms for money (inv 4's spirit):
# the daemon structurally cannot exceed its budget. Fail-closed by design: the id-window is the
# whole log, so cumulative spend only latches the daemon OFF (never under-counts); a rolling
# day-window is a later refinement.
DAEMON_SPEND_CEILING_USD = 1.0


async def daemon_spend_usd(conn: aiosqlite.Connection) -> float:
    """Total USD spent across ALL runs (id-scoped), folded from the whole log's `UsageEvent`s
    (invariant 5 — a projection of the log, never a second counter). The SAME `cost.fold_cost` the
    per-run breaker uses, over the id-window (`db.read_usage`) instead of one run's seq-window."""
    return cost.fold_cost(await db.read_usage(conn))


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
    publish: db.PublishFn | None = None,
) -> str | None:
    """Run one hunt under the concurrency cap, on its own connection, fully isolated.

    Returns the run_id, `None` if the daemon spend ceiling refused this launch (over budget — a
    clean pause, not a crash), or `None` if the hunt escaped `run_hunt`'s backstop (a bug — logged
    loudly, contained so it cannot cancel siblings; under DoD #2 it never happens).
    """
    async with sem:  # surplus hunts wait here for a slot (bound + connection cap)
        conn = await db.connect(db_path)  # one writer connection per run (invariant 7)
        try:
            # The id-scoped budget gate (Tiny Arms for money, inv 4): fold EVERY run's spend from
            # the log (inv 5) BEFORE starting this run. Over the ceiling → refuse to launch (no
            # run_hunt, no start_run, no spend); in-flight hunts on their own connections finish
            # untouched. The per-run seq-scoped ceiling still guards inside run_hunt.
            if (
                daemon_spend_ceiling_usd is not None
                and await daemon_spend_usd(conn) >= daemon_spend_ceiling_usd
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
                publish=publish,
            )
        except Exception:  # defence in depth over run_hunt's own backstop — never expected
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
    publish: db.PublishFn | None = None,
) -> list[str | None]:
    """Run one hunt per territory in a bounded task group; return their run_ids in order (`None` for
    a territory the daemon budget refused).

    `brain_for(territory)` yields a FRESH brain per hunt (each hunt owns its decision stream). The
    per-run `cost_ceiling_usd` guards each hunt; `daemon_spend_ceiling_usd` (default `None` =
    ungated, the daemon bound being `run_scheduler`'s concern) applies the id-scoped gate when set.
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
            publish=publish,
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
) -> None:
    """Run forever: each territory in `schedule` (territory -> interval seconds) hunts on its own
    deadline, all bounded by `max_concurrent`. Cancellation (daemon shutdown) tears down every
    per-territory loop and any in-flight hunt; each in-flight run closes itself (run_hunt's
    shielded cancel path). The deadline is DERIVED from the monotonic clock via `asyncio.sleep`
    and never stored (invariant 5). `publish` (P3) feeds each committed event to the broadcast hub
    post-commit; `None` keeps the loop a pure log-writer (the prototype endpoint still polls).

    Budget guards: `cost_ceiling_usd` bounds ONE hunt (per-run, seq-scoped);
    `daemon_spend_ceiling_usd` bounds the daemon's cumulative spend across ALL hunts (id-scoped,
    folded from the log). Over the daemon ceiling, `_bounded_hunt` refuses to launch — the territory
    loop keeps ticking (a clean pause), never crashes, and any in-flight hunt finishes. Set the
    daemon ceiling `None` to disarm."""
    boot = await db.connect(db_path)
    await boot.close()

    sem = asyncio.Semaphore(max_concurrent)

    async def territory_loop(territory: str, interval: float) -> None:
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
                publish=publish,
            )
            await asyncio.sleep(interval)  # monotonic deadline; derived, never persisted

    async with asyncio.TaskGroup() as tg:
        for territory, interval in schedule.items():
            tg.create_task(territory_loop(territory, interval))
