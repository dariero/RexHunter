"""The hunt scheduler (ADR pillar 2, `P2.3`): bounded concurrent orchestration over the
single-hunt loop built in `P2.2`.

Two concerns under one slice. `run_hunts` is concern (a): a **bounded** task group running N
single-hunt loops at once — the concrete enforcement of invariant 7 (single writer per run).
Each hunt gets its OWN aiosqlite connection (the model the ADR's "SQLite write-lock serialises
writers" presupposes — a single shared connection would serialise every write onto one worker
thread and never exercise the write lock that makes invariant 7 load-bearing). The bound is an
`asyncio.Semaphore`: a connection is opened only after a slot is acquired, so the cap also caps
open connections, and territories past the cap WAIT for a slot (never rejected, never
unbounded).

Failure isolation: `run_hunt` is a total backstop (DoD #2) — it returns normally on every
`Exception`, so the `TaskGroup` never sees a child raise, so its cancel-all-siblings behaviour
never triggers. The per-hunt wrapper here adds defence in depth: it catches `Exception` (never
`CancelledError`, which must still propagate so daemon shutdown tears the group down on
purpose). One hunt's failure marks only its own run; siblings run to completion untouched.
"""

import asyncio
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from rexhunter import db
from rexhunter.loop import Brain, run_hunt
from rexhunter.tools import ToolRegistry

logger = logging.getLogger("rexhunter")


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
) -> list[str | None]:
    """Run one hunt per territory in a bounded task group; return their run_ids in order.

    `brain_for(territory)` yields a FRESH brain per hunt (each hunt owns its decision stream).
    A run_id is `None` only if a hunt escaped `run_hunt`'s backstop entirely (a bug — logged
    loudly, isolated so it cannot cancel siblings); under DoD #2 that never happens.
    """
    boot = await db.connect(db_path)  # bootstrap schema + WAL once, before the per-hunt opens
    await boot.close()

    sem = asyncio.Semaphore(max_concurrent)
    run_ids: list[str | None] = [None] * len(territories)

    async def _one_hunt(index: int, territory: str) -> None:
        async with sem:  # surplus hunts wait here for a slot (bound + connection cap)
            conn = await db.connect(db_path)  # one writer connection per run (invariant 7)
            try:
                run_ids[index] = await run_hunt(
                    conn,
                    territory=territory,
                    brain=brain_for(territory),
                    registry=registry,
                    tool_timeout_s=tool_timeout_s,
                    retry_budget=retry_budget,
                    max_iterations=max_iterations,
                )
            except Exception:  # defence in depth over run_hunt's own backstop — never expected
                logger.exception("hunt for %r escaped run_hunt's backstop", territory)
            finally:
                await conn.close()

    async with asyncio.TaskGroup() as tg:
        for i, territory in enumerate(territories):
            tg.create_task(_one_hunt(i, territory))

    return run_ids
