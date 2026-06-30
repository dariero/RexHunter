"""Production stub driver for the hunt daemon — the `P5` `brain()` placeholder.

`P2.3-wiring` needs *some* brain + registry to drive `run_scheduler` through the real
daemon lifespan. The provider-backed `brain()` is `P5` (paid); until then this stub builds
`Decision` objects directly — no model, no vendor SDK, no network, no spend. Swapping this
for `brain()` is the whole of `P5`; nothing here imports an LLM.

The one seam the lifespan reads is `daemon_config()`; the lifespan gate monkeypatches it to
inject a deterministic blocking tool. Everything else here is the real (if trivial) daemon.
"""

import asyncio
from collections.abc import Callable, Mapping

import aiosqlite

from rexhunter.loop import Brain, Context, Decision, HuntComplete, ToolCallDecision
from rexhunter.tools import ToolRegistry
from rexhunter.verdicts import Drafter

SNIFF_INTERVAL = 5.0  # the per-sniff beat, mirroring the prototype's cadence
DEFAULT_SCHEDULE: dict[str, float] = {"mock-gym": SNIFF_INTERVAL}
MAX_CONCURRENT = 4


async def sniff(prey: str) -> str:
    """The one trivial, no-spend tool. Real board adapters are `P5`.

    No real board yet: a beat, then echo the scent back as the raw response the log stores
    (invariant 6). No network, no spend.
    """
    await asyncio.sleep(SNIFF_INTERVAL)
    return f"posting:{prey}"


def build_registry() -> ToolRegistry:
    """A fresh registry holding the stub tool (instantiable — no global state to leak)."""
    reg = ToolRegistry()
    reg.tool(sniff)
    return reg


def stub_brain_for(territory: str) -> Brain:
    """One sniff, then done. The scheduler re-fires each territory on its own interval.

    A FRESH brain per hunt (each owns its decision stream); exhaustion ends the hunt cleanly
    rather than starving the loop — production must not crash on an over-eager loop.
    """
    decisions: list[Decision] = [
        ToolCallDecision(tool=sniff.__name__, args={"prey": territory}),
        HuntComplete(catch=[f"posting:{territory}"]),  # one penned posting per hunt (P4)
    ]
    queue = iter(decisions)

    async def brain(_context: Context) -> Decision:
        try:
            return next(queue)
        except StopIteration:
            return HuntComplete()

    return brain


async def draft_pitch(conn: aiosqlite.Connection, prey_id: str) -> str:
    """The stub pitch drafter — the `P5` paid drafter's placeholder. Reads the penned posting and
    returns a draft for human editing; no LLM, no spend. Rex drafts, never sends (invariant 4)."""
    async with conn.execute("SELECT posting FROM prey WHERE id = ?", (prey_id,)) as cur:
        row = await cur.fetchone()
    posting = str(row[0]) if row is not None else prey_id
    return f"Draft pitch for {posting} — [stub, edit me]"


def daemon_config() -> tuple[
    Mapping[str, float], Callable[[str], Brain], ToolRegistry, int, Drafter
]:
    """The single seam the lifespan reads (and the lifespan gate monkeypatches).

    Returns (schedule, brain_for, registry, max_concurrent, drafter) — the scheduler's knobs plus
    the follow-up worker's drafter.
    """
    return DEFAULT_SCHEDULE, stub_brain_for, build_registry(), MAX_CONCURRENT, draft_pitch
