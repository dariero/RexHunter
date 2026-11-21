"""Production stub driver for the hunt daemon — the `P5` `brain()` placeholder.

`P2.3-wiring` needs *some* brain + registry to drive `run_scheduler` through the real
daemon lifespan. The provider-backed `brain()` is `P5` (paid); until then this stub builds
`Decision` objects directly — no model, no vendor SDK, no network, no spend. Swapping this
for `brain()` is the whole of `P5`; nothing here imports an LLM.

The one seam the lifespan reads is `daemon_config()`; the lifespan gate monkeypatches it to
inject a deterministic blocking tool. Since `P2.4` (D9) the daemon's default schedule is the six gym
territories, hunted by the scripted keyword baseline (`board.baseline`); `sniff`, `mock-gym` and
`build_registry` stay for the recorded replays and the wire check.
"""

import asyncio
from collections.abc import Callable, Mapping

import aiosqlite

from rexhunter.board import tools as board_tools
from rexhunter.board.baseline import baseline_brain_for
from rexhunter.board.catalogue import GYM_BOARDS
from rexhunter.board.client import BoardClient
from rexhunter.loop import Brain, Context, Decision, HuntComplete, ThinkingSink, ToolCallDecision
from rexhunter.tools import ToolRegistry
from rexhunter.verdicts import Drafter

SNIFF_INTERVAL = 5.0  # the per-sniff beat, mirroring the prototype's cadence
GYM_INTERVAL = 30.0  # each gym territory's interval (D10)
DEFAULT_SCHEDULE: dict[str, float] = dict.fromkeys(GYM_BOARDS, GYM_INTERVAL)
MAX_CONCURRENT = 4

# The board tools over the lifespan's board client: the daemon's registry is built from the client
# the lifespan owns, so this seam builds no client itself (`/viewstate` calls it per request).
type RegistryFor = Callable[[BoardClient], ToolRegistry]


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
    The catch is empty: `sniff` is no posting source, so a reference to what it found would
    never resolve.
    """
    decisions: list[Decision] = [
        ToolCallDecision(tool=sniff.__name__, args={"prey": territory}),
        HuntComplete(),
    ]
    queue = iter(decisions)

    async def brain(_context: Context, _sink: ThinkingSink) -> Decision:
        # No LLM, no stream: the stub emits typed decisions directly and relays no thinking.
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


def gym_registry(client: BoardClient) -> ToolRegistry:
    return board_tools.build_registry(client, GYM_BOARDS)


def daemon_config() -> tuple[
    Mapping[str, float], Callable[[str], Brain], RegistryFor, int, Drafter
]:
    """The single seam the lifespan reads (and the lifespan gate monkeypatches).

    Returns (schedule, brain_for, registry_for, max_concurrent, drafter): the scheduler's knobs,
    the registry built over the lifespan's board client, and the follow-up worker's drafter.
    """
    return (
        DEFAULT_SCHEDULE,
        baseline_brain_for(GYM_BOARDS),
        gym_registry,
        MAX_CONCURRENT,
        draft_pitch,
    )
