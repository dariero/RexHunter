"""Step 2a · the run-finished SSE pulse.

A LIVE run closure (run_hunt's normal terminal — completed / needs_help / a live abort) fans an
id-less notification frame to every open board (hub.notify), so the territory tile and run-card
outcome flip on the finish itself, not on the next unrelated event. Post-commit (write-ahead,
invariant 1: runs.outcome is truth before the pulse), id-less (the trajectory-id resume cursor
never moves), and it appends nothing (invariant 7). NOT a trajectory event — terminal decisions
emit none (the settled ADR reconciliation, rexhunter-adr.md Pillar 2/4) — this relays an
already-committed runs-table fact, exactly as slice C relays a committed verdict
(test_live_verdict.py). Boot's crash-sweep and the shutdown-abort path never pulse.
"""

import json
from pathlib import Path

import pytest

from rexhunter import db, events, loop
from rexhunter.hub import Envelope, Hub
from rexhunter.loop import Brain, Context, Decision, HuntComplete, ThinkingSink, ToolCallDecision
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio


async def echo(prey: str) -> str:
    """An instant no-spend tool — the stub's sniff sleeps SNIFF_INTERVAL=5s (stub.py:32), far too
    slow for a unit test; the run_hunt path exercised is identical."""
    return f"posting:{prey}"


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.tool(echo)
    return reg


def _scripted_brain(decisions: list[Decision]) -> Brain:
    """A zero-sleep decision script (the stub.py:49-62 shape): no LLM, no spend, no delay."""
    queue = iter(decisions)

    async def brain(_context: Context, _sink: ThinkingSink) -> Decision:
        return next(queue)

    return brain


async def test_finish_run_emits_an_idless_hub_frame(tmp_path: Path) -> None:
    """A completing hunt pulses exactly ONE id-less notification (hub.notify, never publish) — the
    board's live tick for the runs-⊕ outcome/tile flip. Fired post-commit: by the time a viewer
    re-fetches /viewstate, runs.outcome is already durable (invariant 1)."""
    conn = await db.connect(tmp_path / "rex.db")
    try:
        hub = Hub()
        _, queue = hub.register()
        run_id = await loop.run_hunt(
            conn,
            territory="mock-gym",
            brain=_scripted_brain([HuntComplete()]),
            registry=_registry(),
            notify=hub.notify,  # publish NOT threaded: the queue isolates the pulse
        )
        env = queue.get_nowait()
        assert env.id is None  # id-less: never advances Last-Event-ID (hub.notify, not publish)
        assert json.loads(env.data) == {
            "type": "run_finished",
            "run_id": run_id,
            "outcome": "completed",
        }
        assert env.sse().startswith("data:")  # a message frame → onmessage → board re-fetch
        assert queue.empty()  # exactly one pulse per closure
    finally:
        await conn.close()


async def test_finish_pulse_leaves_the_resume_cursor_untouched(tmp_path: Path) -> None:
    """The pulse never advances the resume cursor: with publish AND notify threaded, every
    trajectory frame carries its global id, the pulse is the id-less LAST frame, and the log head
    (MAX(id)) equals the max PUBLISHED id — the pulse moved neither. Lossy-safe (the slice-C
    property): a viewer that misses it re-derives the outcome from /snapshot + /viewstate."""
    conn = await db.connect(tmp_path / "rex.db")
    try:
        hub = Hub()
        _, queue = hub.register()
        await loop.run_hunt(
            conn,
            territory="mock-gym",
            brain=_scripted_brain(
                [ToolCallDecision(tool="echo", args={"prey": "mock-gym"}), HuntComplete()]
            ),
            registry=_registry(),
            publish=hub.publish,
            notify=hub.notify,
        )
        frames: list[Envelope] = []
        while not queue.empty():
            frames.append(queue.get_nowait())
        *published, pulse = frames
        assert published  # the tool_call + tool_result rode the publish path…
        assert all(env.id is not None for env in published)  # …each with its global id
        assert pulse.id is None
        assert json.loads(pulse.data)["type"] == "run_finished"
        async with conn.execute("SELECT MAX(id) FROM trajectory_events") as cur:
            row = await cur.fetchone()
        assert row is not None
        # The log head is the max PUBLISHED id: the pulse advanced neither the log nor the cursor.
        assert max(env.id for env in published if env.id is not None) == int(row[0])
    finally:
        await conn.close()


async def test_boot_crash_sweep_does_not_notify(tmp_path: Path) -> None:
    """A structural tripwire, green from birth (like Step 1's deferral pin): mark_crashed_runs —
    the boot sweep — takes no hub and must stay that way. At boot there is no live viewer to serve
    (the lifespan sweeps BEFORE the hub exists, server.py:48 vs :62), and a reconnecting viewer
    recovers the crashed outcome via the snapshot dance. Pins the wiring decision that ONLY
    run_hunt's live terminal pulses (the shutdown-abort path re-raises before the pulse line)."""
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await db.start_run(conn, territory="mock-gym")
        await db.append_event(conn, run_id, events.ToolCallEvent(tool="echo", raw_request=b"{}"))
        hub = Hub()
        _, queue = hub.register()
        assert await db.mark_crashed_runs(conn) == 1  # the dangling run IS marked…
        assert queue.empty()  # …but no frame fans out
    finally:
        await conn.close()
