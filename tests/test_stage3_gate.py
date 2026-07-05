"""P3 gate — ADR Definition-of-done #3 (Streaming).

Three clauses, all driven IN-PROCESS against one real hunt (stub brain, no spend) publishing to a
hub we own — no live browser, no HTTP server:

(a) two viewers connected for the whole hunt render BYTE-IDENTICAL event sequences;
(b) a viewer closed for a full hunt reconnects via Last-Event-ID (snapshot + catch-up + live
    splice) with ZERO gaps and ZERO duplicates;
(c) a viewer whose bounded queue fills is DROPPED and the hunt completes UNSLOWED (the producer
    does the same work it would with no viewer at all).

The stream helpers under test (`server.catch_up`, `server.stream_events`, `server.snapshot_state`)
are the exact code the SSE endpoint runs; the endpoint is a thin `async for … yield env.sse()`
wrapper over `stream_events`, so proving these proves the endpoint.
"""

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path

import aiosqlite
import pytest

from rexhunter import db, server
from rexhunter.events import SniffEvent
from rexhunter.hub import Envelope, Hub
from rexhunter.loop import Brain, Decision, HuntComplete, ToolCallDecision, run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

BrainFactory = Callable[[Sequence[Decision]], Brain]
TERRITORY = "mock-gym"


# ── A fast, deterministic hunt: tool_call → tool_result → prey_captured (3 events) ────────────


async def peek() -> str:
    return "seen"  # a no-sleep tool so the gate runs at full speed (unlike stub.sniff's beat)


def gate_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.tool(peek)
    return reg


def gate_decisions() -> list[Decision]:
    # one tool round then done-with-a-catch: emits ToolCallEvent + ToolResultEvent + a
    # PreyCapturedEvent (the SECOND trajectory writer, verdicts.capture_prey — proves the whole
    # publish choke set streams, not just db.append_event).
    return [ToolCallDecision(tool=peek.__name__, args={}), HuntComplete(catch=["job-1"])]


async def _hunt(conn: aiosqlite.Connection, hub: Hub, brain: Brain) -> str:
    return await run_hunt(
        conn,
        territory=TERRITORY,
        brain=brain,
        registry=gate_registry(),
        publish=hub.publish,
    )


def _drain(queue: asyncio.Queue[Envelope]) -> list[Envelope]:
    out: list[Envelope] = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


async def _all_event_ids(conn: aiosqlite.Connection) -> list[int]:
    async with conn.execute("SELECT id FROM trajectory_events ORDER BY id") as cur:
        return [int(r[0]) for r in await cur.fetchall()]


# ── (a) two viewers render byte-identical feeds ───────────────────────────────


async def test_two_viewers_render_byte_identical_feeds(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    hub = Hub(maxsize=64)
    _, qa = hub.register()
    _, qb = hub.register()
    conn = await db.connect(tmp_path / "rex.db")
    try:
        await _hunt(conn, hub, scripted_brain(gate_decisions()))

        frames_a = [e.sse() for e in _drain(qa)]
        frames_b = [e.sse() for e in _drain(qb)]

        assert frames_a == frames_b  # byte-identical fan-out
        assert len(frames_a) == 3  # tool_call, tool_result, prey_captured
        assert all(f.startswith("id: ") for f in frames_a)  # each a real, id-carrying frame
    finally:
        await conn.close()


# ── (b) reconnect: snapshot + catch-up + live splice, zero gaps / zero dups ────


async def test_reconnect_catches_up_and_splices_with_no_gap_no_dup(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    hub = Hub(maxsize=64)
    conn = await db.connect(tmp_path / "rex.db")
    try:
        # A hunt runs while this viewer is CLOSED (no one registered on the hub).
        run_id = await _hunt(conn, hub, scripted_brain(gate_decisions()))
        assert len(hub) == 0  # nothing was streamed live
        missed = await _all_event_ids(conn)
        assert len(missed) == 3

        snap = await server.snapshot_state(conn)
        assert snap["latest_id"] == missed[-1]  # snapshot hands the client its Last-Event-ID

        # Reconnect from scratch (Last-Event-ID = 0): register → catch-up the whole hunt from the
        # log → then live-splice. Drive the REAL generator the endpoint runs.
        gen = server.stream_events(conn, hub, last_seen=0)
        caught = [await anext(gen) for _ in range(len(missed))]  # the catch-up backfill

        # a NEW event arrives live, after the reconnect (a fresh hunt turn)
        live_run = await db.start_run(conn, territory=TERRITORY)
        live_id = await db.append_event(
            conn, live_run, SniffEvent(prey="live"), publish=hub.publish
        )
        caught.append(await anext(gen))  # spliced live off the queue

        ids = [e.id for e in caught if e.id is not None]
        assert len(ids) == len(caught)  # every frame a real event (no heartbeat leaked in)
        assert ids == [*missed, live_id]  # zero gaps: exactly the log tail, then the live event
        assert len(ids) == len(set(ids))  # zero duplicates
        assert ids == sorted(ids)  # monotonic, in commit order
        assert snap["latest_id"] == missed[-1] and run_id  # (bind names; snapshot from the log)

        await gen.aclose()
        assert len(hub) == 0  # the generator deregistered its viewer on close — no leak
    finally:
        await conn.close()


async def test_live_splice_drops_ids_at_or_below_high_water(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    """The dedup line, white-box: an id already delivered by catch-up (≤ high-water) that later
    arrives on the live queue is DROPPED; a genuinely new id (> high-water) passes. This is the
    monotonic-id dedup the ADR relies on — the browser never double-applies. The overlap window is
    internal to the generator, so we publish a stale-then-fresh pair onto its live queue (via the
    public hub) to exercise the exact predicate."""
    hub = Hub(maxsize=64)
    conn = await db.connect(tmp_path / "rex.db")
    try:
        await _hunt(conn, hub, scripted_brain(gate_decisions()))
        log_ids = await _all_event_ids(conn)

        gen = server.stream_events(conn, hub, last_seen=0)
        caught = [await anext(gen) for _ in range(len(log_ids))]  # catch-up sets high-water
        high_water = caught[-1].id
        assert high_water == log_ids[-1]
        assert high_water is not None

        # The generator registered exactly one viewer; publish through the PUBLIC hub so both land
        # on its live queue. A replayed id (== high-water) then a genuinely new id (> high-water).
        hub.publish(high_water, "STALE-DUP")  # ≤ high-water → must be dropped by the dedup
        hub.publish(high_water + 1, "FRESH")  # > high-water → must pass

        spliced = await anext(gen)  # the stale dup is skipped; the fresh one is yielded
        assert spliced.id == high_water + 1
        assert spliced.data == "FRESH"

        await gen.aclose()
    finally:
        await conn.close()


# ── (c) a full-queue viewer is dropped; the hunt completes unslowed ───────────


async def test_full_queue_viewer_dropped_hunt_completes_unslowed(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    conn = await db.connect(tmp_path / "rex.db")
    try:
        # control: a hunt with NO viewers — the yardstick for "how much work a hunt does".
        control_hub = Hub(maxsize=64)
        control_run = await _hunt(conn, control_hub, scripted_brain(gate_decisions()))
        control_events = await db.read_events(conn, control_run)

        # stress: a viewer whose maxsize-1 queue overflows immediately and is never drained.
        stuck_hub = Hub(maxsize=1)
        stuck_id, _stuck_q = stuck_hub.register()
        stuck_run = await _hunt(conn, stuck_hub, scripted_brain(gate_decisions()))
        stuck_events = await db.read_events(conn, stuck_run)

        async with conn.execute("SELECT outcome FROM runs WHERE id = ?", (stuck_run,)) as cur:
            row = await cur.fetchone()
        assert row is not None and row[0] == "completed"  # terminal, never blocked on the viewer
        assert stuck_id not in stuck_hub  # the slow viewer was dropped (drop-and-resync)
        # unslowed: the producer committed the SAME events it would with no viewer watching.
        assert len(stuck_events) == len(control_events)
        assert [e.type for e in stuck_events] == [e.type for e in control_events]
    finally:
        await conn.close()


async def test_dropped_viewer_stream_terminates_so_client_can_resync(tmp_path: Path) -> None:
    """Drop-and-resync, consumer side: once the hub drops a full-queue viewer it sends that viewer
    nothing further (not even heartbeats), so `stream_events` must END its stream rather than hang
    on `queue.get()` forever — that closed response is exactly what makes EventSource reconnect into
    the snapshot dance (ADR line 166). A tiny heartbeat interval keeps the drop-detection fast."""
    hub = Hub(maxsize=1, heartbeat_interval_s=0.05)
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run = await db.start_run(conn, territory=TERRITORY)
        e1 = await db.append_event(conn, run, SniffEvent(prey="a"), publish=hub.publish)

        gen = server.stream_events(conn, hub, last_seen=0)
        first = await anext(gen)  # registers the viewer, catches e1 up from the log
        assert first.id == e1

        # Overflow the maxsize-1 live queue while the generator is parked → the hub drops us.
        for i in range(5):
            await db.append_event(conn, run, SniffEvent(prey=f"x{i}"), publish=hub.publish)
        assert len(hub) == 0  # dropped (drop-and-resync)

        async def drain_rest() -> list[Envelope]:
            return [env async for env in gen]  # runs to StopAsyncIteration — or hangs if unfixed

        # Must terminate on its own; wait_for fails the test if the stream ever hangs.
        rest = await asyncio.wait_for(drain_rest(), timeout=3.0)
        assert all(env.id is None or env.id > e1 for env in rest)  # only real, post-catch-up frames
        assert len(hub) == 0  # the generator deregistered on close
    finally:
        await conn.close()
