"""P3.1 gate — the in-process broadcast hub (ADR pillar 3), offline.

The hub is deliberately simple and lossy: one bounded asyncio.Queue per viewer, events
*offered* (never awaited) after the DB commit, a full queue dropped rather than blocked.
Correctness lives in the log (invariant 1), not the hub — so these tests pin exactly four
load-bearing properties and nothing more:

1. Write-ahead ordering (the invariant-1 gate, structural): an envelope reaches a viewer queue
   ONLY after its row is committed and visible to a separate reader connection.
2. Lossy drop, no back-pressure: a viewer with a full queue is dropped; the producer never blocks.
3. Fan-out: N viewers all receive the same envelope; deregister is clean and idempotent.
4. Heartbeat: the interval loop fires a keep-alive to every queue (clock injected — no wall-clock).
"""

import asyncio
from contextlib import suppress
from pathlib import Path

import pytest

from rexhunter import db, verdicts
from rexhunter.events import SniffEvent
from rexhunter.hub import Envelope, Hub

pytestmark = pytest.mark.anyio


# ── 1. Write-ahead ordering: the invariant-1 gate (structural) ────────────────


async def test_publish_reaches_queue_only_after_commit(tmp_path: Path) -> None:
    """append_event publishes strictly post-commit: the row is visible to a SEPARATE reader
    connection by the time the envelope is on the queue. Reorder publish before commit and the
    fresh reader would see zero rows — so this is structural, not incidental."""
    hub = Hub(maxsize=8)
    _, queue = hub.register()
    writer = await db.connect(tmp_path / "rex.db")
    reader = await db.connect(tmp_path / "rex.db")  # a DIFFERENT connection — sees only commits
    try:
        run_id = await db.start_run(writer, territory="gate")
        event_id = await db.append_event(writer, run_id, SniffEvent(prey="p0"), publish=hub.publish)

        env = queue.get_nowait()  # the envelope reached the viewer
        assert env.id == event_id

        async with reader.execute(
            "SELECT payload FROM trajectory_events WHERE id = ?", (env.id,)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None  # committed & visible before the publish landed (write-ahead)
        assert row[0] == env.data  # the envelope carries the exact stored payload string
    finally:
        await writer.close()
        await reader.close()


async def test_capture_prey_also_publishes_post_commit(tmp_path: Path) -> None:
    """capture_prey is the SECOND trajectory writer (it bypasses append_event for one-txn
    atomicity). It too publishes its PreyCapturedEvent post-commit — else every stub hunt's
    capture would reach the log but never the live stream."""
    hub = Hub(maxsize=8)
    _, queue = hub.register()
    writer = await db.connect(tmp_path / "rex.db")
    reader = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await db.start_run(writer, territory="gate")
        prey_id = await verdicts.capture_prey(
            writer, run_id, territory="gate", posting="job-1", publish=hub.publish
        )

        env = queue.get_nowait()
        assert env.id is not None
        async with reader.execute(
            "SELECT type, payload FROM trajectory_events WHERE id = ?", (env.id,)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row[0] == "prey_captured"  # the one trajectory event a capture writes
        assert row[1] == env.data
        # the prey projection row landed in the SAME committed transaction
        async with reader.execute("SELECT status FROM prey WHERE id = ?", (prey_id,)) as cur:
            prow = await cur.fetchone()
        assert prow is not None and prow[0] == verdicts.AWAITING
    finally:
        await writer.close()
        await reader.close()


# ── 2. Lossy drop, no back-pressure (the load-bearing property) ───────────────


async def test_full_queue_viewer_is_dropped_producer_never_blocks(tmp_path: Path) -> None:
    """A viewer whose bounded queue overflows is dropped; publish never awaits a full queue, so
    the producer runs to completion unslowed. The events still land in the log — the dropped
    viewer recovers from there (invariant 1)."""
    hub = Hub(maxsize=2)
    vid, _queue = hub.register()  # registered but never drained -> it will overflow
    writer = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await db.start_run(writer, territory="gate")
        # If publish ever blocked on the full queue, this timeout would fire.
        async with asyncio.timeout(2.0):
            for i in range(10):
                await db.append_event(writer, run_id, SniffEvent(prey=f"p{i}"), publish=hub.publish)

        assert vid not in hub  # overflowed -> dropped (drop-and-resync)
        assert len(hub) == 0  # no leaked queue left behind
        logged = await db.read_events(writer, run_id)
        assert len(logged) == 10  # every event committed despite the drop (log is truth)
    finally:
        await writer.close()


# ── 3. Fan-out + clean register / deregister ──────────────────────────────────


async def test_fanout_every_viewer_receives_the_same_event(tmp_path: Path) -> None:
    hub = Hub(maxsize=8)
    viewers = [hub.register() for _ in range(5)]
    writer = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await db.start_run(writer, territory="gate")
        event_id = await db.append_event(writer, run_id, SniffEvent(prey="p"), publish=hub.publish)
        assert len(hub) == 5
        for _vid, q in viewers:
            env = q.get_nowait()
            assert env.id == event_id
            assert q.empty()  # exactly one, no duplication across the fan-out
    finally:
        await writer.close()


async def test_deregister_is_idempotent_and_stops_delivery(tmp_path: Path) -> None:
    hub = Hub(maxsize=8)
    vid, q = hub.register()
    assert len(hub) == 1

    hub.deregister(vid)
    assert len(hub) == 0  # no leaked queue
    hub.deregister(vid)  # idempotent: a second deregister is a no-op, never a KeyError
    assert len(hub) == 0

    writer = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await db.start_run(writer, territory="gate")
        await db.append_event(writer, run_id, SniffEvent(prey="p"), publish=hub.publish)
        assert q.empty()  # a deregistered viewer receives nothing further
    finally:
        await writer.close()


async def test_publish_with_no_viewers_is_a_noop(tmp_path: Path) -> None:
    """The daemon publishes on every event whether or not anyone is watching (invariant 1: the
    log is written regardless). With zero viewers, publish is a harmless no-op."""
    hub = Hub(maxsize=8)
    writer = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await db.start_run(writer, territory="gate")
        await db.append_event(writer, run_id, SniffEvent(prey="p"), publish=hub.publish)
        assert len(hub) == 0
    finally:
        await writer.close()


# ── 4. Heartbeat (injected clock — no wall-clock sleeps) ──────────────────────


async def test_heartbeat_fires_to_every_queue_on_interval() -> None:
    """run_heartbeats emits a keep-alive to every viewer queue once per interval. The clock is
    injected: reaching the SECOND sleep proves the first beat fired between the two — no
    wall-clock, fully deterministic."""
    hub = Hub(maxsize=8, heartbeat_interval_s=999)  # interval irrelevant; the clock is injected
    _, q1 = hub.register()
    _, q2 = hub.register()

    reached_second_sleep = asyncio.Event()
    calls = 0

    async def fake_sleep(_seconds: float) -> None:
        nonlocal calls
        calls += 1
        if calls >= 2:  # the beat between sleep #1 and sleep #2 has already run to get here
            reached_second_sleep.set()
            await asyncio.Event().wait()  # park forever; the test cancels the task
        # first call returns immediately -> exactly one beat fires

    task = asyncio.create_task(hub.run_heartbeats(sleep=fake_sleep))
    try:
        await asyncio.wait_for(reached_second_sleep.wait(), timeout=1.0)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    env1 = q1.get_nowait()
    env2 = q2.get_nowait()
    assert env1.id is None and env2.id is None  # a heartbeat carries no id...
    assert env1.sse() == ": keep-alive\n\n"  # ...so it never advances Last-Event-ID
    assert q1.empty() and q2.empty()  # exactly one beat


# ── Envelope rendering (SSE framing) ──────────────────────────────────────────


def test_envelope_sse_framing() -> None:
    assert Envelope(id=7, data='{"type":"sniff","prey":"x"}').sse() == (
        'id: 7\ndata: {"type":"sniff","prey":"x"}\n\n'
    )
    assert Envelope(id=None, data="").sse() == ": keep-alive\n\n"


def test_envelope_notification_framing() -> None:
    """An id-less envelope WITH data is a NOTIFICATION (slice C): a `data:` frame that fires the
    browser's onmessage (→ board refresh) but carries no `id:`, so it never advances Last-Event-ID
    or collides with the trajectory-id stream (pen_events has its own id sequence). An id-less EMPTY
    envelope stays a heartbeat comment (which does NOT fire onmessage)."""
    assert Envelope(id=None, data='{"type":"verdict"}').sse() == 'data: {"type":"verdict"}\n\n'
    assert Envelope(id=None, data="").sse() == ": keep-alive\n\n"


def test_notify_fans_an_idless_frame_to_every_viewer() -> None:
    """hub.notify broadcasts an id-less notification to every viewer — the same lossy
    fan-out as publish, but carrying no id so the trajectory resume cursor is untouched. Post-commit
    at the call site keeps write-ahead (inv 1) intact."""
    hub = Hub(maxsize=8)
    viewers = [hub.register() for _ in range(3)]
    hub.notify('{"type":"verdict","verdict":"feast"}')
    for _vid, q in viewers:
        env = q.get_nowait()
        assert env.id is None
        assert env.data == '{"type":"verdict","verdict":"feast"}'
        assert env.sse().startswith("data:")  # a real message frame, not a keep-alive comment
        assert q.empty()  # exactly one, no duplication
