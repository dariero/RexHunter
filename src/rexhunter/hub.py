"""The in-process broadcast hub (ADR pillar 3): hunt task → SQLite append (write-ahead) → HUB →
SSE endpoint → browser projection.

A small object holding one bounded ``asyncio.Queue`` per connected viewer. After a DB commit
returns the new global event id, the event is *offered* to every viewer queue — offered, never
awaited: a viewer whose bounded queue is full is **dropped** from the hub (drop-and-resync), so a
slow viewer can never block or slow the hunter. The hub is allowed to be lossy precisely because
of the write-ahead rule (invariant 1) — correctness lives in the log, and a dropped viewer
recovers everything it missed via snapshot + catch-up (``WHERE id > :last_seen``).

Deliberately dumb: no durable state, no delivery guarantee, no per-viewer buffering beyond the
bounded queue. It imports neither the DB nor the event models — it carries opaque
``(id, payload-string)`` envelopes that the two trajectory writers (``db.append_event`` and
``verdicts.capture_prey``) hand it post-commit via the ``publish`` callback. That inversion keeps
pillar 3 decoupled from pillars 1/2 (only ``server.py`` wires the two together).

Accepted limit (ADR): the in-process hub binds streaming to a single backend process. Horizontal
scale would need external pub/sub between writer and SSE servers; the terrarium is single-process
by design — this is the design, not a debt to pre-solve.
"""

import asyncio
import itertools
import os
from collections.abc import Awaitable, Callable
from typing import NamedTuple

# Knobs. VIEWER_QUEUE_MAXSIZE is the per-viewer bounded-queue depth (the drop threshold): a viewer
# more than this many events behind is dropped rather than allowed to apply back-pressure.
# HEARTBEAT_INTERVAL_S is the keep-alive cadence that stops intermediaries killing idle connections.
VIEWER_QUEUE_MAXSIZE = int(os.environ.get("REXHUNTER_VIEWER_QUEUE_MAXSIZE", "256"))
HEARTBEAT_INTERVAL_S = float(os.environ.get("REXHUNTER_HEARTBEAT_INTERVAL_S", "15"))


class Envelope(NamedTuple):
    """One item on a viewer queue. ``id`` is the global event cursor (``Last-Event-ID``); ``data``
    is the event's stored JSON payload string (the exact bytes in the log, so a live-spliced viewer
    and a catch-up viewer render byte-identical feeds). A heartbeat is ``id=None`` — it renders to a
    bare SSE comment and therefore never advances the client's high-water mark."""

    id: int | None
    data: str

    def sse(self) -> str:
        """Render to a Server-Sent-Events frame. A real event carries its ``id:`` (the resume
        cursor); a heartbeat is a bare comment line the browser ignores but proxies do not."""
        if self.id is None:
            return ": keep-alive\n\n"
        return f"id: {self.id}\ndata: {self.data}\n\n"


class Hub:
    """Fan-out of committed events to registered viewers over bounded, lossy queues."""

    def __init__(
        self,
        *,
        maxsize: int = VIEWER_QUEUE_MAXSIZE,
        heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S,
    ) -> None:
        self._maxsize = maxsize
        self.heartbeat_interval_s = heartbeat_interval_s
        self._viewers: dict[str, asyncio.Queue[Envelope]] = {}
        self._ids = itertools.count()

    def __len__(self) -> int:
        return len(self._viewers)

    def __contains__(self, viewer_id: str) -> bool:
        return viewer_id in self._viewers

    def register(self) -> tuple[str, asyncio.Queue[Envelope]]:
        """Add a viewer; return its id and its bounded queue. The SSE endpoint drains the queue and
        must ``deregister`` on disconnect so no queue is leaked."""
        viewer_id = str(next(self._ids))
        queue: asyncio.Queue[Envelope] = asyncio.Queue(maxsize=self._maxsize)
        self._viewers[viewer_id] = queue
        return viewer_id, queue

    def deregister(self, viewer_id: str) -> None:
        """Remove a viewer. Idempotent — a double deregister (disconnect racing a drop) is a
        no-op, never a KeyError."""
        self._viewers.pop(viewer_id, None)

    def _offer(self, envelope: Envelope) -> None:
        # Offer to every viewer without ever awaiting. A full queue means the viewer fell too far
        # behind: drop it (drop-and-resync) rather than block the producer. Iterate over a COPY of
        # the items because a drop mutates the dict mid-iteration.
        for viewer_id, queue in list(self._viewers.items()):
            try:
                queue.put_nowait(envelope)
            except asyncio.QueueFull:
                self.deregister(viewer_id)

    def publish(self, event_id: int, payload: str) -> None:
        """Fan a committed event out to every viewer. Called by the trajectory writers AFTER their
        commit (write-ahead, invariant 1) — this is the ``publish`` callback they thread down.
        Non-blocking by construction: a full queue is dropped, never awaited."""
        self._offer(Envelope(event_id, payload))

    def _beat(self) -> None:
        # One keep-alive to every queue, via the same lossy offer path.
        self._offer(Envelope(None, ""))

    async def run_heartbeats(
        self, *, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    ) -> None:
        """Emit a keep-alive to every viewer once per ``heartbeat_interval_s``, forever (until the
        daemon cancels it on shutdown). ``sleep`` is injectable so tests drive the cadence without
        a wall-clock wait; it defaults to ``asyncio.sleep``."""
        while True:
            await sleep(self.heartbeat_interval_s)
            self._beat()
