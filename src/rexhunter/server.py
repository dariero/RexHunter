"""FastAPI host: lifespan-launched hunt daemon + SSE feed projected from the log."""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager, suppress

import aiosqlite
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from rexhunter import db, stub, verdicts
from rexhunter.events import Verdict
from rexhunter.hub import Envelope, Hub
from rexhunter.scheduler import run_scheduler

DB_PATH = os.environ.get("REXHUNTER_DB", "rexhunter.db")

logger = logging.getLogger("rexhunter")


def surface_crash(task: asyncio.Task[None]) -> None:
    # A daemon task that dies must die loudly, not leave the server serving a dead stream.
    if not task.cancelled():
        task.result()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    # Boot crash-sweep on its OWN short-lived connection; the scheduler opens one connection
    # per hunt (invariant 7), so the lifespan shares no connection with it and spawns it once.
    sweep = await db.connect(DB_PATH)
    try:
        crashed = await db.mark_crashed_runs(sweep)
        if crashed:
            logger.warning("boot: marked %d dangling run(s) as crashed", crashed)
        # The jobs analogue: a job left 'running' from a prior process is requeued, not stranded.
        requeued = await verdicts.requeue_running_jobs(sweep)
        if requeued:
            logger.warning("boot: requeued %d running job(s)", requeued)
    finally:
        await sweep.close()

    # The broadcast hub (ADR pillar 3): the scheduler publishes each committed event to it
    # post-commit (write-ahead, invariant 1), and the heartbeat task keeps idle viewers alive.
    # Stored on app.state so the SSE endpoint can register viewers against it. P3.1 wires it
    # ALONGSIDE the prototype /events poll — that endpoint stays until P3.2 retires it.
    hub = Hub()
    app.state.hub = hub

    schedule, brain_for, registry, cap, drafter = stub.daemon_config()
    tasks = [
        asyncio.create_task(
            run_scheduler(
                DB_PATH,
                schedule,
                brain_for=brain_for,
                registry=registry,
                max_concurrent=cap,
                publish=hub.publish,
            )
        ),
        asyncio.create_task(verdicts.run_job_worker(DB_PATH, drafter=drafter)),
        asyncio.create_task(hub.run_heartbeats()),
    ]
    for task in tasks:
        task.add_done_callback(surface_crash)
    try:
        yield
    finally:
        # Graceful shutdown: cancel every daemon task, then AWAIT to drain. run_hunt's shielded
        # cleanup marks every in-flight run aborted/"daemon shutdown" before the group tears down,
        # and the worker closes its connection in its finally - no run left outcome IS NULL (which
        # the next boot would mislabel 'crashed', breaking DoD #1), no orphaned task, no leak.
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task


app = FastAPI(lifespan=lifespan)


# ── SSE stream plumbing (ADR pillar 3): snapshot + catch-up + live splice ──────────────────────


async def catch_up(conn: aiosqlite.Connection, last_seen: int) -> list[Envelope]:
    """Replay the log tail a viewer missed: WHERE id > last_seen, in global-id order — the SSE
    stream cursor (invariant 2). Monotonic ids make this a gapless backfill the caller can dedup
    against its high-water mark."""
    async with conn.execute(
        "SELECT id, payload FROM trajectory_events WHERE id > ? ORDER BY id", (last_seen,)
    ) as cur:
        rows = await cur.fetchall()
    return [Envelope(int(row[0]), str(row[1])) for row in rows]


async def stream_events(
    conn: aiosqlite.Connection, hub: Hub, *, last_seen: int
) -> AsyncGenerator[Envelope]:
    """Catch-up + live splice as one stream of envelopes (the SSE endpoint renders each `.sse()`).

    Register with the hub FIRST so no live event is missed during the backfill, THEN replay the
    log tail (catch-up), THEN drain the live queue — dropping anything at or below the high-water
    mark so an event present in BOTH the backfill and the queue is emitted exactly once
    (monotonic-id dedup: the browser never sees a gap and never double-applies). A heartbeat
    (id=None) passes straight through and never advances the mark.

    Drop-and-resync (ADR line 166): if this viewer's bounded queue overflowed, the hub dropped it
    and sends it nothing further — not even heartbeats. A bare `await queue.get()` would then hang
    forever and the HTTP response would never close, so EventSource would never reconnect (exactly
    what heartbeats exist to prevent). Instead we wait at most two heartbeat intervals: a still-live
    viewer is fed at least one keep-alive per interval, so a timeout with the viewer already gone
    from the hub means we were dropped — END the stream so the browser reconnects into the snapshot
    dance. A timeout while still registered is just an idle terrarium: keep waiting. The finally
    deregisters, so a normal disconnect leaks no queue.
    """
    viewer_id, queue = hub.register()
    try:
        high_water = last_seen
        for env in await catch_up(conn, last_seen):
            yield env
            if env.id is not None:
                high_water = max(high_water, env.id)
        while True:
            try:
                env = await asyncio.wait_for(queue.get(), timeout=hub.heartbeat_interval_s * 2)
            except TimeoutError:
                if viewer_id not in hub:
                    break  # dropped (drop-and-resync): end the stream → EventSource resyncs
                continue  # still registered, just idle: keep the connection open
            if env.id is None:
                yield env  # heartbeat: keep-alive, carries no id, bypasses the dedup
            elif env.id > high_water:
                high_water = env.id
                yield env
            # else: already delivered by catch-up — drop the duplicate (id <= high-water)
    finally:
        hub.deregister(viewer_id)


async def snapshot_state(conn: aiosqlite.Connection) -> dict[str, object]:
    """The reconnect snapshot, rendered from the log (invariant 2): the latest global event id
    (the client's Last-Event-ID seed), open runs, the prey pen, and each territory's latest state.
    A fresh client renders this instantly, then opens the stream from `latest_id`."""
    async with conn.execute("SELECT COALESCE(MAX(id), 0) FROM trajectory_events") as cur:
        row = await cur.fetchone()
    latest_id = int(row[0]) if row is not None else 0

    async with conn.execute(
        "SELECT id, territory, started_at FROM runs WHERE outcome IS NULL ORDER BY started_at"
    ) as cur:
        open_runs = [
            {"id": str(r[0]), "territory": str(r[1]), "started_at": str(r[2])}
            for r in await cur.fetchall()
        ]

    async with conn.execute(
        "SELECT id, territory, posting, status, captured_at FROM prey ORDER BY captured_at"
    ) as cur:
        pen = [
            {
                "id": str(r[0]),
                "territory": str(r[1]),
                "posting": str(r[2]),
                "status": str(r[3]),  # prey.status IS the maintained projection (invariant 2)
                "captured_at": str(r[4]),
            }
            for r in await cur.fetchall()
        ]

    # Territory state derived (not stored, invariant 5): the latest run per territory.
    async with conn.execute(
        "SELECT territory, outcome, MAX(started_at) FROM runs GROUP BY territory ORDER BY territory"
    ) as cur:
        territories = [
            {
                "territory": str(r[0]),
                "last_outcome": None if r[1] is None else str(r[1]),
                "last_started_at": str(r[2]),
            }
            for r in await cur.fetchall()
        ]

    return {
        "latest_id": latest_id,
        "open_runs": open_runs,
        "pen": pen,
        "territories": territories,
    }


class VerdictRequest(BaseModel):
    """The verdict POST body — the one Pydantic boundary the raw request bytes cross (inv. 3)."""

    prey_id: str
    verdict: Verdict
    reason: str | None = None
    provenance: str | None = None


@app.post("/verdict")
async def post_verdict(req: VerdictRequest) -> dict[str, bool]:
    """Apply a human verdict (Feast/Release/Amber) as a guarded, idempotent DB transition. There is
    no apply/send/submit tool anywhere in Rex — this POST IS the only path the state can move
    (invariant 4, Tiny Arms). Returns {"applied": False} on a no-op (replay / already-resolved)."""
    conn = await db.connect(DB_PATH)
    try:
        applied = await verdicts.submit_verdict(
            conn, req.prey_id, req.verdict, reason=req.reason, provenance=req.provenance
        )
    except ValueError as exc:  # e.g. RELEASE without a reason — a boundary error, not a 500
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await conn.close()
    return {"applied": applied}


@app.get("/snapshot")
async def snapshot() -> dict[str, object]:
    """Snapshot the terrarium from the log so a fresh client renders instantly, then opens the
    stream from `latest_id` (ADR pillar 3; invariant 2)."""
    conn = await db.connect(DB_PATH)
    try:
        return await snapshot_state(conn)
    finally:
        await conn.close()


@app.get("/events")
async def stream(request: Request) -> StreamingResponse:
    """The live consciousness stream over the real broadcast hub (ADR pillar 3): catch-up + live
    splice, resuming exactly where the client left off. The prototype poll that stood in for this
    is retired. The resume cursor is the `Last-Event-ID` header (set by EventSource on reconnect)
    or, on the first connect, the `?since=` query param the snapshot dance passes."""
    raw = request.headers.get("last-event-id", "") or request.query_params.get("since", "")
    last_seen = int(raw) if raw.isdigit() else 0
    hub: Hub = request.app.state.hub

    async def feed() -> AsyncIterator[str]:
        # Own read connection: under WAL, readers never block the single writer (invariant 7).
        reader = await db.connect(DB_PATH)
        try:
            async for env in stream_events(reader, hub, last_seen=last_seen):
                yield env.sse()
        finally:
            await reader.close()

    return StreamingResponse(feed(), media_type="text/event-stream")


@app.get("/")
async def page() -> HTMLResponse:
    # Snapshot + catch-up + live splice (ADR pillar 3): render the snapshot, then stream from
    # latest_id so the first connect resumes at the log head instead of replaying everything.
    return HTMLResponse("""
        <pre id="log"></pre>
        <script>
          fetch("/snapshot").then(r => r.json()).then(snap => {
            log.textContent = JSON.stringify(snap, null, 2) + "\\n--- live ---\\n";
            new EventSource("/events?since=" + snap.latest_id).onmessage =
              e => log.textContent += e.data + "\\n";
          });
        </script>
    """)
