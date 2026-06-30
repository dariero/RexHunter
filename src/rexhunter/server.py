"""FastAPI host: lifespan-launched hunt daemon + SSE feed projected from the log."""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from rexhunter import db, stub, verdicts
from rexhunter.events import Verdict
from rexhunter.scheduler import run_scheduler

DB_PATH = os.environ.get("REXHUNTER_DB", "rexhunter.db")
POLL_INTERVAL = 0.5

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

    schedule, brain_for, registry, cap, drafter = stub.daemon_config()
    tasks = [
        asyncio.create_task(
            run_scheduler(
                DB_PATH, schedule, brain_for=brain_for, registry=registry, max_concurrent=cap
            )
        ),
        asyncio.create_task(verdicts.run_job_worker(DB_PATH, drafter=drafter)),
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


@app.get("/events")
async def stream(request: Request) -> StreamingResponse:
    raw = request.headers.get("last-event-id", "")
    resume_from = int(raw) if raw.isdigit() else 0

    async def feed(last_seen: int) -> AsyncIterator[str]:
        # Own read connection: under WAL, readers never block the single writer (invariant 7).
        reader = await db.connect(DB_PATH)
        try:
            while True:
                async with reader.execute(
                    "SELECT id, payload FROM trajectory_events WHERE id > ? ORDER BY id",
                    (last_seen,),
                ) as cursor:
                    rows = list(await cursor.fetchall())
                for event_id, payload in rows:
                    yield f"id: {event_id}\ndata: {payload}\n\n"
                    last_seen = int(event_id)
                await asyncio.sleep(POLL_INTERVAL)
        finally:
            await reader.close()

    return StreamingResponse(feed(resume_from), media_type="text/event-stream")


@app.get("/")
async def page() -> HTMLResponse:
    return HTMLResponse("""
        <pre id="log"></pre>
        <script>
          new EventSource("/events").onmessage =
            e => log.textContent += e.data + "\\n";
        </script>
    """)
