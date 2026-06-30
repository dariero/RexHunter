"""FastAPI host: lifespan-launched hunt daemon + SSE feed projected from the log."""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from rexhunter import db, stub
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
    finally:
        await sweep.close()

    schedule, brain_for, registry, cap = stub.daemon_config()
    task = asyncio.create_task(
        run_scheduler(DB_PATH, schedule, brain_for=brain_for, registry=registry, max_concurrent=cap)
    )
    task.add_done_callback(surface_crash)
    try:
        yield
    finally:
        # Graceful shutdown: cancel, then AWAIT to drain. run_hunt's shielded cleanup marks
        # every in-flight run aborted/"daemon shutdown" before the group tears down - no run is
        # left outcome IS NULL (which the next boot would mislabel 'crashed', breaking DoD #1).
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(lifespan=lifespan)


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
