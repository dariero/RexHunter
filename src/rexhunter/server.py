"""FastAPI host: lifespan-launched hunt daemon + SSE feed projected from the log."""

import asyncio
import logging
import os
import random
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager, suppress

import aiosqlite
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from rexhunter import db
from rexhunter.events import SniffEvent

DB_PATH = os.environ.get("REXHUNTER_DB", "rexhunter.db")
SNIFF_INTERVAL = 5.0
POLL_INTERVAL = 0.5

logger = logging.getLogger("rexhunter")


async def rex_loop(conn: aiosqlite.Connection) -> None:
    run_id = await db.start_run(conn, territory="mock-gym")
    try:
        while True:
            await asyncio.sleep(SNIFF_INTERVAL)
            prey = random.choice(["AI Engineer", "ML Platform Eng", "Eval Engineer"])
            await db.append_event(conn, run_id, SniffEvent(prey=prey))
    except asyncio.CancelledError:
        await db.finish_run(conn, run_id, outcome="aborted", abort_reason="daemon shutdown")
        raise


def surface_crash(task: asyncio.Task[None]) -> None:
    # A daemon task that dies must die loudly, not leave the server serving a dead stream.
    if not task.cancelled():
        task.result()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    conn = await db.connect(DB_PATH)
    crashed = await db.mark_crashed_runs(conn)
    if crashed:
        logger.warning("boot: marked %d dangling run(s) as crashed", crashed)
    task = asyncio.create_task(rex_loop(conn))
    task.add_done_callback(surface_crash)
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        await conn.close()


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
