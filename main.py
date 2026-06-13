import asyncio
import random
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

events: list[str] = []


async def rex_loop() -> None:
    while True:
        await asyncio.sleep(5)
        prey = random.choice(["AI Engineer", "ML Platform Eng", "Eval Engineer"])
        events.append(f"Rex sniffs the air... fresh {prey} scent!")


def surface_crash(task: asyncio.Task[None]) -> None:
    # A daemon task that dies must die loudly, not leave the server serving a dead stream.
    if not task.cancelled():
        task.result()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    task = asyncio.create_task(rex_loop())
    task.add_done_callback(surface_crash)
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


app = FastAPI(lifespan=lifespan)


@app.get("/events")
async def stream() -> StreamingResponse:
    async def feed() -> AsyncIterator[str]:
        sent = 0
        while True:
            while sent < len(events):
                yield f"data: {events[sent]}\n\n"
                sent += 1
            await asyncio.sleep(0.5)

    return StreamingResponse(feed(), media_type="text/event-stream")


@app.get("/")
async def page() -> HTMLResponse:
    return HTMLResponse("""
        <pre id="log"></pre>
        <script>
          new EventSource("/events").onmessage =
            e => log.textContent += e.data + "\\n";
        </script>
    """)
