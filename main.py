import asyncio
import random
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

events = []  # Rex's diary - everything he's ever done


async def rex_loop():
    while True:
        await asyncio.sleep(5)
        prey = random.choice(["AI Engineer", "ML Platform Eng", "Eval Engineer"])
        events.append(f"Rex sniffs the air... fresh {prey} scent!")


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(rex_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/events")
async def stream():
    async def feed():
        sent = 0
        while True:
            while sent < len(events):  # anything new in the diary?
                yield f"data: {events[sent]}\n\n"
                sent += 1
            await asyncio.sleep(0.5)  # check again shortly

    return StreamingResponse(feed(), media_type="text/event-stream")


@app.get("/")
async def page():
    return HTMLResponse("""
        <pre id="log"></pre>
        <script>
          new EventSource("/events").onmessage =
            e => log.textContent += e.data + "\\n";
        </script>
    """)
