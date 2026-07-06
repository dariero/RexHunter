"""FastAPI host: lifespan-launched hunt daemon + SSE feed projected from the log."""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime

import aiosqlite
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from rexhunter import brain, db, render, scheduler, stub, verdicts, viewstate
from rexhunter.events import Verdict
from rexhunter.hub import Envelope, Hub
from rexhunter.loop import COST_CEILING_USD
from rexhunter.scheduler import run_scheduler

DB_PATH = os.environ.get("REXHUNTER_DB", "rexhunter.db")

logger = logging.getLogger("rexhunter")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


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

    schedule, stub_brain_for, registry, cap, drafter = stub.daemon_config()
    # Select the daemon's brain by REXHUNTER_BRAIN (the autonomous-spender containment): the stub
    # default routes daemon_config's brain through (the injectable seam the lifespan gate patches),
    # while `live` swaps in the STREAMING/THINKING adapter (Unit 3) — Rex's reasoning relays to
    # /events. `client` is the paid client's handle (None on stub); we own its close on shutdown.
    brain_for, client = brain.select_brain_for(registry, default=stub_brain_for)
    tasks = [
        asyncio.create_task(
            run_scheduler(
                DB_PATH,
                schedule,
                brain_for=brain_for,
                registry=registry,
                max_concurrent=cap,
                max_iterations=_env_int("MAX_ITERATIONS", 50),
                cost_ceiling_usd=_env_float("COST_CEILING_USD", COST_CEILING_USD),
                daemon_spend_ceiling_usd=_env_float(
                    "DAEMON_SPEND_CEILING_USD", scheduler.DAEMON_SPEND_CEILING_USD
                ),
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
        # Close the paid client AFTER the scheduler has drained — it drove brain calls on this
        # client, so an earlier close could abort an in-flight request. None on the stub path.
        if client is not None:
            await client.aclose()


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


@app.get("/viewstate")
async def get_viewstate() -> HTMLResponse:
    """The server-rendered game board (ADR invariant 2): build_viewstate folds the log into a
    ViewState, render draws it. The browser re-fetches this on each SSE tick — a dumb painter, never
    re-running the reducer. now() is injected HERE, the live clock (inv 5), never inside project."""
    conn = await db.connect(DB_PATH)
    try:
        state = await viewstate.build_viewstate(conn, datetime.now(UTC))
    finally:
        await conn.close()
    return HTMLResponse(render.render(state))


# The page shell (CSS + fetch/tick JS). A raw string: the ``\n`` in the JS stays literal for the
# browser, not a Python newline. It renders the server board and re-fetches it on each SSE tick
# (the projection stays server-side, invariant 2 — no reducer in the browser).
_SHELL = r"""<!doctype html>
<title>RexHunter 🦖</title>
<style>
  body{background:#0b0f0a;color:#7dd87d;font-family:'Courier New',monospace;margin:0;padding:1rem}
  .board{display:flex;flex-direction:column;gap:.75rem}
  .hud{display:flex;justify-content:space-between;font-size:1.2rem;
    border-bottom:1px solid #2f4f2f;padding-bottom:.4rem}
  .runs{display:flex;flex-wrap:wrap;gap:.5rem}
  .run{border:1px solid #2f4f2f;padding:.5rem;min-width:12rem;background:#0f150e}
  .run h3{margin:.1rem 0;color:#9effa0;font-size:.9rem}
  .thinking{color:#5a8a5a;font-size:.75rem;max-height:3rem;overflow:auto;white-space:pre-wrap}
  .pen h2{color:#9effa0;border-bottom:1px solid #2f4f2f}
  .prey{display:flex;gap:.6rem;align-items:center;padding:.3rem .5rem;
    border-left:3px solid #666;margin:.2rem 0;background:#0f150e}
  .prey .posting{flex:1;color:#cceeff;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .badge{font-size:.7rem;padding:.1rem .4rem;border-radius:.2rem;color:#000}
  .status-awaiting_verdict{border-left-color:#e0a020}
  .status-awaiting_verdict .badge{background:#e0a020}
  .status-feasted{border-left-color:#3caa3c}
  .status-feasted .badge{background:#3caa3c}
  .status-released{border-left-color:#777}
  .status-released .badge{background:#777}
  .status-ambered{border-left-color:#c8862a}
  .status-ambered .badge{background:#c8862a}
  .actions{display:flex;gap:.3rem;align-items:center}
  .actions button{background:#1b3a1b;color:#9effa0;border:1px solid #2f4f2f;
    font-family:inherit;font-size:.7rem;padding:.15rem .5rem;cursor:pointer}
  .actions button:hover{background:#2f5f2f}
  .reason{background:#000;color:#9effa0;border:1px solid #2f4f2f;font-family:inherit;
    font-size:.7rem;padding:.15rem .3rem;width:8rem}
  .empty{color:#4a6a4a;font-style:italic}
  #feed{margin-top:1rem;background:#000;color:#4a7a4a;font-size:.7rem;max-height:12rem;
    overflow:auto;padding:.5rem;border-top:1px solid #2f4f2f;white-space:pre-wrap}
  [data-phase="night"]{filter:brightness(.75) hue-rotate(200deg)}
</style>
<div id="board"><p class="empty">loading…</p></div>
<pre id="feed"></pre>
<script>
  const board = document.getElementById('board');
  async function refresh(){ board.innerHTML = await (await fetch('/viewstate')).text(); }
  board.addEventListener('click', async (ev) => {
    const btn = ev.target.closest('button[data-verdict]');
    if(!btn) return;
    const verdict = btn.dataset.verdict;
    const rEl = btn.closest('.prey').querySelector('.reason');
    const note = rEl ? rEl.value.trim() : '';
    if(verdict === 'release' && !note){ rEl.focus(); return; }
    const body = {prey_id: btn.dataset.preyId, verdict};
    if(verdict === 'release') body.reason = note;
    if(verdict === 'amber' && note) body.provenance = note;
    await fetch('/verdict', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body)});
    refresh();
  });
  (async () => {
    await refresh();
    const snap = await (await fetch('/snapshot')).json();
    const feed = document.getElementById('feed');
    const es = new EventSource('/events?since=' + snap.latest_id);
    es.onmessage = (e) => {
      feed.textContent += e.data + '\n';
      feed.scrollTop = feed.scrollHeight;
      refresh();
    };
  })();
</script>
"""


@app.get("/")
async def page() -> HTMLResponse:
    """The terrarium shell: render the server-drawn board (/viewstate), then open the live SSE feed
    from the log head (?since=latest_id). Each event ticks a board re-fetch (projection stays
    server-side, invariant 2 — no reducer in the browser) and appends to the raw feed."""
    return HTMLResponse(_SHELL)
