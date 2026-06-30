"""P2.3-wiring gate — the scheduler runs through the daemon's REAL ASGI lifespan.

"Gate-green" and "has run in the real server" are different facts. The invariant-7 proof
(`test_invariant7.py`) cancels `run_hunt`/`run_scheduler` tasks DIRECTLY; here the *lifespan*
is the caller — startup spawns the scheduler, shutdown cancels-and-awaits it through a
`finally:` that also tears down a connection. We boot the app through
`app.router.lifespan_context(app)` — the exact callable uvicorn invokes on startup/shutdown,
not a hand-made harness — so no httpx/TestClient dependency is needed, and because it runs in
THIS event loop the gate can await the hunts directly and introspect `asyncio.all_tasks()`.

The property the harness could never test: a graceful lifespan shutdown drains every in-flight
hunt through `run_hunt`'s shielded cleanup BEFORE the lifespan returns — no run left
`outcome IS NULL` (which the next boot would mislabel `'crashed'`), no orphaned task, no leaked
connection.
"""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import aiosqlite
import pytest

from rexhunter import db, server, stub
from rexhunter.events import ToolCallEvent, ToolResultEvent
from rexhunter.loop import Brain, Decision, HuntComplete, ToolCallDecision
from rexhunter.tools import ToolRegistry
from rexhunter.verdicts import Drafter

pytestmark = pytest.mark.anyio

BrainFactory = Callable[[Sequence[Decision]], Brain]


async def _runs(conn: aiosqlite.Connection) -> list[tuple[str, str | None, str | None]]:
    async with conn.execute("SELECT id, outcome, abort_reason FROM runs") as cur:
        return [(str(r[0]), r[1], r[2]) for r in await cur.fetchall()]


async def test_scheduler_runs_through_real_lifespan_and_drains_on_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted_brain: BrainFactory,
) -> None:
    db_path = tmp_path / "rex.db"

    # ── inject a deterministic, blocking daemon: two territories, a tool that parks in-flight
    #    until shutdown cancels it. Proves GENUINE concurrency (both parked at once), not a
    #    sequence of fast hunts that never overlap.
    started = asyncio.Semaphore(0)  # released once per hunt that reaches the tool
    release = asyncio.Event()  # never set in this test -> the tool hangs until cancelled
    reg = ToolRegistry()

    @reg.tool
    async def blocker() -> str:
        started.release()
        await release.wait()  # cancellable; daemon shutdown tears it down
        return "ok"

    def brain_for(_territory: str) -> Brain:
        # the conftest stub-brain factory (no model, no spend); tool by .__name__, never a literal
        return scripted_brain([ToolCallDecision(tool=blocker.__name__, args={}), HuntComplete()])

    async def drafter(_conn: aiosqlite.Connection, _prey_id: str) -> str:
        return "stub draft"  # the job worker runs alongside; no FEAST here, so it just idles

    def fake_config() -> tuple[
        Mapping[str, float], Callable[[str], Brain], ToolRegistry, int, Drafter
    ]:
        return {"t0": 0.01, "t1": 0.01}, brain_for, reg, 4, drafter

    monkeypatch.setattr(server, "DB_PATH", str(db_path))
    monkeypatch.setattr(stub, "daemon_config", fake_config)

    # ── connection-leak accounting: wrap db.connect, counting opens and (per-instance) closes.
    real_connect = db.connect
    opened: list[aiosqlite.Connection] = []
    closed: list[int] = []

    async def counting_connect(path: str | Path) -> aiosqlite.Connection:
        conn = await real_connect(path)
        opened.append(conn)
        orig_close = conn.close

        async def close_and_count() -> None:
            await orig_close()
            closed.append(id(conn))

        conn.close = close_and_count  # type: ignore[method-assign]
        return conn

    monkeypatch.setattr(db, "connect", counting_connect)

    baseline_tasks = asyncio.all_tasks()

    # ── boot the app through its REAL lifespan; assert inside, shut down on exit. ───────────
    async with server.app.router.lifespan_context(server.app):
        # startup spawned run_scheduler; wait until TWO hunts are parked in the blocking tool.
        for _ in range(2):
            await asyncio.wait_for(started.acquire(), timeout=5)

        reader = await real_connect(db_path)
        try:
            runs_mid = await _runs(reader)
            assert len(runs_mid) == 2, f"expected 2 concurrent hunts, got {len(runs_mid)}"
            # each is genuinely parked: a ToolCallEvent written, no ToolResultEvent yet.
            for run_id, _outcome, _ in runs_mid:
                evs = await db.read_events(reader, run_id)  # via patched connect is fine (read)
                assert any(isinstance(e, ToolCallEvent) for e in evs)
                assert not any(isinstance(e, ToolResultEvent) for e in evs)
        finally:
            await reader.close()
    # ── exiting the context manager ran the REAL shutdown (cancel -> shielded finish -> drain).

    await asyncio.sleep(0)  # let any done-callbacks settle

    # (1) no orphaned tasks: nothing the wiring spawned is still alive.
    leftover = {
        t
        for t in asyncio.all_tasks()
        if t not in baseline_tasks and t is not asyncio.current_task() and not t.done()
    }
    assert leftover == set(), f"orphaned tasks survived shutdown: {leftover}"

    # (2) no leaked connections: every connection opened by the lifespan/scheduler was closed.
    assert len(closed) == len(opened), (
        f"connection leak: {len(opened)} opened, {len(closed)} closed"
    )

    # (3) every in-flight run drained to aborted/"daemon shutdown" — none left NULL.
    final = await real_connect(db_path)
    try:
        runs = await _runs(final)
        assert len(runs) == 2
        for run_id, outcome, abort_reason in runs:
            assert outcome == "aborted", (
                f"{run_id}: outcome={outcome!r} (a NULL would mislabel as crashed)"
            )
            assert abort_reason == "daemon shutdown"

        # (4) projection (invariant 2): the SSE feed's own query sees the scheduler's writes.
        async with final.execute(
            "SELECT id, payload FROM trajectory_events WHERE id > 0 ORDER BY id"
        ) as cur:
            projected = list(await cur.fetchall())
        assert len(projected) >= 2  # at least the two ToolCallEvents

        # (5) no dangling write txn: a fresh writer succeeds without "database is locked".
        live_run = await db_write_smoke(final)
        assert live_run is not None
    finally:
        await final.close()


async def db_write_smoke(conn: aiosqlite.Connection) -> str:
    from rexhunter.events import SniffEvent

    run_id = await db.start_run(conn, territory="smoke")
    await db.append_event(conn, run_id, SniffEvent(prey="post-shutdown write"))
    return run_id
