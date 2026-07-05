"""Daemon live-wiring gate (`P5` Daemon live-wiring).

W.1 (offline, no spend): the lifespan must select its brain by `REXHUNTER_BRAIN` (stub default,
`live` = the Unit-3 STREAMING/THINKING brain, not the non-streaming one), and the scheduler must
be bounded by an **id-scoped** daemon spend ceiling — the seq-scoped per-run ceiling's analogue,
folded from ALL runs' `UsageEvent`s along the global `id` cursor (inv 5: a fold over the log, never
a second counter). Over the ceiling, the scheduler refuses to LAUNCH a new hunt (pauses, never
crashes); an in-flight hunt is untouched. This is Tiny Arms for money (inv 4's spirit): the daemon
structurally cannot exceed its budget.

W.2 (offline): the loop-built `ThinkingSink` closure, exercised under the REAL lifespan.
"""

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import aiosqlite
import httpx
import pytest

from rexhunter import brain, cost, db, scheduler, server, stub
from rexhunter.events import UsageEvent
from rexhunter.hub import Envelope
from rexhunter.loop import Brain, Context, Decision, HuntComplete, ThinkingSink
from rexhunter.tools import ToolRegistry
from rexhunter.verdicts import Drafter

pytestmark = pytest.mark.anyio


# ── W.1 · brain selection: stub default, live = streaming/thinking (Unit 3) ────


def test_default_brain_is_stub_and_no_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default (no env) → the no-spend stub and NO client: a default start constructs nothing that
    could hit the network (the autonomous-spender containment, unchanged)."""
    monkeypatch.delenv("REXHUNTER_BRAIN", raising=False)
    brain_for, client = brain.select_brain_for(stub.build_registry())
    assert brain_for is stub.stub_brain_for
    assert client is None


def test_stub_mode_returns_the_injected_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `default=` seam W.1 adds: in stub mode `select_brain_for` returns the caller's brain
    (the lifespan threads `daemon_config`'s brain through — the injection seam the lifespan gate
    monkeypatches), still with no client."""
    monkeypatch.delenv("REXHUNTER_BRAIN", raising=False)

    def sentinel_brain_for(_territory: str) -> Brain:
        async def _brain(_ctx: Context, _sink: ThinkingSink) -> Decision:
            return HuntComplete()

        return _brain

    brain_for, client = brain.select_brain_for(stub.build_registry(), default=sentinel_brain_for)
    assert brain_for is sentinel_brain_for  # the injected default, not the module stub
    assert client is None


async def test_live_selects_the_streaming_thinking_brain(monkeypatch: pytest.MonkeyPatch) -> None:
    """`REXHUNTER_BRAIN=live` selects the STREAMING/THINKING adapter (Unit 3), not the non-streaming
    one. Proven by spying on `adapter_brain_for`: it must receive `stream=True`, the adaptive
    thinking config, and the raised max_tokens. No live call is made; the client is closed unused.
    """
    monkeypatch.setenv("REXHUNTER_BRAIN", "live")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")

    captured: dict[str, object] = {}

    def spy(**kwargs: object):
        captured.update(kwargs)

        def brain_for(_territory: str) -> Brain:
            async def _brain(_ctx: Context, _sink: ThinkingSink) -> Decision:
                return HuntComplete()

            return _brain

        return brain_for

    monkeypatch.setattr(brain, "adapter_brain_for", spy)
    brain_for, client = brain.select_brain_for(stub.build_registry())
    try:
        assert captured["stream"] is True
        assert captured["thinking"] == brain.HUNT_THINKING
        assert captured["max_tokens"] == brain.HUNT_MAX_TOKENS
        assert brain_for is not stub.stub_brain_for
    finally:
        assert isinstance(client, httpx.AsyncClient)
        await client.aclose()  # closed without ever calling brain() → zero spend


# ── W.1 · the id-scoped daemon spend ceiling (Tiny Arms for money) ─────────────


async def _seed_usage(db_path: Path, *, input_tokens: int, output_tokens: int) -> None:
    """Append one Sonnet-5 UsageEvent (under a throwaway run) so the whole-log fold is non-zero."""
    conn = await db.connect(db_path)
    try:
        run_id = await db.start_run(conn, territory="seed")
        await db.append_event(
            conn,
            run_id,
            UsageEvent(
                model="claude-sonnet-5", input_tokens=input_tokens, output_tokens=output_tokens
            ),
        )
    finally:
        await conn.close()


async def test_daemon_spend_usd_folds_usage_across_all_runs(tmp_path: Path) -> None:
    """The id-scoped analogue of `fold_cost`: total USD folded from EVERY run's UsageEvents along
    the global id cursor (inv 5). 1M input @ $3 (a) + 1M output @ $15 (b) = $18 across two runs."""
    db_path = tmp_path / "rex.db"
    conn = await db.connect(db_path)
    try:
        r1 = await db.start_run(conn, territory="a")
        await db.append_event(
            conn, r1, UsageEvent(model="claude-sonnet-5", input_tokens=1_000_000, output_tokens=0)
        )
        r2 = await db.start_run(conn, territory="b")
        await db.append_event(
            conn, r2, UsageEvent(model="claude-sonnet-5", input_tokens=0, output_tokens=1_000_000)
        )
        spend = await scheduler.daemon_spend_usd(conn)
        assert spend == pytest.approx(18.0)
        # Same primitive, id-scoped: it equals fold_cost over the whole-log usage sub-set.
        assert spend == pytest.approx(cost.fold_cost(await db.read_usage(conn)))
    finally:
        await conn.close()


def _counting_brain_for(flag: dict[str, bool]) -> Callable[[str], Brain]:
    def brain_for(_territory: str) -> Brain:
        async def _brain(_ctx: Context, _sink: ThinkingSink) -> Decision:
            flag["launched"] = True
            return HuntComplete()  # terminal, no tool dispatch → fast, no sniff sleep

        return _brain

    return brain_for


async def _run_one(
    db_path: Path, flag: dict[str, bool], *, daemon_spend_ceiling_usd: float
) -> str | None:
    """One hunt for `mock-gym` through the PUBLIC scheduler seam (`run_hunts` — runs once and
    returns, no `while True`), gated by the id-scoped daemon ceiling. Returns its run_id or None."""
    (run_id,) = await scheduler.run_hunts(
        db_path,
        ["mock-gym"],
        brain_for=_counting_brain_for(flag),
        registry=ToolRegistry(),
        max_concurrent=1,
        tool_timeout_s=5.0,
        retry_budget=0,
        max_iterations=3,
        daemon_spend_ceiling_usd=daemon_spend_ceiling_usd,
    )
    return run_id


async def test_daemon_ceiling_refuses_the_next_hunt(tmp_path: Path) -> None:
    """Replay UsageEvents over a low ceiling → the scheduler refuses the next hunt: the launch gate
    returns None BEFORE running (the brain is never called), and no new run row is created. The
    daemon pauses, never crashes."""
    db_path = tmp_path / "rex.db"
    await _seed_usage(db_path, input_tokens=1_000_000, output_tokens=0)  # $3.00 >> ceiling

    flag = {"launched": False}
    run_id = await _run_one(db_path, flag, daemon_spend_ceiling_usd=0.01)
    assert run_id is None  # refused: over budget, no hunt launched
    assert flag["launched"] is False  # the brain was never called

    conn = await db.connect(db_path)
    try:
        async with conn.execute("SELECT COUNT(*) FROM runs") as cur:
            row = await cur.fetchone()
        assert row is not None and int(row[0]) == 1  # only the seed run; no new run created
    finally:
        await conn.close()


async def test_daemon_launches_under_the_ceiling(tmp_path: Path) -> None:
    """Under the ceiling (fresh DB → fold $0), the daemon launches normally: a run_id comes back and
    the brain ran."""
    db_path = tmp_path / "rex.db"
    flag = {"launched": False}
    run_id = await _run_one(db_path, flag, daemon_spend_ceiling_usd=1.0)
    assert run_id is not None  # a run_id — the hunt launched
    assert flag["launched"] is True


# ── W.2 · the ThinkingSink closure under the REAL lifespan (offline, no spend) ─


async def test_thinking_sink_is_write_ahead_under_the_lifespan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loop-built `ThinkingSink` (closed over the run's own conn/run_id + the shared hub
    publish) exercised under the REAL daemon lifespan — the callable uvicorn invokes on startup.

    An offline delta-emitting brain (no model, no spend) is injected via the `daemon_config` seam;
    it holds on a test `asyncio.Event` so the viewer registers FIRST (deltas fire DURING the brain
    call, before any tool — registering after the hunt starts would race and miss them). Once the
    gate is set, each streamed delta must (a) reach the viewer via the shared `hub.publish`, (b)
    land under THIS run's run_id (the sink threaded the run's own conn/run_id, not a test closure),
    and (c) be write-ahead: a SEPARATE reader connection sees the committed row by the time the
    envelope is on the queue (the `test_hub` / `test_thinking_relay` invariant-1 shape, now under
    the daemon's task lifecycle).
    """
    db_path = tmp_path / "rex.db"
    gate = asyncio.Event()
    deltas = ["Rex is ", "sniffing the ", "territory…"]

    reg = ToolRegistry()  # the brain ends with HuntComplete — no tool dispatch needed

    def brain_for(_territory: str) -> Brain:
        async def _brain(_ctx: Context, sink: ThinkingSink) -> Decision:
            await gate.wait()  # hold until the viewer is registered (avoid the delta race)
            for chunk in deltas:
                await sink(chunk)  # the loop-built closure: append write-ahead, then hub broadcast
            return HuntComplete()

        return _brain

    async def drafter(_conn: aiosqlite.Connection, _prey_id: str) -> str:
        return "stub draft"  # the job worker idles alongside; no FEAST here

    def fake_config() -> tuple[
        dict[str, float], Callable[[str], Brain], ToolRegistry, int, Drafter
    ]:
        # a large interval → exactly one hunt fires within the test window.
        return {"mock-gym": 3600.0}, brain_for, reg, 4, drafter

    monkeypatch.delenv("REXHUNTER_BRAIN", raising=False)  # stub mode → the injected brain is used
    monkeypatch.setattr(server, "DB_PATH", str(db_path))
    monkeypatch.setattr(stub, "daemon_config", fake_config)

    async with server.app.router.lifespan_context(server.app):
        hub = server.app.state.hub
        viewer_id, queue = hub.register()  # register BEFORE releasing the brain
        gate.set()  # now the brain may stream its deltas to the registered viewer

        collected: list[Envelope] = []
        while len(collected) < len(deltas):
            env = await asyncio.wait_for(queue.get(), timeout=5.0)
            if env.id is not None and json.loads(env.data)["type"] == "thinking_delta":
                collected.append(env)

        # (a) each delta reached the viewer via the shared hub.publish — in order.
        assert [json.loads(e.data)["text"] for e in collected] == deltas

        reader = await db.connect(db_path)  # a DIFFERENT connection — sees only commits
        try:
            async with reader.execute("SELECT id FROM runs") as cur:
                run_ids = [str(r[0]) for r in await cur.fetchall()]
            assert len(run_ids) == 1  # the one hunt this lifespan fired
            run_id = run_ids[0]
            for env in collected:
                async with reader.execute(
                    "SELECT run_id, payload FROM trajectory_events "
                    "WHERE id = ? AND type = 'thinking_delta'",
                    (env.id,),
                ) as cur:
                    row = await cur.fetchone()
                assert row is not None  # (c) committed BEFORE the broadcast reached the queue
                assert str(row[0]) == run_id  # (b) the sink threaded THIS run's own run_id
                assert row[1] == env.data  # the envelope carries the exact stored payload
        finally:
            await reader.close()
        hub.deregister(viewer_id)
