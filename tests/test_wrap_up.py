"""The wrap-up notice (`P5.1`).

With `run_hunt`'s `wrap_up` setting the loop appends one `WrapUpNoticeEvent` before the brain call
that should be the run's last: when the run's spend plus twice its last call's cost reaches its
ceiling, or when the call is its last iteration. The projection sends the notice as a text block
after the tool result it follows, and keeps it there in every later request. The setting is off by
default, recorded on the run's row, and turned on by the live brain's callers. Every brain here is
scripted or answers from a `MockTransport`; nothing leaves the process.
"""

import asyncio
import functools
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
import anthropic
import httpx2
import pytest

from rexhunter import brain, db, eval_wave, scheduler, server, view
from rexhunter.events import (
    PostingRef,
    SniffEvent,
    ToolCallEvent,
    ToolResultEvent,
    TrajectoryEvent,
    UsageEvent,
    WrapUpNoticeEvent,
)
from rexhunter.loop import (
    WRAP_UP_TEXT,
    Brain,
    Context,
    Decision,
    HuntComplete,
    ThinkingSink,
    ToolCallDecision,
    run_hunt,
)
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

BrainFactory = Callable[[Sequence[Decision]], Brain]
MockAnthropic = Callable[..., anthropic.AsyncAnthropic]

TERRITORY = "gym-greenhouse"
# 125 000 Sonnet 5.5 input tokens cost $0.25, exact in binary, so the rule's sums are exact.
QUARTER = 125_000
CEILING_USD = 1.0


def _usage() -> UsageEvent:
    return UsageEvent(model="claude-sonnet-5-5", input_tokens=QUARTER, output_tokens=0)


def _noop_registry() -> ToolRegistry:
    async def noop() -> str:
        """Do nothing."""
        return "ok"

    reg = ToolRegistry()
    reg.tool(noop)
    return reg


def _notices(evs: Sequence[TrajectoryEvent]) -> list[WrapUpNoticeEvent]:
    return [e for e in evs if isinstance(e, WrapUpNoticeEvent)]


def _kinds(evs: Sequence[TrajectoryEvent]) -> list[str]:
    return [e.type for e in evs if e.type in ("tool_call", "tool_result", "wrap_up_notice")]


async def _row(conn: aiosqlite.Connection, run_id: str) -> tuple[Any, ...]:
    rows = list(
        await conn.execute_fetchall(
            "SELECT outcome, abort_reason, wrap_up FROM runs WHERE id = ?", (run_id,)
        )
    )
    return tuple(rows[0])


async def test_the_notice_comes_once_at_the_rules_point(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    # Each call costs $0.25 against $1.00. Before call 3 the run has spent $0.50, and $0.50 plus
    # twice $0.25 reaches the ceiling, so the notice goes in after call 2's result. Call 4 gets no
    # second one, and the ceiling stops the run before call 5.
    calls = [ToolCallDecision(tool="noop", args={}, usage=_usage()) for _ in range(5)]
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await run_hunt(
            conn,
            territory=TERRITORY,
            brain=scripted_brain([*calls, HuntComplete()]),
            registry=_noop_registry(),
            retry_budget=0,
            cost_ceiling_usd=CEILING_USD,
            wrap_up=True,
        )
        evs = await db.read_events(conn, run_id)
        row = await _row(conn, run_id)
    finally:
        await conn.close()
    assert _kinds(evs) == [
        *["tool_call", "tool_result"] * 2,
        "wrap_up_notice",
        *["tool_call", "tool_result"] * 2,
    ]
    assert _notices(evs) == [WrapUpNoticeEvent(text=WRAP_UP_TEXT, reason="cost")]
    assert row == ("aborted", "cost ceiling", 1)


async def test_the_cost_rule_prices_the_last_call(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    # Calls of $0.0625 then $0.375 against $1.00: $0.4375 spent plus twice the last call reaches
    # the ceiling, so call 3 gets the notice, where twice the first call would leave it at $0.5625.
    def call(input_tokens: int) -> ToolCallDecision:
        usage = UsageEvent(model="claude-sonnet-5-5", input_tokens=input_tokens, output_tokens=0)
        return ToolCallDecision(tool="noop", args={}, usage=usage)

    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await run_hunt(
            conn,
            territory=TERRITORY,
            brain=scripted_brain([call(31_250), call(187_500), HuntComplete()]),
            registry=_noop_registry(),
            retry_budget=0,
            cost_ceiling_usd=CEILING_USD,
            wrap_up=True,
        )
        evs = await db.read_events(conn, run_id)
    finally:
        await conn.close()
    assert _kinds(evs) == [*["tool_call", "tool_result"] * 2, "wrap_up_notice"]
    assert _notices(evs) == [WrapUpNoticeEvent(text=WRAP_UP_TEXT, reason="cost")]


def _tool_use(call: int) -> bytes:
    return json.dumps(
        {
            "id": f"msg_{call}",
            "type": "message",
            "role": "assistant",
            "model": brain.SMOKE_MODEL,
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": f"toolu_{call}", "name": "noop", "input": {}}],
            "usage": {"input_tokens": QUARTER, "output_tokens": 0},
        }
    ).encode()


async def test_the_notice_stays_where_it_was_placed(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # Through the real adapter: the notice joins call 3's last user turn after its tool_result,
    # and call 4's request carries call 3's bytes unchanged up to the end of the notice block.
    sent: list[bytes] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.content)
        return httpx2.Response(200, content=_tool_use(len(sent)))

    registry = _noop_registry()
    conn = await db.connect(tmp_path / "rex.db")
    try:
        async with mock_anthropic(handler) as client:
            brain_for = brain.adapter_brain_for(
                client=client, model=brain.SMOKE_MODEL, registry=registry
            )
            await run_hunt(
                conn,
                territory=TERRITORY,
                brain=brain_for(TERRITORY),
                registry=registry,
                retry_budget=0,
                cost_ceiling_usd=CEILING_USD,
                wrap_up=True,
            )
    finally:
        await conn.close()
    assert len(sent) == 4
    third, fourth = (json.loads(body)["messages"] for body in sent[2:])
    block = {"type": "text", "text": WRAP_UP_TEXT}
    assert [m["role"] for m in third] == ["user", "assistant", "user", "assistant", "user"]
    assert [b["type"] for b in third[-1]["content"]] == ["tool_result", "text"]
    assert third[-1]["content"][-1] == block
    assert fourth[:5] == third
    needle = json.dumps(block, separators=(",", ":")).encode()
    assert (sent[2].count(needle), sent[3].count(needle)) == (1, 1)
    end = sent[2].index(needle) + len(needle)
    assert sent[3][:end] == sent[2][:end]
    assert all(needle not in body for body in sent[:2])


def _posting_registry(bodies: dict[str, bytes]) -> ToolRegistry:
    async def read_posting(board: str, posting_id: str) -> bytes:
        """Read one posting."""
        assert board == TERRITORY
        return bodies[posting_id]

    reg = ToolRegistry()
    reg.tool(read_posting, posting_source=True)
    return reg


def _heeding_brain() -> Brain:
    # Reads the postings one by one at $0.25 a call, and on the notice catches what it has read.
    read: list[PostingRef] = []

    async def heed(context: Context, _sink: ThinkingSink) -> Decision:
        if any(isinstance(e, WrapUpNoticeEvent) for e in context):
            return HuntComplete(catch=list(read), usage=_usage())
        ref = PostingRef(board=TERRITORY, posting_id=str(len(read) + 1))
        read.append(ref)
        return ToolCallDecision(tool="read_posting", args=ref.model_dump(), usage=_usage())

    return heed


@pytest.mark.parametrize(
    ("wrap_up", "ending", "penned"),
    [(True, ("completed", None), ["1", "2"]), (False, ("aborted", "cost ceiling"), [])],
    ids=["with-the-notice", "without-it"],
)
async def test_a_run_that_heeds_the_notice_catches_what_it_read(
    wrap_up: bool, ending: tuple[str, str | None], penned: list[str], tmp_path: Path
) -> None:
    # The same brain, $1.00 a run: with the notice it ends at call 3 with the two postings it read;
    # without it, it reads on until the ceiling stops the run and pens nothing.
    bodies = {str(n): f"posting {n}".encode() for n in range(1, 6)}
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await run_hunt(
            conn,
            territory=TERRITORY,
            brain=_heeding_brain(),
            registry=_posting_registry(bodies),
            retry_budget=0,
            cost_ceiling_usd=CEILING_USD,
            wrap_up=wrap_up,
        )
        row = await _row(conn, run_id)
        pen = [str(r[0]) for r in await conn.execute_fetchall("SELECT posting_id FROM prey")]
    finally:
        await conn.close()
    assert (row[:2], sorted(pen)) == (ending, penned)


def _recording_brain(seen: list[Context]) -> Brain:
    async def record(context: Context, _sink: ThinkingSink) -> Decision:
        seen.append(list(context))
        return ToolCallDecision(tool="noop", args={})

    return record


@pytest.mark.parametrize("max_iterations", [3, 1])
async def test_the_notice_comes_before_the_last_iteration(
    max_iterations: int, tmp_path: Path
) -> None:
    # No spend at all, so only the iteration rule can fire: the notice comes before the last call,
    # after the last tool result, or after the seed when the first call is the last.
    seen: list[Context] = []
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await run_hunt(
            conn,
            territory=TERRITORY,
            brain=_recording_brain(seen),
            registry=_noop_registry(),
            retry_budget=0,
            max_iterations=max_iterations,
            wrap_up=True,
        )
        evs = await db.read_events(conn, run_id)
        row = await _row(conn, run_id)
    finally:
        await conn.close()
    assert _kinds(evs) == [
        *["tool_call", "tool_result"] * (max_iterations - 1),
        "wrap_up_notice",
        "tool_call",
        "tool_result",
    ]
    assert _notices(evs) == [WrapUpNoticeEvent(text=WRAP_UP_TEXT, reason="iterations")]
    assert row == ("aborted", "max iterations", 1)
    assert [len(_notices(context)) for context in seen] == [0] * (max_iterations - 1) + [1]
    last_turn = brain.project_messages(TERRITORY, seen[-1], _noop_registry())[-1]
    block = {"type": "text", "text": WRAP_UP_TEXT}
    if max_iterations == 1:
        seed = {"type": "text", "text": brain.project_messages(TERRITORY, [])[0]["content"]}
        assert last_turn == {"role": "user", "content": [seed, block]}
    else:
        assert [b["type"] for b in last_turn["content"]] == ["tool_result", "text"]
        assert last_turn["content"][-1] == block


async def _spending_hunt(conn: aiosqlite.Connection, *, wrap_up: bool | None = None) -> str:
    # Two calls at $0.25 against $0.50: the second is the last iteration and over the spend rule.
    # With `wrap_up` None, `run_hunt` is called without the setting, so its default is what runs.
    async def spend(_context: Context, _sink: ThinkingSink) -> Decision:
        return ToolCallDecision(tool="noop", args={}, usage=_usage())

    hunt = functools.partial(
        run_hunt,
        conn,
        territory=TERRITORY,
        brain=spend,
        registry=_noop_registry(),
        retry_budget=0,
        max_iterations=2,
        cost_ceiling_usd=0.5,
    )
    return await (hunt() if wrap_up is None else hunt(wrap_up=wrap_up))


async def test_the_notice_is_off_by_default_and_recorded(tmp_path: Path) -> None:
    # Off by default: such a run gets no notice and its row records 0; on, the row records 1; a
    # run started outside `run_hunt` records NULL.
    conn = await db.connect(tmp_path / "rex.db")
    try:
        off = await _spending_hunt(conn)
        on = await _spending_hunt(conn, wrap_up=True)
        bare = await db.start_run(conn, territory=TERRITORY)
        notices = [len(_notices(await db.read_events(conn, r))) for r in (off, on)]
        recorded = [(await _row(conn, r))[2] for r in (off, on, bare)]
    finally:
        await conn.close()
    assert (notices, recorded) == ([0, 1], [0, 1, None])


async def test_an_old_runs_table_gains_the_column(tmp_path: Path) -> None:
    # A log from before `P5.1` keeps its rows; the column-ensure adds `wrap_up`, NULL on them.
    path = tmp_path / "old.db"
    async with aiosqlite.connect(path) as old:
        await old.execute(
            "CREATE TABLE runs (id TEXT PRIMARY KEY, territory TEXT NOT NULL, started_at TEXT NOT"
            " NULL, ended_at TEXT, outcome TEXT, abort_reason TEXT, cost_ceiling_usd REAL,"
            " max_iterations INTEGER)"
        )
        await old.execute("INSERT INTO runs (id, territory, started_at) VALUES ('r0', 't', 'x')")
        await old.commit()
    conn = await db.connect(path)
    try:
        rows = list(await conn.execute_fetchall("SELECT id, wrap_up FROM runs"))
    finally:
        await conn.close()
    assert [tuple(r) for r in rows] == [("r0", None)]


def test_the_view_folds_a_notice_to_nothing() -> None:
    # The notice is the brain's cue; the board draws nothing for it, as for a SniffEvent.
    clock = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)

    def row(event: TrajectoryEvent) -> view.LogRow:
        return view.LogRow(id=1, seq=0, run_id="r", created_at=clock.isoformat(), event=event)

    notice = WrapUpNoticeEvent(text=WRAP_UP_TEXT, reason="cost")
    assert view.project([row(notice)], clock) == view.project([row(SniffEvent(prey="x"))], clock)


def test_the_context_read_carries_the_notice() -> None:
    # The brain reads the notice with the conversation (`db.read_context`), and only the turn types
    # and the notice are projected: a tool call, its result and the notice make two turns. The
    # block carries the stored text, not today's `WRAP_UP_TEXT`, so a later rewording cannot
    # change a request that already holds a notice.
    evs: list[TrajectoryEvent] = [
        ToolCallEvent(tool="noop", raw_request=b"{}", tool_use_id="toolu_1"),
        ToolResultEvent(tool="noop", raw_request=b"{}", raw_response=b"ok", tool_use_id="toolu_1"),
        WrapUpNoticeEvent(text="An earlier wording.", reason="cost"),
    ]
    assert "wrap_up_notice" in db.CONTEXT_TYPES
    messages = brain.project_messages(TERRITORY, evs, _noop_registry())
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[-1]["content"][-1] == {"type": "text", "text": "An earlier wording."}


@pytest.mark.parametrize("live", [True, False], ids=["live", "stub"])
async def test_the_live_callers_turn_the_notice_on(
    live: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The daemon (the lifespan's `run_scheduler`) and the wave (`run_hunts`) get `wrap_up` exactly
    # when `select_brain_for` gave them the live brain. Neither spy calls a brain, so no request
    # is made.
    if live:
        monkeypatch.setenv("REXHUNTER_BRAIN", "live")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    else:
        monkeypatch.delenv("REXHUNTER_BRAIN", raising=False)
    monkeypatch.setattr(server, "DB_PATH", str(tmp_path / "rex.db"))
    daemon: list[dict[str, Any]] = []
    started = asyncio.Event()

    async def scheduler_spy(*_args: Any, **kwargs: Any) -> None:
        daemon.append(kwargs)
        started.set()
        await asyncio.Event().wait()

    wave: list[dict[str, Any]] = []

    async def hunts_spy(path: Path, territories: Sequence[str], **kwargs: Any) -> list[None]:
        wave.append(kwargs)
        boot = await db.connect(path)  # an empty log for the score that follows
        await boot.close()
        return [None] * len(territories)

    monkeypatch.setattr(server, "run_scheduler", scheduler_spy)
    monkeypatch.setattr(scheduler, "run_hunts", hunts_spy)
    async with server.app.router.lifespan_context(server.app):
        await asyncio.wait_for(started.wait(), timeout=5)
    await eval_wave.run_live_wave(tmp_path / "wave.db")
    await eval_wave.run_baseline_wave(tmp_path / "baseline.db")
    assert ([d["wrap_up"] for d in daemon], [w["wrap_up"] for w in wave]) == (
        [live],
        [live, False],
    )
