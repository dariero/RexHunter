"""W3.3-W (fragility #1): the brain reads only the events its call uses.

H1 at 070ed9a: before every brain attempt `_call_brain` read the whole run (`db.read_events`),
every row decoded through the boundary, ThinkingDeltas included, though a call uses four types:
`brain.project_messages` turns `tool_call`, `tool_result` and fatal `error` events into the
`messages` array, and the per-run spend fold reads `usage`. A stored row of any other type was
decoded on every attempt, so one that does not validate ended the run at the `<loop>` backstop.
W3.3-W: `db.read_context` reads those four types in SQL (`db.CONTEXT_TYPES`), and the model still
gets the whole conversation, every turn (Darie, 17 Nov 2026). On both recorded Sonnet 5.5 runs the
narrow read projects the same messages as the whole read, at every point of the run.

Every response is a recorded fixture's bytes or built here; nothing under tests/fixtures/ is
written. Zero network: a `MockTransport` answers every call.
"""

from collections.abc import Callable
from pathlib import Path

import anthropic
import httpx2
import pytest

from rexhunter import brain, db, hunt_smoke, hunt_smoke_thinking
from rexhunter.events import (
    ErrorEvent,
    ThinkingDelta,
    ToolCallEvent,
    ToolResultEvent,
    TrajectoryEvent,
    UsageEvent,
    WrapUpNoticeEvent,
)
from rexhunter.loop import Context, Decision, NeedsHelp, ThinkingSink, ToolCallDecision, run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

FIX = Path(__file__).parent / "fixtures"
PLAIN_RUN = FIX / "hunt_5b62bc48-7577-4a52-b003-bdabf592061d"  # Sonnet 5.5, plain, needs_help
STREAM_RUN = FIX / "hunt_54590aec-de50-4727-a0a2-9e533fb02efa"  # Sonnet 5.5, streamed, needs_help
RECORDED_MODEL = "claude-sonnet-5-5"  # both runs were recorded on it (W2.5-R1)
MockAnthropic = Callable[..., anthropic.AsyncAnthropic]


async def sniff(prey: str) -> str:
    """Sniff a territory for postings."""
    return f"posting:{prey}"


def _reg() -> ToolRegistry:
    reg = ToolRegistry()
    reg.tool(sniff)
    return reg


# ── H1: the brain reads the whole run ────────────────────────────────────────────────────


async def test_h1_the_context_holds_only_the_events_the_call_uses(tmp_path: Path) -> None:
    seen: list[set[str]] = []
    decisions = iter([ToolCallDecision(tool="sniff", args={"prey": "x"}), NeedsHelp()])

    async def brain_(context: Context, sink: ThinkingSink) -> Decision:
        seen.append({e.type for e in context})
        await sink("thinking out loud")  # a ThinkingDelta in the run, before the next call
        return next(decisions)

    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await run_hunt(
            conn, territory="t", brain=brain_, registry=_reg(), tool_timeout_s=5.0
        )
        logged = {e.type for e in await db.read_events(conn, run_id)}
    finally:
        await conn.close()
    assert "thinking_delta" in logged  # never vacuous
    assert seen[1] <= {"tool_call", "tool_result", "error", "usage"}


async def test_h1_a_row_of_a_type_the_call_does_not_use_is_never_decoded(tmp_path: Path) -> None:
    conn = await db.connect(tmp_path / "rex.db")
    decisions = iter([ToolCallDecision(tool="sniff", args={"prey": "x"}), NeedsHelp()])

    async def brain_(_context: Context, _sink: ThinkingSink) -> Decision:
        decision = next(decisions)
        if isinstance(decision, ToolCallDecision):
            # a stored row of a type the call never uses, whose payload does not decode
            await conn.execute(
                "INSERT INTO trajectory_events (run_id, seq, type, payload, created_at)"
                " SELECT id, (SELECT COALESCE(MAX(seq) + 1, 0) FROM trajectory_events"
                " WHERE run_id = runs.id), 'thinking_delta', '{\"type\": \"thinking_delta\"}', ''"
                " FROM runs WHERE outcome IS NULL"
            )
            await conn.commit()
        return decision

    try:
        run_id = await run_hunt(
            conn, territory="t", brain=brain_, registry=_reg(), tool_timeout_s=5.0
        )
        async with conn.execute(
            "SELECT outcome, abort_reason FROM runs WHERE id = ?", (run_id,)
        ) as cur:
            row = await cur.fetchone()
    finally:
        await conn.close()
    assert row is not None and (row[0], row[1]) == ("needs_help", None)


# ── W: the context types, and the same conversation ──────────────────────────────────────


def test_the_context_types_are_the_discriminators_of_the_events_a_call_uses() -> None:
    used = (ToolCallEvent, ToolResultEvent, ErrorEvent, UsageEvent, WrapUpNoticeEvent)
    assert db.CONTEXT_TYPES == tuple(cls.model_fields["type"].default for cls in used)


def _recorded_bodies(run_dir: Path) -> list[bytes]:
    return [p.read_bytes() for p in sorted(run_dir.glob("call_*.json"))]


async def _recorded_run(
    tmp_path: Path, mock_anthropic: MockAnthropic, run_dir: Path, *, stream: bool
) -> tuple[list[TrajectoryEvent], list[TrajectoryEvent], str, int]:
    """A recorded run's responses through the real adapter, on the model and settings it was
    recorded with: (the whole run as `read_events` returns it, the narrow read, its territory,
    the number of brain calls)."""
    bodies = _recorded_bodies(run_dir)
    served: list[int] = []

    def handler(_request: httpx2.Request) -> httpx2.Response:
        served.append(1)
        return httpx2.Response(200, content=bodies[len(served) - 1])

    smoke = hunt_smoke_thinking if stream else hunt_smoke
    async with mock_anthropic(handler, api_key="offline") as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            model=RECORDED_MODEL,
            registry=_reg(),
            max_tokens=smoke.HUNT_MAX_TOKENS,
            thinking=hunt_smoke_thinking.THINKING if stream else None,
            stream=stream,
        )
        conn = await db.connect(tmp_path / "rex.db")
        try:
            run_id = await run_hunt(
                conn,
                territory=smoke.TERRITORY,
                brain=brain_for(smoke.TERRITORY),
                registry=_reg(),
                cost_ceiling_usd=smoke.COST_CEILING_USD,
                max_iterations=smoke.MAX_ITERATIONS,
                tool_timeout_s=5.0,
            )
            whole = await db.read_events(conn, run_id)
            narrow = await db.read_context(conn, run_id)
        finally:
            await conn.close()
    return whole, narrow, smoke.TERRITORY, len(served)


@pytest.mark.parametrize(("run_dir", "stream"), [(PLAIN_RUN, False), (STREAM_RUN, True)])
async def test_the_narrow_read_projects_the_same_conversation_as_the_whole_read(
    run_dir: Path, stream: bool, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    whole, narrow, territory, calls = await _recorded_run(
        tmp_path, mock_anthropic, run_dir, stream=stream
    )
    assert calls == 3  # three brain calls, as recorded
    # The narrow read is the whole read's events of the context types, in seq order.
    assert narrow == [e for e in whole if e.type in db.CONTEXT_TYPES]
    if stream:  # the streamed run's whole read holds types the narrow read leaves out
        assert any(isinstance(e, ThinkingDelta) for e in whole)
    # At every point of the run (each brain call saw a prefix of this log), the narrow events
    # project the same messages as the whole run's events.
    for k in range(len(whole) + 1):
        prefix = whole[:k]
        kept = [e for e in prefix if e.type in db.CONTEXT_TYPES]
        assert brain.project_messages(territory, kept) == brain.project_messages(territory, prefix)
