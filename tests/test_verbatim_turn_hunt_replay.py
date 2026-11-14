"""W2.3 — offline replay of the thinking-on hunt recorded after the whole-turn echo (DoD #5).

`tests/fixtures/hunt_<run_id>/call_{01,02,03}.json` are the raw SSE STREAMS of the ONE gated live
hunt run on branch w2.3-verbatim-turn (run `c1b41835…`, `claude-sonnet-5`, adaptive thinking on,
streamed, wire `Authorization: Bearer` + `anthropic-version: 2026-10-21`, 3 brain calls, outcome
`aborted` / `max iterations`, printed spend $0.0269 under the $2/$10 table). Re-driving them through
the loop over a replay transport — no network, no spend — reproduces the run.

What the live run settled, and this replay pins:

  - call one is the first STREAMED response under 2026-10-21; it assembles with known event and
    delta types only (`[thinking, tool_use]`, the tool_use with `caller`);
  - call two is the first live request built from a STORED WHOLE TURN (fork E1): call one's verbatim
    signed thinking block, then its tool_use with `caller` and the model's own `input`. Accepted;
  - call three echoed TWO stored turns, each in its own place, earlier messages unchanged
    (append-only). Accepted.

The model sniffed three times and never called hunt_complete, so the run ended on the iteration
breaker: model behaviour, not a failure.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from rexhunter import brain, cost, db, hunt_smoke_thinking
from rexhunter.events import (
    ThinkingDelta,
    ToolCallEvent,
    ToolResultEvent,
    TrajectoryEvent,
    UsageEvent,
)
from rexhunter.loop import ToolCallDecision, run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

FIXTURE_DIR = Path(__file__).parent / "fixtures"
RUN_ID = "c1b41835-9206-4878-896f-fa13601a1f91"  # the W2.3 gated thinking-on hunt
FIXTURES = sorted((FIXTURE_DIR / f"hunt_{RUN_ID}").glob("call_*.json"))
# The recording, pinned: a rewritten fixture fails here before anything else (invariant 6).
SHA256 = {
    "call_01.json": "7ea00d35fd2ff4344fa2432cd8cb302cb91821747c9ec86186856d0de0eafdea",
    "call_02.json": "9cb9d9551c3c9d05de6a5bf11c9a9da5ff04a62bc95d11948952f47d0b01bee9",
    "call_03.json": "9359079e9aaa4b8320d3b37df21f830152a39724e8880e26baf12fb6b450aa8e",
}
# Facts read off the recorded streams (the oracle is the recording, not the assembler).
TOOL_USE_IDS = [
    "toolu_01PA9irUadRNfsrqcMqHKdQV",
    "toolu_01TGvAb3xTnXpvheEdmb6rAw",
    "toolu_01VaHppgC3dd4eXMm1Mgr5yM",
]
PREY = ["mock-gym", "AI engineer", "Machine Learning Engineer"]
USAGE = [(1810, 148), (1974, 462), (2452, 833)]
LIVE_SPEND_USD = 0.0269  # the [hunt] report (4 places), priced at today's table
SPEND_TODAY_USD = 0.026902  # (1810 + 1974 + 2452) x $2/M + (148 + 462 + 833) x $10/M

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]


def _assemble(raw: bytes) -> dict[str, Any]:
    assembler = brain.StreamAssembler()
    for event in brain.iter_sse_events(raw):
        assembler.feed(event)
    return assembler.assembled()


async def sniff(prey: str) -> str:
    """The mock-gym board tool the live hunt ran (stub.sniff without its 5 s beat): echoes the
    scent, so every tool_result replays byte for byte."""
    return f"posting:{prey}"


def _replay_handler(
    requests: list[httpx2.Request],
) -> Callable[[httpx2.Request], httpx2.Response]:
    """Feed the captured streams in order (zero network), keeping each request as the transport got
    it, so a test can check what was folded into calls two and three and which wire it went over."""
    responses = [f.read_bytes() for f in FIXTURES]

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, content=responses[len(requests) - 1])

    return handler


async def _replay(
    db_path: Path, mock_anthropic: MockAnthropic
) -> tuple[list[TrajectoryEvent], tuple[str, str | None], list[httpx2.Request]]:
    """Re-drive run_hunt over the fixtures under the paid entrypoint's own caps."""
    requests: list[httpx2.Request] = []
    reg = ToolRegistry()
    reg.tool(sniff)
    async with mock_anthropic(_replay_handler(requests), api_key="replay") as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            model=brain.SMOKE_MODEL,
            registry=reg,
            max_tokens=hunt_smoke_thinking.HUNT_MAX_TOKENS,
            thinking=hunt_smoke_thinking.THINKING,
            stream=True,
        )
        conn = await db.connect(db_path)
        try:
            run_id = await run_hunt(
                conn,
                territory=hunt_smoke_thinking.TERRITORY,
                brain=brain_for(hunt_smoke_thinking.TERRITORY),
                registry=reg,
                cost_ceiling_usd=hunt_smoke_thinking.COST_CEILING_USD,
                max_iterations=hunt_smoke_thinking.MAX_ITERATIONS,
                tool_timeout_s=5.0,
            )
            events = await db.read_events(conn, run_id)
            async with conn.execute(
                "SELECT outcome, abort_reason FROM runs WHERE id = ?", (run_id,)
            ) as cur:
                row = await cur.fetchone()
            outcome = (str(row[0]), row[1]) if row else ("unknown", None)
        finally:
            await conn.close()
    return events, outcome, requests


def _messages(request: httpx2.Request) -> list[dict[str, Any]]:
    return json.loads(request.content)["messages"]


def _keys(blocks: list[dict[str, Any]]) -> list[list[str]]:
    return [list(block) for block in blocks]


def test_three_streaming_fixtures_were_captured_and_are_unchanged() -> None:
    assert [f.name for f in FIXTURES] == ["call_01.json", "call_02.json", "call_03.json"]
    for f in FIXTURES:
        assert hashlib.sha256(f.read_bytes()).hexdigest() == SHA256[f.name]


def test_the_recorded_streams_carry_the_facts_the_replay_relies_on() -> None:
    # Read off the raw bytes: each stream opens a tool_use with its id and `caller`.
    for raw_path, tool_use_id in zip(FIXTURES, TOOL_USE_IDS, strict=True):
        raw = raw_path.read_bytes()
        assert f'"id":"{tool_use_id}"'.encode() in raw
        assert b'"caller":{"type":"direct"}' in raw
        assert b'"type":"thinking"' in raw and b'"signature_delta"' in raw


def test_each_stream_assembles_to_the_live_decision() -> None:
    for raw_path, tool_use_id, prey in zip(FIXTURES, TOOL_USE_IDS, PREY, strict=True):
        assembled = _assemble(raw_path.read_bytes())
        assert assembled["model"] == "claude-sonnet-5"
        assert assembled["stop_reason"] == "tool_use"
        assert [b["type"] for b in assembled["content"]] == ["thinking", "tool_use"]
        assert len(assembled["content"][0]["signature"]) > 100  # a real opaque signature
        decision = brain.parse_decision(json.dumps(assembled).encode())
        assert isinstance(decision, ToolCallDecision)
        assert (decision.tool, decision.tool_use_id, decision.args) == (
            "sniff",
            tool_use_id,
            {"prey": prey},
        )
        assert json.loads(decision.assistant_content) == assembled["content"]


async def test_replay_reproduces_the_live_hunt_trajectory(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    events, (outcome, reason), requests = await _replay(tmp_path / "rex.db", mock_anthropic)
    assert len(requests) == 3  # three brain calls, as live
    assert (outcome, reason) == ("aborted", "max iterations")  # three sniffs, no hunt_complete

    # Three UsageEvents with the recorded tokens; the fold under today's table is the printed spend.
    usage = [e for e in events if isinstance(e, UsageEvent)]
    assert [(u.input_tokens, u.output_tokens) for u in usage] == USAGE
    assert cost.fold_cost(events) == pytest.approx(LIVE_SPEND_USD, abs=1e-4)
    assert cost.fold_cost(events) == pytest.approx(SPEND_TODAY_USD)

    assert [e for e in events if isinstance(e, ThinkingDelta)]  # the reasoning was relayed live

    # Each call's sniff: one ToolCallEvent storing the WHOLE assembled turn (W1: `thinking` empty),
    # paired by tool_use_id with one ToolResultEvent.
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert [c.tool_use_id for c in calls] == TOOL_USE_IDS
    assert [r.tool_use_id for r in results] == TOOL_USE_IDS
    for call, result, raw_path, prey in zip(calls, results, FIXTURES, PREY, strict=True):
        assert call.tool == result.tool == "sniff"
        assert json.loads(call.raw_request) == {"prey": prey}
        assert result.raw_response == json.dumps(f"posting:{prey}").encode()
        assert json.loads(call.assistant_content) == _assemble(raw_path.read_bytes())["content"]
        assert call.thinking == b""


async def test_calls_two_and_three_echo_the_stored_turns_on_the_wire(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # THE GATE: what the real API accepted on calls two and three, rebuilt offline from the log and
    # checked at the transport (after the SDK serialised the body).
    _events, _outcome, requests = await _replay(tmp_path / "rex.db", mock_anthropic)
    turns = [_assemble(f.read_bytes())["content"] for f in FIXTURES]

    call_two, call_three = _messages(requests[1]), _messages(requests[2])
    assert [m["role"] for m in call_two] == ["user", "assistant", "user"]
    assert [m["role"] for m in call_three] == ["user", "assistant", "user", "assistant", "user"]
    # Each assistant turn is its stored turn exactly: dicts and key order.
    for sent, turn in (
        (call_two[1], turns[0]),
        (call_three[1], turns[0]),
        (call_three[3], turns[1]),
    ):
        assert sent["content"] == turn
        assert _keys(sent["content"]) == _keys(turn)
    # Append-only: call three opens with call two's messages unchanged.
    assert call_three[:3] == call_two

    # Recorded facts on the wire: the tool_use goes back with the model's id, `caller` and input.
    for sent, tool_use_id, prey in (
        (call_two[1], TOOL_USE_IDS[0], PREY[0]),
        (call_three[3], TOOL_USE_IDS[1], PREY[1]),
    ):
        tool_use = sent["content"][1]
        assert tool_use["id"] == tool_use_id
        assert tool_use["caller"] == {"type": "direct"}
        assert tool_use["input"] == {"prey": prey}
    # Each tool_result pairs with the turn before it.
    assert call_two[2]["content"][0]["tool_use_id"] == TOOL_USE_IDS[0]
    assert call_three[4]["content"][0]["tool_use_id"] == TOOL_USE_IDS[1]
    assert call_three[4]["content"][0]["content"] == json.dumps(f"posting:{PREY[1]}")

    # The request shape and the wire the live run used.
    for request in requests:
        body = json.loads(request.content)
        assert body["model"] == "claude-sonnet-5"
        assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert body["tool_choice"]["disable_parallel_tool_use"] is True
        assert body["stream"] is True
        assert request.headers.get_list("authorization") == ["Bearer replay"]
        assert request.headers.get_list("anthropic-version") == ["2026-10-21"]


async def test_replay_is_deterministic_across_two_passes(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    def _norm(events: list[TrajectoryEvent]) -> list[dict[str, Any]]:
        return [json.loads(e.model_dump_json()) for e in events]

    first, _, _ = await _replay(tmp_path / "a.db", mock_anthropic)
    second, _, _ = await _replay(tmp_path / "b.db", mock_anthropic)
    assert any(isinstance(e, ToolCallEvent) for e in first)  # never vacuous
    assert _norm(first) == _norm(second)
