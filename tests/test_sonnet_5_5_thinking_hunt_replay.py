"""W2.5 — offline replay of the first hunt on Claude Sonnet 5.5, thinking on (DoD #5, no network).

`tests/fixtures/hunt_<run_id>/call_{01,02,03}.json` are the raw SSE STREAMS of the ONE gated live
hunt run on branch w2.5-sonnet-5-5 (run `54590aec…`, `claude-sonnet-5-5`, adaptive thinking with
`display: "summarized"`, streamed, wire `Authorization: Bearer` + `anthropic-version: 2026-10-21`,
3 brain calls, outcome `needs_help`, printed spend $0.0145 under the $2/$10 table). Re-driving them
through the loop over a replay transport reproduces the run.

What the live run settled, and this replay pins:

  - adaptive thinking with `display: "summarized"` is accepted on `messages`, streamed;
  - the streams use the same SSE event and delta types as the Sonnet 5 recordings (`ping` included);
  - adaptive thinking did not think on call one (`[tool_use]` only, 0 thinking tokens); calls two
    and three each carry one summarized, signed thinking block before the tool_use;
  - call three is the first live request that sent back a stored Sonnet 5.5 signed thinking block
    (call two's turn, exactly as returned). Accepted.

The model sniffed twice, saw that the mock board only echoes its query, and called needs_help.
"""

import hashlib
import json
from collections import Counter
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
from rexhunter.loop import NeedsHelp, ToolCallDecision, run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

FIXTURE_DIR = Path(__file__).parent / "fixtures"
RUN_ID = "54590aec-de50-4727-a0a2-9e533fb02efa"  # the W2.5 gated thinking-on hunt
RUN_DIR = FIXTURE_DIR / f"hunt_{RUN_ID}"
FIXTURES = [RUN_DIR / name for name in ("call_01.json", "call_02.json", "call_03.json")]
# The recording, pinned: a rewritten fixture fails here before anything else (invariant 6).
SHA256 = {
    "call_01.json": "032964f88e166abe0d0c5b5aabcc2802037f0467774cf1afec088e11e7a8b9fc",
    "call_02.json": "06662275ab731ceceb93817736e74e95f5351593f332a4596a1eea3997ac1e79",
    "call_03.json": "016fb771d3bd2dc9e6c602b656da473209979af66db6bdf59ff982bc32394099",
}
RECORDED_MODEL = "claude-sonnet-5-5"  # the model the run was recorded on (each stream's "model")
# Facts read off the recorded streams (the oracle is the recording, not the assembler).
TOOL_USES: list[tuple[str, str, dict[str, Any]]] = [
    ("toolu_01LtoRw49PMV8HGW3LnLQ1M6", "sniff", {"prey": "mock-gym AI-engineering job postings"}),
    ("toolu_01MNhH6cKQoBnJDYF7kEV5Aa", "sniff", {"prey": "mock-gym machine learning engineer"}),
    ("toolu_011Sf7WF688cZwD6mZZRA2h5", "needs_help", {}),
]
SNIFF_IDS = [TOOL_USES[0][0], TOOL_USES[1][0]]
# What the live run's sniffs returned (the live DB holds these exact strings, JSON-encoded).
SNIFF_OUTPUTS = [
    "posting:mock-gym AI-engineering job postings",
    "posting:mock-gym machine learning engineer",
]
CONTENT_TYPES = [["tool_use"], ["thinking", "tool_use"], ["thinking", "tool_use"]]
# (thinking text length, signature length) of calls two and three's one thinking block each.
THINKING_SIZES = [(187, 1200), (252, 1288)]
# SSE deltas per (block index, delta type), as recorded.
DELTAS: list[dict[tuple[int, str], int]] = [
    {(0, "input_json_delta"): 8},
    {(0, "thinking_delta"): 18, (0, "signature_delta"): 1, (1, "input_json_delta"): 8},
    {(0, "thinking_delta"): 19, (0, "signature_delta"): 1, (1, "input_json_delta"): 1},
]
SSE_TYPES = {
    "message_start",
    "content_block_start",
    "ping",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
}
USAGE = [(1744, 59, 0), (1829, 155, 97), (2008, 124, 98)]  # (input, output, thinking_tokens)
LIVE_SPEND_USD = 0.0145  # the [hunt] report (4 places), priced at today's table
SPEND_TODAY_USD = 0.014542  # (1744 + 1829 + 2008) x $2/M + (59 + 155 + 124) x $10/M
# The live event sequence: each thinking call's deltas are written during the call, before its
# UsageEvent. Calls two and three relay 18 and 19 deltas, at events[3:21] and events[24:43].
EVENT_TYPES: list[type] = [
    UsageEvent,
    ToolCallEvent,
    ToolResultEvent,
    *[ThinkingDelta] * 18,
    UsageEvent,
    ToolCallEvent,
    ToolResultEvent,
    *[ThinkingDelta] * 19,
    UsageEvent,
]
DELTA_RUNS = [(3, 21), (24, 43)]  # [start, stop) of each call's ThinkingDelta run

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
    it, so a test can check what was sent back on calls two and three and over which wire."""
    responses = [f.read_bytes() for f in FIXTURES]

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, content=responses[len(requests) - 1])

    return handler


async def _replay(
    db_path: Path, mock_anthropic: MockAnthropic
) -> tuple[list[TrajectoryEvent], tuple[str, str | None], list[httpx2.Request]]:
    """Re-drive run_hunt over the streams under the paid entrypoint's own caps, on the recorded
    model with the entrypoint's thinking and streaming, as the live run did
    (hunt_smoke_thinking.py)."""
    requests: list[httpx2.Request] = []
    reg = ToolRegistry()
    reg.tool(sniff)
    async with mock_anthropic(_replay_handler(requests), api_key="replay") as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            model=RECORDED_MODEL,
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


def test_the_run_directory_holds_the_three_recorded_streams_unchanged() -> None:
    assert sorted(p.name for p in RUN_DIR.glob("*")) == list(SHA256)
    for f in FIXTURES:
        assert hashlib.sha256(f.read_bytes()).hexdigest() == SHA256[f.name]


def test_the_streams_use_only_the_known_sse_event_and_delta_types() -> None:
    # The same event and delta types as the Sonnet 5 recordings: nothing the assembler skips.
    assert len(FIXTURES) == len(DELTAS) == 3  # never vacuous: three streams, three tables
    for f, deltas in zip(FIXTURES, DELTAS, strict=True):
        events = list(brain.iter_sse_events(f.read_bytes()))
        assert {event["type"] for event in events} == SSE_TYPES
        seen = Counter(
            (int(event["index"]), str(event["delta"]["type"]))
            for event in events
            if event["type"] == "content_block_delta"
        )
        assert dict(seen) == deltas


def test_each_stream_assembles_to_the_recorded_turn() -> None:
    assert len(FIXTURES) == len(TOOL_USES) == len(CONTENT_TYPES) == len(USAGE) == 3  # never vacuous
    thinking_sizes = iter(THINKING_SIZES)
    for f, (tool_use_id, name, tool_input), types, (n_in, n_out, n_think) in zip(
        FIXTURES, TOOL_USES, CONTENT_TYPES, USAGE, strict=True
    ):
        assembled = _assemble(f.read_bytes())
        assert assembled["model"] == RECORDED_MODEL
        assert assembled["stop_reason"] == "tool_use"
        assert [block["type"] for block in assembled["content"]] == types
        tool_use = assembled["content"][-1]
        assert (tool_use["id"], tool_use["name"], tool_use["input"]) == (
            tool_use_id,
            name,
            tool_input,
        )
        assert tool_use["caller"] == {"type": "direct"}
        if types[0] == "thinking":
            block = assembled["content"][0]
            assert list(block) == ["type", "thinking", "signature"]
            assert (len(block["thinking"]), len(block["signature"])) == next(thinking_sizes)
        usage = assembled["usage"]
        assert (usage["input_tokens"], usage["output_tokens"]) == (n_in, n_out)
        assert usage["output_tokens_details"] == {"thinking_tokens": n_think}
    assert next(thinking_sizes, None) is None  # both thinking blocks were checked


def test_each_stream_parses_to_the_live_decision() -> None:
    decisions = [
        brain.parse_decision(json.dumps(_assemble(f.read_bytes())).encode()) for f in FIXTURES
    ]
    assert len(decisions) == 3
    for decision, f, (tool_use_id, name, tool_input) in zip(
        decisions[:2], FIXTURES[:2], TOOL_USES[:2], strict=True
    ):
        assert isinstance(decision, ToolCallDecision)
        assert (decision.tool, decision.tool_use_id, decision.args) == (
            name,
            tool_use_id,
            tool_input,
        )
        assert json.loads(decision.assistant_content) == _assemble(f.read_bytes())["content"]
    assert isinstance(decisions[2], NeedsHelp)  # call three's terminal decision


async def test_replay_reproduces_the_live_hunt_trajectory(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    events, (outcome, reason), requests = await _replay(tmp_path / "rex.db", mock_anthropic)
    assert len(requests) == 3  # three brain calls, as live
    assert (outcome, reason) == ("needs_help", None)
    # The live run's 44 events in order: no ErrorEvent, no extra event.
    assert [type(e) for e in events] == EVENT_TYPES

    # Each thinking call's relayed deltas sit in one run and join to that call's assembled thinking.
    turns = [_assemble(f.read_bytes())["content"] for f in FIXTURES]
    assert len(DELTA_RUNS) == 2
    for (start, stop), turn in zip(DELTA_RUNS, turns[1:], strict=True):
        run = events[start:stop]
        assert run and all(isinstance(e, ThinkingDelta) for e in run)
        assert "".join(e.text for e in run if isinstance(e, ThinkingDelta)) == turn[0]["thinking"]

    # Three UsageEvents with the recorded tokens; the fold under today's table is the printed spend
    # (4 places) and the exact product of the recorded tokens.
    usage = [e for e in events if isinstance(e, UsageEvent)]
    assert [(u.model, u.input_tokens, u.output_tokens) for u in usage] == [
        (RECORDED_MODEL, n_in, n_out) for n_in, n_out, _n_think in USAGE
    ]
    assert cost.fold_cost(events) == pytest.approx(LIVE_SPEND_USD, abs=1e-4)
    assert cost.fold_cost(events) == pytest.approx(SPEND_TODAY_USD, abs=1e-9)

    # Each sniff: one ToolCallEvent storing the WHOLE assembled turn (W1: `thinking` empty), paired
    # by tool_use_id with one ToolResultEvent carrying the output the live run got.
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert [c.tool_use_id for c in calls] == SNIFF_IDS
    assert [r.tool_use_id for r in results] == SNIFF_IDS
    for call, result, turn, (_id, _name, tool_input), output in zip(
        calls, results, turns[:2], TOOL_USES[:2], SNIFF_OUTPUTS, strict=True
    ):
        assert call.tool == result.tool == "sniff"
        assert json.loads(call.raw_request) == tool_input
        assert json.loads(result.raw_response) == output
        assert json.loads(call.assistant_content) == turn
        assert call.thinking == b""


async def test_calls_two_and_three_send_back_the_stored_turns_exactly(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # THE GATE: what the real API accepted on calls two and three, rebuilt offline from the log and
    # checked at the transport (after the SDK serialised the body).
    _events, _outcome, requests = await _replay(tmp_path / "rex.db", mock_anthropic)
    assert len(requests) == 3
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
    # Call one's turn goes back with no text and no thinking block: the tool_use alone.
    assert [block["type"] for block in call_two[1]["content"]] == ["tool_use"]
    # Call three sends back call two's signed thinking block unchanged, signature included.
    sent_block = call_three[3]["content"][0]
    assert sent_block == turns[1][0]
    assert list(sent_block) == ["type", "thinking", "signature"]
    assert sent_block["signature"] == turns[1][0]["signature"]
    # Append-only: call three opens with call two's messages unchanged.
    assert call_three[:3] == call_two
    # Each tool_result pairs with the turn before it and carries the live sniff output.
    for result_message, tool_use_id, output in (
        (call_two[2], SNIFF_IDS[0], SNIFF_OUTPUTS[0]),
        (call_three[4], SNIFF_IDS[1], SNIFF_OUTPUTS[1]),
    ):
        result_block = result_message["content"][0]
        assert result_block["type"] == "tool_result"
        assert result_block["tool_use_id"] == tool_use_id
        assert result_block["content"] == json.dumps(output)

    # The request shape and the wire the live run used: the recorded model, adaptive thinking with
    # summarized display, auto tool choice with one tool call per turn, streamed.
    for request in requests:
        body = json.loads(request.content)
        assert body["model"] == RECORDED_MODEL
        assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
        assert body["max_tokens"] == 4096
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
    assert any(isinstance(e, ThinkingDelta) for e in first)  # never vacuous
    assert _norm(first) == _norm(second)
