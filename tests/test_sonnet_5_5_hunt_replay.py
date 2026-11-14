"""W2.5 — offline replay of the first hunt on Claude Sonnet 5.5, thinking off (DoD #5, no network).

`tests/fixtures/hunt_<run_id>/call_{01,02,03}.json` are the raw response bytes of the ONE gated live
hunt run on branch w2.5-sonnet-5-5 (run `5b62bc48…`, `claude-sonnet-5-5`, plain, not streamed,
`thinking: {type: "between_tools"}` (the model's lowest setting, W2.2), wire `Authorization:
Bearer` + `anthropic-version: 2026-10-21`, 3 brain calls, outcome `needs_help`, printed spend
$0.0135 under the $2/$10 table). Re-driving them through the loop over a replay transport
reproduces the run.

What the live run settled, and this replay pins:

  - `between_tools` is accepted on `messages`, not only by `count_tokens`;
  - a thinking-off turn on Sonnet 5.5 came back as `[text, tool_use]` (the tool_use with `caller`),
    with no thinking block and 0 thinking tokens, in all three calls;
  - call two is the first live `messages` request that sent back a stored thinking-off turn, and
    call three sent back two; both were accepted, so the prefix check on a model with preserved
    thinking passed for the exact echo.

The model sniffed twice, saw that the mock board only echoes its query, and called needs_help.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from rexhunter import brain, cost, db, hunt_smoke
from rexhunter.events import ToolCallEvent, ToolResultEvent, TrajectoryEvent, UsageEvent
from rexhunter.loop import NeedsHelp, ToolCallDecision, run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

FIXTURE_DIR = Path(__file__).parent / "fixtures"
RUN_ID = "5b62bc48-7577-4a52-b003-bdabf592061d"  # the W2.5 gated thinking-off hunt
RUN_DIR = FIXTURE_DIR / f"hunt_{RUN_ID}"
FIXTURES = [RUN_DIR / name for name in ("call_01.json", "call_02.json", "call_03.json")]
# The recording, pinned: a rewritten fixture fails here before anything else (invariant 6).
SHA256 = {
    "call_01.json": "6f09106f4c09ce2cbc23aff8ad0bcc2a9b811a3ee6a3805b0a0e6d6a5940754c",
    "call_02.json": "8129fb8bcb7aa0c43bae0c167c00374b1615b7a5c7f29a2afcdf5811ab3eb4a4",
    "call_03.json": "7e63afc5eeed2fc2ed541e3f4a48042e1338f48a5de9053ff25baf8820999791",
}
RECORDED_MODEL = "claude-sonnet-5-5"  # the model the run was recorded on (each fixture's "model")
# Facts read off the recorded responses (the oracle is the recording, not the parser).
TOOL_USES: list[tuple[str, str, dict[str, Any]]] = [
    ("toolu_01K6B2iWWRy8DdUsMv88x5Vk", "sniff", {"prey": "mock-gym AI-engineering job postings"}),
    ("toolu_01Hi1VX2vY6Gf5MtUCN9p58K", "sniff", {"prey": "mock-gym"}),
    ("toolu_01UEKeyEA2YTXLMH633CzmbF", "needs_help", {}),
]
SNIFF_IDS = [TOOL_USES[0][0], TOOL_USES[1][0]]
# What the live run's sniffs returned (the live DB holds these exact strings, JSON-encoded).
SNIFF_OUTPUTS = ["posting:mock-gym AI-engineering job postings", "posting:mock-gym"]
USAGE = [(1743, 73), (1842, 89), (1948, 79)]
LIVE_SPEND_USD = 0.0135  # the [hunt] report (4 places), priced at today's table
SPEND_TODAY_USD = 0.013476  # (1743 + 1842 + 1948) x $2/M + (73 + 89 + 79) x $10/M

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]


async def sniff(prey: str) -> str:
    """The mock-gym board tool the live hunt ran (stub.sniff without its 5 s beat): echoes the
    scent, so every tool_result replays byte for byte."""
    return f"posting:{prey}"


def _payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def _replay_handler(
    requests: list[httpx2.Request],
) -> Callable[[httpx2.Request], httpx2.Response]:
    """Feed the captured responses in order (zero network), keeping each request as the transport
    got it, so a test can check what was sent back on calls two and three and over which wire."""
    responses = [f.read_bytes() for f in FIXTURES]

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, content=responses[len(requests) - 1])

    return handler


async def _replay(
    db_path: Path, mock_anthropic: MockAnthropic
) -> tuple[list[TrajectoryEvent], tuple[str, str | None], list[httpx2.Request]]:
    """Re-drive run_hunt over the fixtures under the paid entrypoint's own caps. The adapter runs
    on the recorded model with no thinking override, as the live run did (hunt_smoke.py), so the
    thinking value on the wire is whatever the adapter picks for that model."""
    requests: list[httpx2.Request] = []
    reg = ToolRegistry()
    reg.tool(sniff)
    async with mock_anthropic(_replay_handler(requests), api_key="replay") as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            model=RECORDED_MODEL,
            registry=reg,
            max_tokens=hunt_smoke.HUNT_MAX_TOKENS,
        )
        conn = await db.connect(db_path)
        try:
            run_id = await run_hunt(
                conn,
                territory=hunt_smoke.TERRITORY,
                brain=brain_for(hunt_smoke.TERRITORY),
                registry=reg,
                cost_ceiling_usd=hunt_smoke.COST_CEILING_USD,
                max_iterations=hunt_smoke.MAX_ITERATIONS,
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


def test_the_run_directory_holds_the_three_recorded_calls_unchanged() -> None:
    assert sorted(p.name for p in RUN_DIR.glob("*")) == list(SHA256)
    for f in FIXTURES:
        assert hashlib.sha256(f.read_bytes()).hexdigest() == SHA256[f.name]


def test_each_fixture_names_the_recorded_model() -> None:
    for f in FIXTURES:
        assert _payload(f)["model"] == RECORDED_MODEL


def test_the_recorded_responses_carry_the_facts_the_replay_relies_on() -> None:
    # A thinking-off turn on Sonnet 5.5: a text block, then the one tool_use with `caller`; no
    # thinking block, no thinking tokens, a tool_use stop in every call.
    for f, (tool_use_id, name, tool_input), (n_in, n_out) in zip(
        FIXTURES, TOOL_USES, USAGE, strict=True
    ):
        payload = _payload(f)
        assert payload["stop_reason"] == "tool_use"
        assert [block["type"] for block in payload["content"]] == ["text", "tool_use"]
        tool_use = payload["content"][1]
        assert (tool_use["id"], tool_use["name"], tool_use["input"]) == (
            tool_use_id,
            name,
            tool_input,
        )
        assert tool_use["caller"] == {"type": "direct"}
        usage = payload["usage"]
        assert (usage["input_tokens"], usage["output_tokens"]) == (n_in, n_out)
        assert usage["output_tokens_details"] == {"thinking_tokens": 0}


def test_fixtures_round_trip_through_parse_decision() -> None:
    decisions = [brain.parse_decision(f.read_bytes()) for f in FIXTURES]
    for decision, f, (tool_use_id, name, tool_input) in zip(
        decisions[:2], FIXTURES[:2], TOOL_USES[:2], strict=True
    ):
        assert isinstance(decision, ToolCallDecision)
        assert (decision.tool, decision.tool_use_id, decision.args) == (
            name,
            tool_use_id,
            tool_input,
        )
        assert json.loads(decision.assistant_content) == _payload(f)["content"]
    assert isinstance(decisions[2], NeedsHelp)  # call three's terminal decision


async def test_replay_reproduces_the_live_hunt_trajectory(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    events, (outcome, reason), requests = await _replay(tmp_path / "rex.db", mock_anthropic)
    assert len(requests) == 3  # three brain calls, as live
    assert (outcome, reason) == ("needs_help", None)
    # The live run's event sequence: no ErrorEvent, no extra event.
    assert [type(e) for e in events] == [
        UsageEvent,
        ToolCallEvent,
        ToolResultEvent,
        UsageEvent,
        ToolCallEvent,
        ToolResultEvent,
        UsageEvent,
    ]

    # Three UsageEvents with the recorded tokens; the fold under today's table is the printed spend
    # (4 places) and the exact product of the recorded tokens.
    usage = [e for e in events if isinstance(e, UsageEvent)]
    assert [(u.model, u.input_tokens, u.output_tokens) for u in usage] == [
        (RECORDED_MODEL, n_in, n_out) for n_in, n_out in USAGE
    ]
    assert cost.fold_cost(events) == pytest.approx(LIVE_SPEND_USD, abs=1e-4)
    assert cost.fold_cost(events) == pytest.approx(SPEND_TODAY_USD, abs=1e-9)

    # Each sniff: one ToolCallEvent storing the WHOLE turn as recorded, paired by tool_use_id with
    # one ToolResultEvent carrying the output the live run got.
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert [c.tool_use_id for c in calls] == SNIFF_IDS
    assert [r.tool_use_id for r in results] == SNIFF_IDS
    for call, result, f, (_id, _name, tool_input), output in zip(
        calls, results, FIXTURES[:2], TOOL_USES[:2], SNIFF_OUTPUTS, strict=True
    ):
        assert call.tool == result.tool == "sniff"
        assert json.loads(call.raw_request) == tool_input
        assert json.loads(result.raw_response) == output
        assert json.loads(call.assistant_content) == _payload(f)["content"]
        assert call.thinking == b""


async def test_calls_two_and_three_send_back_the_stored_turns_exactly(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # THE GATE: what the real API accepted on calls two and three, rebuilt offline from the log and
    # checked at the transport (after the SDK serialised the body).
    _events, _outcome, requests = await _replay(tmp_path / "rex.db", mock_anthropic)
    assert len(requests) == 3
    turns = [_payload(f)["content"] for f in FIXTURES]

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
    # Each tool_result pairs with the turn before it and carries the live sniff output.
    for result_message, tool_use_id, output in (
        (call_two[2], SNIFF_IDS[0], SNIFF_OUTPUTS[0]),
        (call_three[4], SNIFF_IDS[1], SNIFF_OUTPUTS[1]),
    ):
        result_block = result_message["content"][0]
        assert result_block["type"] == "tool_result"
        assert result_block["tool_use_id"] == tool_use_id
        assert result_block["content"] == json.dumps(output)

    # The request shape and the wire the live run used: the recorded model, its lowest thinking
    # setting, auto tool choice with one tool call per turn, plain (not streamed).
    for request in requests:
        body = json.loads(request.content)
        assert body["model"] == RECORDED_MODEL
        assert body["thinking"] == {"type": "between_tools"}
        assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
        assert body["max_tokens"] == 1024
        assert "stream" not in body
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
