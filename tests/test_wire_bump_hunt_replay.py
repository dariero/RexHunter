"""W2.1b — offline replay of the hunt recorded under the bumped wire (DoD #5, zero network).

`tests/fixtures/hunt_<run_id>/call_{01,02}.json` are the raw response bytes of the ONE live hunt
run after the wire moved to `Authorization: Bearer` + `anthropic-version: 2026-10-21` (run
`4c47b41b…`, `claude-sonnet-5`, thinking off, outcome `needs_help`, 2 brain calls, printed spend
$0.0096 under the $2/$10 table). Like the 2c.2 hunt (tests/test_hunt_replay.py), call two is the
gate: call one's `sniff` tool_use is folded back as assistant(tool_use) + user(tool_result), paired
by `tool_use_id`, and the real API accepted that turn over the new wire. Re-driving the fixed bytes
through the loop reproduces the run's events every time.

The responses also record what 2026-10-21 sends back: extra top-level `container` and
`diagnostics`, `usage.speed`, and a `caller` on each tool_use block. The lenient envelope ignores
them, so they parse to the same decisions as the 2023-06-01 recordings.

Since W2.3 the replayed call two sends call one's WHOLE turn (its text block, then the tool_use with
`caller`). The live run sent the pre-W2.3 rebuild (the tool_use alone), so the W2.3 shape is pinned
here offline only.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from rexhunter import brain, cost, db
from rexhunter.events import ToolCallEvent, ToolResultEvent, TrajectoryEvent, UsageEvent
from rexhunter.loop import NeedsHelp, ToolCallDecision, run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

FIXTURE_DIR = Path(__file__).parent / "fixtures"
RUN_ID = "4c47b41b-5268-4da3-b787-9483b21d2c3f"  # the W2.1b live hunt; fixtures pinned to this run
FIXTURES = sorted((FIXTURE_DIR / f"hunt_{RUN_ID}").glob("call_*.json"))
CALL1_TOOL_USE_ID = "toolu_01PzYiMeMjrpwgia2hmXY6mv"  # call 1's sniff id, the correlation key
LIVE_SPEND_USD = 0.0096  # the [hunt] report (printed to 4 places), priced at today's table
SPEND_TODAY_USD = 0.009624  # (1810 + 1877) x $2/M + (67 + 158) x $10/M, an exact product

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]


async def sniff(prey: str) -> str:
    """The mock-gym board tool: the deterministic sniff the live hunt ran."""
    return f"posting:{prey}"


def _sniff_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.tool(sniff)
    return reg


def _replay_handler(
    requests: list[httpx2.Request],
) -> Callable[[httpx2.Request], httpx2.Response]:
    """Feed the captured responses in order for each MESSAGES_URL call (zero network), keeping each
    request so a test can check what was folded into call two and which wire it went over."""
    responses = [f.read_bytes() for f in FIXTURES]

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, content=responses[len(requests) - 1])

    return handler


async def _replay(
    db_path: Path, mock_anthropic: MockAnthropic
) -> tuple[list[TrajectoryEvent], tuple[str, str | None], list[httpx2.Request]]:
    """Re-drive run_hunt over the fixtures; return (events, (outcome, reason), requests)."""
    requests: list[httpx2.Request] = []
    reg = _sniff_registry()
    async with mock_anthropic(_replay_handler(requests), api_key="replay") as client:
        brain_for = brain.adapter_brain_for(client=client, model=brain.SMOKE_MODEL, registry=reg)
        conn = await db.connect(db_path)
        try:
            run_id = await run_hunt(
                conn,
                territory="mock-gym",
                brain=brain_for("mock-gym"),
                registry=reg,
                cost_ceiling_usd=0.05,
                max_iterations=3,
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


def test_two_fixtures_were_captured() -> None:
    assert len(FIXTURES) == 2  # the live hunt made two brain calls


def test_the_fixtures_carry_the_2026_10_21_response_shape() -> None:
    # What the bumped API version sends back, recorded: fields the 2023-06-01 hunt did not have,
    # in the same block order (text, then the one tool_use).
    assert len(FIXTURES) == 2  # never vacuous: no fixture, no pass
    for path in FIXTURES:
        payload = json.loads(path.read_bytes())
        assert payload["model"] == "claude-sonnet-5"
        assert {"container", "diagnostics"} <= payload.keys()
        assert [block["type"] for block in payload["content"]] == ["text", "tool_use"]
        assert payload["content"][1]["caller"] == {"type": "direct"}


def test_fixtures_round_trip_through_parse_decision() -> None:
    # The extra fields are ignored at the boundary: the same Decisions the live run produced.
    call1 = brain.parse_decision(FIXTURES[0].read_bytes())
    call2 = brain.parse_decision(FIXTURES[1].read_bytes())
    assert isinstance(call1, ToolCallDecision)
    assert call1.tool == "sniff" and call1.tool_use_id == CALL1_TOOL_USE_ID
    assert isinstance(call2, NeedsHelp)  # call two's terminal decision


async def test_replay_reproduces_the_live_hunt_trajectory(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    events, (outcome, reason), _requests = await _replay(tmp_path / "rex.db", mock_anthropic)

    # Two brain calls, two UsageEvents; the fold under today's table reproduces the printed spend
    # (4 places) and the exact product of the recorded tokens.
    usage = [e for e in events if isinstance(e, UsageEvent)]
    assert len(usage) == 2
    assert (usage[0].input_tokens, usage[0].output_tokens) == (1810, 67)
    assert (usage[1].input_tokens, usage[1].output_tokens) == (1877, 158)
    assert cost.fold_cost(events) == pytest.approx(LIVE_SPEND_USD, abs=1e-4)
    assert cost.fold_cost(events) == pytest.approx(SPEND_TODAY_USD)

    # Call 1's sniff → one ToolCallEvent + one ToolResultEvent, paired by tool_use_id.
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(calls) == 1 and len(results) == 1
    assert calls[0].tool == "sniff"
    assert calls[0].tool_use_id == results[0].tool_use_id == CALL1_TOOL_USE_ID

    # The typed terminal outcome the live run reached.
    assert (outcome, reason) == ("needs_help", None)


async def test_call_two_carried_the_reconstructed_turn_over_the_new_wire(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # THE GATE: call two folds call one's turn back as assistant + user(tool_result), paired by
    # tool_use_id. The real API accepted the pre-W2.3 turn (the tool_use alone) over Bearer +
    # 2026-10-21; since W2.3 the whole turn goes back (text, then the tool_use), checked here
    # offline. The replay goes through the same factory, so both requests carry that wire too.
    _events, _outcome, requests = await _replay(tmp_path / "rex.db", mock_anthropic)
    assert len(requests) == 2  # exactly two brain calls, no more
    for request in requests:
        assert request.headers.get_list("authorization") == ["Bearer replay"]
        assert request.headers.get_list("anthropic-version") == ["2026-10-21"]

    messages: list[dict[str, Any]] = json.loads(requests[1].content)["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assistant_block = messages[1]["content"][1]
    result_block = messages[2]["content"][0]
    assert assistant_block["type"] == "tool_use"
    assert assistant_block["id"] == CALL1_TOOL_USE_ID
    assert assistant_block["name"] == "sniff"
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == CALL1_TOOL_USE_ID  # correlation survives the fold
    assert "posting:" in result_block["content"]  # the sniff output threaded back as observation
    # W2.3: the whole turn goes back, so call one's text block leads it, as the API returned it.
    assert messages[1]["content"][0] == json.loads(FIXTURES[0].read_bytes())["content"][0]


async def test_replay_is_deterministic_across_two_passes(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # Determinism lives in the fixed bytes: two replays yield identical event payloads.
    def _norm(events: list[TrajectoryEvent]) -> list[dict[str, Any]]:
        return [json.loads(e.model_dump_json()) for e in events]

    first, _, _ = await _replay(tmp_path / "a.db", mock_anthropic)
    second, _, _ = await _replay(tmp_path / "b.db", mock_anthropic)
    assert any(isinstance(e, ToolCallEvent) for e in first)  # never vacuous: a real trajectory
    assert _norm(first) == _norm(second)
