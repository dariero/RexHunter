"""P5 Unit 2c.2 — offline replay of the captured live hunt (DoD #5, stage 3's payoff).

`tests/fixtures/hunt_<run_id>/call_{01,02}.json` are the raw response bytes of the ONE live hunt
(run `1ab26132…`, outcome `needs_help`, 2 brain calls, $0.0149). Re-driving them through the loop
via a replay transport — REXHUNTER_BRAIN not live, zero network, zero spend — reproduces the run.
The gate this certifies is CALL TWO: call 1's `sniff` tool_use is folded back into call 2's messages
(assistant `tool_use` + user `tool_result`, paired by `tool_use_id`), and the loop drives that
reconstructed turn to a typed outcome deterministically. Determinism lives in replaying fixed bytes,
not in the model — the live call is one sample; this test certifies the harness handles it the same
every time. DoD #5: "replaying a recorded run's raw payloads … reproduces identical events."
"""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from rexhunter import brain, cost, db
from rexhunter.events import ToolCallEvent, ToolResultEvent, TrajectoryEvent, UsageEvent
from rexhunter.loop import NeedsHelp, ToolCallDecision, run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

FIXTURE_DIR = Path(__file__).parent / "fixtures"
RUN_ID = "1ab26132-953e-4a14-864e-e00817ebce38"  # THE gated live hunt; fixtures pinned to this run
FIXTURES = sorted((FIXTURE_DIR / f"hunt_{RUN_ID}").glob("call_*.json"))
CALL1_TOOL_USE_ID = "toolu_01LvxWhUt8t37c6vKAtZYWt7"  # call 1's sniff id — the correlation key
LIVE_SPEND_USD = 0.0149  # the [hunt] report; fold_cost over the two UsageEvents must reproduce it


async def sniff(prey: str) -> str:
    """The mock-gym board tool — the deterministic sniff the live hunt ran."""
    return f"posting:{prey}"


def _sniff_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.tool(sniff)
    return reg


def _replay_transport(requests: list[dict[str, Any]]) -> httpx.MockTransport:
    """Feed the captured hunt responses in order for each MESSAGES_URL call (zero network), saving
    each request body so a test can prove what was folded into call two."""
    responses = [f.read_bytes() for f in FIXTURES]
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        idx = state["i"]
        state["i"] += 1
        return httpx.Response(200, content=responses[idx])

    return httpx.MockTransport(handler)


async def _replay(
    db_path: Path,
) -> tuple[list[TrajectoryEvent], tuple[str, str | None], list[dict[str, Any]]]:
    """Re-drive run_hunt over the fixtures; return (events, (outcome, reason), requests)."""
    requests: list[dict[str, Any]] = []
    reg = _sniff_registry()
    async with httpx.AsyncClient(transport=_replay_transport(requests)) as client:
        brain_for = brain.adapter_brain_for(
            client=client, api_key="replay", model=brain.SMOKE_MODEL, registry=reg
        )
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


def test_fixtures_round_trip_through_parse_decision() -> None:
    # Phase 1: the captured bytes parse to the same Decisions the live run produced.
    call1 = brain.parse_decision(FIXTURES[0].read_bytes())
    call2 = brain.parse_decision(FIXTURES[1].read_bytes())
    assert isinstance(call1, ToolCallDecision)
    assert call1.tool == "sniff" and call1.tool_use_id == CALL1_TOOL_USE_ID
    assert isinstance(call2, NeedsHelp)  # call two's terminal decision


async def test_replay_reproduces_the_live_hunt_trajectory(tmp_path: Path) -> None:
    events, (outcome, reason), _requests = await _replay(tmp_path / "rex.db")

    # Two brain calls → two UsageEvents, priced to the live spend.
    usage = [e for e in events if isinstance(e, UsageEvent)]
    assert len(usage) == 2
    assert (usage[0].input_tokens, usage[0].output_tokens) == (1810, 69)
    assert (usage[1].input_tokens, usage[1].output_tokens) == (1877, 190)
    assert cost.fold_cost(events) == pytest.approx(LIVE_SPEND_USD, abs=1e-4)

    # Call 1's sniff → one ToolCallEvent + one ToolResultEvent, paired by tool_use_id.
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(calls) == 1 and len(results) == 1
    assert calls[0].tool == "sniff"
    assert calls[0].tool_use_id == results[0].tool_use_id == CALL1_TOOL_USE_ID

    # The typed terminal outcome the live run reached.
    assert (outcome, reason) == ("needs_help", None)


async def test_call_two_carried_the_reconstructed_turn(tmp_path: Path) -> None:
    # THE GATE: call two's request must fold call one's tool_use back as assistant(tool_use) +
    # user(tool_result), paired by tool_use_id — the reconstructed turn the real API accepted live.
    _events, _outcome, requests = await _replay(tmp_path / "rex.db")
    assert len(requests) == 2  # exactly two brain calls, no more

    messages = requests[1]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assistant_block = messages[1]["content"][0]
    result_block = messages[2]["content"][0]
    assert assistant_block["type"] == "tool_use"
    assert assistant_block["id"] == CALL1_TOOL_USE_ID
    assert assistant_block["name"] == "sniff"
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == CALL1_TOOL_USE_ID  # correlation survives the fold
    assert "posting:" in result_block["content"]  # the sniff output threaded back as observation


async def test_replay_is_deterministic_across_two_passes(tmp_path: Path) -> None:
    # Determinism lives in the fixed bytes: two replays yield identical event payloads. The event
    # models carry no run_id / created_at (those are DB columns), so equality is exact.
    def _norm(events: list[TrajectoryEvent]) -> list[dict[str, Any]]:
        return [json.loads(e.model_dump_json()) for e in events]

    first, _, _ = await _replay(tmp_path / "a.db")
    second, _, _ = await _replay(tmp_path / "b.db")
    assert _norm(first) == _norm(second)
