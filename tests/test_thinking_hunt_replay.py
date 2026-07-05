"""P5 Unit 3c — offline replay of the captured THINKING-ON live hunt (DoD #5, the replay gate).

`tests/fixtures/hunt_<run_id>/call_{01,02}.json` are the raw SSE STREAMS of the ONE gated live hunt
(run `32975e98…`, `claude-sonnet-5`, adaptive thinking on, 2 brain calls, outcome `completed`,
$0.0187). Re-driving them through the loop over a replay transport — no network, no spend —
reproduces the run and certifies THE GATE:

    call two was ACCEPTED by the real API, so the reconstructed assistant turn must — and does —
    lead with call one's VERBATIM signed thinking block (invariant 6). A rebuilt/stripped block
    would have 400'd ("thinking blocks cannot be modified") and the hunt would read `aborted`,
    not `completed`.

Determinism lives in replaying fixed bytes, not in the model — the live call is one sample; this
re-drives it to identical events every time (DoD #5: "replaying a recorded run's raw payloads …
reproduces identical events"). Each fixture is the raw stream, so the streaming assembler + the
signed-block fold are exercised end to end offline.
"""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from rexhunter import brain, cost, db
from rexhunter.events import (
    ThinkingDelta,
    ToolCallEvent,
    ToolResultEvent,
    TrajectoryEvent,
    UsageEvent,
)
from rexhunter.loop import HuntComplete, ToolCallDecision, run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

FIXTURE_DIR = Path(__file__).parent / "fixtures"
RUN_ID = "32975e98-4f30-4763-9d6a-a34541d0e1ef"  # THE gated thinking-on hunt; fixtures pin this run
FIXTURES = sorted((FIXTURE_DIR / f"hunt_{RUN_ID}").glob("call_*.json"))
LIVE_SPEND_USD = 0.0187  # the [hunt] report; fold_cost over the two UsageEvents must reproduce it
THINKING: dict[str, Any] = {"type": "adaptive", "display": "summarized"}


def _assemble(raw: bytes) -> dict[str, Any]:
    a = brain.StreamAssembler()
    for event in brain.iter_sse_events(raw):
        a.feed(event)
    return a.assembled()


# Call one's stream-assembled signed thinking block — the exact bytes the real API accepted on the
# reconstructed turn. Derived from the fixture (never hand-copied), so the signature is verbatim.
CALL1: dict[str, Any] = _assemble(FIXTURES[0].read_bytes()) if FIXTURES else {}
CALL1_THINKING: dict[str, Any] = (
    next(b for b in CALL1["content"] if b["type"] == "thinking") if FIXTURES else {}
)
CALL1_TOOL_USE_ID = "toolu_01YQAE7FHSfnyAqaNaHR2mB2"


async def sniff(prey: str) -> str:
    """The mock-gym board tool — the deterministic sniff the live hunt ran (echoes the scent)."""
    return f"posting:{prey}"


def _sniff_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.tool(sniff)
    return reg


def _replay_transport(requests: list[dict[str, Any]]) -> httpx.MockTransport:
    """Feed the captured SSE streams in order for each MESSAGES_URL call (zero network), saving each
    request body so a test can prove what was folded into call two."""
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
    """Re-drive run_hunt over the streaming fixtures; return (events, (outcome, reason), reqs)."""
    requests: list[dict[str, Any]] = []
    reg = _sniff_registry()
    async with httpx.AsyncClient(transport=_replay_transport(requests)) as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            api_key="replay",
            model=brain.SMOKE_MODEL,
            registry=reg,
            thinking=THINKING,
            stream=True,
        )
        conn = await db.connect(db_path)
        try:
            run_id = await run_hunt(
                conn,
                territory="mock-gym",
                brain=brain_for("mock-gym"),
                registry=reg,
                cost_ceiling_usd=0.2,
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


def test_two_streaming_fixtures_were_captured() -> None:
    assert len(FIXTURES) == 2  # the live hunt made two brain calls


def test_fixtures_assemble_to_the_live_decisions() -> None:
    d1 = brain.parse_decision(json.dumps(_assemble(FIXTURES[0].read_bytes())).encode())
    d2 = brain.parse_decision(json.dumps(_assemble(FIXTURES[1].read_bytes())).encode())
    assert isinstance(d1, ToolCallDecision) and d1.tool == "sniff"
    assert d1.tool_use_id == CALL1_TOOL_USE_ID
    assert isinstance(d2, HuntComplete)  # the model scouted, then closed the hunt
    # The signed thinking block is present with a real opaque signature (never empty, display on).
    assert CALL1_THINKING["signature"] and len(CALL1_THINKING["signature"]) > 100


async def test_replay_reproduces_the_live_hunt_trajectory(tmp_path: Path) -> None:
    events, (outcome, reason), _requests = await _replay(tmp_path / "rex.db")

    # Two brain calls → two UsageEvents priced to the live spend (thinking tokens included).
    usage = [e for e in events if isinstance(e, UsageEvent)]
    assert len(usage) == 2
    assert (usage[0].input_tokens, usage[0].output_tokens) == (1810, 121)
    assert (usage[1].input_tokens, usage[1].output_tokens) == (1947, 374)
    assert cost.fold_cost(events) == pytest.approx(LIVE_SPEND_USD, abs=1e-4)

    # The streamed reasoning was relayed — ThinkingDelta events, the live feed.
    assert [e for e in events if isinstance(e, ThinkingDelta)]

    # Call one's sniff → one ToolCallEvent carrying its VERBATIM signed block + one paired result.
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(calls) == 1 and len(results) == 1
    assert calls[0].tool == "sniff"
    assert calls[0].tool_use_id == results[0].tool_use_id == CALL1_TOOL_USE_ID
    assert json.loads(calls[0].thinking) == CALL1_THINKING  # the signed block stored on the event

    assert (outcome, reason) == ("completed", None)  # NOT aborted — call two was accepted live


async def test_call_two_led_with_call_ones_verbatim_signed_block(tmp_path: Path) -> None:
    # THE GATE: call two's request folds call one's turn back as assistant[thinking, tool_use] —
    # the reconstructed turn LEADS with the byte-identical signed block the real API accepted.
    _events, _outcome, requests = await _replay(tmp_path / "rex.db")
    assert len(requests) == 2  # exactly two brain calls, no more

    messages = requests[1]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assistant = messages[1]["content"]
    assert assistant[0] == CALL1_THINKING  # verbatim signed block leads the reconstructed turn
    assert assistant[0]["signature"] == CALL1_THINKING["signature"]  # exact — echoed, never rebuilt
    assert assistant[1]["type"] == "tool_use"
    assert assistant[1]["id"] == CALL1_TOOL_USE_ID  # tool_use follows the block, correlation intact
    assert messages[2]["content"][0]["tool_use_id"] == CALL1_TOOL_USE_ID  # result paired

    # Request shape with thinking ON: the 2c guards still hold.
    assert requests[1]["thinking"] == THINKING
    assert requests[1]["tool_choice"]["disable_parallel_tool_use"] is True
    assert "temperature" not in requests[1] and "top_p" not in requests[1]


async def test_replay_is_deterministic_across_two_passes(tmp_path: Path) -> None:
    def _norm(events: list[TrajectoryEvent]) -> list[dict[str, Any]]:
        return [json.loads(e.model_dump_json()) for e in events]

    first, _, _ = await _replay(tmp_path / "a.db")
    second, _, _ = await _replay(tmp_path / "b.db")
    assert _norm(first) == _norm(second)
