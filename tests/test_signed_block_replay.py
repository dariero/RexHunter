"""P5 Unit 3c (offline half) — signed-block replay + the gate.

Sonnet 5 runs adaptive thinking by default, so every reconstructed assistant turn must lead with
the model's VERBATIM signed thinking block — invariant 6 (raw I/O snapshots), reinforced by
invariant 3 (the signature is opaque, never interpreted). The gate proves `project_messages`
ECHOES the stored block rather than rebuilding it:

- The reconstructed assistant turn leads with the byte-identical signed block (signature exact).
- Mutation: strip or tamper the block → the reconstructed turn differs in exactly the signed bytes.
  That difference is what the real API rejects with a 400 ("thinking blocks cannot be modified") on
  a thinking-on multi-turn hunt — asserted OFFLINE on the messages array, never a live call.
- End to end: a streaming hunt folds call one's stream-assembled signed block into call two's
  request (the never-run path the gated live hunt will exercise), and the request keeps the 2c
  guards (adaptive+summarized thinking, disable_parallel_tool_use, no sampling params).
"""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from rexhunter import brain, db
from rexhunter.events import ToolCallEvent
from rexhunter.loop import run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

SIGNED_BLOCK = {
    "type": "thinking",
    "thinking": "I'll scout mock-gym.",
    "signature": "OPAQUE-SiGnAtUre-do/not/interpret+VERBATIM==",
}
THINKING_BYTES = json.dumps(SIGNED_BLOCK).encode()


def _tool_call(thinking: bytes) -> list[Any]:
    return [
        ToolCallEvent(
            tool="sniff",
            raw_request=b'{"prey": "mock-gym"}',
            tool_use_id="toolu_1",
            thinking=thinking,
        )
    ]


# ── The gate: project_messages echoes the verbatim signed block ───────────────


def test_reconstructed_assistant_turn_leads_with_the_verbatim_signed_block() -> None:
    messages = brain.project_messages("mock-gym", _tool_call(THINKING_BYTES))
    assistant = messages[1]
    assert assistant["role"] == "assistant"
    # content[0] is the signed block — byte-identical, echoed not rebuilt; the signature is exact.
    assert assistant["content"][0] == SIGNED_BLOCK
    assert assistant["content"][0]["signature"] == SIGNED_BLOCK["signature"]
    # content[1] is the tool_use it precedes.
    assert assistant["content"][1]["type"] == "tool_use"
    assert assistant["content"][1]["id"] == "toolu_1"


def test_stripping_or_rebuilding_the_block_changes_the_turn_and_would_400() -> None:
    verbatim = brain.project_messages("mock-gym", _tool_call(THINKING_BYTES))

    # (a) STRIP — a turn with no signed block (the pre-Unit-3 / bare-tool_use shape). The assistant
    # turn no longer leads with the thinking block; on a thinking-on hunt the API 400s.
    stripped = brain.project_messages("mock-gym", _tool_call(b""))
    assert stripped[1]["content"][0]["type"] == "tool_use"  # no leading signed block
    assert verbatim != stripped  # the reconstructed turns differ — exactly the missing signed block

    # (b) REBUILD — a tampered signature (any re-serialisation that drifts the signed bytes). The
    # echoed block carries the mutated signature, which no longer matches what the provider signed.
    tampered = dict(SIGNED_BLOCK, signature="TAMPERED==")
    rebuilt = brain.project_messages("mock-gym", _tool_call(json.dumps(tampered).encode()))
    assert rebuilt[1]["content"][0]["signature"] != SIGNED_BLOCK["signature"]  # drift → would 400
    assert rebuilt != verbatim


# ── End to end: a streaming hunt folds the signed block into the next call ─────

SIG1 = "SiGnAtUre-CALL-ONE=="


def _sse(
    thinking: str, signature: str, tool_id: str, name: str, tool_input: dict[str, Any]
) -> bytes:
    frames: list[str] = []

    def frame(event: str, data: dict[str, Any]) -> None:
        frames.append(f"event: {event}\ndata: {json.dumps(data)}\n\n")

    frame(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "model": "claude-sonnet-5",
                "content": [],
                "usage": {"input_tokens": 100, "output_tokens": 1},
            },
        },
    )
    frame(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
    )
    frame(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": thinking},
        },
    )
    frame(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": signature},
        },
    )
    frame("content_block_stop", {"type": "content_block_stop", "index": 0})
    frame(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
        },
    )
    frame(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool_input)},
        },
    )
    frame("content_block_stop", {"type": "content_block_stop", "index": 1})
    frame(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 20},
        },
    )
    frame("message_stop", {"type": "message_stop"})
    return "".join(frames).encode()


CALL1 = _sse("Scouting mock-gym.", SIG1, "toolu_1", "sniff", {"prey": "mock-gym"})
CALL2 = _sse("Enough — capturing.", "SIG2==", "toolu_2", "hunt_complete", {"catch": ["posting:mg"]})
CALL1_BLOCK = {"type": "thinking", "thinking": "Scouting mock-gym.", "signature": SIG1}


async def sniff(prey: str) -> str:
    return f"posting:{prey}"


async def _run_capturing(db_path: Path) -> list[dict[str, Any]]:
    """Drive a two-call streaming hunt; return the captured request bodies."""
    requests: list[dict[str, Any]] = []
    responses = [CALL1, CALL2]
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        i = state["i"]
        state["i"] += 1
        return httpx.Response(200, content=responses[i])

    reg = ToolRegistry()
    reg.tool(sniff)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            api_key="replay",
            model=brain.SMOKE_MODEL,
            registry=reg,
            thinking={"type": "adaptive", "display": "summarized"},
            stream=True,
        )
        conn = await db.connect(db_path)
        try:
            await run_hunt(
                conn,
                territory="mock-gym",
                brain=brain_for("mock-gym"),
                registry=reg,
                max_iterations=4,
            )
        finally:
            await conn.close()
    return requests


async def test_call_two_leads_with_call_ones_stream_assembled_signed_block(tmp_path: Path) -> None:
    requests = await _run_capturing(tmp_path / "rex.db")
    assert len(requests) == 2  # sniff on call one, then hunt_complete on call two

    messages = requests[1]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assistant = messages[1]["content"]
    # Call two's reconstructed assistant turn LEADS with call one's verbatim signed block, assembled
    # off the stream — the never-run path the gated live hunt proves against the real API.
    assert assistant[0] == CALL1_BLOCK
    assert assistant[0]["signature"] == SIG1  # signature carried through the stream + the fold
    assert assistant[1]["type"] == "tool_use" and assistant[1]["id"] == "toolu_1"


async def test_streaming_request_keeps_the_2c_guards_with_thinking_on(tmp_path: Path) -> None:
    body = (await _run_capturing(tmp_path / "rex.db"))[0]
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}  # thinking back ON
    assert body["tool_choice"]["disable_parallel_tool_use"] is True  # one tool_use per turn intact
    assert "temperature" not in body and "top_p" not in body and "top_k" not in body  # Sonnet 5
