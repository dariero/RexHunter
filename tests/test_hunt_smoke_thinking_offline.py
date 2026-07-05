"""P5 Unit 3c — the thinking-on live-hunt HARNESS, proven offline before any spend.

`hunt_smoke_thinking.run_live_hunt` is the paid entrypoint. This drives its ENTIRE path — free
pre-flight → streaming hunt through the `CapturingTransport` tee → fixture write — with an injected
`httpx.MockTransport` and a tmp fixture dir, so at spend time the ONLY untested unknown is "does the
real API accept the stream-assembled signed block". The load-bearing thing this catches that the
per-adapter tests can't: the tee reads the full body, and we prove that still feeds the streaming
adapter (a buffered response iterated by `aiter_lines`).
"""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from rexhunter import brain, hunt_smoke_thinking

pytestmark = pytest.mark.anyio


def _sse(thinking: str, sig: str, tool_id: str, name: str, tool_input: dict[str, Any]) -> bytes:
    f: list[str] = []

    def e(event: str, data: dict[str, Any]) -> None:
        f.append(f"event: {event}\ndata: {json.dumps(data)}\n\n")

    e(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "model": "claude-sonnet-5",
                "content": [],
                "usage": {"input_tokens": 200, "output_tokens": 1},
            },
        },
    )
    e(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
    )
    e(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": thinking},
        },
    )
    e(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": sig},
        },
    )
    e("content_block_stop", {"type": "content_block_stop", "index": 0})
    e(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
        },
    )
    e(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool_input)},
        },
    )
    e("content_block_stop", {"type": "content_block_stop", "index": 1})
    e(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 30},
        },
    )
    e("message_stop", {"type": "message_stop"})
    return "".join(f).encode()


CALL1 = _sse("Scouting.", "SIG-1==", "toolu_1", "sniff", {"prey": "mock-gym"})
CALL2 = _sse("Done.", "SIG-2==", "toolu_2", "hunt_complete", {"catch": ["posting:mock-gym"]})


async def test_harness_full_path_runs_offline_and_writes_fixtures(tmp_path: Path) -> None:
    msg = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == brain.COUNT_TOKENS_URL:
            return httpx.Response(200, json={"input_tokens": 1234})  # the free pre-flight
        i = msg["i"]
        msg["i"] += 1
        return httpx.Response(200, content=[CALL1, CALL2][i])  # the two streaming brain calls

    summary = await hunt_smoke_thinking.run_live_hunt(
        "offline-key", inner=httpx.MockTransport(handler), fixture_dir=tmp_path
    )

    # Two brain calls captured through the tee — the streaming path survives the full-body read.
    assert summary["brain_calls"] == 2
    assert summary["outcome"] == "completed"  # call two's hunt_complete ended the run cleanly
    assert summary["spend"] > 0  # usage folded from the streamed message_delta

    # Two golden fixtures written UNDER the run id, in the tmp dir (never the repo).
    fixtures = sorted((tmp_path / f"hunt_{summary['run_id']}").glob("call_*.json"))
    assert [p.read_bytes() for p in fixtures] == [CALL1, CALL2]  # the raw SSE streams, verbatim

    # And a captured fixture re-drives through the streaming assembler (it's a valid replay source).
    assembler = brain.StreamAssembler()
    for event in brain.iter_sse_events(fixtures[0].read_bytes()):
        assembler.feed(event)
    blocks = {b["type"]: b for b in assembler.assembled()["content"]}
    assert (
        blocks["thinking"]["signature"] == "SIG-1=="
    )  # signature captured verbatim off the stream


async def test_preflight_non_200_stops_before_any_paid_call(tmp_path: Path) -> None:
    """A non-200 count_tokens (auth / schema / a rejected thinking param) raises SystemExit BEFORE
    the messages endpoint is ever hit — the free call is where a bad payload surfaces, for $0."""
    hit = {"messages": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == brain.COUNT_TOKENS_URL:
            return httpx.Response(400, json={"error": {"message": "bad thinking param"}})
        hit["messages"] += 1
        return httpx.Response(200, content=CALL1)

    with pytest.raises(SystemExit):
        await hunt_smoke_thinking.run_live_hunt(
            "offline-key", inner=httpx.MockTransport(handler), fixture_dir=tmp_path
        )
    assert hit["messages"] == 0  # never spent — stopped at the free pre-flight
