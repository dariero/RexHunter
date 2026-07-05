"""P5 Unit 3b — the write-ahead ThinkingDelta relay (offline).

Thinking is back on: as the model streams its reasoning, the loop appends each delta to the log
(write-ahead, invariant 1) and the P3 hub broadcasts it — reusing `publish()`, no new fan-out.
This is Rex's chain of thought as the live hunt feed.

Driven through the REAL loop: a streaming `httpx.MockTransport` feeds two SSE responses (each a
thinking block + a tool_use), `adapter_brain_for(..., stream=True)` assembles them, and the
loop-built sink relays each delta. Asserts the three load-bearing properties:

- Write-ahead ordering (the invariant-1 gate, same shape as `test_hub.py`): a ThinkingDelta reaches
  a viewer queue only after a SEPARATE reader connection can see its committed row.
- Live, not batched: a turn's ThinkingDeltas are committed DURING the brain call, so their global
  ids precede that turn's ToolCallEvent id (appended only after the decision returns).
- Fan-out: two viewers render byte-identical ThinkingDelta sequences.
"""

import json
from pathlib import Path

import httpx
import pytest

from rexhunter import brain, db
from rexhunter.events import ThinkingDelta, ToolCallEvent
from rexhunter.hub import Envelope, Hub
from rexhunter.loop import run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

SIG1 = "SiGnAtUre-CALL-ONE-opaque=="
SIG2 = "SiGnAtUre-CALL-TWO-opaque=="


async def sniff(prey: str) -> str:
    return f"posting:{prey}"


def _sniff_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.tool(sniff)
    return reg


def _sse(
    thinking: list[str], signature: str, tool_id: str, name: str, tool_input: dict[str, object]
) -> bytes:
    """Build an Anthropic SSE response: a thinking block (deltas + signature) then one tool_use."""
    frames: list[str] = []

    def frame(event: str, data: dict[str, object]) -> None:
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
    for chunk in thinking:
        frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": chunk},
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


# Call 1 reasons then sniffs; call 2 reasons then ends the hunt.
CALL1 = _sse(["Scouting ", "mock-gym."], SIG1, "toolu_1", "sniff", {"prey": "mock-gym"})
CALL2 = _sse(["Enough — capturing."], SIG2, "toolu_2", "hunt_complete", {"catch": ["posting:mg"]})
EXPECTED_DELTAS = ["Scouting ", "mock-gym.", "Enough — capturing."]


def _streaming_transport(responses: list[bytes]) -> httpx.MockTransport:
    state = {"i": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        i = state["i"]
        state["i"] += 1
        return httpx.Response(200, content=responses[i])

    return httpx.MockTransport(handler)


def _drain(queue: object) -> list[Envelope]:
    out: list[Envelope] = []
    q = queue  # asyncio.Queue[Envelope]
    while not q.empty():  # type: ignore[attr-defined]
        out.append(q.get_nowait())  # type: ignore[attr-defined]
    return out


async def _run(db_path: Path, hub: Hub) -> str:
    reg = _sniff_registry()
    async with httpx.AsyncClient(transport=_streaming_transport([CALL1, CALL2])) as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            api_key="relay",
            model=brain.SMOKE_MODEL,
            registry=reg,
            thinking={"type": "adaptive", "display": "summarized"},
            stream=True,
        )
        conn = await db.connect(db_path)
        try:
            return await run_hunt(
                conn,
                territory="mock-gym",
                brain=brain_for("mock-gym"),
                registry=reg,
                publish=hub.publish,
                max_iterations=4,
            )
        finally:
            await conn.close()


async def test_two_viewers_render_identical_thinking_delta_streams(tmp_path: Path) -> None:
    hub = Hub(maxsize=256)
    _, qa = hub.register()
    _, qb = hub.register()
    await _run(tmp_path / "rex.db", hub)

    frames_a = [e for e in _drain(qa) if e.id is not None]
    frames_b = [e for e in _drain(qb) if e.id is not None]
    assert frames_a == frames_b  # byte-identical fan-out

    deltas = [
        json.loads(e.data) for e in frames_a if json.loads(e.data)["type"] == "thinking_delta"
    ]
    assert [d["text"] for d in deltas] == EXPECTED_DELTAS  # the reasoning, streamed live


async def test_thinking_deltas_are_write_ahead(tmp_path: Path) -> None:
    """Invariant 1: a delta reaches a viewer only after its row is committed — a separate reader
    connection sees every relayed delta's row (the test_hub.py write-ahead shape)."""
    hub = Hub(maxsize=256)
    _, queue = hub.register()
    db_path = tmp_path / "rex.db"
    await _run(db_path, hub)

    reader = await db.connect(db_path)  # a DIFFERENT connection — sees only commits
    try:
        delta_envelopes = [
            e
            for e in _drain(queue)
            if e.id is not None and json.loads(e.data)["type"] == "thinking_delta"
        ]
        assert len(delta_envelopes) == len(EXPECTED_DELTAS)
        for env in delta_envelopes:
            async with reader.execute(
                "SELECT payload FROM trajectory_events WHERE id = ? AND type = 'thinking_delta'",
                (env.id,),
            ) as cur:
                row = await cur.fetchone()
            assert row is not None  # committed before the broadcast reached the queue
            assert row[0] == env.data  # the envelope carries the exact stored payload
    finally:
        await reader.close()


async def test_deltas_are_relayed_during_the_stream_not_batched_after(tmp_path: Path) -> None:
    """Live, not batched: call one's thinking deltas are committed DURING the brain call, so their
    global ids precede that turn's ToolCallEvent (appended only after the decision returns)."""
    hub = Hub(maxsize=256)
    _, queue = hub.register()
    run_id = await _run(tmp_path / "rex.db", hub)

    ordered = [(e.id, json.loads(e.data)) for e in _drain(queue) if e.id is not None]
    first_delta_id = next(i for i, p in ordered if p["type"] == "thinking_delta")
    first_call_id = next(i for i, p in ordered if p["type"] == "tool_call")
    assert first_delta_id is not None and first_call_id is not None
    assert first_delta_id < first_call_id  # reasoning streamed BEFORE the tool dispatched

    # And the run reached its terminal outcome cleanly (call two's hunt_complete).
    conn = await db.connect(tmp_path / "rex.db")
    try:
        events = await conn.execute("SELECT outcome FROM runs WHERE id = ?", (run_id,))
        row = await events.fetchone()
        await events.close()
        assert row is not None and row[0] == "completed"
        typed = await db.read_events(conn, run_id)
        assert [type(e).__name__ for e in typed].count(ThinkingDelta.__name__) == len(
            EXPECTED_DELTAS
        )
        assert any(isinstance(e, ToolCallEvent) for e in typed)
    finally:
        await conn.close()
