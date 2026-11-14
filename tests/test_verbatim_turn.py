"""W2.3 (D7) — the whole assistant turn, stored on the tool call and echoed exactly as returned.

Until W2.3 a ToolCallEvent kept only the turn's first `thinking` block (and only on the streaming
path), and `project_messages` rebuilt the assistant turn from it plus a `tool_use` made from
`id`/`name`/`raw_request`. Every other block was dropped: text, a second thinking block, a
`redacted_thinking` block, the tool_use's own `caller` and the model's own `input`.

Since W2.3 the event carries `assistant_content` — the JSON bytes of the response's `content` array,
every block in order (invariant 6) — and the projection echoes it exactly as stored (fork E1): the
docs say to "Send each assistant turn back exactly as the API returned it" (prefix checks), and a
`thinking` block that is reordered, edited or dropped is a 400. An event written before W2.3 (no
`assistant_content`) keeps the old reconstruction, unchanged.
"""

import base64
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from rexhunter import brain, db
from rexhunter.events import (
    ErrorEvent,
    ToolCallEvent,
    ToolResultEvent,
    TrajectoryEvent,
    decode_event,
)
from rexhunter.loop import ToolCallDecision, run_hunt
from rexhunter.tools import ToolFn, ToolRegistry

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]

THINKING_A = {"type": "thinking", "thinking": "Reasoning first.", "signature": "SIG-A/opaque=="}
THINKING_B = {"type": "thinking", "thinking": "", "signature": "SIG-B/progress-update=="}
REDACTED = {"type": "redacted_thinking", "data": "ENCRYPTED/opaque+data=="}
TEXT = {"type": "text", "text": "Let me sniff mock-gym first."}
# The model's own tool_use, as the API returns it: `caller` included, `input` as the model wrote it.
TOOL_USE = {
    "type": "tool_use",
    "id": "toolu_1",
    "name": "sniff",
    "input": {"prey": "mock-gym"},
    "caller": {"type": "direct"},
}
TEXT_2 = {"type": "text", "text": "Now greenhouse."}
TOOL_USE_2 = {
    "type": "tool_use",
    "id": "toolu_9",
    "name": "sniff",
    "input": {"prey": "greenhouse"},
    "caller": {"type": "direct"},
}


def _call(
    content: list[dict[str, Any]],
    *,
    raw_request: bytes = b'{"prey":"mock-gym"}',
    tool_use_id: str = "toolu_1",
) -> ToolCallEvent:
    """A ToolCallEvent that stores the whole turn (W2.3)."""
    return ToolCallEvent(
        tool="sniff",
        raw_request=raw_request,
        tool_use_id=tool_use_id,
        assistant_content=json.dumps(content).encode(),
    )


def _assistant(context: list[Any]) -> list[dict[str, Any]]:
    messages = brain.project_messages("mock-gym", context)
    assert messages[1]["role"] == "assistant"
    return messages[1]["content"]


# ── B1: the echo ──────────────────────────────────────────────────────────────


def test_two_thinking_blocks_are_echoed_in_order_before_the_tool_use() -> None:
    # Reasoning plus a progress update: two signed blocks before one tool_use. Dropping or
    # reordering either is the "partially drop" / "rearrange" the docs reject with a 400.
    turn = [THINKING_A, THINKING_B, TOOL_USE]
    assert _assistant([_call(turn)]) == turn


def test_a_text_block_is_echoed_before_the_tool_use() -> None:
    # The thinking-off shape every recorded plain response has: [text, tool_use].
    turn = [TEXT, TOOL_USE]
    result = ToolResultEvent(
        tool="sniff", raw_request=b"{}", raw_response=b"posting:mock-gym", tool_use_id="toolu_1"
    )
    messages = brain.project_messages("mock-gym", [_call(turn), result])
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == turn
    assert messages[2]["content"][0]["tool_use_id"] == "toolu_1"  # the pairing is unchanged


def test_a_redacted_thinking_block_is_echoed_as_stored_in_its_position() -> None:
    turn = [THINKING_A, REDACTED, TOOL_USE]
    echoed = _assistant([_call(turn)])
    assert echoed == turn
    assert echoed[1] == REDACTED  # `data` verbatim, between the two blocks it came between


def test_a_block_field_the_loop_does_not_know_is_echoed_with_it() -> None:
    # Echoed, never rebuilt: a field the API adds to a thinking block reaches the wire unchanged.
    future = dict(THINKING_A, zz_future_field={"k": 1})
    turn = [future, TOOL_USE]
    assert _assistant([_call(turn)])[0] == future


def test_the_stored_turn_wins_over_the_pre_w2_3_thinking_field() -> None:
    # Both fields set (no producer writes this, but the projection must not mix them): the stored
    # turn is the whole turn, so the old single block is not added in front of it.
    event = ToolCallEvent(
        tool="sniff",
        raw_request=b'{"prey":"mock-gym"}',
        tool_use_id="toolu_1",
        assistant_content=json.dumps([TEXT, TOOL_USE]).encode(),
        thinking=json.dumps(THINKING_A).encode(),
    )
    assert _assistant([event]) == [TEXT, TOOL_USE]


def test_an_event_written_before_w2_3_decodes_and_keeps_the_old_reconstruction() -> None:
    # A payload exactly as the log stored it before W2.3: no `assistant_content` key.
    def b64(raw: bytes) -> str:
        return base64.b64encode(raw).decode()

    old = {
        "type": "tool_call",
        "tool": "sniff",
        "raw_request": b64(b'{"prey": "mock-gym"}'),
        "tool_use_id": "toolu_1",
        "thinking": b64(json.dumps(THINKING_A).encode()),
    }
    event = decode_event(json.dumps(old))
    assert isinstance(event, ToolCallEvent)
    assert event.assistant_content == b""
    # The old shape: the signed block, then a tool_use rebuilt from id/name/raw_request (no caller).
    rebuilt = {"type": "tool_use", "id": "toolu_1", "name": "sniff", "input": {"prey": "mock-gym"}}
    assert _assistant([event]) == [THINKING_A, rebuilt]
    # A stub-era event (no thinking either) still projects to the bare rebuilt tool_use.
    del old["thinking"]
    assert _assistant([decode_event(json.dumps(old))]) == [rebuilt]


def test_the_tool_use_is_echoed_as_the_model_returned_it_not_rebuilt() -> None:
    # E1: the tool_use goes back with its `caller` and the model's own `input`, not the validated
    # args the tool ran on (`raw_request`, made to differ here). A rebuild would change an earlier
    # message, which the prefix check on preserved-thinking models counts as a change.
    event = _call([THINKING_A, TOOL_USE], raw_request=b'{"prey":"validated-args"}')
    echoed = _assistant([event])[1]
    assert echoed == TOOL_USE
    assert echoed["caller"] == {"type": "direct"}
    assert echoed["input"] == {"prey": "mock-gym"}


def test_a_fatal_tool_error_still_pairs_with_the_stored_turns_tool_use() -> None:
    event = _call([TEXT, TOOL_USE])
    error = ErrorEvent(tool="sniff", retryable=False, error="boom", raw_request=b"{}")
    messages = brain.project_messages("mock-gym", [event, error])
    assert messages[2]["content"][0]["tool_use_id"] == "toolu_1"
    assert messages[2]["content"][0]["is_error"] is True


def test_each_stored_turn_is_echoed_in_its_own_position() -> None:
    # Two tool calls in one hunt: each assistant turn is its own stored turn, never another's.
    turn_a, turn_b = [TEXT, TOOL_USE], [TEXT_2, TOOL_USE_2]

    def result(tool_use_id: str) -> ToolResultEvent:
        return ToolResultEvent(
            tool="sniff", raw_request=b"{}", raw_response=b"posting:x", tool_use_id=tool_use_id
        )

    context = [
        _call(turn_a),
        result("toolu_1"),
        _call(turn_b, raw_request=b'{"prey":"greenhouse"}', tool_use_id="toolu_9"),
        result("toolu_9"),
    ]
    messages = brain.project_messages("mock-gym", context)
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[1]["content"] == turn_a
    assert messages[3]["content"] == turn_b


# ── B2: the store ─────────────────────────────────────────────────────────────
# `parse_decision` stores the response's `content` array on the ToolCallDecision from the validated
# envelope, on BOTH paths (the plain one kept no block at all before W2.3), and the loop copies it
# onto the ToolCallEvent. W1: new events leave the old `thinking` field empty.

FIXTURE_DIR = Path(__file__).parent / "fixtures"
PLAIN_RUN = "4c47b41b-5268-4da3-b787-9483b21d2c3f"  # W2.1b: thinking off, [text, tool_use]
STREAM_RUN = "32975e98-4f30-4763-9d6a-a34541d0e1ef"  # Unit 3: thinking on, [thinking, tool_use]
RECORDED_MODEL = "claude-sonnet-5"  # the model both recorded runs ran on (each fixture's "model")
NEEDS_HELP_TURN: dict[str, Any] = {
    "type": "tool_use",
    "id": "toolu_2",
    "name": "needs_help",
    "input": {},
}


async def sniff(prey: str) -> str:
    """The mock-gym board tool (echoes the scent)."""
    return f"posting:{prey}"


def _plain(content: list[dict[str, Any]]) -> bytes:
    """A non-streaming Messages response carrying `content`."""
    body = {
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "stop_reason": "tool_use",
        "content": content,
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }
    return json.dumps(body).encode()


def _sse(content: list[dict[str, Any]]) -> bytes:
    """A streaming response whose assembled `content` is exactly `content`: each block opens with
    its empty start shape, streams its deltas, and closes (signatures as one `signature_delta`).
    A `redacted_thinking` block arrives whole in its start frame, with no delta; any other key a
    thinking block carries rides on its start frame."""
    frames: list[str] = []

    def frame(data: dict[str, Any]) -> None:
        frames.append(f"event: {data['type']}\ndata: {json.dumps(data)}\n\n")

    frame(
        {
            "type": "message_start",
            "message": {
                "model": "claude-sonnet-5",
                "content": [],
                "usage": {"input_tokens": 100, "output_tokens": 1},
            },
        }
    )
    for i, block in enumerate(content):
        if block["type"] == "thinking":
            start: dict[str, Any] = {**block, "thinking": "", "signature": ""}
            deltas = [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": block["signature"]},
            ]
        elif block["type"] == "text":
            start = {"type": "text", "text": ""}
            deltas = [{"type": "text_delta", "text": block["text"]}]
        elif block["type"] == "redacted_thinking":
            start, deltas = dict(block), []
        else:  # tool_use: everything but `input` arrives at the start
            start = {**block, "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
        frame({"type": "content_block_start", "index": i, "content_block": start})
        for delta in deltas:
            frame({"type": "content_block_delta", "index": i, "delta": delta})
        frame({"type": "content_block_stop", "index": i})
    frame(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 20},
        }
    )
    frame({"type": "message_stop"})
    return "".join(frames).encode()


def _assemble(raw: bytes) -> dict[str, Any]:
    assembler = brain.StreamAssembler()
    for event in brain.iter_sse_events(raw):
        assembler.feed(event)
    return assembler.assembled()


async def scout(prey: str, depth: int = 1) -> str:
    """A tool with a defaulted parameter: its validated args (`raw_request`) differ from the
    model's own `input` whenever the model leaves `depth` out."""
    return f"posting:{prey}:{depth}"


async def _hunt(
    db_path: Path,
    mock_anthropic: MockAnthropic,
    responses: list[bytes],
    *,
    stream: bool,
    tool: ToolFn = sniff,
    model: str | None = None,
) -> tuple[list[TrajectoryEvent], list[httpx2.Request]]:
    """Drive run_hunt over `responses` in order (zero network); return (events, requests), the
    requests exactly as the mock transport received them (after the SDK serialised the body).
    `model` defaults to SMOKE_MODEL (synthetic streams); a recorded run passes its own (R1)."""
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, content=responses[len(requests) - 1])

    reg = ToolRegistry()
    reg.tool(tool)
    async with mock_anthropic(handler, api_key="replay") as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            model=brain.SMOKE_MODEL if model is None else model,
            registry=reg,
            thinking=brain.HUNT_THINKING if stream else None,
            stream=stream,
        )
        conn = await db.connect(db_path)
        try:
            run_id = await run_hunt(
                conn,
                territory="mock-gym",
                brain=brain_for("mock-gym"),
                registry=reg,
                cost_ceiling_usd=1.0,
                max_iterations=len(responses),
                tool_timeout_s=5.0,
            )
            events = await db.read_events(conn, run_id)
        finally:
            await conn.close()
    return events, requests


def _only_call(events: list[TrajectoryEvent]) -> ToolCallEvent:
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(calls) == 1
    return calls[0]


def _sent_assistant_turn(request: httpx2.Request) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = json.loads(request.content)["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    return messages[1]["content"]


@pytest.mark.anyio
async def test_the_plain_path_stores_the_whole_turn(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    turn = [TEXT, TOOL_USE]
    events, _ = await _hunt(
        tmp_path / "rex.db",
        mock_anthropic,
        [_plain(turn), _plain([NEEDS_HELP_TURN])],
        stream=False,
    )
    call = _only_call(events)
    assert json.loads(call.assistant_content) == turn  # both blocks, in order
    assert call.thinking == b""  # W1: new events fill only the new field


@pytest.mark.anyio
async def test_the_streaming_path_stores_every_block_with_exact_signatures(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    turn = [THINKING_A, THINKING_B, TOOL_USE]
    events, _ = await _hunt(
        tmp_path / "rex.db",
        mock_anthropic,
        [_sse(turn), _sse([NEEDS_HELP_TURN])],
        stream=True,
    )
    call = _only_call(events)
    stored = json.loads(call.assistant_content)
    assert stored == turn  # all three blocks, in order
    assert [b.get("signature") for b in stored[:2]] == [
        THINKING_A["signature"],
        THINKING_B["signature"],
    ]
    assert call.thinking == b""  # W1


def _keys(blocks: list[dict[str, Any]]) -> list[list[str]]:
    return [list(block) for block in blocks]


@pytest.mark.anyio
async def test_the_recorded_plain_hunt_echoes_call_ones_turn_on_the_wire(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # The W2.1b recording (thinking off). Call two's assistant turn, as the SDK serialised it onto
    # the transport, is call one's `content` exactly: the text block, then the tool_use with its
    # `caller` and the model's own `input` (fork E1), keys in the provider's own order.
    fixtures = sorted((FIXTURE_DIR / f"hunt_{PLAIN_RUN}").glob("call_*.json"))
    assert len(fixtures) == 2
    responses = [f.read_bytes() for f in fixtures]
    events, requests = await _hunt(
        tmp_path / "rex.db", mock_anthropic, responses, stream=False, model=RECORDED_MODEL
    )
    call_one = json.loads(responses[0])["content"]
    assert len(requests) == 2
    # R1 (W2.2): each request on the model the run was recorded on, thinking off.
    assert [json.loads(r)["model"] for r in responses] == [RECORDED_MODEL, RECORDED_MODEL]
    for request in requests:
        body = json.loads(request.content)
        assert body["model"] == RECORDED_MODEL
        assert body["thinking"] == {"type": "disabled"}
    sent = _sent_assistant_turn(requests[1])
    assert sent == call_one
    assert _keys(sent) == _keys(call_one)  # stored and sent as returned, key order included
    stored = json.loads(_only_call(events).assistant_content)
    assert stored == call_one
    assert _keys(stored) == _keys(call_one)


@pytest.mark.anyio
async def test_the_recorded_thinking_hunt_echoes_call_ones_assembled_turn_on_the_wire(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # The Unit 3 recording (thinking on, streamed). Call two's assistant turn on the wire is call
    # one's ASSEMBLED content exactly: the signed thinking block, then the tool_use with `caller`.
    fixtures = sorted((FIXTURE_DIR / f"hunt_{STREAM_RUN}").glob("call_*.json"))
    assert len(fixtures) == 2
    responses = [f.read_bytes() for f in fixtures]
    _events, requests = await _hunt(
        tmp_path / "rex.db", mock_anthropic, responses, stream=True, model=RECORDED_MODEL
    )
    assert len(requests) == 2
    # R1 (W2.2): each request on the model the run was recorded on.
    assert [_assemble(r)["model"] for r in responses] == [RECORDED_MODEL, RECORDED_MODEL]
    assert [json.loads(r.content)["model"] for r in requests] == [RECORDED_MODEL, RECORDED_MODEL]
    sent = _sent_assistant_turn(requests[1])
    assembled = _assemble(responses[0])["content"]
    assert sent == assembled
    assert _keys(sent) == _keys(assembled)
    # Facts read from the recording itself, not from the assembler: the tool_use goes back with
    # the id, `caller` and `input` the model streamed.
    assert [b["type"] for b in sent] == ["thinking", "tool_use"]
    assert sent[1]["id"] == "toolu_01YQAE7FHSfnyAqaNaHR2mB2"
    assert sent[1]["caller"] == {"type": "direct"}
    assert sent[1]["input"] == {"prey": "mock-gym"}
    assert b'"caller":{"type":"direct"}' in responses[0]  # the stream itself carried `caller`


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True], ids=["plain", "streaming"])
async def test_a_redacted_thinking_block_is_stored_and_sent_back_in_place(
    tmp_path: Path, mock_anthropic: MockAnthropic, stream: bool
) -> None:
    turn = [THINKING_A, REDACTED, TOOL_USE]
    wire = _sse if stream else _plain
    events, requests = await _hunt(
        tmp_path / "rex.db", mock_anthropic, [wire(turn), wire([NEEDS_HELP_TURN])], stream=stream
    )
    assert json.loads(_only_call(events).assistant_content) == turn
    assert _sent_assistant_turn(requests[1]) == turn


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True], ids=["plain", "streaming"])
async def test_a_block_field_the_loop_does_not_know_is_stored_and_reaches_the_wire(
    tmp_path: Path, mock_anthropic: MockAnthropic, stream: bool
) -> None:
    future = dict(THINKING_A, zz_future_field={"k": 1})
    turn = [future, TOOL_USE]
    wire = _sse if stream else _plain
    events, requests = await _hunt(
        tmp_path / "rex.db", mock_anthropic, [wire(turn), wire([NEEDS_HELP_TURN])], stream=stream
    )
    assert json.loads(_only_call(events).assistant_content) == turn
    assert _sent_assistant_turn(requests[1])[0] == future


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True], ids=["plain", "streaming"])
async def test_two_tool_calls_each_store_and_send_back_their_own_turn(
    tmp_path: Path, mock_anthropic: MockAnthropic, stream: bool
) -> None:
    turn_a = [THINKING_A, TOOL_USE] if stream else [TEXT, TOOL_USE]
    turn_b = [THINKING_B, TOOL_USE_2] if stream else [TEXT_2, TOOL_USE_2]
    wire = _sse if stream else _plain
    events, requests = await _hunt(
        tmp_path / "rex.db",
        mock_anthropic,
        [wire(turn_a), wire(turn_b), wire([NEEDS_HELP_TURN])],
        stream=stream,
    )
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert [json.loads(c.assistant_content) for c in calls] == [turn_a, turn_b]
    messages: list[dict[str, Any]] = json.loads(requests[2].content)["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[1]["content"] == turn_a
    assert messages[3]["content"] == turn_b


@pytest.mark.anyio
async def test_the_stored_tool_use_keeps_the_models_input_not_the_validated_args(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # The model leaves `depth` out; validation fills the default, so `raw_request` (what the tool
    # ran on) differs from the model's `input`. The stored and sent turn keep the model's input.
    tool_use = dict(TOOL_USE, name="scout")
    turn = [TEXT, tool_use]
    events, requests = await _hunt(
        tmp_path / "rex.db",
        mock_anthropic,
        [_plain(turn), _plain([NEEDS_HELP_TURN])],
        stream=False,
        tool=scout,
    )
    call = _only_call(events)
    assert json.loads(call.raw_request) == {"prey": "mock-gym", "depth": 1}
    assert json.loads(call.assistant_content) == turn
    assert _sent_assistant_turn(requests[1]) == turn


@pytest.mark.parametrize(
    "content",
    [
        [TEXT, TOOL_USE],
        [TEXT, TOOL_USE, TEXT_2],
        [THINKING_A, REDACTED, THINKING_B, TOOL_USE],
    ],
    ids=["text-then-tool-use", "a-block-after-the-tool-use", "redacted-between-thinking"],
)
def test_parse_decision_stores_the_content_array_as_received(content: list[dict[str, Any]]) -> None:
    # The fill site itself, on a payload with no `usage` and no `model`: every block, in order,
    # whatever comes after the tool_use.
    decision = brain.parse_decision(json.dumps({"content": content}).encode())
    assert decision.usage is None
    assert isinstance(decision, ToolCallDecision)
    assert json.loads(decision.assistant_content) == content


def test_no_tool_schema_offers_the_stored_turn_to_the_model() -> None:
    # The field lives on the tool-call decision only; a terminal decision's schema is model-facing
    # (build_tools derives it from HuntComplete / NeedsHelp), so the field must not leak into it.
    reg = ToolRegistry()
    reg.tool(sniff)
    assert "assistant_content" not in json.dumps(brain.build_tools(reg))
