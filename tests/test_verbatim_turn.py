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
from typing import Any

from rexhunter import brain
from rexhunter.events import ErrorEvent, ToolCallEvent, ToolResultEvent, decode_event

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
