"""P5 Unit 3a — the hand-rolled SSE streaming transport (offline, fixture-driven).

Sonnet 5 streams its response as a sequence of SSE events (`message_start`, `content_block_*`,
`message_delta`, `message_stop`). This unit assembles that stream into content blocks WITHOUT the
vendor SDK — the loop stays SDK-free (ADR §What point 1), and we need the raw signed-block bytes
in hand (invariants 3 + 6) which the SDK's typed-delta layer hides.

Three properties, all against a hand-authored fixture (`fixtures/stream_thinking.sse`) whose
thinking block carries a `signature_delta` frame — the one event the streaming docs omit and the
whole point of the fidelity check:

1. Assembly: the stream folds into a `thinking` block (+ signature), a `tool_use` block (name +
   assembled `input`), and usage, in the exact shape `parse_decision` already consumes.
2. Signed-block fidelity (inv 6): the signature is captured verbatim and round-trips byte-identical
   through parse → serialise → parse. It is opaque — never interpreted or mutated.
3. Display streams, execution waits: partial `input_json_delta` frames buffer, but the tool_use
   `input` is finalized only at `content_block_stop` — a truncated stream leaves it un-dispatchable.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from rexhunter import brain
from rexhunter.loop import ToolCallDecision

FIXTURE = Path(__file__).parent / "fixtures" / "stream_thinking.sse"
# The exact opaque signature the fixture streams via signature_delta — captured verbatim, not read.
EXPECTED_SIGNATURE = "ErUBCkYIBRgCKkD3SiGnAtUrEopaqueBlob+do/not/interpret+VERBATIM=="
TOOL_USE_ID = "toolu_01STREAMSNIFF"


def _assemble(raw: bytes) -> dict[str, Any]:
    assembler = brain.StreamAssembler()
    for event in brain.iter_sse_events(raw):
        assembler.feed(event)
    return assembler.assembled()


def _content(msg: dict[str, Any]) -> list[dict[str, Any]]:
    return msg["content"]


def test_stream_assembles_into_content_blocks() -> None:
    msg = _assemble(FIXTURE.read_bytes())
    by_type = {b["type"]: b for b in _content(msg)}

    thinking = by_type["thinking"]
    assert (
        thinking["thinking"]
        == "The user wants me to scout the mock-gym territory. I'll call sniff."
    )
    assert thinking["signature"] == EXPECTED_SIGNATURE

    tool_use = by_type["tool_use"]
    assert tool_use["id"] == TOOL_USE_ID
    assert tool_use["name"] == "sniff"
    assert tool_use["input"] == {"prey": "mock-gym"}  # assembled from the input_json_delta frames

    assert msg["stop_reason"] == "tool_use"
    assert msg["usage"] == {"input_tokens": 1500, "output_tokens": 57}
    assert msg["model"] == "claude-sonnet-5"


def test_assembled_message_parses_to_a_decision() -> None:
    # parse_decision counts tool_use blocks only, so the thinking sibling doesn't disturb it (2b).
    msg = _assemble(FIXTURE.read_bytes())
    decision = brain.parse_decision(json.dumps(msg).encode())
    assert isinstance(decision, ToolCallDecision)
    assert decision.tool == "sniff"
    assert decision.args == {"prey": "mock-gym"}
    assert decision.tool_use_id == TOOL_USE_ID
    assert decision.usage is not None
    assert (decision.usage.input_tokens, decision.usage.output_tokens) == (1500, 57)


def test_signed_block_round_trips_byte_identical() -> None:
    """Invariant 6: the thinking block's text AND signature survive parse → serialise → parse
    unchanged. The signature is opaque bytes we never touch."""
    thinking = next(b for b in _content(_assemble(FIXTURE.read_bytes())) if b["type"] == "thinking")
    serialised = json.dumps(thinking)
    reparsed = json.loads(serialised)
    assert reparsed == thinking
    assert reparsed["signature"] == EXPECTED_SIGNATURE  # verbatim through the round-trip


def test_iter_sse_events_skips_non_data_frames() -> None:
    # `event:` lines and the `ping` keep-alive carry no assembled state; only `data:` JSON is read.
    events = list(brain.iter_sse_events(FIXTURE.read_bytes()))
    types = [e["type"] for e in events]
    assert types[0] == "message_start"
    assert types[-1] == "message_stop"
    assert "ping" in types  # ping is surfaced as an event but folds to nothing


def test_display_streams_execution_waits() -> None:
    """A tool_use's `input` is finalized only at content_block_stop: feed the stream truncated just
    before the tool_use's stop and the partial JSON stays un-parsed (not dispatchable), and the
    assembler reports the message incomplete."""
    raw = FIXTURE.read_bytes()
    # Cut on a FRAME boundary just before the tool_use's closing content_block_stop (index 1), so
    # its input_json_delta is fed but its stop is not. Back up to the preceding blank line so the
    # `rest` half resumes on a whole `event:`/`data:` frame.
    stop = raw.index(b'data: {"type":"content_block_stop","index":1}')
    cut = raw.rindex(b"\n\n", 0, stop) + 2
    truncated, rest = raw[:cut], raw[cut:]

    assembler = brain.StreamAssembler()
    for event in brain.iter_sse_events(truncated):
        assembler.feed(event)

    assert assembler.complete is False  # no message_stop / stop_reason seen
    tool_use = next(b for b in _content(assembler.assembled()) if b["type"] == "tool_use")
    assert tool_use["input"] == {}  # partial JSON buffered but NOT applied — execution waits

    # Feeding the rest finalizes it: input parsed, message complete.
    for event in brain.iter_sse_events(rest):
        assembler.feed(event)
    assert assembler.complete is True
    tool_use = next(b for b in _content(assembler.assembled()) if b["type"] == "tool_use")
    assert tool_use["input"] == {"prey": "mock-gym"}


@pytest.mark.parametrize("blank", [b"", b"   "])
def test_iter_sse_events_tolerates_trailing_blank_frames(blank: bytes) -> None:
    # A trailing blank/whitespace frame (SSE streams end with a blank line) must not raise.
    events = list(brain.iter_sse_events(FIXTURE.read_bytes() + b"\n\n" + blank))
    assert events[-1]["type"] == "message_stop"
