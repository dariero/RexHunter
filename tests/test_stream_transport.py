"""P5 Unit 3a — the hand-rolled SSE streaming transport (offline, fixture-driven).

Sonnet 5.5 streams its response as a sequence of SSE events (`message_start`, `content_block_*`,
`message_delta`, `message_stop`). This unit assembles that stream into content blocks WITHOUT the
vendor SDK — the loop stays SDK-free (ADR §What point 1), and we need the raw signed-block bytes
in hand (invariants 3 + 6) which the SDK's typed-delta layer hides.

Three properties, all against a recorded Sonnet 5.5 stream (call two of the W2.5 thinking-on hunt,
`fixtures/hunt_54590aec…/call_02.json`) whose thinking block carries a `signature_delta` frame,
the one event the streaming docs omit and the whole point of the fidelity check:

1. Assembly: the stream folds into a `thinking` block (+ signature), a `tool_use` block (name +
   assembled `input`), and usage, in the exact shape `parse_decision` already consumes.
2. Signed-block fidelity (inv 6): the signature is captured verbatim and round-trips byte-identical
   through parse → serialise → parse. It is opaque — never interpreted or mutated.
3. Display streams, execution waits: partial `input_json_delta` frames buffer, but the tool_use
   `input` is finalized only at `content_block_stop` — a truncated stream leaves it un-dispatchable.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from rexhunter import brain
from rexhunter.loop import ToolCallDecision

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "hunt_54590aec-de50-4727-a0a2-9e533fb02efa"
    / "call_02.json"
)
TOOL_USE_ID = "toolu_01MNhH6cKQoBnJDYF7kEV5Aa"
TOOL_INPUT = {"prey": "mock-gym machine learning engineer"}
# The signature the stream carries is 1200 characters; its sha256 pins it as recorded.
SIGNATURE_SHA256 = "ebc610f21cd88c081e0cb9023554337078b14f88eef1a9fc146d2f4b49a33c5d"


def _frames(raw: bytes) -> list[dict[str, Any]]:
    # Every `data:` frame, read with a plain split and json.loads, never the code under test, so
    # what the assembler must produce is not taken from the assembler.
    return [
        json.loads(line[len(b"data: ") :])
        for frame in raw.split(b"\n\n")
        for line in frame.split(b"\n")
        if line.startswith(b"data: ")
    ]


def _expected_signature() -> str:
    # The opaque signature as its signature_delta frames carry it, joined verbatim, not read.
    return "".join(
        frame["delta"]["signature"]
        for frame in _frames(FIXTURE.read_bytes())
        if frame["type"] == "content_block_delta" and frame["delta"]["type"] == "signature_delta"
    )


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
    assert thinking["thinking"] == (
        "The tool only echoed my input back, with no actual postings in it. I'm weighing one more "
        "attempt with a different approach, or stopping here and reporting that the lookup came "
        "up empty.\n\n"
    )
    signature = _expected_signature()
    assert len(signature) == 1200
    assert hashlib.sha256(signature.encode()).hexdigest() == SIGNATURE_SHA256
    assert thinking["signature"] == signature

    tool_use = by_type["tool_use"]
    assert tool_use["id"] == TOOL_USE_ID
    assert tool_use["name"] == "sniff"
    assert tool_use["input"] == TOOL_INPUT  # assembled from the input_json_delta frames

    assert msg["stop_reason"] == "tool_use"
    # The message_start usage with the message_delta usage over it, every key as the stream carries
    # them (the API repeats the cache counts in both and adds the thinking count at the end).
    assert msg["usage"] == {
        "input_tokens": 1829,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 155,
        "service_tier": "standard",
        "inference_geo": "global",
        "speed": "standard",
        "output_tokens_details": {"thinking_tokens": 97},
    }
    assert msg["model"] == "claude-sonnet-5-5"


def test_assembled_message_parses_to_a_decision() -> None:
    # parse_decision counts tool_use blocks only, so the thinking sibling doesn't disturb it (2b).
    msg = _assemble(FIXTURE.read_bytes())
    decision = brain.parse_decision(json.dumps(msg).encode())
    assert isinstance(decision, ToolCallDecision)
    assert decision.tool == "sniff"
    assert decision.args == TOOL_INPUT
    assert decision.tool_use_id == TOOL_USE_ID
    assert decision.usage is not None
    assert (decision.usage.input_tokens, decision.usage.output_tokens) == (1829, 155)


def test_signed_block_round_trips_byte_identical() -> None:
    """Invariant 6: the thinking block's text AND signature survive parse → serialise → parse
    unchanged. The signature is opaque bytes we never touch."""
    thinking = next(b for b in _content(_assemble(FIXTURE.read_bytes())) if b["type"] == "thinking")
    serialised = json.dumps(thinking)
    reparsed = json.loads(serialised)
    assert reparsed == thinking
    assert reparsed["signature"] == _expected_signature()  # verbatim through the round-trip


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
    marker = b'data: {"type":"content_block_stop","index":1'
    assert raw.count(marker) == 1
    stop = raw.index(marker)
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
    assert tool_use["input"] == TOOL_INPUT


@pytest.mark.parametrize("blank", [b"", b"   "])
def test_iter_sse_events_tolerates_trailing_blank_frames(blank: bytes) -> None:
    # A trailing blank/whitespace frame (SSE streams end with a blank line) must not raise.
    events = list(brain.iter_sse_events(FIXTURE.read_bytes() + b"\n\n" + blank))
    assert events[-1]["type"] == "message_stop"
