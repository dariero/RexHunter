"""W3.1 — the cut-stream guard: a turn that did not finish never reaches a tool or a terminal.

Before W3.1 the loop could act on a turn that never finished. A `tool_use` block keeps its start
`input` (`{}`) until its `content_block_stop`, and `parse_decision` acted on a `tool_use` whatever
the stop reason, so a cut `hunt_complete` closed the run `completed` with an empty catch (H1), a
cut `sniff` aborted as "invalid tool args" with no response bytes (H2), and a turn that stopped on
`max_tokens` after its tool call was dispatched and stored (H5).

Fork S1 (W3.1): the loop acts only on a turn whose `stop_reason` is `tool_use`. Any other value,
`None` included, is a `BrainParseError` carrying the raw bytes, so the run ends `aborted`,
"malformed provider response", with one `<brain>` ErrorEvent that names the stop reason. The docs:
the canonical loop runs tools "while `stop_reason == "tool_use"`" and "exits on any other stop
reason" (agents-and-tools/tool-use/how-tool-use-works), and a turn cut by `max_tokens` with an
incomplete tool use needs a retry with a higher `max_tokens`
(build-with-claude/handling-stop-reasons), both read 15 Nov 2026.

Fork T3 (W3.1): only a whole stream is a turn. A body that ends before `message_stop` is a
transport cut, retried within the budget like a reset mid-stream; a body that reached
`message_stop` with a block left open or no stop reason is malformed, and the run ends.

Every stream here is built in the test, from literals or by cutting the read-only
`fixtures/stream_thinking.sse` at a frame boundary; no file under tests/fixtures/ is written.
The request model is `brain.SMOKE_MODEL`. Zero network: a `MockTransport` answers every call.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import anthropic
import httpx2
import pytest

from rexhunter import brain, db
from rexhunter.events import (
    ErrorEvent,
    PreyCapturedEvent,
    ThinkingDelta,
    ToolCallEvent,
    ToolResultEvent,
    TrajectoryEvent,
)
from rexhunter.loop import run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]
RETRY_BUDGET = 2  # run_hunt's default, passed explicitly: a retried call makes 3 requests
MALFORMED = ("aborted", "malformed provider response")
BRAIN_CALL_FAILED = ("aborted", "brain call failed")
# The hand-authored stream test_stream_transport.py cuts: a thinking block (index 0, two
# thinking deltas), then a sniff tool_use (index 1), message_delta(tool_use), message_stop.
STREAM_THINKING = Path(__file__).parent / "fixtures" / "stream_thinking.sse"

SNIFF: dict[str, Any] = {
    "type": "tool_use",
    "id": "toolu_cut",
    "name": "sniff",
    "input": {"prey": "mock-gym"},
}
CUT_HUNT_COMPLETE: dict[str, Any] = {
    "type": "tool_use",
    "id": "toolu_done",
    "name": "hunt_complete",
    "input": {},  # a cut call's input: its start value, the catch never arrived
}
# The thinking docs' stand-in for work cut short (build-with-claude/thinking, "Progress updates
# between tool calls"): the last block of a turn that stops on max_tokens soon after a tool call.
INTERRUPTED: dict[str, Any] = {
    "type": "thinking",
    "thinking": "This part of the response was interrupted before it finished.",
    "signature": "SIG-interrupted==",
}


async def sniff(prey: str) -> str:
    """Sniff a territory for postings."""
    return f"posting:{prey}"


class Hunt(NamedTuple):
    events: list[TrajectoryEvent]
    row: tuple[str, str | None]
    requests: int


async def _hunt(
    tmp_path: Path, mock_anthropic: MockAnthropic, body: bytes, *, stream: bool
) -> Hunt:
    """One run_hunt whose every brain call is answered with `body`."""
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, content=body)

    reg = ToolRegistry()
    reg.tool(sniff)
    async with mock_anthropic(handler, api_key="offline") as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            model=brain.SMOKE_MODEL,
            registry=reg,
            thinking=brain.HUNT_THINKING if stream else None,
            stream=stream,
        )
        conn = await db.connect(tmp_path / "rex.db")
        try:
            run_id = await run_hunt(
                conn,
                territory="mock-gym",
                brain=brain_for("mock-gym"),
                registry=reg,
                retry_budget=RETRY_BUDGET,
                max_iterations=3,
                cost_ceiling_usd=1000.0,
                tool_timeout_s=5.0,
            )
            events = await db.read_events(conn, run_id)
            async with conn.execute(
                "SELECT outcome, abort_reason FROM runs WHERE id = ?", (run_id,)
            ) as cur:
                row = await cur.fetchone()
        finally:
            await conn.close()
    assert row is not None
    return Hunt(events, (str(row[0]), row[1]), len(requests))


def _plain(
    stop_reason: str | None,
    content: list[dict[str, Any]],
    *,
    usage: dict[str, Any] | None = None,
) -> bytes:
    """A whole Messages-API response, as the plain path reads it."""
    return json.dumps(
        {
            "id": "msg_w31",
            "type": "message",
            "role": "assistant",
            "model": brain.SMOKE_MODEL,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "content": content,
            "usage": {"input_tokens": 100, "output_tokens": 20} if usage is None else usage,
        }
    ).encode()


def _frame(data: dict[str, Any]) -> bytes:
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()


def _stream(
    content: list[dict[str, Any]],
    *,
    stop_reason: str | None = "tool_use",
    partial_json: str | None = None,
    close_tool_use: bool = True,
) -> list[bytes]:
    """A stream as frames: `message_start`, each block (start, deltas, stop), `message_delta`,
    `message_stop`. `partial_json` replaces each tool_use's input text; `close_tool_use=False`
    leaves out each tool_use's `content_block_stop`."""
    frames = [
        _frame(
            {
                "type": "message_start",
                "message": {
                    "model": brain.SMOKE_MODEL,
                    "content": [],
                    "usage": {"input_tokens": 100, "output_tokens": 1},
                },
            }
        )
    ]
    for i, block in enumerate(content):
        if block["type"] == "thinking":
            start: dict[str, Any] = {**block, "thinking": "", "signature": ""}
            deltas: list[dict[str, Any]] = [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": block["signature"]},
            ]
        else:  # tool_use: everything but `input` arrives at the start
            start = {**block, "input": {}}
            text = json.dumps(block["input"]) if partial_json is None else partial_json
            deltas = [{"type": "input_json_delta", "partial_json": text}]
        frames.append(_frame({"type": "content_block_start", "index": i, "content_block": start}))
        frames.extend(
            _frame({"type": "content_block_delta", "index": i, "delta": d}) for d in deltas
        )
        if close_tool_use or block["type"] != "tool_use":
            frames.append(_frame({"type": "content_block_stop", "index": i}))
    frames.append(
        _frame(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 20},
            }
        )
    )
    frames.append(_frame({"type": "message_stop"}))
    return frames


def _assembled(raw: bytes) -> bytes:
    """The bytes the streaming path hands `parse_decision`: the stream folded by the assembler."""
    assembler = brain.StreamAssembler()
    for event in brain.iter_sse_events(raw):
        assembler.feed(event)
    return json.dumps(assembler.assembled()).encode()


def _errors(hunt: Hunt) -> list[ErrorEvent]:
    return [e for e in hunt.events if isinstance(e, ErrorEvent)]


def _assert_nothing_dispatched(hunt: Hunt) -> None:
    acted = [
        e for e in hunt.events if isinstance(e, ToolCallEvent | ToolResultEvent | PreyCapturedEvent)
    ]
    assert acted == []


def _assert_one_malformed_turn(
    hunt: Hunt, *, raw: bytes, cause: str, content: list[dict[str, Any]] | None = None
) -> None:
    """One brain call, refused at the boundary: no dispatch, one fatal `<brain>` ErrorEvent with
    the bytes that crossed it and the cause in `detail`, the run ended malformed. `content` pins
    the turn those bytes hold, block by block (for the stream tests, whose bytes the assembler
    built)."""
    _assert_nothing_dispatched(hunt)
    assert hunt.requests == 1
    errors = _errors(hunt)
    assert len(errors) == 1, errors
    (error,) = errors
    assert (error.tool, error.retryable, error.error) == (
        "<brain>",
        False,
        "malformed provider response",
    )
    assert error.raw_response == raw
    if content is not None:
        assert json.loads(raw)["content"] == content
    assert cause in (error.detail or "")
    assert hunt.row == MALFORMED


# ── S1: the loop acts only on a turn that stopped for its tool call ──────────────────


@pytest.mark.parametrize(
    "stop_reason",
    [
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "pause_turn",
        "refusal",
        "model_context_window_exceeded",
        None,
    ],
)
async def test_a_plain_turn_that_did_not_stop_for_its_tool_call_ends_the_run_with_its_bytes(
    stop_reason: str | None, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # H5 on the plain path: a whole, valid sniff call whose turn stopped for another reason.
    body = _plain(stop_reason, [SNIFF])
    hunt = await _hunt(tmp_path, mock_anthropic, body, stream=False)
    _assert_one_malformed_turn(hunt, raw=body, cause=f"stop_reason={stop_reason!r}")


async def test_a_turn_that_did_not_stop_for_its_tool_call_ends_typed_whatever_its_usage_holds(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # The stop reason is checked before the usage is read: a token count that is not a number
    # (`int()` in `_extract_usage` raises ValueError) must not carry a cut turn to the `<loop>`
    # catch-all without its bytes.
    body = _plain("max_tokens", [SNIFF], usage={"input_tokens": "x", "output_tokens": 20})
    hunt = await _hunt(tmp_path, mock_anthropic, body, stream=False)
    _assert_one_malformed_turn(hunt, raw=body, cause="stop_reason='max_tokens'")


async def test_a_streamed_turn_cut_by_max_tokens_after_its_tool_call_is_never_dispatched(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # H5 on the stream, in the thinking docs' shape: the tool_use is whole, then the turn stops on
    # max_tokens with the interrupted progress-update block last.
    body = b"".join(_stream([SNIFF, INTERRUPTED], stop_reason="max_tokens"))
    hunt = await _hunt(tmp_path, mock_anthropic, body, stream=True)
    _assert_one_malformed_turn(
        hunt,
        raw=_assembled(body),
        cause="stop_reason='max_tokens'",
        content=[SNIFF, INTERRUPTED],  # the whole tool_use, then the interrupted block, last
    )


async def test_a_cut_hunt_complete_never_completes_the_hunt_on_the_plain_path(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # H1: before W3.1, `HuntComplete.model_validate({})` gave `catch=[]` and the run completed.
    body = _plain("max_tokens", [CUT_HUNT_COMPLETE])
    hunt = await _hunt(tmp_path, mock_anthropic, body, stream=False)
    _assert_one_malformed_turn(hunt, raw=body, cause="stop_reason='max_tokens'")


async def test_a_cut_hunt_complete_never_completes_the_hunt_on_the_stream(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # H1 on the stream: the catch's first characters arrive, the block's content_block_stop never
    # does, then the stream finishes on max_tokens. Its start input `{}` must not become a catch.
    frames = _stream(
        [CUT_HUNT_COMPLETE],
        stop_reason="max_tokens",
        partial_json='{"catch": ["po',
        close_tool_use=False,
    )
    body = b"".join(frames)
    hunt = await _hunt(tmp_path, mock_anthropic, body, stream=True)
    _assert_one_malformed_turn(
        hunt,
        raw=_assembled(body),
        cause="stop_reason='max_tokens'",
        content=[CUT_HUNT_COMPLETE],  # the block keeps its start input: the partial catch is unused
    )


# ── T3: a stream that did not finish is never a turn ────────────────────────────────


def _cut_before(raw: bytes, marker: bytes) -> bytes:
    """`raw` cut at the frame boundary before `marker`: whole frames only, as a body that closes
    cleanly before its end would arrive."""
    at = raw.index(marker)
    return raw[: raw.rindex(b"\n\n", 0, at) + 2]


@pytest.mark.parametrize(
    ("marker", "stop_reason", "open_blocks", "tool_input"),
    [
        (b'data: {"type":"content_block_stop","index":1}', None, [1], {}),
        (b'data: {"type":"message_delta"', None, [], {"prey": "mock-gym"}),
        (b'data: {"type":"message_stop"}', "tool_use", [], {"prey": "mock-gym"}),
    ],
    ids=["before-the-tool-use-stop", "before-message-delta", "before-message-stop"],
)
async def test_a_stream_cut_before_message_stop_is_retried_and_never_dispatched(
    marker: bytes,
    stop_reason: str | None,
    open_blocks: list[int],
    tool_input: dict[str, Any],
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
) -> None:
    # H2: before W3.1 the first cut dispatched `sniff` with its start input `{}` ("invalid tool
    # args", no response bytes) and the other two dispatched it outright. A body that ends before
    # `message_stop` is a transport cut: retried within the budget, like a reset mid-stream.
    body = _cut_before(STREAM_THINKING.read_bytes(), marker)
    hunt = await _hunt(tmp_path, mock_anthropic, body, stream=True)
    _assert_nothing_dispatched(hunt)
    assert hunt.requests == RETRY_BUDGET + 1
    errors = _errors(hunt)
    assert len(errors) == RETRY_BUDGET + 1, errors
    for error in errors:
        assert (error.tool, error.retryable) == ("<brain>", True)
        assert error.raw_response is not None
        assert error.raw_response == _assembled(body)
        stored = json.loads(error.raw_response)  # what was kept is what arrived before the cut
        assert stored["stop_reason"] == stop_reason
        assert stored["content"][1]["input"] == tool_input
        cause = f"stream cut before message_stop (stop_reason={stop_reason!r}, open blocks "
        assert f"{cause}{open_blocks})" in (error.detail or "")
    assert hunt.row == BRAIN_CALL_FAILED
    # The thinking each attempt relayed before its cut stays in the log, before its error.
    relayed = [
        e.text if isinstance(e, ThinkingDelta) else "<error>"
        for e in hunt.events
        if isinstance(e, ThinkingDelta | ErrorEvent)
    ]
    assert relayed == [
        "The user wants me to scout ",
        "the mock-gym territory. I'll call sniff.",
        "<error>",
    ] * (RETRY_BUDGET + 1)


async def test_a_stream_that_ended_with_a_block_left_open_is_malformed_not_retried(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # The T3 split: `message_stop` arrived, so the provider sent the whole body, but the tool_use
    # never got its `content_block_stop`. That is a malformed response (fatal, one request), not
    # a network blip to retry; before W3.1 it dispatched `sniff` with its start input `{}`.
    body = b"".join(_stream([SNIFF], close_tool_use=False))
    hunt = await _hunt(tmp_path, mock_anthropic, body, stream=True)
    _assert_one_malformed_turn(
        hunt,
        raw=_assembled(body),
        cause="open blocks [0]",
        content=[{**SNIFF, "input": {}}],  # the open block keeps its start input
    )


async def test_a_stream_that_ended_with_no_stop_reason_is_malformed_not_retried(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # The other T3 arm: every block closed and `message_stop` arrived, but no `message_delta`
    # carried a stop reason. The stream ended without a whole turn, so the T3 check ends it.
    body = b"".join(_stream([SNIFF], stop_reason=None))
    hunt = await _hunt(tmp_path, mock_anthropic, body, stream=True)
    _assert_one_malformed_turn(
        hunt,
        raw=_assembled(body),
        cause="stream ended without a whole turn: open blocks [], stop_reason=None",
        content=[SNIFF],
    )
