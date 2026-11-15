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
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import anthropic
import httpx2
import pytest

from rexhunter import brain, db, hunt_smoke_thinking
from rexhunter.events import (
    ErrorEvent,
    PreyCapturedEvent,
    ThinkingDelta,
    ToolCallEvent,
    ToolResultEvent,
    TrajectoryEvent,
)
from rexhunter.loop import BrainCallError, run_hunt
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
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
    body: bytes,
    *,
    stream: bool,
    respond: Callable[[], httpx2.Response] | None = None,
) -> Hunt:
    """One run_hunt whose every brain call is answered with `body` (or by `respond()`)."""
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, content=body) if respond is None else respond()

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


# ── P1: a closed tool_use whose input is not JSON ends the run with its bytes ──────────

# A closed block whose input does not parse: a complete key, then a value cut short (the shape
# fine-grained streaming can send). Under default buffering a cut more likely leaves a whole
# pair without its closing brace; both fail `json.loads` the same way.
BAD_INPUT = '{"prey": "mo'


@pytest.mark.parametrize(
    ("stop_reason", "cause"),
    [
        # A cut by max_tokens. S names it too, so this case holds without P's check (M7).
        ("max_tokens", "stop_reason='max_tokens'"),
        # A whole stream that says `tool_use` yet closed a block with broken input: P alone.
        ("tool_use", BAD_INPUT),
    ],
    ids=["max_tokens", "tool_use"],
)
async def test_a_tool_use_input_that_is_not_json_ends_the_run_with_its_bytes(
    stop_reason: str, cause: str, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # H3: before W3.1 `_finalize` raised `json.JSONDecodeError` at the block's stop, which is not
    # a BrainParseError, so the run ended in the `<loop>` catch-all with no response bytes.
    body = b"".join(_stream([SNIFF], stop_reason=stop_reason, partial_json=BAD_INPUT))
    hunt = await _hunt(tmp_path, mock_anthropic, body, stream=True)
    assert [(e.tool, e.error) for e in _errors(hunt)] == [
        ("<brain>", "malformed provider response")
    ]
    _assert_one_malformed_turn(
        hunt,
        raw=_assembled(body),
        cause=cause,
        content=[{**SNIFF, "input": {}}],  # the failed block keeps its start input
    )


async def test_the_thinking_harness_keeps_its_fixtures_when_a_tool_input_is_not_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # H3's second half: hunt_smoke_thinking re-assembles every captured stream in
    # `_report_findings` before `write_fixtures`, so a raising fold cost a paid run its
    # fixtures. Offline: a MockTransport answers, the fixtures go to `tmp_path`.
    # run_live_hunt makes its run DB with `tempfile.mkdtemp()`: under tmp_path here, so this test
    # adds no `rexhunter-think-*` dir to the system temp dir.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    body = b"".join(_stream([SNIFF], stop_reason="max_tokens", partial_json=BAD_INPUT))

    def handler(request: httpx2.Request) -> httpx2.Response:
        if str(request.url) == brain.COUNT_TOKENS_URL:
            return httpx2.Response(200, json={"input_tokens": 1})  # the free pre-flight
        return httpx2.Response(200, content=body)

    summary = await hunt_smoke_thinking.run_live_hunt(
        "offline-key", inner=httpx2.MockTransport(handler), fixture_dir=tmp_path
    )
    fixtures = sorted((tmp_path / f"hunt_{summary['run_id']}").glob("call_*.json"))
    assert [p.read_bytes() for p in fixtures] == [body]  # the raw stream, kept
    assert "[truncation] call 1" in capsys.readouterr().out  # the reporter ran to its end
    assert (summary["outcome"], summary["reason"], summary["brain_calls"]) == (
        "aborted",
        "malformed provider response",
        1,
    )


async def test_a_tool_use_input_that_json_loads_refuses_with_a_value_error_ends_typed(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # Valid JSON that `json.loads` still refuses: an integer over 4300 digits raises a plain
    # ValueError (CPython's int max-str-digits limit), not a JSONDecodeError. P catches it too.
    huge = '{"prey": ' + "1" * 4301 + "}"
    body = b"".join(_stream([SNIFF], partial_json=huge))
    hunt = await _hunt(tmp_path, mock_anthropic, body, stream=True)
    assert [(e.tool, e.error) for e in _errors(hunt)] == [
        ("<brain>", "malformed provider response")
    ]
    _assert_one_malformed_turn(
        hunt,
        raw=_assembled(body),
        cause="does not parse as JSON",
        content=[{**SNIFF, "input": {}}],
    )


async def test_a_bad_tool_input_in_a_stream_cut_short_is_retried_as_a_cut(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # The order of the checks: T runs before P. A body that ends before `message_stop` is a cut,
    # retried within the budget, even when a block it did close holds input that does not parse.
    body = b"".join(_stream([SNIFF], partial_json=BAD_INPUT)[:-2])  # no message_delta, message_stop
    hunt = await _hunt(tmp_path, mock_anthropic, body, stream=True)
    _assert_nothing_dispatched(hunt)
    assert hunt.requests == RETRY_BUDGET + 1
    errors = _errors(hunt)
    assert len(errors) == RETRY_BUDGET + 1, errors
    for error in errors:
        assert (error.tool, error.retryable) == ("<brain>", True)
        assert error.raw_response == _assembled(body)
        assert "stream cut before message_stop" in (error.detail or "")
    assert hunt.row == BRAIN_CALL_FAILED


# ── Z1: an SSE error frame fails the brain call, with the verdict its type stands for ─────


def _error_frame(error_type: str) -> dict[str, Any]:
    """An SSE `error` event's data, in the shape the streaming docs show."""
    return {"type": "error", "error": {"type": error_type, "message": "An error mid-stream"}}


def _error_data(frame: dict[str, Any]) -> bytes:
    """The frame's `data` bytes as this test sends them: uneven spacing and a `\\u` escape in the
    message, bytes no `json.dumps` call reproduces (spaced or compact). So `raw_response` can match
    only if it is the frame's own text, never a re-serialisation of the parsed frame. (No error
    frame is recorded; the streaming docs' example is spaced.)"""
    error = frame["error"]
    message = str(error["message"]).replace("-", "\\u002d")
    return (
        f'{{"type":"{frame["type"]}", "error":{{"type": "{error["type"]}","message":"{message}"}}}}'
    ).encode()


def _error_event(frame: dict[str, Any]) -> bytes:
    return b"event: error\ndata: " + _error_data(frame) + b"\n\n"


def _error_after_the_tool_use(frame: dict[str, Any]) -> bytes:
    """A sniff tool_use that closed, then the error frame in place of message_delta/stop."""
    return b"".join(_stream([SNIFF])[:-2]) + _error_event(frame)


def _assert_the_frames_verdict(hunt: Hunt, *, frame: dict[str, Any], retryable: bool) -> None:
    """Nothing dispatched; one `<brain>` ErrorEvent per attempt carrying the frame's own data
    bytes and naming its type; retried within the budget only when the type is retryable."""
    _assert_nothing_dispatched(hunt)
    attempts = RETRY_BUDGET + 1 if retryable else 1
    assert hunt.requests == attempts
    errors = _errors(hunt)
    assert len(errors) == attempts, errors
    for error in errors:
        assert (error.tool, error.retryable) == ("<brain>", retryable)
        assert error.raw_response == _error_data(frame)  # the frame's own `data` bytes, as sent
        assert f"stream error frame: {frame['error']['type']!r}" in (error.detail or "")
    assert hunt.row == BRAIN_CALL_FAILED


async def test_an_overloaded_error_frame_after_the_tool_call_is_retried_never_dispatched(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # H4: before W3.1 `feed` ignored the error frame and the closed tool_use was dispatched. An
    # `overloaded_error` "would normally correspond to an HTTP 529": retried, like a 529.
    frame = _error_frame("overloaded_error")
    hunt = await _hunt(tmp_path, mock_anthropic, _error_after_the_tool_use(frame), stream=True)
    _assert_the_frames_verdict(hunt, frame=frame, retryable=True)


@pytest.mark.parametrize("error_type", ["invalid_request_error", "teapot_error"])
async def test_a_fatal_error_frame_is_not_retried(
    error_type: str, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # A 400-class type is fatal, as its status is; a type with no row fails closed (fatal).
    frame = _error_frame(error_type)
    hunt = await _hunt(tmp_path, mock_anthropic, _error_after_the_tool_use(frame), stream=True)
    _assert_the_frames_verdict(hunt, frame=frame, retryable=False)


async def test_a_fatal_error_frame_before_the_tool_use_closes_is_not_retried(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # The frame's verdict wins wherever it sits: here the body is also unfinished, which T alone
    # would retry.
    frame = _error_frame("invalid_request_error")
    body = b"".join(_stream([SNIFF])[:3]) + _error_event(frame)  # message_start, block, a delta
    hunt = await _hunt(tmp_path, mock_anthropic, body, stream=True)
    _assert_the_frames_verdict(hunt, frame=frame, retryable=False)


async def test_an_error_frame_that_ends_the_body_without_a_blank_line_is_still_seen(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # `_drive_stream` folds a final frame with no trailing blank line on a second path; the error
    # frame is caught on that path too.
    frame = _error_frame("invalid_request_error")
    body = _error_after_the_tool_use(frame).rstrip(b"\n")
    hunt = await _hunt(tmp_path, mock_anthropic, body, stream=True)
    _assert_the_frames_verdict(hunt, frame=frame, retryable=False)


async def test_a_conflict_error_frame_is_fatal(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # 409 `conflict_error`: fatal under the 429-or-5xx rule, as a 409 status already is.
    frame = _error_frame("conflict_error")
    hunt = await _hunt(tmp_path, mock_anthropic, _error_after_the_tool_use(frame), stream=True)
    _assert_the_frames_verdict(hunt, frame=frame, retryable=False)


def test_the_error_type_table_has_the_rows_of_the_errors_page() -> None:
    # https://platform.claude.com/docs/en/api/errors ("HTTP errors"), read 15 Nov 2026, in the
    # page's order.
    assert brain.STREAM_ERROR_STATUS == {
        "invalid_request_error": 400,
        "authentication_error": 401,
        "billing_error": 402,
        "permission_error": 403,
        "not_found_error": 404,
        "conflict_error": 409,
        "request_too_large": 413,
        "rate_limit_error": 429,
        "api_error": 500,
        "timeout_error": 504,
        "overloaded_error": 529,
    }


async def _drop(_text: str) -> None:
    """A thinking sink that keeps nothing (the adapter-level tests read only the error)."""


async def _the_brain_call_error(mock_anthropic: MockAnthropic, body: bytes) -> BrainCallError:
    """One streamed adapter call answered with `body`; the BrainCallError it raises."""

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=body)

    reg = ToolRegistry()
    reg.tool(sniff)
    async with mock_anthropic(handler, api_key="offline") as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            model=brain.SMOKE_MODEL,
            registry=reg,
            thinking=brain.HUNT_THINKING,
            stream=True,
        )
        with pytest.raises(BrainCallError) as raised:
            await brain_for("mock-gym")([], _drop)
    return raised.value


@pytest.mark.parametrize(
    ("error_type", "status", "retryable"),
    [("overloaded_error", 529, True), ("teapot_error", None, False)],
)
async def test_a_stream_error_carries_the_status_its_type_stands_for(
    error_type: str, status: int | None, retryable: bool, mock_anthropic: MockAnthropic
) -> None:
    # ErrorEvent keeps no status, so this reads the BrainCallError the adapter raises.
    frame = _error_frame(error_type)
    error = await _the_brain_call_error(mock_anthropic, _error_after_the_tool_use(frame))
    assert (error.status, error.retryable, error.raw_response) == (
        status,
        retryable,
        _error_data(frame),
    )


@pytest.mark.parametrize(
    ("error_type", "status", "retryable"),
    [
        ("rate_limit_error", 429, True),
        ("api_error", 500, True),
        ("timeout_error", 504, True),
        ("overloaded_error", 529, True),
        ("invalid_request_error", 400, False),
        ("authentication_error", 401, False),
        ("conflict_error", 409, False),
        ("request_too_large", 413, False),
    ],
)
async def test_an_error_frame_is_retryable_exactly_when_its_status_is(
    error_type: str, status: int, retryable: bool, mock_anthropic: MockAnthropic
) -> None:
    # The rule `brain_call_error` applies to a status: 429 or 5xx is transient, any other fatal.
    error = await _the_brain_call_error(
        mock_anthropic, _error_after_the_tool_use(_error_frame(error_type))
    )
    assert (error.status, error.retryable) == (status, retryable)


class _ErrorFrameThenReset(httpx2.AsyncByteStream):
    """A 200 body that sends `head`, then the connection resets."""

    def __init__(self, head: bytes) -> None:
        self._head = head

    async def __aiter__(self):  # type: ignore[override]
        yield self._head
        raise httpx2.ReadError("connection reset after the error frame")

    async def aclose(self) -> None:
        pass


async def test_the_first_error_frame_ends_the_read(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # Z-stop: nothing after an error frame can make the turn usable, and the docs do not say what
    # follows one. Reading on would let a later transport failure (here a reset) replace the
    # frame's verdict with a transport retry.
    frame = _error_frame("invalid_request_error")
    head = _error_after_the_tool_use(frame)
    hunt = await _hunt(
        tmp_path,
        mock_anthropic,
        b"",
        stream=True,
        respond=lambda: httpx2.Response(200, stream=_ErrorFrameThenReset(head)),
    )
    _assert_the_frames_verdict(hunt, frame=frame, retryable=False)


@pytest.mark.parametrize("where", ["after-message-stop", "inside-a-whole-turn"])
async def test_an_error_frame_in_a_stream_that_reached_message_stop_is_never_dispatched(
    where: str, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # H4's own dispatch shape: the turn is whole, so T passes it and only Z keeps it from running.
    # Either after `message_stop`, or between the tool_use's stop and a `message_delta(tool_use)`
    # and `message_stop`.
    frame = _error_frame("invalid_request_error")
    frames = _stream([SNIFF])
    if where == "after-message-stop":
        body = b"".join(frames) + _error_event(frame)
    else:
        body = b"".join(frames[:-2]) + _error_event(frame) + b"".join(frames[-2:])
    hunt = await _hunt(tmp_path, mock_anthropic, body, stream=True)
    _assert_the_frames_verdict(hunt, frame=frame, retryable=False)


async def test_the_thinking_harness_keeps_its_fixture_when_a_stream_carries_an_error_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `feed` records an error frame and never raises, so hunt_smoke_thinking's reporter, which
    # re-folds every captured stream, still reaches `write_fixtures`. Offline, under tmp_path.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    body = _error_after_the_tool_use(_error_frame("invalid_request_error"))

    def handler(request: httpx2.Request) -> httpx2.Response:
        if str(request.url) == brain.COUNT_TOKENS_URL:
            return httpx2.Response(200, json={"input_tokens": 1})  # the free pre-flight
        return httpx2.Response(200, content=body)

    summary = await hunt_smoke_thinking.run_live_hunt(
        "offline-key", inner=httpx2.MockTransport(handler), fixture_dir=tmp_path
    )
    fixtures = sorted((tmp_path / f"hunt_{summary['run_id']}").glob("call_*.json"))
    assert [p.read_bytes() for p in fixtures] == [body]  # the raw stream, kept
    assert (summary["outcome"], summary["reason"], summary["brain_calls"]) == (
        *BRAIN_CALL_FAILED,
        1,
    )


@pytest.mark.parametrize(
    ("data", "named"),
    [
        (b'{"type":"error"}', "None"),
        (b'{"type":"error", "error":"boom"}', "None"),
        (b'{"type":"error", "error":{"type": ["x"]}}', "['x']"),
    ],
    ids=["no-error-object", "error-not-an-object", "type-not-a-string"],
)
async def test_a_malformed_error_frame_is_fatal_with_its_bytes(
    data: bytes, named: str, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # Untrusted input (invariant 3): an error frame whose `error` is missing, not an object, or
    # names a type that is not a string fails closed (fatal, status None), never `<loop>`.
    body = b"".join(_stream([SNIFF])[:-2]) + b"event: error\ndata: " + data + b"\n\n"
    hunt = await _hunt(tmp_path, mock_anthropic, body, stream=True)
    _assert_nothing_dispatched(hunt)
    assert hunt.requests == 1
    errors = _errors(hunt)
    assert len(errors) == 1, errors
    (error,) = errors
    assert (error.tool, error.retryable, error.raw_response) == ("<brain>", False, data)
    assert f"stream error frame: {named}" in (error.detail or "")
    assert hunt.row == BRAIN_CALL_FAILED
    raised = await _the_brain_call_error(mock_anthropic, body)
    assert (raised.status, raised.retryable, raised.raw_response) == (None, False, data)


def test_the_assembler_keeps_the_first_error_frame() -> None:
    # The verdict and `raw_response` must come from the same frame: `_drive_stream` keeps the first
    # frame's bytes, so the assembler keeps the first frame too, even with Z-stop out of the way.
    first, second = _error_frame("overloaded_error"), _error_frame("api_error")
    assembler = brain.StreamAssembler()
    assert assembler.feed(first) is None
    assert assembler.feed(second) is None
    assert assembler.error_frame == first
