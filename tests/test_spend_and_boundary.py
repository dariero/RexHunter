"""W3.4 Part A: the spend and the boundary, one section per commit.

Every response is built here from literals; nothing under tests/fixtures/ is read or written. The
request model is `brain.SMOKE_MODEL`. Zero network: a `MockTransport` answers every call. A test
whose name starts `test_guard_` passed before W3.4 too: it pins what must not change.
"""

import json
import random
import sys
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import anthropic
import httpx2
import pytest

from rexhunter import brain, cost, db, loop
from rexhunter.events import BrainExchangeEvent, ErrorEvent, TrajectoryEvent, UsageEvent
from rexhunter.loop import (
    Brain,
    BrainCallError,
    Decision,
    NeedsHelp,
    RetryableToolError,
    ToolCallDecision,
    run_hunt,
)
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]
Respond = Callable[[httpx2.Request], httpx2.Response]

MODEL = brain.SMOKE_MODEL
RETRY_BUDGET = 2  # run_hunt's default, passed explicitly: a retried call makes 3 requests
MALFORMED = ("aborted", "malformed provider response")
BRAIN_CALL_FAILED = ("aborted", "brain call failed")
E, U, ERR = "brain_exchange", "usage", "error"

SNIFF: dict[str, Any] = {
    "type": "tool_use",
    "id": "toolu_w34",
    "name": "sniff",
    "input": {"prey": "mock-gym"},
}
# What a whole literal stream reports: 100 input tokens; 1 output token at message_start and 20 by
# the end (message_delta's cumulative count).
START_USAGE: dict[str, Any] = {"input_tokens": 100, "output_tokens": 1}


async def sniff(prey: str) -> str:
    """Sniff a territory for postings."""
    return f"posting:{prey}"


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.tool(sniff)
    return reg


def _plain(stop_reason: str, content: list[dict[str, Any]], usage: dict[str, Any]) -> bytes:
    """A whole Messages-API response, as the plain path reads it."""
    return json.dumps(
        {
            "id": "msg_w34",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "content": content,
            "usage": usage,
        }
    ).encode()


def _frame(data: dict[str, Any]) -> bytes:
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()


def _message_start(usage: dict[str, Any]) -> bytes:
    return _frame(
        {"type": "message_start", "message": {"model": MODEL, "content": [], "usage": usage}}
    )


def _message_delta(usage: dict[str, Any]) -> bytes:
    return _frame(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": usage,
        }
    )


# A whole streamed sniff turn, frame by frame.
SNIFF_STREAM = [
    _message_start(START_USAGE),
    _frame({"type": "content_block_start", "index": 0, "content_block": {**SNIFF, "input": {}}}),
    _frame(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(SNIFF["input"])},
        }
    ),
    _frame({"type": "content_block_stop", "index": 0}),
    _message_delta({"output_tokens": 20}),
    _frame({"type": "message_stop"}),
]


def _ok(body: bytes) -> Respond:
    return lambda _request: httpx2.Response(200, content=body)


class Hunt(NamedTuple):
    events: list[TrajectoryEvent]  # every event, BrainExchangeEvents included
    row: tuple[str, str | None]
    requests: int


async def _hunt(
    tmp_path: Path, mock_anthropic: MockAnthropic, respond: Respond, *, stream: bool
) -> Hunt:
    """One run_hunt whose every brain request is answered by `respond`."""
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return respond(request)

    reg = _registry()
    async with mock_anthropic(handler, api_key="offline") as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            model=MODEL,
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


def _types(events: Sequence[TrajectoryEvent]) -> list[str]:
    return [e.type for e in events]


def _errors(events: Sequence[TrajectoryEvent]) -> list[ErrorEvent]:
    return [e for e in events if isinstance(e, ErrorEvent)]


def _usages(events: Sequence[TrajectoryEvent]) -> list[tuple[int, int]]:
    return [(e.input_tokens, e.output_tokens) for e in events if isinstance(e, UsageEvent)]


def _assert_malformed(hunt: Hunt, *, types: list[str], cause: str) -> ErrorEvent:
    """One attempt, refused "malformed provider response" with its bytes and its exchange first."""
    assert _types(hunt.events) == types
    assert hunt.row == MALFORMED
    assert hunt.requests == 1  # fatal: never retried
    [error] = _errors(hunt.events)
    assert (error.tool, error.error, error.retryable) == (
        "<brain>",
        "malformed provider response",
        False,
    )
    assert error.raw_response  # the bytes (invariant 6)
    assert cause in (error.detail or "")
    [exchange] = [e for e in hunt.events if isinstance(e, BrainExchangeEvent)]
    assert exchange.raw_response  # the attempt's bytes as received (W3.3)
    return error


# ── counts: a token count crosses the boundary only as a bounded non-negative int ──────────
# W3.2-RV12 (a negative count lowered the folds), the count types (a bool, a numeric string or a
# float read as a number), W3.2-RV1b (a 315 to 4300 digit count on an accepted turn was written and
# a later fold raised) and W3.1-O5 (an accepted turn's count that is not a number reached the
# `<loop>` catch-all without its bytes). `brain.MAX_TOKEN_COUNT` is 10,000,000: one past it is
# refused on both paths, the bound itself is read.

BAD_COUNTS = [True, "100", 100.0, -1, 10_000_001, 10**400, "x", None, [1]]
BAD_IDS = [
    "bool",
    "numeric-string",
    "float",
    "negative",
    "bound+1",
    "401-digits",
    "word",
    "null",
    "list",
]


@pytest.mark.parametrize("bad", BAD_COUNTS, ids=BAD_IDS)
async def test_a_refused_turn_whose_count_is_not_a_token_count_is_not_counted(
    bad: object, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # A turn refused for its stop reason. Before W3.4 `int()` read a bool, a numeric string, a
    # float, a negative count and one past the bound as tokens; a word, null and a list went
    # uncounted already, and so did a 401-digit count (the pricing check W3.4 drops).
    body = _plain("max_tokens", [SNIFF], {"input_tokens": bad, "output_tokens": 20})
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=False)
    error = _assert_malformed(hunt, types=[E, ERR], cause="stop_reason='max_tokens'")
    assert "usage not counted" in (error.detail or "")


async def test_a_negative_count_never_lowers_the_fold(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # W3.2-RV12: a refused turn's negative count was counted, so the spend folds went down.
    body = _plain("max_tokens", [SNIFF], {"input_tokens": -5_000_000, "output_tokens": 20})
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=False)
    assert cost.fold_cost(hunt.events) == 0.0
    assert _usages(hunt.events) == []


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
@pytest.mark.parametrize("bad", BAD_COUNTS, ids=BAD_IDS)
async def test_an_accepted_turn_whose_count_is_not_a_token_count_is_refused_with_its_bytes(
    bad: object, stream: bool, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # W3.1-O5 (a word, null, a list: `int()` raised, so the run ended at the `<loop>` catch-all
    # without its bytes), the count types and a negative count or one past the bound (the turn was
    # dispatched and the count written), and W3.2-RV1b (401 digits: written, then the next
    # attempt's fold raised). Now the turn is refused with its bytes and writes no UsageEvent.
    usage = {"input_tokens": bad, "output_tokens": 20}
    if stream:
        body = b"".join([_message_start(usage), *SNIFF_STREAM[1:]])
    else:
        body = _plain("tool_use", [SNIFF], usage)
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=stream)
    error = _assert_malformed(hunt, types=[E, ERR], cause="usage not counted")
    if not stream:
        assert error.raw_response == body


async def test_a_huge_count_on_an_accepted_turn_is_never_written(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # W3.2-RV1b: a 401-digit count on an accepted turn was written; the next attempt's fold raised
    # OverflowError and the run ended at the `<loop>` catch-all.
    body = _plain("tool_use", [SNIFF], {"input_tokens": 10**400, "output_tokens": 20})
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=False)
    assert hunt.row == MALFORMED
    assert _usages(hunt.events) == []
    assert cost.fold_cost(hunt.events) == 0.0  # the fold never raises


@pytest.mark.parametrize("stop_reason", ["tool_use", "max_tokens"], ids=["accepted", "refused"])
@pytest.mark.parametrize("digits", [4301, 5000])
async def test_guard_a_count_over_4300_digits_is_refused_at_the_envelope(
    digits: int, stop_reason: str, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # Past Python's default `int_max_str_digits` (4300), `repr` of an int raises ValueError, so a
    # refusal whose message showed such a count would raise inside itself. pydantic's JSON parser
    # refuses the literal first ("number out of range"): the body fails the envelope, refused with
    # its bytes and its exchange, not counted. `json.dumps` cannot write the literal, so it is put
    # into the body's bytes here.
    marker = "COUNT_LITERAL"
    body = _plain(stop_reason, [SNIFF], {"input_tokens": marker, "output_tokens": 20})
    body = body.replace(f'"{marker}"'.encode(), b"9" * digits)
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=False)
    error = _assert_malformed(hunt, types=[E, ERR], cause="number out of range")
    assert error.raw_response == body


@pytest.mark.parametrize(
    ("usage", "counted"),
    [
        ({"input_tokens": 10_000_000, "output_tokens": 10_000_000}, (10_000_000, 10_000_000)),
        ({"input_tokens": 100}, (100, 0)),  # an absent count reads as 0, as before W3.4
        ({}, (0, 0)),
        (
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": None,
                "service_tier": "standard",
                "server_tool_use": {"web_search_requests": 1},
            },
            (100, 20),
        ),
    ],
    ids=["the-bound", "absent-output", "empty", "other-keys"],
)
async def test_guard_a_token_count_that_reads_is_counted_as_before(
    usage: dict[str, Any],
    counted: tuple[int, int],
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
) -> None:
    # Only `input_tokens` and `output_tokens` are read; the other keys of `usage` are not.
    body = _plain("tool_use", [{**SNIFF, "name": "needs_help", "input": {}}], usage)
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=False)
    assert hunt.row == ("needs_help", None)
    assert _usages(hunt.events) == [counted]


# ── N1: a null count in message_delta keeps the earlier count, as the SDK does (W3.2-N1) ─────
# The assembler wrote every key of message_delta's `usage` over the counts so far, a null one too,
# so a null `input_tokens` replaced message_start's count. The SDK's accumulator keeps the earlier
# count when the new one is null (anthropic 2.6.0 lib/streaming/_messages.py:547-556). A count that
# is not null still replaces the earlier one: they are cumulative (the streaming page, :334).

NULL_COUNTS_DELTA = _message_delta(
    {"input_tokens": None, "cache_read_input_tokens": None, "output_tokens": 20}
)


async def test_a_null_count_in_message_delta_keeps_message_starts(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    body = b"".join([*SNIFF_STREAM[:4], NULL_COUNTS_DELTA, SNIFF_STREAM[5]])
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    assert _usages(hunt.events)[:1] == [(100, 20)]
    assert _types(hunt.events)[:4] == [E, U, "tool_call", "tool_result"]


async def test_a_null_count_in_message_delta_keeps_message_starts_on_a_refused_stream(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # A stream cut before message_stop (W3.1-T, retried): each attempt is counted with
    # message_start's input count, not left uncounted for a null.
    body = b"".join([*SNIFF_STREAM[:4], NULL_COUNTS_DELTA])
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    assert _types(hunt.events) == [E, U, ERR] * (RETRY_BUDGET + 1)
    assert _usages(hunt.events) == [(100, 20)] * (RETRY_BUDGET + 1)


async def test_guard_a_count_in_message_delta_still_replaces_message_starts(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    delta = _message_delta({"input_tokens": 150, "output_tokens": 20})
    body = b"".join([*SNIFF_STREAM[:4], delta, SNIFF_STREAM[5]])
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    assert _usages(hunt.events)[:1] == [(150, 20)]


# ── frames: a malformed stream frame ends the call typed, with its bytes (W3.1-O1 to O4) ─────
# Before W3.4 `json.loads` (O1), `.get` on a value that is not an object (O2), a missing key (O3)
# and `int()`, `dict()` or `dict.update` on a value of the wrong type (O4) raised past
# `_drive_stream`, so the run ended at the `<loop>` catch-all without its bytes. frames-1 with 1a:
# a frame whose closing blank line never came, last on a body that ended before `message_stop`,
# is the cut (retried, W3.1-T); every other malformed frame ends "malformed provider response",
# with the bytes assembled so far, the counts so far and the exchange. An `index` is an int, and
# a `content_block`, `delta`, `message` or `usage` an object. An absent `message` or `usage`, an
# unknown event type or field, and a delta or stop for an index with no start are read as before.


def _data(text: str) -> bytes:
    """One SSE frame whose `data:` line is `text`, as sent (it need not be JSON)."""
    return f"data: {text}\n\n".encode()


def _with(at: int, *inserted: bytes, replace: bool = False) -> bytes:
    """SNIFF_STREAM with `inserted` before index `at`, or put in place of that frame."""
    frames = list(SNIFF_STREAM)
    frames[at : at + 1 if replace else at] = inserted
    return b"".join(frames)


@pytest.mark.parametrize(
    "body",
    [
        _with(1, _data("{not json")),
        b"".join(SNIFF_STREAM[:-1]) + _data('{"type": "message_st'),
        b"".join(SNIFF_STREAM) + b'data: {"type": "ping"',
    ],
    ids=["terminated-mid-stream", "terminated-last", "unterminated-after-message-stop"],
)
async def test_o1_a_data_line_that_is_not_json_is_a_malformed_response(
    body: bytes, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    error = _assert_malformed(hunt, types=[E, U, ERR], cause="not JSON")
    assert json.loads(error.raw_response or b"")["usage"]["input_tokens"] == 100  # counts so far


async def test_o1_a_body_cut_inside_its_last_data_line_is_a_cut(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # The body ends inside its last `data:` line, before message_stop: the cut class (W3.1-T),
    # retried within the budget, each attempt counted.
    body = b"".join(SNIFF_STREAM[:-1]) + b'data: {"type": "message_st'
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    assert _types(hunt.events) == [E, U, ERR] * (RETRY_BUDGET + 1)
    assert _usages(hunt.events) == [(100, 20)] * (RETRY_BUDGET + 1)
    assert all(e.tool == "<brain>" and e.retryable for e in _errors(hunt.events))
    assert hunt.row == BRAIN_CALL_FAILED


def _start_with_a_count_of(digits: int) -> bytes:
    """A message_start whose `input_tokens` is a literal of `digits` digits (spliced into the
    bytes: `json.dumps` cannot write it past `int_max_str_digits`)."""
    frame = _message_start({"input_tokens": "COUNT_LITERAL", "output_tokens": 1})
    return frame.replace(b'"COUNT_LITERAL"', b"9" * digits)


async def test_o1_a_count_over_4300_digits_in_a_frame_is_a_malformed_response(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # `json.loads` raises a plain ValueError (not JSONDecodeError) for an int over
    # `int_max_str_digits` digits (4300 by default): the guard catches ValueError.
    body = _with(0, _start_with_a_count_of(4301), replace=True)
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    _assert_malformed(hunt, types=[E, ERR], cause="not JSON")


@pytest.fixture
def digit_limit_640() -> Iterator[None]:
    """Python's `int_max_str_digits` lowered to 640 for one test (as PYTHONINTMAXSTRDIGITS can
    set it), then restored."""
    before = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(640)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(before)


async def test_o1_a_count_over_a_lowered_digit_limit_in_a_frame_is_a_malformed_response(
    tmp_path: Path, mock_anthropic: MockAnthropic, digit_limit_640: None
) -> None:
    # The stream path stays typed under a lowered limit: `json.loads` enforces it (W3.4-D1 is
    # the plain path's, where pydantic's parser does not).
    body = _with(0, _start_with_a_count_of(1000), replace=True)
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    _assert_malformed(hunt, types=[E, ERR], cause="not JSON")


@pytest.mark.parametrize(
    "text", ["[1, 2]", "42", "null", '"text"'], ids=["list", "number", "null", "string"]
)
async def test_o2_a_frame_that_is_not_an_object_is_a_malformed_response(
    text: str, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(_with(1, _data(text))), stream=True)
    _assert_malformed(hunt, types=[E, U, ERR], cause="not a JSON object")


@pytest.mark.parametrize(
    "frame",
    [
        {"type": "content_block_start", "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_start", "index": 1},
        {"type": "content_block_delta", "index": 0},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "x"}},
        {"type": "content_block_stop"},
    ],
    ids=[
        "start-without-index",
        "start-without-block",
        "delta-without-delta",
        "delta-without-index",
        "stop-without-index",
    ],
)
async def test_o3_a_block_frame_missing_a_field_is_a_malformed_response(
    frame: dict[str, Any], tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(_with(2, _frame(frame))), stream=True)
    _assert_malformed(hunt, types=[E, U, ERR], cause="without")


BLOCK: dict[str, Any] = {"type": "text", "text": ""}


@pytest.mark.parametrize(
    ("at", "frame", "types"),
    [
        (2, {"type": "content_block_start", "index": None, "content_block": BLOCK}, [E, U, ERR]),
        (2, {"type": "content_block_start", "index": "a", "content_block": BLOCK}, [E, U, ERR]),
        (2, {"type": "content_block_start", "index": [1], "content_block": BLOCK}, [E, U, ERR]),
        (2, {"type": "content_block_start", "index": 1, "content_block": None}, [E, U, ERR]),
        (2, {"type": "content_block_start", "index": 1, "content_block": "x"}, [E, U, ERR]),
        (2, {"type": "content_block_delta", "index": 0, "delta": None}, [E, U, ERR]),
        (2, {"type": "content_block_delta", "index": 0, "delta": "x"}, [E, U, ERR]),
        (0, {"type": "message_start", "message": None}, [E, ERR]),
        (0, {"type": "message_start", "message": {"model": MODEL, "usage": None}}, [E, ERR]),
        (0, {"type": "message_start", "message": {"model": MODEL, "usage": [1]}}, [E, ERR]),
        (4, {"type": "message_delta", "delta": None, "usage": {"output_tokens": 20}}, [E, U, ERR]),
        (
            4,
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": None},
            [E, U, ERR],
        ),
        (
            4,
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": "x"},
            [E, U, ERR],
        ),
    ],
    ids=[
        "index-null",
        "index-word",
        "index-list",
        "block-null",
        "block-string",
        "delta-null",
        "delta-string",
        "message-null",
        "start-usage-null",
        "start-usage-list",
        "message-delta-null",
        "delta-usage-null",
        "delta-usage-string",
    ],
)
async def test_o4_a_field_of_the_wrong_type_is_a_malformed_response(
    at: int,
    frame: dict[str, Any],
    types: list[str],
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
) -> None:
    # A message_start or message_delta of the wrong shape takes that frame's place.
    replace = frame["type"] in ("message_start", "message_delta")
    body = _with(at, _frame(frame), replace=replace)
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    _assert_malformed(hunt, types=types, cause="is not an")


@pytest.mark.parametrize(
    "frame",
    [
        {"type": "content_block_start", "index": "1", "content_block": BLOCK},
        {"type": "content_block_start", "index": 1.0, "content_block": BLOCK},
        {"type": "content_block_start", "index": True, "content_block": BLOCK},
        {"type": "content_block_start", "index": 1, "content_block": [["type", "text"]]},
    ],
    ids=["index-numeric-string", "index-float", "index-bool", "block-pair-list"],
)
async def test_frames_1a_a_value_int_or_dict_would_coerce_is_refused(
    frame: dict[str, Any], tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # Not an O-item: before W3.4 `int()` and `dict()` read these, so the turn (a tool_use and a
    # closed text block) was accepted. Under frames-1a they are refused like any wrong type.
    stop = _frame({"type": "content_block_stop", "index": 1})
    body = _with(2, _frame(frame), stop)
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    _assert_malformed(hunt, types=[E, U, ERR], cause="is not an")


@pytest.mark.parametrize(
    ("at", "frame", "counted"),
    [
        (0, {"type": "message_start"}, (0, 20)),
        (0, {"type": "message_start", "message": {"model": MODEL}}, (0, 20)),
        (4, {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}, (100, 1)),
    ],
    ids=["no-message", "no-start-usage", "no-delta-usage"],
)
async def test_guard_an_absent_message_or_usage_reads_as_before(
    at: int,
    frame: dict[str, Any],
    counted: tuple[int, int],
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
) -> None:
    body = _with(at, _frame(frame), replace=True)
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    assert _types(hunt.events)[:4] == [E, U, "tool_call", "tool_result"]
    assert _usages(hunt.events)[:1] == [counted]


async def test_guard_unknown_events_and_fields_and_unstarted_blocks_are_ignored(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    ignored = [
        _frame({"type": "a_future_event", "index": "x", "delta": 5}),
        b'event: ping\ndata: {"type": "ping"}\n\n',
        _frame({"type": "content_block_delta", "index": 7, "delta": {"type": "text_delta"}}),
        _frame({"type": "content_block_stop", "index": 7}),
    ]
    body = _with(1, *ignored)
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    assert _types(hunt.events)[:4] == [E, U, "tool_call", "tool_result"]


# ── reset: a reset mid-stream counts what the stream reported (W3.2-U2b) ──────────────────
# A transport failure while the body streams raised a bare `httpx2.TransportError` past the
# assembler, so the refused attempt, which can still be billed, appended no UsageEvent. reset-1
# with 1a: `_drive_stream` raises the retryable BrainCallError with the counts its frames reported,
# so each attempt appends its UsageEvent before its ErrorEvent (invariant 1). The ErrorEvent's text
# is as before, plus the note when the counts do not read; `raw_response` stays None (the bytes as
# received are in the exchange, W3.3). A frame whose closing blank line never came is not folded.

RESET = httpx2.ReadError("connection reset mid-stream")
RESET_TEXT = repr(BrainCallError(repr(RESET), retryable=True))  # the ErrorEvent's text before W3.4


class _ResetAfter(httpx2.AsyncByteStream):
    """A 200 stream that sends `chunks`, then the connection resets."""

    def __init__(self, chunks: Sequence[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        raise httpx2.ReadError("connection reset mid-stream")

    async def aclose(self) -> None:
        pass


def _reset_after(*chunks: bytes) -> Respond:
    return lambda _request: httpx2.Response(200, stream=_ResetAfter(chunks))


@pytest.mark.parametrize(
    ("chunks", "counted"),
    [
        (SNIFF_STREAM[:2], (100, 1)),
        (SNIFF_STREAM[:5], (100, 20)),
        # this message_delta's closing blank line never came: dropped, its count not read
        (
            [
                SNIFF_STREAM[0],
                b'data: {"type": "message_delta", "delta": {}, "usage": {"output_tokens": 20}}\n',
            ],
            (100, 1),
        ),
        # the turn was whole, then the connection reset before the body ended: counted, retried
        (SNIFF_STREAM, (100, 20)),
    ],
    ids=["after-message-start", "after-message-delta", "inside-a-frame", "after-message-stop"],
)
async def test_a_reset_mid_stream_counts_what_the_stream_reported(
    chunks: list[bytes],
    counted: tuple[int, int],
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
) -> None:
    hunt = await _hunt(tmp_path, mock_anthropic, _reset_after(*chunks), stream=True)
    assert _types(hunt.events) == [E, U, ERR] * (RETRY_BUDGET + 1)
    assert _usages(hunt.events) == [counted] * (RETRY_BUDGET + 1)
    assert hunt.row == BRAIN_CALL_FAILED
    for error in _errors(hunt.events):
        assert (error.tool, error.error, error.retryable) == ("<brain>", RESET_TEXT, True)
        assert error.raw_response is None


async def test_a_reset_whose_counts_do_not_read_says_so(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    respond = _reset_after(_message_start({"input_tokens": "x", "output_tokens": 1}))
    hunt = await _hunt(tmp_path, mock_anthropic, respond, stream=True)
    assert _types(hunt.events) == [E, ERR] * (RETRY_BUDGET + 1)
    # The text is the transport cause, then the note: no count read, none appended.
    for error in _errors(hunt.events):
        assert f"{RESET!r}; usage not counted" in error.error


async def test_guard_a_reset_before_message_start_counts_nothing(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # No count arrived, so none is appended; the text is as before W3.4.
    respond = _reset_after(b'event: ping\ndata: {"type": "ping"}\n\n')
    hunt = await _hunt(tmp_path, mock_anthropic, respond, stream=True)
    assert _types(hunt.events) == [E, ERR] * (RETRY_BUDGET + 1)
    assert [e.error for e in _errors(hunt.events)] == [RESET_TEXT] * (RETRY_BUDGET + 1)


# ── Cap60: a brain retry waits for retry-after up to 60 s ──────────────────────────────────
# Before W3.4 a `retry-after` above 8 s was cut to the jitter's 8 s cap (W3.2-B2); the
# rate-limits page says to wait that many seconds before retrying (:89, :204) and that "Earlier
# retries might fail" (:309). Now a brain retry waits at least min(`retry-after`, 60 s); the
# jitter (base 0.5 s, cap 8 s) and the tool retries, which carry no `retry-after`, are unchanged.
# The waits are recorded, never slept (tests/conftest.py's `retry_waits`, W3.2-Y1).


def _status(code: int, headers: dict[str, str]) -> Respond:
    return lambda _request: httpx2.Response(code, content=b"status body", headers=headers)


@pytest.mark.parametrize(
    ("header", "waits"),
    [("30", [30.0, 30.0]), ("60", [60.0, 60.0]), ("90", [60.0, 60.0])],
    ids=["30s-honoured", "60s-honoured", "90s-cut-to-60s"],
)
async def test_a_brain_retry_waits_for_retry_after_up_to_60_s(
    header: str,
    waits: list[float],
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
    retry_waits: list[float],
) -> None:
    await _hunt(tmp_path, mock_anthropic, _status(429, {"retry-after": header}), stream=False)
    assert retry_waits == waits


class _Top(random.Random):
    """A jitter source that always draws the top of its range, to read the backoff's bounds."""

    def uniform(self, a: float, b: float) -> float:
        return b


def test_guard_the_jitter_bound_still_stops_at_8_s(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loop, "jitter_source", _Top())
    assert [loop.backoff_delay(n) for n in range(1, 9)] == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0, 8.0, 8.0]
    assert loop.backoff_delay(8, retry_after=0.0) == 8.0


async def test_guard_a_tool_retry_waits_the_jitter_only(
    tmp_path: Path,
    scripted_brain: Callable[[Sequence[Decision]], Brain],
    retry_waits: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(loop, "jitter_source", _Top())

    async def flaky() -> str:
        """A tool that always fails, retryably."""
        raise RetryableToolError("busy")

    reg = ToolRegistry()
    reg.tool(flaky)
    conn = await db.connect(tmp_path / "rex.db")
    try:
        await run_hunt(
            conn,
            territory="mock-gym",
            brain=scripted_brain([ToolCallDecision(tool="flaky", args={}), NeedsHelp()]),
            registry=reg,
            retry_budget=6,
        )
    finally:
        await conn.close()
    assert retry_waits == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]
