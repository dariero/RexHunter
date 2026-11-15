"""W3.2 — a brain call the boundary refuses is counted (H4, W3.2-U2a); both spend ceilings are
checked before every brain attempt (H3, H5, W3.2-C1); a retry waits (H1, H2, W3.2-B2).

H4 (W3.1-U): before W3.2 a brain call the boundary refused appended no UsageEvent, so a 200 that can
still be billed (a turn refused for its stop reason, a stream cut before `message_stop`, an error
frame, an unfinished stream, a tool input that does not parse, a turn with no single tool_use) was
missed by `cost.fold_cost`, the per-run ceiling and `scheduler.daemon_spend_usd`. W3.2-U2a: the
refusal carries the response's own usage, and the loop appends it as a UsageEvent before the
ErrorEvent. A count that does not read is not counted, and the ErrorEvent says so; a stream cut by
a transport failure (W3.2-U2b), a status error and an envelope-invalid body carry no usage.

H3 and H5 (fragility #4): the daemon ceiling was checked once per hunt, at launch, and the per-run
ceiling once per iteration, so a running hunt never saw the daemon ceiling and a retried call made
`retry_budget + 1` requests with no check between them. W3.2-C1: both are checked before every
brain attempt, each a fold over the log, so a run or the daemon goes past its ceiling by at most
the one call each hunt has in flight, of the calls the log counts.

H1 and H2 (fragility #2): a retried brain call or tool ran again at once. W3.2-B2: each retry
waits a jittered backoff, at least a status response's `retry-after` (cut to 8 s), after its
events are committed. The waits are recorded, never slept: tests/conftest.py's autouse `retry_waits`
(W3.2-Y1).

Every response is built here from literals; nothing under tests/fixtures/ is read or written. The
request model is `brain.SMOKE_MODEL`. Zero network: a `MockTransport` answers every call.
"""

import asyncio
import json
import random
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import anthropic
import httpx2
import pytest

from rexhunter import brain, cost, db, loop, scheduler
from rexhunter.events import ErrorEvent, TrajectoryEvent, UsageEvent
from rexhunter.loop import (
    Brain,
    Context,
    Decision,
    HuntComplete,
    RetryableToolError,
    ThinkingSink,
    ToolCallDecision,
    run_hunt,
)
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]
Respond = Callable[[httpx2.Request], httpx2.Response]
BrainFactory = Callable[[Sequence[Decision]], Brain]

MODEL = brain.SMOKE_MODEL
RETRY_BUDGET = 2  # run_hunt's default, passed explicitly: a retried call makes 3 requests
MALFORMED = ("aborted", "malformed provider response")
BRAIN_CALL_FAILED = ("aborted", "brain call failed")
COST_CEILING = ("aborted", "cost ceiling")
DAEMON_CEILING = ("aborted", "daemon spend ceiling")

SNIFF: dict[str, Any] = {
    "type": "tool_use",
    "id": "toolu_w32",
    "name": "sniff",
    "input": {"prey": "mock-gym"},
}
NO_ID_TOOL_USE: dict[str, Any] = {"type": "tool_use", "name": "sniff", "input": {}}  # no `id`
BAD_CATCH: dict[str, Any] = {**SNIFF, "name": "hunt_complete", "input": {"catch": 7}}  # not a list
# What every literal response here reports: 100 input tokens; 1 output token at message_start and
# 20 by the end (message_delta's cumulative count, or the plain body's).
END_USAGE: dict[str, Any] = {"input_tokens": 100, "output_tokens": 20}
START_USAGE: dict[str, Any] = {"input_tokens": 100, "output_tokens": 1}


async def sniff(prey: str) -> str:
    """Sniff a territory for postings."""
    return f"posting:{prey}"


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.tool(sniff)
    return reg


def _usage(input_tokens: int, output_tokens: int) -> UsageEvent:
    return UsageEvent(model=MODEL, input_tokens=input_tokens, output_tokens=output_tokens)


def _plain(
    stop_reason: str | None,
    content: list[dict[str, Any]],
    *,
    usage: dict[str, Any] | None = None,
) -> bytes:
    """A whole Messages-API response, as the plain path reads it."""
    return json.dumps(
        {
            "id": "msg_w32",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "content": content,
            "usage": END_USAGE if usage is None else usage,
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
    start_usage: dict[str, Any] | None = None,
) -> list[bytes]:
    """A tool_use stream as frames: `message_start`, each block (start, one input delta, stop),
    `message_delta`, `message_stop`. `partial_json` replaces the input text; `close_tool_use=False`
    leaves out each block's `content_block_stop`."""
    frames = [
        _frame(
            {
                "type": "message_start",
                "message": {
                    "model": MODEL,
                    "content": [],
                    "usage": START_USAGE if start_usage is None else start_usage,
                },
            }
        )
    ]
    for i, block in enumerate(content):
        text = json.dumps(block["input"]) if partial_json is None else partial_json
        frames.append(
            _frame(
                {"type": "content_block_start", "index": i, "content_block": {**block, "input": {}}}
            )
        )
        frames.append(
            _frame(
                {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {"type": "input_json_delta", "partial_json": text},
                }
            )
        )
        if close_tool_use:
            frames.append(_frame({"type": "content_block_stop", "index": i}))
    frames.append(
        _frame(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": END_USAGE["output_tokens"]},
            }
        )
    )
    frames.append(_frame({"type": "message_stop"}))
    return frames


def _error_frame(error_type: str) -> bytes:
    data = {"type": "error", "error": {"type": error_type, "message": "An error mid-stream"}}
    return f"event: error\ndata: {json.dumps(data)}\n\n".encode()


def _ok(body: bytes) -> Respond:
    return lambda _request: httpx2.Response(200, content=body)


def _status(code: int, headers: dict[str, str] | None = None) -> Respond:
    return lambda _request: httpx2.Response(code, content=b"status body", headers=headers or {})


class Hunt(NamedTuple):
    events: list[TrajectoryEvent]
    row: tuple[str, str | None]
    requests: int
    db_path: Path


async def _row(conn: Any, run_id: str) -> tuple[str, str | None]:
    async with conn.execute(
        "SELECT outcome, abort_reason FROM runs WHERE id = ?", (run_id,)
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    return str(row[0]), row[1]


async def _hunt(
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
    respond: Respond,
    *,
    stream: bool,
    retry_budget: int = RETRY_BUDGET,
    cost_ceiling_usd: float = 1000.0,
    published: list[str] | None = None,
) -> Hunt:
    """One run_hunt whose every brain request is answered by `respond`; `published`, if given,
    collects the `type` of every event the loop publishes to the hub, in order."""

    def publish(_event_id: int, payload: str) -> None:
        if published is not None:
            published.append(str(json.loads(payload)["type"]))

    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return respond(request)

    db_path = tmp_path / "rex.db"
    reg = _registry()
    async with mock_anthropic(handler, api_key="offline") as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            model=MODEL,
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
                retry_budget=retry_budget,
                max_iterations=3,
                cost_ceiling_usd=cost_ceiling_usd,
                tool_timeout_s=5.0,
                publish=publish,
            )
            events = await db.read_events(conn, run_id)
            row = await _row(conn, run_id)
        finally:
            await conn.close()
    return Hunt(events, row, len(requests), db_path)


def _types(events: Sequence[TrajectoryEvent]) -> list[str]:
    return [e.type for e in events]


def _errors(events: Sequence[TrajectoryEvent]) -> list[ErrorEvent]:
    return [e for e in events if isinstance(e, ErrorEvent)]


# ── U2a: a 200 the boundary refuses is counted, before its ErrorEvent (H4) ─────────────


async def test_a_plain_turn_refused_for_its_stop_reason_is_counted_before_its_error(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(_plain("max_tokens", [SNIFF])), stream=False)
    assert _types(hunt.events) == ["usage", "error"]
    assert hunt.events[0] == _usage(100, 20)
    assert hunt.row == MALFORMED


async def test_a_streamed_turn_refused_for_its_stop_reason_is_counted_before_its_error(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    body = b"".join(_stream([SNIFF], stop_reason="max_tokens"))
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    assert _types(hunt.events) == ["usage", "error"]
    assert hunt.events[0] == _usage(100, 20)  # message_start's input, message_delta's output
    assert hunt.row == MALFORMED


@pytest.mark.parametrize(
    ("frames", "counted"),
    [(-1, _usage(100, 20)), (-2, _usage(100, 1))],
    ids=["before-message-stop", "before-message-delta"],
)
async def test_a_stream_cut_before_message_stop_is_counted_on_every_attempt(
    frames: int, counted: UsageEvent, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # Retried within the budget (W3.1-T): each attempt is counted, at the usage its stream reported
    # before the cut, before its error.
    body = b"".join(_stream([SNIFF])[:frames])
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    assert hunt.requests == RETRY_BUDGET + 1
    assert _types(hunt.events) == ["usage", "error"] * (RETRY_BUDGET + 1)
    assert [e for e in hunt.events if isinstance(e, UsageEvent)] == [counted] * (RETRY_BUDGET + 1)
    assert hunt.row == BRAIN_CALL_FAILED


async def test_a_retryable_error_frame_is_counted_on_every_attempt(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    body = b"".join(_stream([SNIFF])[:-2]) + _error_frame("overloaded_error")
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    assert hunt.requests == RETRY_BUDGET + 1
    assert _types(hunt.events) == ["usage", "error"] * (RETRY_BUDGET + 1)
    assert all(e == _usage(100, 1) for e in hunt.events if isinstance(e, UsageEvent))
    assert hunt.row == BRAIN_CALL_FAILED


async def test_a_fatal_error_frame_is_counted_once(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    body = b"".join(_stream([SNIFF])[:-2]) + _error_frame("invalid_request_error")
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    assert hunt.requests == 1
    assert _types(hunt.events) == ["usage", "error"]
    assert hunt.events[0] == _usage(100, 1)
    assert hunt.row == BRAIN_CALL_FAILED


@pytest.mark.parametrize(
    "frames",
    [
        _stream([SNIFF], close_tool_use=False),  # W3.1-T3: a block left open at message_stop
        _stream([SNIFF], stop_reason=None),  # W3.1-T3: no stop reason at message_stop
        _stream([SNIFF], partial_json='{"prey": "mock-'),  # W3.1-P: the input does not parse
    ],
    ids=["block-left-open", "no-stop-reason", "input-not-json"],
)
async def test_a_whole_stream_refused_at_the_boundary_is_counted_once(
    frames: list[bytes], tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(b"".join(frames)), stream=True)
    assert hunt.requests == 1
    assert _types(hunt.events) == ["usage", "error"]
    assert hunt.events[0] == _usage(100, 20)
    assert hunt.row == MALFORMED


@pytest.mark.parametrize(
    "content",
    [
        [SNIFF, {**SNIFF, "id": "toolu_w32_2"}],
        [{"type": "text", "text": "no tool call"}],
        [{"type": "tool_use", "name": "sniff", "input": {}}],  # no `id`: _ToolUseBlock refuses
        [{"type": "tool_use", "id": "toolu_w32", "name": "hunt_complete", "input": {"catch": 7}}],
    ],
    ids=["two-tool-uses", "no-tool-use", "tool-use-without-id", "hunt-complete-bad-catch"],
)
async def test_a_turn_refused_after_its_stop_reason_is_counted_once(
    content: list[dict[str, Any]], tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # The older refusals in parse_decision, on turns that did stop for a tool call.
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(_plain("tool_use", content)), stream=False)
    assert _types(hunt.events) == ["usage", "error"]
    assert hunt.events[0] == _usage(100, 20)
    assert hunt.row == MALFORMED


async def test_refused_calls_fold_into_the_run_and_the_daemon_spend(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    body = b"".join(_stream([SNIFF])[:-1])  # cut before message_stop: three counted attempts
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    one_call = cost.cost_of(_usage(100, 20), prices=cost.PRICE_PER_MTOK)
    assert one_call == pytest.approx(100 * 2.0 / 1e6 + 20 * 10.0 / 1e6)  # $2/$10 (SMOKE_MODEL)
    assert cost.fold_cost(hunt.events, prices=cost.PRICE_PER_MTOK) == pytest.approx(3 * one_call)
    conn = await db.connect(hunt.db_path)
    try:
        assert await scheduler.daemon_spend_usd(conn) == pytest.approx(3 * one_call)
    finally:
        await conn.close()


@pytest.mark.parametrize(
    ("stop_reason", "content", "input_tokens"),
    [
        ("max_tokens", [SNIFF], "x"),
        ("max_tokens", [SNIFF], float("inf")),  # `Infinity` in the JSON: int() overflows
        ("tool_use", [SNIFF, {**SNIFF, "id": "toolu_w32_2"}], "x"),
        ("tool_use", [NO_ID_TOOL_USE], "x"),
        ("tool_use", [BAD_CATCH], "x"),
    ],
    ids=[
        "stop-reason",
        "stop-reason-infinite-count",
        "two-tool-uses",
        "tool-use-without-id",
        "hunt-complete-bad-catch",
    ],
)
async def test_a_refused_turn_whose_usage_does_not_read_is_not_counted_and_says_so(
    stop_reason: str,
    content: list[dict[str, Any]],
    input_tokens: object,
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
) -> None:
    # Not counted, the run ends typed with its bytes, and the ErrorEvent says why. The last three
    # stop for their tool call and are refused for their blocks: before W3.2 their count was read
    # before those refusals, raised, and reached the `<loop>` catch-all without its bytes (X1-move).
    usage = {"input_tokens": input_tokens, "output_tokens": 20}
    body = _plain(stop_reason, content, usage=usage)
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=False)
    assert _types(hunt.events) == ["error"]
    (error,) = _errors(hunt.events)
    assert error.raw_response == body
    assert "usage not counted" in (error.detail or "")
    assert hunt.row == MALFORMED


async def test_a_cut_stream_whose_usage_does_not_read_is_not_counted_and_says_so(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    frames = _stream([SNIFF], start_usage={"input_tokens": [100], "output_tokens": 1})[:-2]
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(b"".join(frames)), stream=True)
    assert _types(hunt.events) == ["error"] * (RETRY_BUDGET + 1)
    says_so = ["usage not counted" in (e.detail or "") for e in _errors(hunt.events)]
    assert says_so == [True] * (RETRY_BUDGET + 1)
    assert hunt.row == BRAIN_CALL_FAILED


_UNREADABLE_START: dict[str, Any] = {"input_tokens": [100], "output_tokens": 1}


@pytest.mark.parametrize(
    ("frames", "row"),
    [
        (
            [
                *_stream([SNIFF], start_usage=_UNREADABLE_START)[:-2],
                _error_frame("invalid_request_error"),
            ],
            BRAIN_CALL_FAILED,
        ),
        (_stream([SNIFF], close_tool_use=False, start_usage=_UNREADABLE_START), MALFORMED),
        (
            _stream([SNIFF], partial_json='{"prey": "mock-', start_usage=_UNREADABLE_START),
            MALFORMED,
        ),
    ],
    ids=["fatal-error-frame", "block-left-open", "input-not-json"],
)
async def test_a_refused_stream_whose_usage_does_not_read_says_so(
    frames: list[bytes], row: tuple[str, str], tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # The note rides on every stream refusal (W3.1-Z, T3, P), as on the cut stream above.
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(b"".join(frames)), stream=True)
    assert _types(hunt.events) == ["error"]
    assert "usage not counted" in (_errors(hunt.events)[0].detail or "")
    assert hunt.row == row


async def test_a_refused_turn_whose_count_cannot_be_priced_is_not_counted(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # A count int() reads but the fold cannot price (its float overflows) must not reach the log:
    # every later fold would raise, the run's ceiling check and the daemon's launch gate with it.
    usage = {"input_tokens": int("9" * 400), "output_tokens": 20}
    body = _plain("max_tokens", [SNIFF], usage=usage)
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=False)
    assert _types(hunt.events) == ["error"]
    assert "usage not counted" in (_errors(hunt.events)[0].detail or "")
    conn = await db.connect(hunt.db_path)
    try:
        assert await scheduler.daemon_spend_usd(conn) == 0.0  # the daemon's fold still prices
    finally:
        await conn.close()


async def test_a_refused_turn_with_no_model_is_counted_at_the_fallback_price(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # A model that is not a string prices at cost.py's fallback ($15/$75), the safe direction.
    envelope = json.loads(_plain("max_tokens", [SNIFF]))
    envelope["model"] = None
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(json.dumps(envelope).encode()), stream=False)
    assert _types(hunt.events) == ["usage", "error"]
    assert hunt.events[0] == UsageEvent(model="", input_tokens=100, output_tokens=20)
    assert cost.fold_cost(hunt.events) == pytest.approx(100 * 15.0 / 1e6 + 20 * 75.0 / 1e6)


@pytest.mark.parametrize(
    ("body", "stream", "published"),
    [
        (_plain("max_tokens", [SNIFF]), False, ["usage", "error"]),
        (b"".join(_stream([SNIFF])[:-1]), True, ["usage", "error"] * (RETRY_BUDGET + 1)),
    ],
    ids=["plain-refusal", "retried-cut-stream"],
)
async def test_a_refused_calls_usage_is_published_before_its_error(
    body: bytes,
    stream: bool,
    published: list[str],
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
) -> None:
    # The hub hears each counted refusal too, after its commit and before its ErrorEvent (inv 1).
    seen: list[str] = []
    await _hunt(tmp_path, mock_anthropic, _ok(body), stream=stream, published=seen)
    assert seen == published


# Pins: what U2a does not count, and the accepted path it leaves alone.


async def test_a_stream_that_errs_before_message_start_is_not_counted(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # No message_start, so no count: the assembled usage is `{}`, and `{}` gives no UsageEvent.
    hunt = await _hunt(
        tmp_path, mock_anthropic, _ok(_error_frame("invalid_request_error")), stream=True
    )
    assert _types(hunt.events) == ["error"]
    assert hunt.row == BRAIN_CALL_FAILED


async def test_a_status_error_is_not_counted(tmp_path: Path, mock_anthropic: MockAnthropic) -> None:
    hunt = await _hunt(tmp_path, mock_anthropic, _status(529), stream=False)
    assert _types(hunt.events) == ["error"] * (RETRY_BUDGET + 1)
    assert hunt.row == BRAIN_CALL_FAILED


async def test_a_body_that_fails_the_envelope_is_not_counted(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    body = json.dumps(
        {"type": "message", "model": MODEL, "usage": END_USAGE}
    ).encode()  # no content
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=False)
    assert _types(hunt.events) == ["error"]
    assert hunt.row == MALFORMED


class _ResetAfterMessageStart(httpx2.AsyncByteStream):
    """A 200 stream that sends its message_start (with a usage), then the connection resets."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield _stream([SNIFF])[0]
        raise httpx2.ReadError("connection reset mid-stream")

    async def aclose(self) -> None:
        pass


async def test_a_reset_mid_stream_is_not_counted(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # W3.2-U2b, Darie's: the assembler's state is lost with the stream, so nothing is counted.
    def respond(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, stream=_ResetAfterMessageStart())

    hunt = await _hunt(tmp_path, mock_anthropic, respond, stream=True)
    assert _types(hunt.events) == ["error"] * (RETRY_BUDGET + 1)
    assert hunt.row == BRAIN_CALL_FAILED


async def test_an_accepted_turn_whose_usage_does_not_read_still_fails_closed(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # W3.1-O5, unchanged: the accepted path keeps `_extract_usage`, which raises, so the turn is
    # never dispatched uncounted; it reaches the `<loop>` catch-all (without its bytes, O5).
    body = _plain("tool_use", [SNIFF], usage={"input_tokens": "x", "output_tokens": 20})
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=False)
    assert _types(hunt.events) == ["error"]
    assert _errors(hunt.events)[0].tool == "<loop>"
    assert hunt.row == ("aborted", "unhandled exception in loop")


# ── C1: both ceilings before every brain attempt (H3, H5) ──────────────────────────────

PER_CALL = _usage(4000, 0)  # $0.008 at SMOKE_MODEL's $2/MTok input
PER_CALL_USD = 0.008
DAEMON_CEILING_USD = 0.01  # one call fits under it, two do not


def _spending_brain(calls: list[int], *, gate: Callable[[], Any] | None = None) -> Brain:
    """A brain that sniffs forever, each call reporting PER_CALL; `gate` runs before it answers."""

    async def spend(_context: Context, _sink: ThinkingSink) -> Decision:
        calls.append(1)
        if gate is not None:
            await gate()
        return ToolCallDecision(tool="sniff", args={"prey": "mock-gym"}, usage=PER_CALL)

    return spend


async def _spend_and_rows(db_path: Path, run_ids: Sequence[str | None]) -> tuple[float, list[Any]]:
    conn = await db.connect(db_path)
    try:
        rows = [await _row(conn, rid) for rid in run_ids if rid is not None]
        return await scheduler.daemon_spend_usd(conn), rows
    finally:
        await conn.close()


async def test_one_hunt_stops_at_the_daemon_ceiling_mid_run(tmp_path: Path) -> None:
    # H3: before W3.2 the launch gate was the daemon ceiling's only check, so this hunt ran its
    # three calls ($0.024) past a $0.01 ceiling and ended on "max iterations".
    db_path = tmp_path / "rex.db"
    calls: list[int] = []
    run_ids = await scheduler.run_hunts(
        db_path,
        ["mock-gym"],
        brain_for=lambda _territory: _spending_brain(calls),
        registry=_registry(),
        max_concurrent=1,
        retry_budget=0,
        max_iterations=3,
        daemon_spend_ceiling_usd=DAEMON_CEILING_USD,
    )
    spend, rows = await _spend_and_rows(db_path, run_ids)
    assert spend <= DAEMON_CEILING_USD + PER_CALL_USD  # at most one call past the ceiling
    assert spend == pytest.approx(2 * PER_CALL_USD)
    assert len(calls) == 2  # $0.008 < $0.01 lets the second call go; $0.016 stops the third
    assert rows == [DAEMON_CEILING]


async def test_two_concurrent_hunts_stay_within_the_daemon_ceiling_plus_one_call_each(
    tmp_path: Path,
) -> None:
    # H3: before W3.2, hunts that passed the launch gate together never met the daemon ceiling
    # again: here each ran its three calls to max iterations ($0.048 for two). Both first calls
    # are held until both are in flight, so both passed the check at $0: the bound is the ceiling
    # plus one call per hunt, whatever the order after that.
    db_path = tmp_path / "rex.db"
    calls: list[int] = []
    in_first_call = 0
    both_in = asyncio.Event()

    async def hold_first_calls() -> None:
        nonlocal in_first_call
        if len(calls) <= 2:
            in_first_call += 1
            if in_first_call == 2:
                both_in.set()
            await asyncio.wait_for(both_in.wait(), timeout=5)

    run_ids = await scheduler.run_hunts(
        db_path,
        ["a", "b"],
        brain_for=lambda _territory: _spending_brain(calls, gate=hold_first_calls),
        registry=_registry(),
        max_concurrent=2,
        retry_budget=0,
        max_iterations=3,
        daemon_spend_ceiling_usd=DAEMON_CEILING_USD,
    )
    spend, rows = await _spend_and_rows(db_path, run_ids)
    assert both_in.is_set()  # the two first calls overlapped
    assert spend <= DAEMON_CEILING_USD + 2 * PER_CALL_USD + 1e-12  # one call past it per hunt
    assert rows == [DAEMON_CEILING, DAEMON_CEILING]


def _cut_stream() -> bytes:
    """A stream cut before message_stop: a retryable refusal whose usage ($0.0004) is counted."""
    return b"".join(_stream([SNIFF])[:-1])


async def test_the_per_run_ceiling_stops_a_retried_call_between_attempts(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # H5: before W3.2 the per-run ceiling was checked once per iteration, so a retried call made
    # all three requests whatever the first one spent.
    hunt = await _hunt(
        tmp_path, mock_anthropic, _ok(_cut_stream()), stream=True, cost_ceiling_usd=1e-9
    )
    assert hunt.row == COST_CEILING
    assert hunt.requests == 1
    assert _types(hunt.events) == ["usage", "error"]


async def test_the_daemon_ceiling_stops_a_retried_call_between_attempts(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # H5 for the daemon ceiling: the fresh log passes the launch gate ($0 < the ceiling), then the
    # first refused attempt's usage stops the retry.
    db_path = tmp_path / "rex.db"
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, content=_cut_stream())

    reg = _registry()
    async with mock_anthropic(handler, api_key="offline") as client:
        brain_for = brain.adapter_brain_for(
            client=client, model=MODEL, registry=reg, thinking=brain.HUNT_THINKING, stream=True
        )
        run_ids = await scheduler.run_hunts(
            db_path,
            ["mock-gym"],
            brain_for=brain_for,
            registry=reg,
            max_concurrent=1,
            retry_budget=RETRY_BUDGET,
            max_iterations=3,
            daemon_spend_ceiling_usd=1e-9,
        )
    _spend, rows = await _spend_and_rows(db_path, run_ids)
    assert rows == [DAEMON_CEILING]
    assert len(requests) == 1


# Pins: no daemon ceiling leaves a run ungated by other runs' spend; the per-run one is named first.


async def test_without_a_daemon_ceiling_a_hunt_ignores_the_spend_of_other_runs(
    tmp_path: Path,
) -> None:
    # Pin: run_hunt's daemon ceiling defaults to None, ungated, as before W3.2; and the per-run
    # fold counts this run's spend only ($1.00 ceiling, $2.00 spent in another run).
    db_path = tmp_path / "rex.db"
    conn = await db.connect(db_path)
    try:
        seed = await db.start_run(conn, territory="seed")
        await db.append_event(conn, seed, _usage(1_000_000, 0))  # $2.00 in another run
        calls: list[int] = []
        run_id = await run_hunt(
            conn,
            territory="mock-gym",
            brain=_spending_brain(calls),
            registry=_registry(),
            retry_budget=0,
            max_iterations=3,
            cost_ceiling_usd=1.0,
        )
        assert await _row(conn, run_id) == ("aborted", "max iterations")
        assert len(calls) == 3
    finally:
        await conn.close()


async def test_the_per_run_ceiling_is_named_when_both_ceilings_trip(tmp_path: Path) -> None:
    # Pin: the per-run ceiling is checked first, so a run over both ends "cost ceiling".
    db_path = tmp_path / "rex.db"
    calls: list[int] = []
    run_ids = await scheduler.run_hunts(
        db_path,
        ["mock-gym"],
        brain_for=lambda _territory: _spending_brain(calls),
        registry=_registry(),
        max_concurrent=1,
        retry_budget=0,
        max_iterations=3,
        cost_ceiling_usd=1e-9,
        daemon_spend_ceiling_usd=1e-9,
    )
    _spend, rows = await _spend_and_rows(db_path, run_ids)
    assert rows == [COST_CEILING]
    assert len(calls) == 1


async def test_a_hunt_started_over_the_daemon_ceiling_makes_no_brain_call(tmp_path: Path) -> None:
    # The daemon check runs on a run's first attempt too, not only behind the launch gate: a
    # sibling can cross the ceiling between this hunt's launch and its first call. Direct
    # run_hunt, so no launch gate stands in front of the in-loop check.
    conn = await db.connect(tmp_path / "rex.db")
    try:
        seed = await db.start_run(conn, territory="seed")
        await db.append_event(conn, seed, _usage(1_000_000, 0))  # $2.00 in another run
        calls: list[int] = []
        run_id = await run_hunt(
            conn,
            territory="mock-gym",
            brain=_spending_brain(calls),
            registry=_registry(),
            retry_budget=0,
            max_iterations=3,
            cost_ceiling_usd=1000.0,
            daemon_spend_ceiling_usd=1.0,
        )
        assert await _row(conn, run_id) == DAEMON_CEILING
        assert calls == []
    finally:
        await conn.close()


# Boundaries: each check is `>=` (and the daemon one an `is not None` test, not truthiness), so a
# $0 ceiling allows no brain call at all.


async def test_a_zero_per_run_ceiling_allows_no_brain_call(tmp_path: Path) -> None:
    conn = await db.connect(tmp_path / "rex.db")
    try:
        calls: list[int] = []
        run_id = await run_hunt(
            conn,
            territory="mock-gym",
            brain=_spending_brain(calls),
            registry=_registry(),
            retry_budget=0,
            max_iterations=3,
            cost_ceiling_usd=0.0,
        )
        assert await _row(conn, run_id) == COST_CEILING
        assert calls == []
    finally:
        await conn.close()


async def test_a_zero_daemon_ceiling_allows_no_brain_call(tmp_path: Path) -> None:
    conn = await db.connect(tmp_path / "rex.db")
    try:
        calls: list[int] = []
        run_id = await run_hunt(
            conn,
            territory="mock-gym",
            brain=_spending_brain(calls),
            registry=_registry(),
            retry_budget=0,
            max_iterations=3,
            cost_ceiling_usd=1000.0,
            daemon_spend_ceiling_usd=0.0,
        )
        assert await _row(conn, run_id) == DAEMON_CEILING
        assert calls == []
    finally:
        await conn.close()


# ── B2 + Y1: a retry waits, after its events are committed (H1, H2) ─────────────────────


async def test_a_retried_brain_call_waits_before_each_retry(
    tmp_path: Path, mock_anthropic: MockAnthropic, retry_waits: list[float]
) -> None:
    # H1: before W3.2 a 503 was re-sent at once. Two retries, two waits, each within its bound.
    hunt = await _hunt(tmp_path, mock_anthropic, _status(503), stream=False)
    assert hunt.requests == RETRY_BUDGET + 1
    assert len(retry_waits) == RETRY_BUDGET  # none before the first attempt, none after the last
    assert 0.0 <= retry_waits[0] <= 0.5
    assert 0.0 <= retry_waits[1] <= 1.0


async def test_a_retried_tool_waits_before_each_retry(
    tmp_path: Path, scripted_brain: BrainFactory, retry_waits: list[float]
) -> None:
    # H2: before W3.2 a RetryableToolError was re-run at once.
    reg = ToolRegistry()
    attempts = 0

    @reg.tool
    async def flaky(n: int) -> str:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise RetryableToolError(f"transient #{attempts}")
        return "ok"

    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await run_hunt(
            conn,
            territory="gate",
            brain=scripted_brain([ToolCallDecision(tool="flaky", args={"n": 1}), HuntComplete()]),
            registry=reg,
            retry_budget=3,
        )
        assert await _row(conn, run_id) == ("completed", None)
    finally:
        await conn.close()
    assert len(retry_waits) == 2
    assert 0.0 <= retry_waits[0] <= 0.5
    assert 0.0 <= retry_waits[1] <= 1.0


async def _tool_hunt(tmp_path: Path, scripted_brain: BrainFactory, tool: Any, **kw: Any) -> None:
    reg = ToolRegistry()
    reg.tool(tool)
    conn = await db.connect(tmp_path / "rex.db")
    try:
        await run_hunt(
            conn,
            territory="gate",
            brain=scripted_brain([ToolCallDecision(tool=tool.__name__, args={}), HuntComplete()]),
            registry=reg,
            **kw,
        )
    finally:
        await conn.close()


async def test_no_wait_after_a_fatal_error_or_a_success(
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
    scripted_brain: BrainFactory,
    retry_waits: list[float],
) -> None:
    # A fatal status, carrying retry-after: no wait, no retry.
    hunt = await _hunt(tmp_path, mock_anthropic, _status(400, {"retry-after": "3"}), stream=False)
    assert hunt.requests == 1

    async def broken() -> str:
        raise ValueError("a bug, fatal")

    await _tool_hunt(tmp_path, scripted_brain, broken)  # a new run in the same log

    async def fine() -> str:
        return "ok"

    await _tool_hunt(tmp_path, scripted_brain, fine)
    assert retry_waits == []


async def test_a_tool_timeout_waits_before_its_retry(
    tmp_path: Path, scripted_brain: BrainFactory, retry_waits: list[float]
) -> None:
    async def hang() -> None:
        await asyncio.sleep(10)  # the per-tool deadline fires; asyncio.sleep itself is not patched

    await _tool_hunt(tmp_path, scripted_brain, hang, tool_timeout_s=0.05, retry_budget=1)
    assert len(retry_waits) == 1


@pytest.mark.parametrize("code", [429, 503, 529])
async def test_a_retry_after_header_sets_the_least_wait(
    code: int, tmp_path: Path, mock_anthropic: MockAnthropic, retry_waits: list[float]
) -> None:
    respond = _status(code, {"retry-after": "3"})
    await _hunt(tmp_path, mock_anthropic, respond, stream=False)
    assert retry_waits == [3.0, 3.0]  # above both jitter bounds (0.5 s, 1 s), under the cap


async def test_a_retry_after_above_the_cap_is_cut_to_the_cap(
    tmp_path: Path, mock_anthropic: MockAnthropic, retry_waits: list[float]
) -> None:
    await _hunt(tmp_path, mock_anthropic, _status(429, {"retry-after": "30"}), stream=False)
    assert retry_waits == [8.0, 8.0]  # the cap, BACKOFF_CAP_S


@pytest.mark.parametrize(
    "headers",
    [
        {"retry-after": "soon"},
        {"retry-after": "Thu, 01 Jan 2099 00:00:00 GMT"},  # in the future: honoured, it would wait
        {"retry-after": "-1"},
        {"retry-after": "nan"},
        {"retry-after": "inf"},
        {"retry-after-ms": "3000"},  # the SDK's own header; not read (seconds only)
    ],
    ids=["word", "http-date", "negative", "nan", "inf", "retry-after-ms"],
)
async def test_a_retry_after_that_is_not_seconds_is_ignored(
    headers: dict[str, str],
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
    retry_waits: list[float],
) -> None:
    await _hunt(tmp_path, mock_anthropic, _status(429, headers), stream=False)
    assert len(retry_waits) == 2
    assert retry_waits[0] <= 0.5 and retry_waits[1] <= 1.0  # the jitter alone


async def test_each_attempt_is_in_the_log_before_its_wait_starts(
    tmp_path: Path, mock_anthropic: MockAnthropic, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Write-ahead (inv 1): at each wait, a separate reader already sees that attempt's UsageEvent
    # and ErrorEvent committed.
    seen: list[tuple[int, int]] = []
    db_path = tmp_path / "rex.db"

    async def look(_delay: float) -> None:
        reader = await db.connect(db_path)
        try:
            async with reader.execute(
                "SELECT SUM(type = 'usage'), SUM(type = 'error') FROM trajectory_events"
            ) as cur:
                row = await cur.fetchone()
            assert row is not None
            seen.append((int(row[0]), int(row[1])))
        finally:
            await reader.close()

    monkeypatch.setattr(loop, "retry_wait", look)
    await _hunt(tmp_path, mock_anthropic, _ok(_cut_stream()), stream=True)
    assert seen == [(1, 1), (2, 2)]


async def test_a_tool_error_is_in_the_log_before_its_wait_starts(
    tmp_path: Path, scripted_brain: BrainFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []
    db_path = tmp_path / "rex.db"

    async def look(_delay: float) -> None:
        reader = await db.connect(db_path)
        try:
            async with reader.execute(
                "SELECT COUNT(*) FROM trajectory_events WHERE type = 'error'"
            ) as cur:
                row = await cur.fetchone()
            assert row is not None
            seen.append(int(row[0]))
        finally:
            await reader.close()

    monkeypatch.setattr(loop, "retry_wait", look)

    async def flaky() -> str:
        raise RetryableToolError("transient")

    reg = ToolRegistry()
    reg.tool(flaky)
    conn = await db.connect(db_path)
    try:
        await run_hunt(
            conn,
            territory="gate",
            brain=scripted_brain([ToolCallDecision(tool="flaky", args={})]),
            registry=reg,
            retry_budget=2,
        )
    finally:
        await conn.close()
    assert seen == [1, 2]


@pytest.mark.parametrize("where", ["brain", "tool"])
async def test_a_cancel_during_a_wait_ends_the_run_as_daemon_shutdown(
    where: str,
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
    scripted_brain: BrainFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waiting = asyncio.Event()

    async def block(_delay: float) -> None:
        waiting.set()
        await asyncio.Event().wait()  # a wait that only a cancel ends

    monkeypatch.setattr(loop, "retry_wait", block)
    db_path = tmp_path / "rex.db"
    reg = _registry()

    async def flaky() -> str:
        raise RetryableToolError("transient")

    reg.tool(flaky)
    async with mock_anthropic(_status(503), api_key="offline") as client:
        if where == "brain":
            hunt_brain = brain.adapter_brain_for(client=client, model=MODEL, registry=reg)("g")
        else:
            hunt_brain = scripted_brain([ToolCallDecision(tool="flaky", args={})])
        conn = await db.connect(db_path)
        task = asyncio.create_task(
            run_hunt(conn, territory="g", brain=hunt_brain, registry=reg, retry_budget=2)
        )
        try:
            await asyncio.wait_for(waiting.wait(), timeout=2)
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=5)  # bounded: a swallowed cancel fails
            assert task in done, "the cancel did not end the hunt"
            with pytest.raises(asyncio.CancelledError):
                task.result()
            async with conn.execute("SELECT outcome, abort_reason FROM runs") as cur:
                rows = list(await cur.fetchall())
            assert [tuple(r) for r in rows] == [("aborted", "daemon shutdown")]
            async with conn.execute(
                "SELECT COUNT(*) FROM trajectory_events WHERE type = 'error'"
            ) as cur:
                count = await cur.fetchone()
            assert count is not None and int(count[0]) == 1  # the first attempt's, then the wait
        finally:
            task.cancel()  # a no-op once the hunt has ended
            await asyncio.wait({task}, timeout=5)
            await conn.close()


async def test_spend_committed_during_the_wait_stops_the_retry(
    tmp_path: Path, mock_anthropic: MockAnthropic, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The wait comes BEFORE the next attempt's ceiling checks (B2 after C1): spend another hunt
    # commits while this one waits is seen before the retry is sent. A 503 carries no usage, so
    # only the sibling's $2.00 can cross the $1.00 daemon ceiling.
    db_path = tmp_path / "rex.db"

    async def sibling_spends(_delay: float) -> None:
        other = await db.connect(db_path)  # another hunt's own connection (inv 7)
        try:
            sibling = await db.start_run(other, territory="sibling")
            await db.append_event(other, sibling, _usage(1_000_000, 0))  # $2.00
        finally:
            await other.close()

    monkeypatch.setattr(loop, "retry_wait", sibling_spends)
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(503, content=b"status body")

    reg = _registry()
    async with mock_anthropic(handler, api_key="offline") as client:
        conn = await db.connect(db_path)
        try:
            run_id = await run_hunt(
                conn,
                territory="mock-gym",
                brain=brain.adapter_brain_for(client=client, model=MODEL, registry=reg)("mock-gym"),
                registry=reg,
                retry_budget=RETRY_BUDGET,
                max_iterations=3,
                cost_ceiling_usd=1000.0,
                daemon_spend_ceiling_usd=1.0,
            )
            assert await _row(conn, run_id) == DAEMON_CEILING
        finally:
            await conn.close()
    assert len(requests) == 1


def _status_error(retry_after: str | None) -> anthropic.APIStatusError:
    headers = {} if retry_after is None else {"retry-after": retry_after}
    request = httpx2.Request("POST", brain.MESSAGES_URL)
    response = httpx2.Response(429, headers=headers, request=request)
    return anthropic.APIStatusError("http 429", response=response, body=None)


def test_retry_after_is_read_as_finite_non_negative_seconds() -> None:
    # The reader's own guards, which no loop-level test can see: backoff_delay's max() already
    # turns a negative or NaN floor into the jitter alone. The reader does not cap (30 stays 30).
    cases: dict[str | None, float | None] = {
        None: None,
        "": None,
        "soon": None,
        "Thu, 01 Jan 2099 00:00:00 GMT": None,
        "-1": None,
        "nan": None,
        "inf": None,
        "1e309": None,
        "0": 0.0,
        "3": 3.0,
        "3.5": 3.5,
        "30": 30.0,
    }
    got = {value: brain.brain_call_error(_status_error(value)).retry_after for value in cases}
    assert got == cases


def test_the_default_wait_is_asyncio_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()  # drop conftest's recorder for this test
    assert loop.retry_wait is asyncio.sleep


async def test_a_seeded_random_source_drives_the_jitter(
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
    retry_waits: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(loop, "jitter_source", random.Random(7))
    await _hunt(tmp_path, mock_anthropic, _status(503), stream=False)
    expected = random.Random(7)
    assert retry_waits == [expected.uniform(0.0, 0.5), expected.uniform(0.0, 1.0)]


class _Top:
    """A jitter source that always draws the top of its range, to read the backoff's bounds."""

    def uniform(self, a: float, b: float) -> float:
        return b


def test_the_backoff_bound_doubles_to_its_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loop, "jitter_source", _Top())
    assert [loop.backoff_delay(n) for n in range(1, 8)] == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0, 8.0]
    assert loop.backoff_delay(10_000) == loop.BACKOFF_CAP_S  # no float overflow on a long budget
    assert loop.backoff_delay(1, retry_after=3.0) == 3.0
    assert loop.backoff_delay(1, retry_after=30.0) == loop.BACKOFF_CAP_S
    assert loop.backoff_delay(3, retry_after=0.1) == 2.0  # the jitter's draw when it is longer


async def test_tool_retries_follow_the_backoff_bound(
    tmp_path: Path,
    scripted_brain: BrainFactory,
    retry_waits: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(loop, "jitter_source", _Top())

    async def always() -> str:
        raise RetryableToolError("transient")

    await _tool_hunt(tmp_path, scripted_brain, always, retry_budget=6)
    assert retry_waits == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]
