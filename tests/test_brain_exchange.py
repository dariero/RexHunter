"""W3.3-E2 (fragility #3, invariant 6): every brain attempt stores the bytes it sent and received.

The hazards at 070ed9a (unchanged by W3.3-W): H2, a brain call's response bytes were not stored as
received (the plain path stored only the parsed turn; the stream path read text lines and kept a
re-encoded assembly); H3, the request body sent was stored nowhere; H4, a terminal decision appends
no event (ADR:128, :220), so the last call's bytes left no trace and a dead run could not become a
pytest fixture from its log alone (ADR:30).

W3.3 (forks chosen 17 Nov 2026 under Darie's delegation: E2, R1, S1, X2): the adapter attaches a
`BrainExchangeEvent` to what it returns or raises, and the loop appends it before the attempt's
other events (its thinking deltas, written during the call, come first). `raw_request` is the body
httpx2 put on the wire (`http_request.content`, or the SDK error's `request`); `raw_response` is the
body as received, content-decoded, every chunk `iter_bytes` gave on the stream path; `status` is the
HTTP status, so a run rebuilds from its own log. The event is never published, and the SSE backfill
and the projection's rows leave it out. A brain that reports no bytes (the stub and scripted brains)
writes none.

Every response is a recorded fixture's bytes or built here; nothing under tests/fixtures/ is
written. Zero network: a `MockTransport` answers every call.
"""

import json
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import anthropic
import httpx2
import pytest

from rexhunter import brain, db, hunt_smoke, hunt_smoke_thinking, server, view, viewstate
from rexhunter.events import (
    BrainExchangeEvent,
    ErrorEvent,
    ThinkingDelta,
    ToolCallEvent,
    TrajectoryEvent,
)
from rexhunter.loop import Context, Decision, NeedsHelp, ThinkingSink, ToolCallDecision, run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

FIX = Path(__file__).parent / "fixtures"
PLAIN_RUN = FIX / "hunt_5b62bc48-7577-4a52-b003-bdabf592061d"  # Sonnet 5.5, plain, needs_help
STREAM_RUN = FIX / "hunt_54590aec-de50-4727-a0a2-9e533fb02efa"  # Sonnet 5.5, streamed, needs_help
MODEL = "claude-sonnet-5-5"  # both runs were recorded on it (W2.5-R1); the built responses too
MockAnthropic = Callable[..., anthropic.AsyncAnthropic]
Respond = Callable[[httpx2.Request], httpx2.Response]

MALFORMED = ("aborted", "malformed provider response")
BRAIN_CALL_FAILED = ("aborted", "brain call failed")
SNIFF: dict[str, Any] = {
    "type": "tool_use",
    "id": "toolu_w33",
    "name": "sniff",
    "input": {"prey": "x"},
}
NEEDS_HELP: dict[str, Any] = {
    "type": "tool_use",
    "id": "toolu_w33h",
    "name": "needs_help",
    "input": {},
}
CATCH: dict[str, Any] = {**SNIFF, "name": "hunt_complete", "input": {"catch": ["posting:x"]}}
USAGE: dict[str, Any] = {"input_tokens": 100, "output_tokens": 20}


async def sniff(prey: str) -> str:
    """Sniff a territory for postings."""
    return f"posting:{prey}"


def _reg() -> ToolRegistry:
    reg = ToolRegistry()
    reg.tool(sniff)
    return reg


def _stored_bytes(evs: Sequence[TrajectoryEvent]) -> list[bytes]:
    return [v for e in evs for v in e.__dict__.values() if isinstance(v, bytes) and v]


def _exchanges(evs: Sequence[TrajectoryEvent]) -> list[BrainExchangeEvent]:
    return [e for e in evs if isinstance(e, BrainExchangeEvent)]


def _types(evs: Sequence[TrajectoryEvent]) -> list[str]:
    return [e.type for e in evs]


def _recorded_bodies(run_dir: Path) -> list[bytes]:
    return [p.read_bytes() for p in sorted(run_dir.glob("call_*.json"))]


# ── Responses, built from literals ───────────────────────────────────────────────────────


def _plain(stop_reason: str, content: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        {
            "id": "msg_w33",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "content": content,
            "usage": USAGE,
        }
    ).encode()


def _frame(data: dict[str, Any]) -> bytes:
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()


def _stream_frames(block: dict[str, Any], *, close: bool = True) -> list[bytes]:
    """One tool_use turn as SSE frames: message_start, the block (start, input, stop),
    message_delta, message_stop. `close=False` leaves out the block's content_block_stop."""
    start: dict[str, Any] = {
        "model": MODEL,
        "content": [],
        "usage": {"input_tokens": 100, "output_tokens": 1},
    }
    frames = [
        _frame({"type": "message_start", "message": start}),
        _frame(
            {"type": "content_block_start", "index": 0, "content_block": {**block, "input": {}}}
        ),
        _frame(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])},
            }
        ),
    ]
    if close:
        frames.append(_frame({"type": "content_block_stop", "index": 0}))
    frames.append(
        _frame(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 20},
            }
        )
    )
    frames.append(_frame({"type": "message_stop"}))
    return frames


def _stream(block: dict[str, Any]) -> bytes:
    return b"".join(_stream_frames(block))


def _error_frame(error_type: str) -> bytes:
    data = {"type": "error", "error": {"type": error_type, "message": "An error mid-stream"}}
    return f"event: error\ndata: {json.dumps(data)}\n\n".encode()


def _ok(body: bytes) -> Respond:
    return lambda _request: httpx2.Response(200, content=body)


def _status(code: int) -> Respond:
    return lambda _request: httpx2.Response(code, content=b"status body")


def _connect_error(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError("connection refused", request=request)


class _ResetAfterMessageStart(httpx2.AsyncByteStream):
    """A 200 stream that sends its message_start, then the connection resets."""

    FIRST = _stream_frames(SNIFF)[0]

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.FIRST
        raise httpx2.ReadError("connection reset mid-stream")

    async def aclose(self) -> None:
        pass


def _reset(_request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, stream=_ResetAfterMessageStart())


class Hunt(NamedTuple):
    events: list[TrajectoryEvent]
    row: tuple[str, str | None]
    requests: list[bytes]  # every request body the transport got, in order
    db_path: Path
    run_id: str


async def _hunt(
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
    responses: Sequence[Respond],
    *,
    stream: bool,
    published: list[str] | None = None,
    name: str = "rex.db",
    max_tokens: int = 1024,
    thinking: dict[str, Any] | None = None,
) -> Hunt:
    """One run_hunt through the real adapter, the n-th request answered by `responses[n]` (the
    last one repeats); `published`, if given, collects the `type` of every event published."""

    def publish(_event_id: int, payload: str) -> None:
        if published is not None:
            published.append(str(json.loads(payload)["type"]))

    got: list[bytes] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        got.append(request.content)
        return responses[min(len(got), len(responses)) - 1](request)

    db_path = tmp_path / name
    async with mock_anthropic(handler, api_key="offline") as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            model=MODEL,
            registry=_reg(),
            max_tokens=max_tokens,
            thinking=thinking,
            stream=stream,
        )
        conn = await db.connect(db_path)
        try:
            run_id = await run_hunt(
                conn,
                territory="mock-gym",
                brain=brain_for("mock-gym"),
                registry=_reg(),
                max_iterations=3,
                tool_timeout_s=5.0,
                publish=publish,
            )
            evs = await db.read_events(conn, run_id)
            async with conn.execute(
                "SELECT outcome, abort_reason FROM runs WHERE id = ?", (run_id,)
            ) as cur:
                row = await cur.fetchone()
        finally:
            await conn.close()
    assert row is not None
    return Hunt(evs, (str(row[0]), row[1]), got, db_path, run_id)


async def _recorded(
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
    run_dir: Path,
    *,
    stream: bool,
    published: list[str] | None = None,
) -> tuple[list[bytes], Hunt]:
    """A recorded Sonnet 5.5 run's responses through the adapter, on its own settings."""
    bodies = _recorded_bodies(run_dir)
    smoke = hunt_smoke_thinking if stream else hunt_smoke
    hunt = await _hunt(
        tmp_path,
        mock_anthropic,
        [_ok(body) for body in bodies],
        stream=stream,
        published=published,
        max_tokens=smoke.HUNT_MAX_TOKENS,
        thinking=hunt_smoke_thinking.THINKING if stream else None,
    )
    return bodies, hunt


async def _adapter_run(
    tmp_path: Path, mock_anthropic: MockAnthropic, run_dir: Path, *, stream: bool
) -> tuple[list[bytes], list[bytes], list[TrajectoryEvent], tuple[str, str | None]]:
    """The hazard tests' driver (cp1 §4): (bodies sent, request bodies the transport got, the run's
    whole log, (outcome, reason))."""
    bodies, hunt = await _recorded(tmp_path, mock_anthropic, run_dir, stream=stream)
    return bodies, hunt.requests, hunt.events, hunt.row


# ── H2 to H4, as written at cp1 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
async def test_h2_every_response_body_is_in_the_log_as_received(
    stream: bool, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    run_dir = STREAM_RUN if stream else PLAIN_RUN
    bodies, _got, evs, _row = await _adapter_run(tmp_path, mock_anthropic, run_dir, stream=stream)
    stored = _stored_bytes(evs)
    assert [body in stored for body in bodies] == [True] * len(bodies)


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
async def test_h3_every_request_body_sent_is_in_the_log(
    stream: bool, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    run_dir = STREAM_RUN if stream else PLAIN_RUN
    _bodies, got, evs, _row = await _adapter_run(tmp_path, mock_anthropic, run_dir, stream=stream)
    stored = _stored_bytes(evs)
    assert len(got) == 3  # never vacuous: three calls, as recorded
    assert [request in stored for request in got] == [True] * len(got)


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
async def test_h4_the_terminal_calls_bytes_are_in_the_log(
    stream: bool, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    run_dir = STREAM_RUN if stream else PLAIN_RUN
    bodies, got, evs, row = await _adapter_run(tmp_path, mock_anthropic, run_dir, stream=stream)
    assert row == ("needs_help", None)  # call three is the terminal decision
    stored = _stored_bytes(evs)
    assert (bodies[-1] in stored, got[-1] in stored) == (True, True)


# ── One exchange per attempt, before the attempt's other events (M4, M5, M6) ─────────────

E, U, E_ = "brain_exchange", "usage", "error"
OUTCOMES: list[tuple[str, list[Respond], bool, list[str], tuple[str, str | None]]] = [
    (
        "plain-tool-call-then-needs-help",
        [_ok(_plain("tool_use", [SNIFF])), _ok(_plain("tool_use", [NEEDS_HELP]))],
        False,
        [E, U, "tool_call", "tool_result", E, U],
        ("needs_help", None),
    ),
    (
        "stream-tool-call-then-needs-help",
        [_ok(_stream(SNIFF)), _ok(_stream(NEEDS_HELP))],
        True,
        [E, U, "tool_call", "tool_result", E, U],
        ("needs_help", None),
    ),
    (
        "plain-hunt-complete",
        [_ok(_plain("tool_use", [CATCH]))],
        False,
        [E, U, "prey_captured"],
        ("completed", None),
    ),
    ("plain-refusal", [_ok(_plain("max_tokens", [SNIFF]))], False, [E, U, E_], MALFORMED),
    (
        "stream-refusal-block-left-open",
        [_ok(b"".join(_stream_frames(SNIFF, close=False)))],
        True,
        [E, U, E_],
        MALFORMED,
    ),
    (
        "stream-cut-before-message-stop-retried",
        [_ok(b"".join(_stream_frames(SNIFF)[:-1]))],
        True,
        [E, U, E_] * 3,
        BRAIN_CALL_FAILED,
    ),
    (
        "stream-retryable-error-frame",
        [_ok(b"".join(_stream_frames(SNIFF)[:-2]) + _error_frame("overloaded_error"))],
        True,
        [E, U, E_] * 3,
        BRAIN_CALL_FAILED,
    ),
    (
        "stream-fatal-error-frame",
        [_ok(b"".join(_stream_frames(SNIFF)[:-2]) + _error_frame("invalid_request_error"))],
        True,
        [E, U, E_],
        BRAIN_CALL_FAILED,
    ),
    ("plain-status-529-retried", [_status(529)], False, [E, E_] * 3, BRAIN_CALL_FAILED),
    ("plain-connection-failure-retried", [_connect_error], False, [E, E_] * 3, BRAIN_CALL_FAILED),
    ("stream-reset-mid-stream-retried", [_reset], True, [E, U, E_] * 3, BRAIN_CALL_FAILED),
]


@pytest.mark.parametrize(
    ("responses", "stream", "types", "row"),
    [case[1:] for case in OUTCOMES],
    ids=[case[0] for case in OUTCOMES],
)
async def test_every_attempt_stores_one_exchange_before_its_other_events(
    responses: list[Respond],
    stream: bool,
    types: list[str],
    row: tuple[str, str | None],
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
) -> None:
    hunt = await _hunt(tmp_path, mock_anthropic, responses, stream=stream)
    assert _types(hunt.events) == types
    assert hunt.row == row
    assert len(_exchanges(hunt.events)) == len(hunt.requests)  # one per attempt that was sent


async def test_a_calls_thinking_deltas_come_before_its_exchange(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # Deltas are written during the call (Unit 3b); the exchange once the call has its answer, then
    # the usage (the recorded thinking-on run: 18 and 19 deltas on calls two and three).
    _bodies, hunt = await _recorded(tmp_path, mock_anthropic, STREAM_RUN, stream=True)
    deltas = ["thinking_delta"]
    turn = [U, "tool_call", "tool_result"]
    assert _types(hunt.events) == [E, *turn, *deltas * 18, E, *turn, *deltas * 19, E, U]


def test_the_exchange_never_reaches_the_tools_the_model_sees() -> None:
    # HuntComplete and NeedsHelp carry the exchange and the usage, and also derive the terminal
    # tools' `input_schema` (`brain.build_tools`). Both carriers stay out of it: `exchange` since
    # W3.3 (the wire check found it there before `SkipJsonSchema`), `usage` since W3.4 (W3.3-F1).
    # So `catch` is the one property left, and no tool carries a `$defs` entry.
    tools = {tool["name"]: tool["input_schema"] for tool in brain.build_tools(_reg())}
    properties = {
        name: set(tools[name].get("properties", {})) for name in ("hunt_complete", "needs_help")
    }
    assert properties == {"hunt_complete": {"catch"}, "needs_help": set()}
    defs = {key for schema in tools.values() for key in schema.get("$defs", {})}
    assert "BrainExchangeEvent" not in defs
    assert defs == set()  # F1 (W3.3-F1): UsageEvent's entry left with `usage`


async def test_a_brain_that_reports_no_bytes_writes_no_exchange(tmp_path: Path) -> None:
    decisions = iter([ToolCallDecision(tool="sniff", args={"prey": "x"}), NeedsHelp()])

    async def scripted(_context: Context, _sink: ThinkingSink) -> Decision:
        return next(decisions)

    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await run_hunt(
            conn, territory="t", brain=scripted, registry=_reg(), tool_timeout_s=5.0
        )
        assert _types(await db.read_events(conn, run_id)) == ["tool_call", "tool_result"]
    finally:
        await conn.close()


# ── What the exchange holds (R1, S1, M7, M8, M12) ────────────────────────────────────────


@pytest.mark.parametrize(
    ("responses", "stream"),
    [
        ([_ok(_plain("tool_use", [SNIFF])), _ok(_plain("tool_use", [NEEDS_HELP]))], False),
        ([_ok(_stream(SNIFF)), _ok(_stream(NEEDS_HELP))], True),
        ([_status(529)], False),
    ],
    ids=["plain", "stream", "status-error"],
)
async def test_the_stored_request_is_the_body_the_transport_got(
    responses: list[Respond], stream: bool, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    hunt = await _hunt(tmp_path, mock_anthropic, responses, stream=stream)
    assert len(hunt.requests) > 1  # never vacuous: a second turn, or a retry
    assert [x.raw_request for x in _exchanges(hunt.events)] == hunt.requests


@pytest.mark.parametrize(("run_dir", "stream"), [(PLAIN_RUN, False), (STREAM_RUN, True)])
async def test_the_stored_response_is_the_body_as_received(
    run_dir: Path, stream: bool, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    bodies, hunt = await _recorded(tmp_path, mock_anthropic, run_dir, stream=stream)
    exchanges = _exchanges(hunt.events)
    assert [x.raw_response for x in exchanges] == bodies
    assert [x.status for x in exchanges] == [200, 200, 200]


async def test_a_stored_stream_body_assembles_to_the_stored_turn(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # The stored bytes are the SSE body itself: fed back through the offline framer and the
    # assembler, each gives the turn the run stored (S1), and the last one its terminal decision.
    _bodies, hunt = await _recorded(tmp_path, mock_anthropic, STREAM_RUN, stream=True)
    assembled: list[dict[str, Any]] = []
    for exchange in _exchanges(hunt.events):
        assert exchange.raw_response is not None
        assembler = brain.StreamAssembler()
        for event in brain.iter_sse_events(exchange.raw_response):
            assembler.feed(event)
        assert assembler.finished
        assembled.append(assembler.assembled())
    calls = [e for e in hunt.events if isinstance(e, ToolCallEvent)]
    assert [json.loads(c.assistant_content) for c in calls] == [a["content"] for a in assembled[:2]]
    assert isinstance(brain.parse_decision(json.dumps(assembled[2]).encode()), NeedsHelp)


async def test_a_reset_mid_stream_stores_the_bytes_received_before_the_cut(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    hunt = await _hunt(tmp_path, mock_anthropic, [_reset], stream=True)
    exchanges = _exchanges(hunt.events)
    assert [(x.raw_response, x.status) for x in exchanges] == [
        (_ResetAfterMessageStart.FIRST, 200)
    ] * 3


async def test_a_connection_failure_stores_the_request_and_no_response(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    hunt = await _hunt(tmp_path, mock_anthropic, [_connect_error], stream=False)
    exchanges = _exchanges(hunt.events)
    assert [(x.raw_request, x.raw_response, x.status) for x in exchanges] == [
        (request, None, None) for request in hunt.requests
    ]


async def test_a_status_error_stores_its_status_and_body(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    hunt = await _hunt(tmp_path, mock_anthropic, [_status(529)], stream=False)
    assert [(x.raw_response, x.status) for x in _exchanges(hunt.events)] == [
        (b"status body", 529)
    ] * 3


class _Chunks(httpx2.AsyncByteStream):
    """A body sent in chunks of `size` bytes."""

    def __init__(self, body: bytes, size: int) -> None:
        self.body, self.size = body, size

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for i in range(0, len(self.body), self.size):
            yield self.body[i : i + self.size]

    async def aclose(self) -> None:
        pass


def _framing_bodies() -> list[bytes]:
    recorded = _recorded_bodies(STREAM_RUN)
    return [
        *recorded,
        recorded[0].replace(b"\n", b"\r\n"),  # CRLF framing
        recorded[0].replace(b"mock-gym", f"mock{chr(0x2028)}gym{chr(0x85)}".encode()),  # in data
        recorded[0][:-7],  # cut inside a line
    ]


@pytest.mark.parametrize("size", [1, 2, 3, 7, 64, 4096])
async def test_the_lines_framed_from_the_received_bytes_are_the_lines_iter_lines_gave(
    size: int,
) -> None:
    # S1 frames the lines for `_drive_stream` from `iter_bytes`; the lines must be those the SDK's
    # `iter_lines` (httpx2's `aiter_lines`) gave before W3.3, wherever a chunk ends, and the kept
    # chunks must be the body exactly.
    for body in _framing_bodies():
        before = httpx2.Response(200, stream=_Chunks(body, size)).aiter_lines()
        want = [line async for line in before]
        received: list[bytes] = []
        framed = brain._received_lines(  # pyright: ignore[reportPrivateUsage]
            _Chunks(body, size).__aiter__(), received, encoding="utf-8"
        )
        assert [line async for line in framed] == want
        assert b"".join(received) == body


# ── The existing events keep their bytes (R0-b) ──────────────────────────────────────────


def _assembled_bytes(body: bytes) -> bytes:
    """What the stream path has always stored on its refusals: the re-encoded assembly."""
    assembler = brain.StreamAssembler()
    for event in brain.iter_sse_events(body):
        assembler.feed(event)
    return json.dumps(assembler.assembled()).encode()


@pytest.mark.parametrize(
    ("body", "stream"),
    [
        (_plain("max_tokens", [SNIFF]), False),
        (b"".join(_stream_frames(SNIFF, close=False)), True),  # BrainParseError
        (b"".join(_stream_frames(SNIFF)[:-1]), True),  # BrainCallError, retried
    ],
    ids=["plain-refusal", "stream-refusal", "stream-cut"],
)
async def test_the_error_events_keep_the_bytes_they_stored_before(
    body: bytes, stream: bool, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    hunt = await _hunt(tmp_path, mock_anthropic, [_ok(body)], stream=stream)
    errors = [e for e in hunt.events if isinstance(e, ErrorEvent)]
    expected = _assembled_bytes(body) if stream else body
    assert [(e.tool, e.raw_request, e.raw_response) for e in errors] == [
        ("<brain>", b"", expected)
    ] * len(errors)
    # The bytes as received go in the exchange only.
    assert [x.raw_response for x in _exchanges(hunt.events)] == [body] * len(errors)


# ── Kept off the live stream and out of the projection (X2: M9, M10, M11) ────────────────


async def test_the_exchange_is_never_published(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    published: list[str] = []
    _bodies, hunt = await _recorded(
        tmp_path, mock_anthropic, STREAM_RUN, stream=True, published=published
    )
    assert _exchanges(hunt.events)  # never vacuous
    assert published == [t for t in _types(hunt.events) if t != E]


async def test_the_sse_backfill_leaves_the_exchange_out(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    _bodies, hunt = await _recorded(tmp_path, mock_anthropic, PLAIN_RUN, stream=False)
    conn = await db.connect(hunt.db_path)
    try:
        backfill = await server.catch_up(conn, 0)
    finally:
        await conn.close()
    assert _exchanges(hunt.events)  # never vacuous
    assert [json.loads(env.data)["type"] for env in backfill] == [
        t for t in _types(hunt.events) if t != E
    ]


async def test_the_projection_rows_leave_the_exchange_out(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    _bodies, hunt = await _recorded(tmp_path, mock_anthropic, PLAIN_RUN, stream=False)
    conn = await db.connect(hunt.db_path)
    try:
        live = await viewstate.read_log_rows(conn)
        ghost = await viewstate.read_log_rows(conn, run_id=hunt.run_id)
    finally:
        await conn.close()
    want = [t for t in _types(hunt.events) if t != E]
    assert [r.event.type for r in live] == want
    assert [r.event.type for r in ghost] == want


def test_the_reducer_folds_an_exchange_to_nothing() -> None:
    row = view.LogRow(
        id=7,
        seq=0,
        run_id="r",
        created_at="",
        event=BrainExchangeEvent(raw_request=b"{}", raw_response=b"{}", status=200),
    )
    assert view.fold([row]) == view.Accumulator(high_water=7)


# ── A run rebuilds from its own log (DoD #5, ADR:30) ─────────────────────────────────────


def _from_log(exchanges: list[BrainExchangeEvent], sent: list[bytes]) -> list[Respond]:
    """A transport answer per stored exchange: its status and its response bytes, as received.
    Each request the rebuild sends is kept, to compare with the stored request."""

    def answer(exchange: BrainExchangeEvent) -> Respond:
        def respond(request: httpx2.Request) -> httpx2.Response:
            sent.append(request.content)
            return httpx2.Response(exchange.status or 200, content=exchange.raw_response or b"")

        return respond

    return [answer(x) for x in exchanges]


def _norm(evs: Sequence[TrajectoryEvent]) -> list[dict[str, Any]]:
    return [json.loads(e.model_dump_json()) for e in evs]


@pytest.mark.parametrize(
    ("responses", "stream"),
    [
        ([_ok(body) for body in _recorded_bodies(PLAIN_RUN)], False),
        ([_ok(body) for body in _recorded_bodies(STREAM_RUN)], True),
        ([_status(529), _ok(_plain("tool_use", [NEEDS_HELP]))], False),
    ],
    ids=["recorded-plain", "recorded-stream", "a-529-then-needs-help"],
)
async def test_a_run_rebuilds_from_its_own_log(
    responses: list[Respond], stream: bool, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    thinking = hunt_smoke_thinking.THINKING if stream else None
    max_tokens = brain.HUNT_MAX_TOKENS if stream else 1024
    first = await _hunt(
        tmp_path, mock_anthropic, responses, stream=stream, thinking=thinking, max_tokens=max_tokens
    )
    exchanges = _exchanges(first.events)
    assert len(exchanges) == len(first.requests) > 1  # never vacuous
    sent: list[bytes] = []
    rebuilt = await _hunt(
        tmp_path,
        mock_anthropic,
        _from_log(exchanges, sent),
        stream=stream,
        name="rebuilt.db",
        thinking=thinking,
        max_tokens=max_tokens,
    )
    assert sent == [x.raw_request for x in exchanges]  # the same requests, byte for byte
    assert (_norm(rebuilt.events), rebuilt.row) == (_norm(first.events), first.row)
    assert any(isinstance(e, ThinkingDelta) for e in first.events) is stream
