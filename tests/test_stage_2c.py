"""P5 · Unit 2c.1 — brain-in-loop, offline half (no spend, no network).

The adapter now DRIVES the loop. Five contracts, each provable offline:

  1. Context → messages projection — the trajectory log folds into a valid Anthropic `messages`
     array (invariant 2, derived per call — not a second stored chat history); `tool_use_id`
     correlation survives the fold.
  2. System prompt — a stable `HUNT_SYSTEM_PROMPT` constant, threaded into the request.
  3. Retryable HTTP classification: real SDK / transport failures, mapped by the adapter to
     BrainCallError (W2.0), flow through the EXISTING `classify` (429/5xx/timeouts/connect →
     retryable; 4xx / parse / validation → fatal).
  4. Cost-ceiling breaker — per-call usage folds into a running spend from `UsageEvent`s
     (invariant 5, no counter); crossing `COST_CEILING_USD` aborts before the next brain call.
     Proven by replaying the 2b golden fixture under a deliberately low ceiling.
  5. Config-gated brain — `REXHUNTER_BRAIN` selects the brain; default `stub` (no client, no
     spend), `live` is opt-in. This is the containment for the autonomous-spender surface.

Every test injects an httpx2 `MockTransport` (through the `mock_anthropic` fixture, the production
SDK client since W2.0) or hand-built events: zero network, zero spend. The 2b fixture
(`tests/fixtures/smoke_sonnet5.json`, a genuine Sonnet 5 response) anchors the breaker.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import ValidationError

from rexhunter import brain, cost, db, hunt_smoke, stub
from rexhunter.events import (
    BrainParseError,
    ErrorEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from rexhunter.loop import BrainCallError, Context, Decision, ThinkingSink, classify, run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

FIXTURE = Path(__file__).parent / "fixtures" / "smoke_sonnet5.json"

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]
Handler = Callable[[httpx2.Request], httpx2.Response]


async def _drop(_text: str) -> None:
    """A no-op thinking sink for the non-streaming request-shape tests (never called there)."""


async def sniff(prey: str) -> str:
    """Sniff a territory for postings."""
    return f"posting:{prey}"


def _sniff_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.tool(sniff)  # registered under the name `sniff` — matches the fixture's tool_use name
    return reg


# ── 1. Context → messages projection (invariant 2) ────────────────────────────


def test_context_projects_to_a_valid_messages_array() -> None:
    context = [
        ToolCallEvent(tool="sniff", raw_request=b'{"prey": "acme"}', tool_use_id="toolu_X"),
        ToolResultEvent(
            tool="sniff",
            raw_request=b'{"prey": "acme"}',
            raw_response=b"posting:acme",
            tool_use_id="toolu_X",
        ),
    ]
    messages = brain.project_messages("acme", context)

    # The seed directive is the first (user) turn — territory named, tools offered.
    assert messages[0]["role"] == "user"
    assert isinstance(messages[0]["content"], str) and "acme" in messages[0]["content"]

    # ToolCallEvent → an assistant turn carrying the native tool_use block.
    assert messages[1]["role"] == "assistant"
    tool_use = messages[1]["content"][0]
    assert tool_use["type"] == "tool_use"
    assert tool_use["name"] == "sniff"
    assert tool_use["input"] == {"prey": "acme"}  # from raw_request, parsed
    assert tool_use["id"] == "toolu_X"

    # ToolResultEvent → a user turn carrying the tool_result, correlated by tool_use_id.
    assert messages[2]["role"] == "user"
    tool_result = messages[2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "toolu_X"  # correlation survives the fold
    assert tool_result["content"] == "posting:acme"  # from raw_response, decoded
    assert not tool_result.get("is_error")


def test_projection_maps_a_fatal_tool_error_to_an_is_error_result() -> None:
    context = [
        ToolCallEvent(tool="sniff", raw_request=b'{"prey": "acme"}', tool_use_id="toolu_Y"),
        ErrorEvent(tool="sniff", retryable=False, error="boom", raw_request=b'{"prey": "acme"}'),
    ]
    messages = brain.project_messages("acme", context)
    tool_result = messages[2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert (
        tool_result["tool_use_id"] == "toolu_Y"
    )  # paired to the open tool_use (ErrorEvent has no id)
    assert tool_result["is_error"] is True


def test_projection_skips_retryable_and_brain_loop_errors() -> None:
    context = [
        ToolCallEvent(tool="sniff", raw_request=b'{"prey": "a"}', tool_use_id="t1"),
        ErrorEvent(tool="sniff", retryable=True, error="blip", raw_request=b'{"prey": "a"}'),
        ToolResultEvent(
            tool="sniff", raw_request=b'{"prey": "a"}', raw_response=b"posting:a", tool_use_id="t1"
        ),
        ErrorEvent(tool="<brain>", retryable=True, error="429", raw_request=b""),
    ]
    messages = brain.project_messages("a", context)
    # seed + assistant(tool_use) + user(tool_result success). The retryable retry error is
    # superseded by the success; the <brain> error is not a conversation turn — both skipped.
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[2]["content"][0]["content"] == "posting:a"
    assert not messages[2]["content"][0].get("is_error")


# ── 2. System prompt ──────────────────────────────────────────────────────────


def test_hunt_system_prompt_is_a_module_constant() -> None:
    assert isinstance(brain.HUNT_SYSTEM_PROMPT, str) and brain.HUNT_SYSTEM_PROMPT.strip()


def test_build_request_body_threads_system_and_omits_it_when_absent() -> None:
    tools = brain.build_tools(_sniff_registry())
    msgs = [{"role": "user", "content": "hi"}]
    with_system = brain.build_request_body(
        model=brain.SMOKE_MODEL,
        max_tokens=1024,
        messages=msgs,
        tools=tools,
        system=brain.HUNT_SYSTEM_PROMPT,
    )
    assert with_system["system"] == brain.HUNT_SYSTEM_PROMPT
    # Omitted when not passed → the 2b smoke request shape (and its guard test) stays green.
    without = brain.build_request_body(
        model=brain.SMOKE_MODEL, max_tokens=1024, messages=msgs, tools=tools
    )
    assert "system" not in without and "thinking" not in without


async def test_adapter_sends_system_prompt_and_disables_thinking(
    mock_anthropic: MockAnthropic,
) -> None:
    captured: dict[str, httpx2.Request] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["req"] = request
        return httpx2.Response(200, content=FIXTURE.read_bytes())

    async with mock_anthropic(handler) as client:
        brain_for = brain.adapter_brain_for(
            client=client, model=brain.SMOKE_MODEL, registry=_sniff_registry()
        )
        await brain_for("mock-gym")([], _drop)

    body = json.loads(captured["req"].content)
    assert body["system"] == brain.HUNT_SYSTEM_PROMPT
    # Bare tool_use replay turns (no thinking sibling) until capture lands in Unit 3; Sonnet 5
    # accepts an explicit disabled, sidestepping the thinking-block replay rule.
    assert body["thinking"] == {"type": "disabled"}


# ── 3. Retryable HTTP classification (extend the taxonomy, don't fork it) ──────


# W2.0: the brain's transport failures reach the loop only as the adapter's BrainCallError, so
# the taxonomy cases move onto the adapter's mapping, with the same cases and the same verdicts:
# the SDK error (or bare transport error) -> brain.brain_call_error -> the loop's classify.

_REQ = httpx2.Request("POST", brain.MESSAGES_URL)


def _sdk_status_error(code: int) -> anthropic.APIStatusError:
    response = httpx2.Response(code, request=_REQ)
    return anthropic.APIStatusError(f"http {code}", response=response, body=None)


def _sdk_connection_error(cause: httpx2.TransportError) -> anthropic.APIConnectionError:
    """The SDK's wrapper for a failure before the response, as `_attempt_request` builds it."""
    if isinstance(cause, httpx2.TimeoutException):
        wrapped: anthropic.APIConnectionError = anthropic.APITimeoutError(request=_REQ)
    else:
        wrapped = anthropic.APIConnectionError(request=_REQ)
    wrapped.__cause__ = cause
    return wrapped


def test_the_adapter_maps_transport_errors_to_retryable() -> None:
    connect_timeout = _sdk_connection_error(httpx2.ConnectTimeout("slow", request=_REQ))
    read_timeout = _sdk_connection_error(httpx2.ReadTimeout("slow", request=_REQ))
    connect_error = _sdk_connection_error(httpx2.ConnectError("refused", request=_REQ))
    assert classify(brain.brain_call_error(connect_timeout)) is True
    assert classify(brain.brain_call_error(read_timeout)) is True
    assert classify(brain.brain_call_error(connect_error)) is True


def test_the_adapter_maps_429_and_5xx_to_retryable_and_4xx_to_fatal() -> None:
    assert classify(brain.brain_call_error(_sdk_status_error(429))) is True
    assert classify(brain.brain_call_error(_sdk_status_error(503))) is True
    assert classify(brain.brain_call_error(_sdk_status_error(500))) is True
    assert classify(brain.brain_call_error(_sdk_status_error(400))) is False
    assert classify(brain.brain_call_error(_sdk_status_error(404))) is False


def test_classify_parse_and_validation_errors_are_fatal() -> None:
    assert classify(BrainParseError(raw=b"{}", detail="bad")) is False
    with pytest.raises(ValidationError) as excinfo:
        ToolCallEvent.model_validate({})  # missing required fields
    assert classify(excinfo.value) is False


def _status_handler(code: int, body: bytes, seen: list[httpx2.Request] | None = None) -> Handler:
    """Every request gets the same non-2xx status; the SDK must surface it as a status error.
    `retry-after-ms` is inert under max_retries=0; it only keeps a retrying mutant fast."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if seen is not None:
            seen.append(request)
        return httpx2.Response(code, content=body, headers={"retry-after-ms": "1"})

    return handler


async def test_brain_retries_a_transient_5xx_then_aborts(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # Integrated proof of the path classify only unit-tests: the SDK's status error → BrainCallError
    # → _call_brain retries within budget, tags "<brain>", captures the body, then aborts.
    reg = _sniff_registry()
    async with mock_anthropic(_status_handler(503, b"upstream unavailable")) as client:
        brain_for = brain.adapter_brain_for(client=client, model=brain.SMOKE_MODEL, registry=reg)
        conn = await db.connect(tmp_path / "rex.db")
        try:
            run_id = await run_hunt(
                conn,
                territory="mock-gym",
                brain=brain_for("mock-gym"),
                registry=reg,
                retry_budget=2,  # 3 attempts total
                max_iterations=10,
                cost_ceiling_usd=1000.0,
                tool_timeout_s=5.0,
            )
            errors = [e for e in await db.read_events(conn, run_id) if isinstance(e, ErrorEvent)]
            assert len(errors) == 3  # each retry attempt is logged (single writer, invariant 7)
            assert all(e.tool == "<brain>" for e in errors)
            # The flag reflects the error's nature, not "we gave up" — the last stays retryable.
            assert [e.retryable for e in errors] == [True, True, True]
            assert errors[-1].raw_response == b"upstream unavailable"  # invariant 6: the error body
            async with conn.execute(
                "SELECT outcome, abort_reason FROM runs WHERE id = ?", (run_id,)
            ) as cur:
                row = await cur.fetchone()
            assert row == ("aborted", "brain call failed")
        finally:
            await conn.close()


async def test_brain_4xx_is_fatal_without_retry(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    reg = _sniff_registry()
    async with mock_anthropic(_status_handler(400, b"bad request")) as client:
        brain_for = brain.adapter_brain_for(client=client, model=brain.SMOKE_MODEL, registry=reg)
        conn = await db.connect(tmp_path / "rex.db")
        try:
            run_id = await run_hunt(
                conn,
                territory="mock-gym",
                brain=brain_for("mock-gym"),
                registry=reg,
                retry_budget=2,
                max_iterations=10,
                cost_ceiling_usd=1000.0,
                tool_timeout_s=5.0,
            )
            errors = [e for e in await db.read_events(conn, run_id) if isinstance(e, ErrorEvent)]
            assert len(errors) == 1  # 4xx is fatal — no retry burns the budget
            assert errors[0].tool == "<brain>" and errors[0].retryable is False
            assert errors[0].raw_response == b"bad request"  # invariant 6
            async with conn.execute(
                "SELECT outcome, abort_reason FROM runs WHERE id = ?", (run_id,)
            ) as cur:
                row = await cur.fetchone()
            assert row == ("aborted", "brain call failed")
        finally:
            await conn.close()


# ── 3b. W2.0 D1a: the provider-neutral brain error (the loop imports no vendor SDK) ──


def _raising_brain(exc: BaseException):
    """A brain whose every call raises `exc`: no adapter, no transport, no spend."""

    async def brain(_context: Context, _sink: ThinkingSink) -> Decision:
        raise exc

    return brain


async def _hunt_errors(db_path: Path, brain_fn: Any) -> tuple[list[ErrorEvent], Any]:
    conn = await db.connect(db_path)
    try:
        run_id = await run_hunt(
            conn,
            territory="mock-gym",
            brain=brain_fn,
            registry=_sniff_registry(),
            retry_budget=2,  # 3 attempts total
            max_iterations=10,
            cost_ceiling_usd=1000.0,
            tool_timeout_s=5.0,
        )
        errors = [e for e in await db.read_events(conn, run_id) if isinstance(e, ErrorEvent)]
        async with conn.execute(
            "SELECT outcome, abort_reason FROM runs WHERE id = ?", (run_id,)
        ) as cur:
            row = await cur.fetchone()
        return errors, row
    finally:
        await conn.close()


def test_classify_honours_the_brain_call_error_verdict() -> None:
    assert classify(BrainCallError("blip", retryable=True)) is True
    assert classify(BrainCallError("http 400", retryable=False, status=400)) is False


async def test_retryable_brain_call_error_is_retried_within_budget(tmp_path: Path) -> None:
    # The adapter's provider-neutral error: retried within budget, each attempt its own
    # ErrorEvent(tool="<brain>") in the owning task (inv 7), the error body kept (inv 6).
    err = BrainCallError(
        "http 503", retryable=True, status=503, raw_response=b"upstream unavailable"
    )
    errors, row = await _hunt_errors(tmp_path / "rex.db", _raising_brain(err))
    assert len(errors) == 3
    assert all(e.tool == "<brain>" for e in errors)
    assert [e.retryable for e in errors] == [True, True, True]
    assert all(e.raw_response == b"upstream unavailable" for e in errors)
    assert row == ("aborted", "brain call failed")


async def test_fatal_brain_call_error_aborts_without_retry(tmp_path: Path) -> None:
    err = BrainCallError("http 400", retryable=False, status=400, raw_response=b"bad request")
    errors, row = await _hunt_errors(tmp_path / "rex.db", _raising_brain(err))
    assert len(errors) == 1
    assert errors[0].tool == "<brain>" and errors[0].retryable is False
    assert errors[0].raw_response == b"bad request"
    assert row == ("aborted", "brain call failed")


class _ResetMidStream(httpx2.AsyncByteStream):
    """A 200 streaming body that yields its first frame, then the connection resets."""

    async def __aiter__(self):  # type: ignore[override]
        yield b'event: ping\ndata: {"type": "ping"}\n\n'
        raise httpx2.ReadError("connection reset mid-stream")

    async def aclose(self) -> None:
        pass


async def test_a_transport_error_mid_stream_is_retried_as_a_brain_error(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # A failure DURING the streamed body (after the 200 arrived) is a transient transport blip,
    # retried within budget like any other, never the `<loop>` backstop. The SDK does not wrap an
    # error raised while the body streams, so the adapter's mapping must catch the transport's own.
    reg = _sniff_registry()

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, stream=_ResetMidStream())

    async with mock_anthropic(handler) as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            model=brain.SMOKE_MODEL,
            registry=reg,
            thinking=brain.HUNT_THINKING,
            stream=True,
        )
        errors, row = await _hunt_errors(tmp_path / "rex.db", brain_for("mock-gym"))
    assert len(errors) == 3
    assert all(e.tool == "<brain>" and e.retryable for e in errors)
    assert row == ("aborted", "brain call failed")


async def test_a_5xx_makes_exactly_retry_budget_plus_one_requests(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # D1b: the SDK client never retries (max_retries=0), so each loop attempt is exactly ONE HTTP
    # request and the wire sees the loop's budget, no more: retry_budget + 1 requests.
    retry_budget = 2  # the budget _hunt_errors runs with
    seen: list[httpx2.Request] = []
    async with mock_anthropic(_status_handler(503, b"upstream unavailable", seen)) as client:
        brain_for = brain.adapter_brain_for(
            client=client, model=brain.SMOKE_MODEL, registry=_sniff_registry()
        )
        errors, row = await _hunt_errors(tmp_path / "rex.db", brain_for("mock-gym"))
    assert len(seen) == retry_budget + 1
    assert len(errors) == retry_budget + 1  # one ErrorEvent per request
    assert row == ("aborted", "brain call failed")


@pytest.mark.parametrize("stream", [False, True])
async def test_a_redirect_is_never_followed(
    stream: bool, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # HEAD's client never followed redirects, so a 3xx was a fatal status error. The SDK's default
    # client follows them, cross-origin, with the x-api-key and the paid POST body: the factory
    # turns that off, and a 3xx stays one fatal <brain> error with its body kept.
    seen: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(str(request.url))
        if request.url.host == "api.anthropic.com":
            location = {"location": "https://elsewhere.invalid/v1/messages"}
            return httpx2.Response(307, headers=location, content=b"moved")
        return httpx2.Response(200, content=FIXTURE.read_bytes())  # reached only if followed

    async with mock_anthropic(handler) as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            model=brain.SMOKE_MODEL,
            registry=_sniff_registry(),
            thinking=brain.HUNT_THINKING if stream else None,
            stream=stream,
        )
        errors, row = await _hunt_errors(tmp_path / "rex.db", brain_for("mock-gym"))
    assert seen == [brain.MESSAGES_URL]  # one request, never re-sent to another origin
    assert len(errors) == 1
    assert errors[0].tool == "<brain>" and errors[0].retryable is False
    assert errors[0].raw_response == b"moved"
    assert row == ("aborted", "brain call failed")


def _connect_error(request: httpx2.Request) -> Exception:
    return httpx2.ConnectError("refused", request=request)


def _connect_timeout(request: httpx2.Request) -> Exception:
    return httpx2.ConnectTimeout("slow", request=request)


def _transport_bug(_request: httpx2.Request) -> Exception:
    return RuntimeError("bug in a transport")


def _decoding_error(request: httpx2.Request) -> Exception:
    return httpx2.DecodingError("bad gzip body", request=request)


@pytest.mark.parametrize(
    ("make_error", "name"),
    [(_connect_error, "ConnectError"), (_connect_timeout, "ConnectTimeout")],
    ids=["connect-error", "connect-timeout"],
)
async def test_a_connection_failure_before_the_response_is_retried(
    make_error: Callable[[httpx2.Request], Exception],
    name: str,
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
) -> None:
    # The SDK wraps these as APIConnectionError / APITimeoutError; the adapter retries them like
    # HEAD's httpx.ConnectError / ConnectTimeout, and the ErrorEvent names the real cause.
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise make_error(request)

    async with mock_anthropic(handler) as client:
        brain_for = brain.adapter_brain_for(
            client=client, model=brain.SMOKE_MODEL, registry=_sniff_registry()
        )
        errors, row = await _hunt_errors(tmp_path / "rex.db", brain_for("mock-gym"))
    assert len(errors) == 3
    assert all(e.tool == "<brain>" and e.retryable for e in errors)
    assert all(name in e.error for e in errors)  # the cause, not the SDK's generic wrapper text
    assert row == ("aborted", "brain call failed")


@pytest.mark.parametrize(
    ("make_error", "name"),
    [(_transport_bug, "RuntimeError"), (_decoding_error, "DecodingError")],
    ids=["bug", "decoding-error"],
)
async def test_a_non_network_failure_inside_the_sdk_send_reaches_the_loop_backstop(
    make_error: Callable[[httpx2.Request], Exception],
    name: str,
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
) -> None:
    # The SDK wraps EVERY exception raised during send as APIConnectionError. Only a transport
    # failure is a transient blip; anything else (a bug, an undecodable body) is re-raised, so it
    # reaches run_hunt's <loop> backstop once, as at HEAD, and is never re-requested.
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        raise make_error(request)

    async with mock_anthropic(handler) as client:
        brain_for = brain.adapter_brain_for(
            client=client, model=brain.SMOKE_MODEL, registry=_sniff_registry()
        )
        errors, row = await _hunt_errors(tmp_path / "rex.db", brain_for("mock-gym"))
    assert len(seen) == 1
    assert len(errors) == 1
    assert errors[0].tool == "<loop>" and errors[0].retryable is False
    assert name in errors[0].error
    assert row == ("aborted", "unhandled exception in loop")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("code", [429, 529])
async def test_429_and_529_are_retried_on_both_paths(
    code: int, stream: bool, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    async with mock_anthropic(_status_handler(code, b"slow down")) as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            model=brain.SMOKE_MODEL,
            registry=_sniff_registry(),
            thinking=brain.HUNT_THINKING if stream else None,
            stream=stream,
        )
        errors, row = await _hunt_errors(tmp_path / "rex.db", brain_for("mock-gym"))
    assert len(errors) == 3
    assert all(e.tool == "<brain>" and e.retryable for e in errors)
    assert errors[-1].raw_response == b"slow down"
    assert row == ("aborted", "brain call failed")


# ── 4. Cost-ceiling breaker (invariant 5 — folded from UsageEvents) ───────────


def _fixture_handler() -> Handler:
    raw = FIXTURE.read_bytes()  # a real Sonnet 5 response: tool=sniff, usage 1004 in / 80 out

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=raw)

    return handler


async def test_cost_ceiling_aborts_before_the_next_brain_call(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    reg = _sniff_registry()
    async with mock_anthropic(_fixture_handler()) as client:
        brain_for = brain.adapter_brain_for(client=client, model=brain.SMOKE_MODEL, registry=reg)
        conn = await db.connect(tmp_path / "rex.db")
        try:
            run_id = await run_hunt(
                conn,
                territory="mock-gym",
                brain=brain_for("mock-gym"),
                registry=reg,
                cost_ceiling_usd=1e-9,  # any real spend trips it
                max_iterations=50,  # high — prove the COST breaker fires first, not max-iter
                tool_timeout_s=5.0,
            )
            events = await db.read_events(conn, run_id)
            usage = [e for e in events if isinstance(e, UsageEvent)]
            assert len(usage) == 1  # one paid call, then the pre-call check aborts
            assert usage[0].input_tokens == 1004 and usage[0].output_tokens == 80

            async with conn.execute(
                "SELECT outcome, abort_reason FROM runs WHERE id = ?", (run_id,)
            ) as cur:
                row = await cur.fetchone()
            assert row == ("aborted", "cost ceiling")
        finally:
            await conn.close()


async def test_high_ceiling_falls_back_to_max_iterations(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # Cost breaker and max-iter breaker coexist: a never-terminating brain (fixture always sniffs)
    # under a slack ceiling is still bounded by the existing max-iterations breaker.
    reg = _sniff_registry()
    async with mock_anthropic(_fixture_handler()) as client:
        brain_for = brain.adapter_brain_for(client=client, model=brain.SMOKE_MODEL, registry=reg)
        conn = await db.connect(tmp_path / "rex.db")
        try:
            run_id = await run_hunt(
                conn,
                territory="mock-gym",
                brain=brain_for("mock-gym"),
                registry=reg,
                cost_ceiling_usd=1000.0,  # never trips
                max_iterations=3,
                tool_timeout_s=5.0,
            )
            async with conn.execute(
                "SELECT outcome, abort_reason FROM runs WHERE id = ?", (run_id,)
            ) as cur:
                row = await cur.fetchone()
            assert row == ("aborted", "max iterations")
        finally:
            await conn.close()


def test_fold_cost_prices_usage_and_ignores_non_usage() -> None:
    events = [
        UsageEvent(model="claude-sonnet-5", input_tokens=1_000_000, output_tokens=0),
        ToolResultEvent(tool="sniff", raw_request=b"{}", raw_response=b"x"),
        UsageEvent(model="claude-sonnet-5", input_tokens=0, output_tokens=1_000_000),
    ]
    # 1M input @ $3 + 1M output @ $15 = $18.00 (sticker Sonnet 5 pricing)
    assert cost.fold_cost(events) == pytest.approx(18.0)
    # Unknown model must not price to 0 (that would silently disable the guard).
    assert cost.cost_of(UsageEvent(model="???", input_tokens=1_000_000, output_tokens=0)) > 0


# ── 5. Config-gated brain (the autonomous-spender containment) ────────────────


def test_default_brain_is_stub_and_constructs_no_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REXHUNTER_BRAIN", raising=False)
    brain_for, client = brain.select_brain_for(stub.build_registry())
    assert brain_for is stub.stub_brain_for  # the no-spend stub
    assert client is None  # nothing that could hit the network was constructed


async def test_live_brain_is_opt_in_and_never_called(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REXHUNTER_BRAIN", "live")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    brain_for, client = brain.select_brain_for(stub.build_registry())
    try:
        assert client is not None  # explicit opt-in constructed the paid adapter's client
        assert brain_for is not stub.stub_brain_for
    finally:
        if client is not None:
            await client.close()  # closed without ever calling brain() → zero spend


async def test_the_live_client_comes_from_the_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    # The daemon's spending client must be the factory's (max_retries=0 + the pinned wire), not a
    # bare AsyncAnthropic with the SDK defaults (2 retries, Bearer auth, another API version).
    monkeypatch.setenv("REXHUNTER_BRAIN", "live")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://elsewhere.invalid")  # a bare client obeys it
    _brain_for, client = brain.select_brain_for(stub.build_registry())
    assert client is not None
    try:
        assert client.max_retries == 0
        assert str(client.base_url) == brain.ANTHROPIC_BASE_URL
        assert client.default_headers["x-api-key"] == "sk-test-not-real"
        assert client.default_headers["anthropic-version"] == "2023-06-01"
    finally:
        await client.close()


# ── 6. Unit 2c.2 — live request shape + count_tokens pre-flight (offline) ──────


async def _capture_adapter_body(
    mock_anthropic: MockAnthropic, registry: ToolRegistry, territory: str = "mock-gym"
) -> dict[str, Any]:
    """Drive one adapter turn over MockTransport and return the request body it actually sent."""
    captured: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, content=FIXTURE.read_bytes())

    async with mock_anthropic(handler) as client:
        brain_for = brain.adapter_brain_for(
            client=client, model=brain.SMOKE_MODEL, registry=registry
        )
        await brain_for(territory)([], _drop)
    return captured["body"]


async def test_live_request_shape(mock_anthropic: MockAnthropic) -> None:
    body = await _capture_adapter_body(mock_anthropic, _sniff_registry())
    for param in ("temperature", "top_p", "top_k"):
        assert param not in body  # any non-default sampling param → 400 on Sonnet 5
    assert body["thinking"] == {"type": "disabled"}  # accepted on Sonnet 5; only "enabled" 400s
    assert body["tool_choice"]["disable_parallel_tool_use"] is True  # one tool per iteration
    assert body["max_tokens"] >= 1024


async def test_the_plain_request_sends_no_stream_key(mock_anthropic: MockAnthropic) -> None:
    # The non-streaming body must stay HEAD's: no `stream` key at all (not even `false`), so the
    # free pre-flight still prices the exact payload the paid call sends.
    body = await _capture_adapter_body(mock_anthropic, _sniff_registry())
    assert "stream" not in body


def test_build_request_body_omits_tool_choice_when_absent() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    tools = brain.build_tools(_sniff_registry())
    without = brain.build_request_body(
        model=brain.SMOKE_MODEL, max_tokens=1024, messages=msgs, tools=tools
    )
    assert "tool_choice" not in without  # 2b smoke shape unchanged
    with_tc = brain.build_request_body(
        model=brain.SMOKE_MODEL,
        max_tokens=1024,
        messages=msgs,
        tools=tools,
        tool_choice=brain.HUNT_TOOL_CHOICE,
    )
    assert with_tc["tool_choice"] == brain.HUNT_TOOL_CHOICE


async def test_count_tokens_preflight_posts_exact_payload_and_returns_count(
    mock_anthropic: MockAnthropic,
) -> None:
    seen: dict[str, httpx2.Request] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["req"] = request
        return httpx2.Response(200, json={"input_tokens": 1234})

    async with mock_anthropic(handler) as client:
        n = await hunt_smoke.count_tokens_preflight(client, _sniff_registry())

    assert n == 1234
    assert str(seen["req"].url) == brain.COUNT_TOKENS_URL
    count_body = json.loads(seen["req"].content)
    assert "max_tokens" not in count_body  # count_tokens takes no max_tokens
    assert count_body["system"] == brain.HUNT_SYSTEM_PROMPT
    assert count_body["thinking"] == {"type": "disabled"}
    assert count_body["tool_choice"]["disable_parallel_tool_use"] is True


async def test_preflight_payload_matches_adapter_body_minus_max_tokens(
    mock_anthropic: MockAnthropic,
) -> None:
    # Drift guard: the free count must price the exact bytes the paid call will send.
    adapter_body = await _capture_adapter_body(mock_anthropic, _sniff_registry())

    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json={"input_tokens": 1})

    async with mock_anthropic(handler) as client:
        await hunt_smoke.count_tokens_preflight(client, _sniff_registry())

    assert seen["body"] == {k: v for k, v in adapter_body.items() if k != "max_tokens"}


async def test_count_tokens_preflight_stops_on_non_200(mock_anthropic: MockAnthropic) -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, text="bad schema: a paid call must not follow")

    async with mock_anthropic(handler) as client:
        with pytest.raises(SystemExit):
            await hunt_smoke.count_tokens_preflight(client, _sniff_registry())


async def test_capturing_transport_tees_only_messages_responses(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    raw = FIXTURE.read_bytes()
    inner = httpx2.MockTransport(lambda _r: httpx2.Response(200, content=raw))
    tee = hunt_smoke.CapturingTransport(inner, brain.MESSAGES_URL)
    reg = _sniff_registry()
    async with mock_anthropic(tee) as client:
        brain_for = brain.adapter_brain_for(client=client, model=brain.SMOKE_MODEL, registry=reg)
        conn = await db.connect(tmp_path / "rex.db")
        try:
            await run_hunt(
                conn,
                territory="mock-gym",
                brain=brain_for("mock-gym"),
                registry=reg,
                cost_ceiling_usd=1e-9,  # one brain call, then abort — one captured response
                max_iterations=50,
                tool_timeout_s=5.0,
            )
        finally:
            await conn.close()
    assert tee.captured == [raw]  # the brain's raw bytes, teed for the fixture (invariant 6)
