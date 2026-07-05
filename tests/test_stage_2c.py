"""P5 · Unit 2c.1 — brain-in-loop, offline half (no spend, no network).

The adapter now DRIVES the loop. Five contracts, each provable offline:

  1. Context → messages projection — the trajectory log folds into a valid Anthropic `messages`
     array (invariant 2, derived per call — not a second stored chat history); `tool_use_id`
     correlation survives the fold.
  2. System prompt — a stable `HUNT_SYSTEM_PROMPT` constant, threaded into the request.
  3. Retryable HTTP classification — real httpx failures flow through the EXISTING `classify`
     (429/5xx/timeouts/connect → retryable; 4xx / parse / validation → fatal).
  4. Cost-ceiling breaker — per-call usage folds into a running spend from `UsageEvent`s
     (invariant 5, no counter); crossing `COST_CEILING_USD` aborts before the next brain call.
     Proven by replaying the 2b golden fixture under a deliberately low ceiling.
  5. Config-gated brain — `REXHUNTER_BRAIN` selects the brain; default `stub` (no client, no
     spend), `live` is opt-in. This is the containment for the autonomous-spender surface.

Every test injects `httpx.MockTransport` or hand-built events — zero network, zero spend. The 2b
fixture (`tests/fixtures/smoke_sonnet5.json`, a genuine Sonnet 5 response) anchors the breaker.
"""

import json
from pathlib import Path
from typing import Any

import httpx
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
from rexhunter.loop import classify, run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

FIXTURE = Path(__file__).parent / "fixtures" / "smoke_sonnet5.json"


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


async def test_adapter_sends_system_prompt_and_disables_thinking() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["req"] = request
        return httpx.Response(200, content=FIXTURE.read_bytes())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        brain_for = brain.adapter_brain_for(
            client=client, api_key="sk-test", model=brain.SMOKE_MODEL, registry=_sniff_registry()
        )
        await brain_for("mock-gym")([], _drop)

    body = json.loads(captured["req"].content)
    assert body["system"] == brain.HUNT_SYSTEM_PROMPT
    # Bare tool_use replay turns (no thinking sibling) until capture lands in Unit 3; Sonnet 5
    # accepts an explicit disabled, sidestepping the thinking-block replay rule.
    assert body["thinking"] == {"type": "disabled"}


# ── 3. Retryable HTTP classification (extend the taxonomy, don't fork it) ──────


def _status_error(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", brain.MESSAGES_URL)
    resp = httpx.Response(code, request=req)
    return httpx.HTTPStatusError(f"http {code}", request=req, response=resp)


def test_classify_httpx_transport_errors_are_retryable() -> None:
    req = httpx.Request("POST", brain.MESSAGES_URL)
    assert classify(httpx.ConnectTimeout("slow", request=req)) is True
    assert classify(httpx.ReadTimeout("slow", request=req)) is True
    assert classify(httpx.ConnectError("refused", request=req)) is True


def test_classify_429_and_5xx_retryable_4xx_fatal() -> None:
    assert classify(_status_error(429)) is True
    assert classify(_status_error(503)) is True
    assert classify(_status_error(500)) is True
    assert classify(_status_error(400)) is False
    assert classify(_status_error(404)) is False


def test_classify_parse_and_validation_errors_are_fatal() -> None:
    assert classify(BrainParseError(raw=b"{}", detail="bad")) is False
    with pytest.raises(ValidationError) as excinfo:
        ToolCallEvent.model_validate({})  # missing required fields
    assert classify(excinfo.value) is False


def _status_transport(code: int, body: bytes) -> httpx.MockTransport:
    """Every request gets the same non-2xx status — raise_for_status must surface it."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(code, content=body)

    return httpx.MockTransport(handler)


async def test_brain_retries_a_transient_5xx_then_aborts(tmp_path: Path) -> None:
    # Integrated proof of the path classify only unit-tests: raise_for_status → HTTPStatusError →
    # _call_brain retries within budget, tags "<brain>", captures the body, then aborts.
    reg = _sniff_registry()
    async with httpx.AsyncClient(
        transport=_status_transport(503, b"upstream unavailable")
    ) as client:
        brain_for = brain.adapter_brain_for(
            client=client, api_key="sk-test", model=brain.SMOKE_MODEL, registry=reg
        )
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


async def test_brain_4xx_is_fatal_without_retry(tmp_path: Path) -> None:
    reg = _sniff_registry()
    async with httpx.AsyncClient(transport=_status_transport(400, b"bad request")) as client:
        brain_for = brain.adapter_brain_for(
            client=client, api_key="sk-test", model=brain.SMOKE_MODEL, registry=reg
        )
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


# ── 4. Cost-ceiling breaker (invariant 5 — folded from UsageEvents) ───────────


def _fixture_transport() -> httpx.MockTransport:
    raw = FIXTURE.read_bytes()  # a real Sonnet 5 response: tool=sniff, usage 1004 in / 80 out

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=raw)

    return httpx.MockTransport(handler)


async def test_cost_ceiling_aborts_before_the_next_brain_call(tmp_path: Path) -> None:
    reg = _sniff_registry()
    async with httpx.AsyncClient(transport=_fixture_transport()) as client:
        brain_for = brain.adapter_brain_for(
            client=client, api_key="sk-test", model=brain.SMOKE_MODEL, registry=reg
        )
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


async def test_high_ceiling_falls_back_to_max_iterations(tmp_path: Path) -> None:
    # Cost breaker and max-iter breaker coexist: a never-terminating brain (fixture always sniffs)
    # under a slack ceiling is still bounded by the existing max-iterations breaker.
    reg = _sniff_registry()
    async with httpx.AsyncClient(transport=_fixture_transport()) as client:
        brain_for = brain.adapter_brain_for(
            client=client, api_key="sk-test", model=brain.SMOKE_MODEL, registry=reg
        )
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
            await client.aclose()  # closed without ever calling brain() → zero spend


# ── 6. Unit 2c.2 — live request shape + count_tokens pre-flight (offline) ──────


async def _capture_adapter_body(
    registry: ToolRegistry, territory: str = "mock-gym"
) -> dict[str, Any]:
    """Drive one adapter turn over MockTransport and return the request body it actually sent."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=FIXTURE.read_bytes())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        brain_for = brain.adapter_brain_for(
            client=client, api_key="sk-test", model=brain.SMOKE_MODEL, registry=registry
        )
        await brain_for(territory)([], _drop)
    return captured["body"]


async def test_live_request_shape() -> None:
    body = await _capture_adapter_body(_sniff_registry())
    for param in ("temperature", "top_p", "top_k"):
        assert param not in body  # any non-default sampling param → 400 on Sonnet 5
    assert body["thinking"] == {"type": "disabled"}  # accepted on Sonnet 5; only "enabled" 400s
    assert body["tool_choice"]["disable_parallel_tool_use"] is True  # one tool per iteration
    assert body["max_tokens"] >= 1024


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


async def test_count_tokens_preflight_posts_exact_payload_and_returns_count() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["req"] = request
        return httpx.Response(200, json={"input_tokens": 1234})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        n = await hunt_smoke.count_tokens_preflight(client, "sk-test", _sniff_registry())

    assert n == 1234
    assert str(seen["req"].url) == brain.COUNT_TOKENS_URL
    count_body = json.loads(seen["req"].content)
    assert "max_tokens" not in count_body  # count_tokens takes no max_tokens
    assert count_body["system"] == brain.HUNT_SYSTEM_PROMPT
    assert count_body["thinking"] == {"type": "disabled"}
    assert count_body["tool_choice"]["disable_parallel_tool_use"] is True


async def test_preflight_payload_matches_adapter_body_minus_max_tokens() -> None:
    # Drift guard: the free count must price the exact bytes the paid call will send.
    adapter_body = await _capture_adapter_body(_sniff_registry())

    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"input_tokens": 1})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await hunt_smoke.count_tokens_preflight(client, "sk-test", _sniff_registry())

    assert seen["body"] == {k: v for k, v in adapter_body.items() if k != "max_tokens"}


async def test_count_tokens_preflight_stops_on_non_200() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad schema — a paid call must not follow")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(SystemExit):
            await hunt_smoke.count_tokens_preflight(client, "sk-test", _sniff_registry())


async def test_capturing_transport_tees_only_messages_responses(tmp_path: Path) -> None:
    raw = FIXTURE.read_bytes()
    inner = httpx.MockTransport(lambda _r: httpx.Response(200, content=raw))
    tee = hunt_smoke.CapturingTransport(inner, brain.MESSAGES_URL)
    reg = _sniff_registry()
    async with httpx.AsyncClient(transport=tee) as client:
        brain_for = brain.adapter_brain_for(
            client=client, api_key="sk-test", model=brain.SMOKE_MODEL, registry=reg
        )
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
