"""P5 Unit 2b — offline guards for the paid smoke (NO spend, no network).

Two locks the paid call depends on:
  1. schema-clean — `build_tools` strips every `title` from the derived `input_schema`s; the
     `@rex_tool` args-models still emit `additionalProperties: false` (kept — Anthropic accepts it);
  2. Sonnet 5 request guards — `build_request_body` omits the params that 400 on Sonnet 5
     (`temperature`/`top_p`/`top_k`/`thinking`) and never ends `messages` on an assistant prefill.

Since W2.0 the smoke runs on the Anthropic SDK (raw mode), so a third lock: 3. the smoke's HTTP
path, driven offline over an httpx2 `MockTransport` with the fixture path redirected to a tmp dir
(the real `FIXTURE_PATH` is never written): one free pre-flight, exactly ONE paid call (the SDK's
max_retries=0 is the hard one-call guard), the raw bytes persisted before parsing.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import anthropic
import httpx2
import pytest

from rexhunter import smoke
from rexhunter.brain import SMOKE_MODEL, build_request_body, build_tools
from rexhunter.events import BrainParseError
from rexhunter.loop import ToolCallDecision
from rexhunter.stub import build_registry

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]
SMOKE_FIXTURE = Path(__file__).parent / "fixtures" / "smoke_sonnet5.json"  # read-only here


def _has_title(node: Any) -> bool:
    """True if a `title` key appears anywhere in the (possibly nested) schema."""
    if isinstance(node, dict):
        return "title" in node or any(_has_title(v) for v in cast("dict[str, Any]", node).values())
    if isinstance(node, list):
        return any(_has_title(v) for v in cast("list[Any]", node))
    return False


# ── 1. schema-clean ──────────────────────────────────────────────────────────


def test_build_tools_strips_title_from_every_schema() -> None:
    for tool in build_tools(build_registry()):
        assert not _has_title(tool["input_schema"]), f"`title` leaked into {tool['name']}'s schema"


def test_rex_tool_schema_keeps_additionalproperties_false() -> None:
    # @rex_tool args-models set extra="forbid" → additionalProperties:false IS emitted, and is KEPT
    # (Anthropic accepts it). The terminal-tool models are plain BaseModels and don't emit it —
    # non-strict tool use, which is fine (strict mode is out of scope for 2b).
    by_name = {t["name"]: t for t in build_tools(build_registry())}
    assert by_name["sniff"]["input_schema"]["additionalProperties"] is False
    assert "additionalProperties" not in by_name["hunt_complete"]["input_schema"]


# ── 2. Sonnet 5 request guards ───────────────────────────────────────────────


def test_smoke_request_body_omits_sonnet5_forbidden_params() -> None:
    messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
    body = build_request_body(
        model=SMOKE_MODEL, max_tokens=4096, messages=messages, tools=build_tools(build_registry())
    )
    assert body["model"] == "claude-sonnet-5"
    assert body["max_tokens"] >= 1024  # room for adaptive-thinking tokens + the tool_use block
    for forbidden in ("temperature", "top_p", "top_k", "thinking"):
        assert forbidden not in body  # any of these → 400 on Sonnet 5
    assert body["messages"][-1]["role"] != "assistant"  # no assistant prefill (→ 400)


# ── 3. the smoke's HTTP path on the SDK (W2.0), offline ──────────────────────


def _smoke_handler(
    paths: list[str], *, messages_status: int = 200
) -> Callable[[httpx2.Request], httpx2.Response]:
    """count_tokens → 200 {input_tokens}; messages → the recorded smoke response (or an error)."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 1234})
        if messages_status != 200:
            return httpx2.Response(messages_status, content=b"upstream unavailable")
        return httpx2.Response(200, content=SMOKE_FIXTURE.read_bytes())

    return handler


@pytest.mark.anyio
async def test_smoke_count_tokens_prices_the_exact_payload(mock_anthropic: MockAnthropic) -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json={"input_tokens": 77})

    tools = build_tools(build_registry())
    messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
    async with mock_anthropic(handler) as client:
        n = await smoke.count_tokens(client, model=SMOKE_MODEL, messages=messages, tools=tools)
    assert n == 77
    assert json.loads(seen[0].content) == {
        "model": SMOKE_MODEL,
        "messages": messages,
        "tools": tools,
    }


@pytest.mark.anyio
async def test_run_smoke_makes_one_paid_call_and_persists_raw_bytes_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_anthropic: MockAnthropic
) -> None:
    out = tmp_path / "smoke.json"
    monkeypatch.setattr(smoke, "FIXTURE_PATH", out)  # never the real fixture
    paths: list[str] = []
    async with mock_anthropic(_smoke_handler(paths)) as client:
        raw, decision = await smoke.run_smoke(client)
    assert paths == ["/v1/messages/count_tokens", "/v1/messages"]  # free pre-flight, ONE paid call
    assert raw == SMOKE_FIXTURE.read_bytes()  # the raw provider bytes, unchanged by the SDK
    assert out.read_bytes() == raw  # persisted (invariant 6)
    assert isinstance(decision, ToolCallDecision) and decision.tool == "sniff"


@pytest.mark.anyio
async def test_run_smoke_never_retries_a_failed_paid_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_anthropic: MockAnthropic
) -> None:
    # The hard one-call guard on the SDK: a 503 is NOT retried (max_retries=0), the smoke stops,
    # and no error body is written over the fixture.
    out = tmp_path / "smoke.json"
    monkeypatch.setattr(smoke, "FIXTURE_PATH", out)
    paths: list[str] = []
    async with mock_anthropic(_smoke_handler(paths, messages_status=503)) as client:
        with pytest.raises(SystemExit):
            await smoke.run_smoke(client)
    assert paths.count("/v1/messages") == 1
    assert not out.exists()


_UNPARSEABLE: list[dict[str, Any]] = [{"type": "text", "text": "no tool call"}]
_TRUNCATED: list[dict[str, Any]] = [{"type": "tool_use", "id": "t", "name": "sniff", "input": {}}]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("content", "stop_reason", "raised"),
    [(_UNPARSEABLE, "end_turn", BrainParseError), (_TRUNCATED, "max_tokens", SystemExit)],
    ids=["unparseable", "truncated"],
)
async def test_run_smoke_persists_the_raw_bytes_before_parsing(
    content: list[dict[str, Any]],
    stop_reason: str,
    raised: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_anthropic: MockAnthropic,
) -> None:
    # A paid 200 is persisted FIRST (invariant 6), even when the stop_reason check or
    # parse_decision then fails: the money-analog of "make the deliverable durable first".
    out = tmp_path / "smoke.json"
    monkeypatch.setattr(smoke, "FIXTURE_PATH", out)
    body = json.dumps(
        {"model": SMOKE_MODEL, "stop_reason": stop_reason, "content": content}
    ).encode()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 1})
        return httpx2.Response(200, content=body)

    async with mock_anthropic(handler) as client:
        with pytest.raises(raised):
            await smoke.run_smoke(client)
    assert out.read_bytes() == body
