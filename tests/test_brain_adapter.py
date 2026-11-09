"""P5 · Unit 2a — the provider adapter (offline half).

The adapter (`brain.py`: `adapter_brain_for`) is a provider-agnostic `Brain` over the Anthropic SDK
in raw mode (W2.0; plain httpx before): it builds an Anthropic Messages-API request, sends it
through an INJECTED client, and feeds the raw response bytes straight into `parse_decision` (one
boundary, invariant 3). Every test injects an httpx2 `MockTransport` through the `mock_anthropic`
fixture, so nothing touches the network and nothing spends: this unit is the offline half; the paid
smoke call is Unit 2b.

Coverage maps to the unit's contract:
  - the brain wraps `parse_decision` over the injected transport (tool_use → ToolCallDecision,
    hunt_complete → HuntComplete, needs_help → NeedsHelp);
  - the adapter presents sniff + hunt_complete + needs_help as tool schemas derived from
    `@rex_tool` signatures (D1), with the right headers + request body;
  - the two branches the lenient scan opened: >1 tool_use block → reject; 0 actionable → reject;
  - sibling-block tolerance: a `thinking` block alongside a valid `tool_use` parses cleanly;
  - the loop imports no vendor SDK, and the adapter is built on `anthropic.AsyncAnthropic`;
  - the wire stays pinned against the SDK's environment overrides (W2.0).
"""

import json
import typing
from collections.abc import Callable
from pathlib import Path

import anthropic
import httpx2
import pytest

from rexhunter import hunt_smoke
from rexhunter.brain import (
    MESSAGES_URL,
    SMOKE_MODEL,
    adapter_brain_for,
    build_tools,
    parse_decision,
)
from rexhunter.events import BrainParseError
from rexhunter.loop import HuntComplete, NeedsHelp, ToolCallDecision
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]
Handler = Callable[[httpx2.Request], httpx2.Response]


async def _drop(_text: str) -> None:
    """A no-op thinking sink — this adapter unit drives the non-streaming path (never relays)."""


async def sniff(prey: str) -> str:
    """Sniff a territory for postings."""
    return f"posting:{prey}"


def _sniff_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.tool(sniff)  # explicit registration (the stub.py idiom) — registered under the name `sniff`
    return reg


def _provider_response(content: list[dict[str, object]]) -> bytes:
    """An Anthropic Messages-API response envelope carrying the given content blocks."""
    return json.dumps(
        {
            "id": "msg_01FIXTURE",
            "type": "message",
            "role": "assistant",
            "model": "claude-fixture",
            "stop_reason": "tool_use",
            "content": content,
        }
    ).encode()


def _canned(handler_response: bytes) -> Handler:
    """A handler that returns the same canned response to any request (zero network)."""

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=handler_response)

    return handler


async def _drive(mock_anthropic: MockAnthropic, handler: Handler, registry: ToolRegistry) -> object:
    """Build the adapter over an injected transport, drive one brain turn, return its Decision."""
    async with mock_anthropic(handler) as client:
        brain_for = adapter_brain_for(client=client, model=SMOKE_MODEL, registry=registry)
        return await brain_for("acme")([], _drop)


# ── the brain wraps parse_decision over the injected transport ────────────────


async def test_brain_wraps_parse_decision_into_a_toolcall_decision(
    mock_anthropic: MockAnthropic,
) -> None:
    raw = _provider_response(
        [{"type": "tool_use", "id": "toolu_01ABC", "name": "sniff", "input": {"prey": "acme"}}]
    )
    decision = await _drive(mock_anthropic, _canned(raw), _sniff_registry())
    assert isinstance(decision, ToolCallDecision)
    assert decision.tool == "sniff"
    assert decision.args == {"prey": "acme"}
    assert decision.tool_use_id == "toolu_01ABC"  # the provider's correlation key survives


async def test_brain_returns_hunt_complete_for_the_terminal_tool(
    mock_anthropic: MockAnthropic,
) -> None:
    raw = _provider_response(
        [{"type": "tool_use", "id": "t", "name": "hunt_complete", "input": {"catch": ["p:acme"]}}]
    )
    decision = await _drive(mock_anthropic, _canned(raw), _sniff_registry())
    assert isinstance(decision, HuntComplete)
    assert decision.catch == ["p:acme"]


async def test_brain_returns_needs_help_for_the_terminal_tool(
    mock_anthropic: MockAnthropic,
) -> None:
    raw = _provider_response([{"type": "tool_use", "id": "t", "name": "needs_help", "input": {}}])
    decision = await _drive(mock_anthropic, _canned(raw), _sniff_registry())
    assert isinstance(decision, NeedsHelp)


# ── the adapter presents the three tools + the right request shape (D1) ───────


def test_build_tools_presents_sniff_and_both_terminal_tools() -> None:
    tools = build_tools(_sniff_registry())
    by_name = {t["name"]: t for t in tools}
    assert set(by_name) == {"sniff", "hunt_complete", "needs_help"}  # D1 cross-unit commitment
    for tool in tools:
        assert "input_schema" in tool and "description" in tool
    # The schemas are DERIVED from the typed signatures (ADR §What point 1), not hand-written:
    # a hand-written stub or empty schema would NOT expose these parameters.
    assert "prey" in by_name["sniff"]["input_schema"]["properties"]  # from sniff(prey: str)
    assert "catch" in by_name["hunt_complete"]["input_schema"]["properties"]  # from HuntComplete


async def test_adapter_sends_anthropic_headers_and_body(mock_anthropic: MockAnthropic) -> None:
    captured: dict[str, httpx2.Request] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["req"] = request
        return httpx2.Response(
            200,
            content=_provider_response(
                [{"type": "tool_use", "id": "t", "name": "hunt_complete", "input": {"catch": []}}]
            ),
        )

    async with mock_anthropic(handler) as client:
        brain_for = adapter_brain_for(client=client, model=SMOKE_MODEL, registry=_sniff_registry())
        await brain_for("acme")([], _drop)

    req = captured["req"]
    assert req.headers["x-api-key"] == "sk-test"
    assert req.headers["anthropic-version"] == "2023-06-01"
    body = json.loads(req.content)
    assert body["model"] == SMOKE_MODEL
    assert isinstance(body["max_tokens"], int) and body["max_tokens"] > 0
    assert body["messages"]  # a well-formed, non-empty messages array
    assert {t["name"] for t in body["tools"]} == {"sniff", "hunt_complete", "needs_help"}


# ── the two branches the lenient scan opened ──────────────────────────────────


def test_two_tool_use_blocks_are_rejected() -> None:
    # One tool per iteration: two actionable blocks is ambiguous, not first-wins.
    raw = _provider_response(
        [
            {"type": "tool_use", "id": "a", "name": "sniff", "input": {"prey": "x"}},
            {"type": "tool_use", "id": "b", "name": "sniff", "input": {"prey": "y"}},
        ]
    )
    with pytest.raises(BrainParseError):
        parse_decision(raw)


def test_a_response_with_no_actionable_block_is_rejected() -> None:
    raw = _provider_response([{"type": "text", "text": "just chatting, no tool call"}])
    with pytest.raises(BrainParseError):
        parse_decision(raw)


# ── sibling-block tolerance (the lenient-scan deviation's positive case) ──────


def test_thinking_sibling_alongside_a_tool_use_parses_cleanly() -> None:
    raw = _provider_response(
        [
            {"type": "thinking", "thinking": "let me sniff acme"},
            {"type": "tool_use", "id": "toolu_x", "name": "sniff", "input": {"prey": "acme"}},
        ]
    )
    decision = parse_decision(raw)
    assert isinstance(decision, ToolCallDecision)
    assert decision.tool == "sniff"
    assert decision.tool_use_id == "toolu_x"


# ── the vendor SDK lives in the adapter only (W2.0 D1, D1a) ───────────────────


def test_loop_imports_no_vendor_sdk() -> None:
    src = (Path(__file__).parents[1] / "src" / "rexhunter" / "loop.py").read_text()
    assert "import anthropic" not in src
    assert "from anthropic" not in src


def test_loop_imports_no_transport_library() -> None:
    # W2.0 D1a: transport failures reach the loop only as BrainCallError, so classify needs no
    # transport types and loop.py imports neither httpx nor httpx2.
    src = (Path(__file__).parents[1] / "src" / "rexhunter" / "loop.py").read_text()
    assert "import httpx" not in src  # also matches `import httpx2`
    assert "from httpx" not in src


def test_the_adapter_is_built_on_the_anthropic_sdk() -> None:
    # Pins the seam's DECLARED type (the raw-mode behaviour is the replay tests' job).
    assert typing.get_type_hints(adapter_brain_for)["client"] is anthropic.AsyncAnthropic


def test_only_the_factory_builds_an_sdk_client() -> None:
    # Every SDK client in src/ comes from brain.anthropic_client: a bare AsyncAnthropic elsewhere
    # would fall back to the SDK defaults (2 retries, Bearer auth, another API version, redirects).
    src = Path(__file__).parents[1] / "src" / "rexhunter"
    builders = ("AsyncAnthropic(", "DefaultAsyncHttpxClient(")
    hits = [
        path.name
        for path in sorted(src.rglob("*.py"))
        for line in path.read_text().splitlines()
        if any(b in line for b in builders)
    ]
    assert hits == ["brain.py", "brain.py"]  # the factory's two constructor calls, nothing else


# ── the wire stays pinned against the SDK's environment overrides (W2.0) ─────


@pytest.mark.parametrize("beta_header", ["anthropic-beta", "Anthropic-Beta"])
async def test_the_wire_is_pinned_against_the_sdks_environment_overrides(
    beta_header: str, monkeypatch: pytest.MonkeyPatch, mock_anthropic: MockAnthropic
) -> None:
    # The SDK reads these variables at construction; the factory must neutralise each one, so
    # the wire is the one every fixture was recorded under, on the adapter AND the pre-flight.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env-must-not-be-sent")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://elsewhere.invalid")
    monkeypatch.setenv(
        "ANTHROPIC_CUSTOM_HEADERS", f"{beta_header}: interleaved-thinking-2025-05-14"
    )
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 1})
        return httpx2.Response(
            200,
            content=_provider_response(
                [{"type": "tool_use", "id": "t", "name": "needs_help", "input": {}}]
            ),
        )

    async with mock_anthropic(handler) as client:
        brain_for = adapter_brain_for(client=client, model=SMOKE_MODEL, registry=_sniff_registry())
        await brain_for("acme")([], _drop)
        await hunt_smoke.count_tokens_preflight(client, _sniff_registry())

    assert [str(r.url) for r in seen] == [MESSAGES_URL, f"{MESSAGES_URL}/count_tokens"]
    for req in seen:
        assert req.headers["x-api-key"] == "sk-test"  # the explicit key, never the env's
        assert req.headers["anthropic-version"] == "2023-06-01"  # pinned, not the SDK default
        assert "authorization" not in req.headers  # no Bearer auth
        assert "anthropic-beta" not in req.headers  # ANTHROPIC_CUSTOM_HEADERS cannot add betas
