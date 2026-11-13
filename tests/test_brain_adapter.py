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
import ssl
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import certifi
import httpx2
import pytest

from rexhunter import brain, hunt_smoke
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
    assert req.headers.get_list("authorization") == ["Bearer sk-test"]
    assert req.headers["anthropic-version"] == "2026-10-21"
    assert "x-api-key" not in req.headers
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
    # would fall back to the SDK defaults (2 retries, an unpinned wire, redirects).
    src = Path(__file__).parents[1] / "src" / "rexhunter"
    builders = ("AsyncAnthropic(", "DefaultAsyncHttpxClient(")
    hits = [
        path.name
        for path in sorted(src.rglob("*.py"))
        for line in path.read_text().splitlines()
        if any(b in line for b in builders)
    ]
    assert hits == ["brain.py", "brain.py"]  # the factory's two constructor calls, nothing else


# ── the trust store stays HEAD's: certifi, not the system Keychain (W2.0 commit 5) ─────────────


def _certifi_cas() -> list[Any]:
    return ssl.create_default_context(cafile=certifi.where()).get_ca_certs()


def _record_transport_verify(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Spy on every real network transport built (by the SDK or by us): record its `verify`."""
    seen: list[object] = []
    original = httpx2.AsyncHTTPTransport.__init__

    def spy(self: httpx2.AsyncHTTPTransport, *args: Any, **kwargs: Any) -> None:
        seen.append(kwargs.get("verify", True))  # True = httpx2's default: truststore on macOS
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "__init__", spy)
    return seen


def _assert_certifi(verify: object) -> None:
    assert isinstance(verify, ssl.SSLContext)
    assert type(verify) is ssl.SSLContext  # a plain context, not truststore's SSLContext subclass
    assert verify.get_ca_certs() == _certifi_cas()


async def test_the_factory_client_trusts_certifi_not_the_system_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # HEAD's httpx verified against certifi's bundle; httpx2's own default on macOS is the system
    # Keychain (truststore), a wider trust set. The daemon's client keeps HEAD's.
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    seen = _record_transport_verify(monkeypatch)
    client = brain.anthropic_client("sk-test")  # the daemon's shape: the SDK builds the transport
    await client.close()
    assert seen  # the SDK built its keep-alive transport
    for verify in seen:
        _assert_certifi(verify)


def test_the_teed_entrypoints_live_transport_trusts_certifi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    seen = _record_transport_verify(monkeypatch)
    brain.live_transport()
    assert len(seen) == 1
    _assert_certifi(seen[0])


def test_the_trust_store_honours_ssl_cert_file_like_httpx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # HEAD's httpx read SSL_CERT_FILE (then SSL_CERT_DIR) before falling back to certifi; kept.
    end = "-----END CERTIFICATE-----"
    one_cert = certifi.contents().split(end)[0] + end + "\n"
    bundle = tmp_path / "one.pem"
    bundle.write_text(one_cert)
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))
    assert len(brain.tls_context().get_ca_certs()) == 1


def test_only_brain_builds_a_network_transport() -> None:
    # Every real transport in src/ comes from brain.py, so none can miss the certifi trust store.
    src = Path(__file__).parents[1] / "src" / "rexhunter"
    hits = [
        path.name
        for path in sorted(src.rglob("*.py"))
        for line in path.read_text().splitlines()
        if "AsyncHTTPTransport(" in line
    ]
    assert hits == ["brain.py"]


# ── the wire stays pinned against the SDK's environment overrides (W2.0) ─────


@pytest.mark.parametrize("beta_header", ["anthropic-beta", "Anthropic-Beta"])
async def test_the_wire_is_pinned_against_the_sdks_environment_overrides(
    beta_header: str, monkeypatch: pytest.MonkeyPatch, mock_anthropic: MockAnthropic
) -> None:
    # These variables could steer the SDK or the factory (credentials, base URL, betas): none may
    # reach the wire, on the adapter AND the pre-flight, and the client keeps the explicit key.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env-must-not-be-sent")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-token-from-env-must-not-be-sent")
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
        assert client.api_key == "sk-test"  # the SDK's own credential is the explicit key too
        brain_for = adapter_brain_for(client=client, model=SMOKE_MODEL, registry=_sniff_registry())
        await brain_for("acme")([], _drop)
        await hunt_smoke.count_tokens_preflight(client, _sniff_registry())

    assert [str(r.url) for r in seen] == [MESSAGES_URL, f"{MESSAGES_URL}/count_tokens"]
    for req in seen:
        assert req.headers.get_list("authorization") == ["Bearer sk-test"]  # never the env's
        assert req.headers["anthropic-version"] == "2026-10-21"  # pinned
        assert "x-api-key" not in req.headers  # Bearer is the only auth header
        assert "anthropic-beta" not in req.headers  # ANTHROPIC_CUSTOM_HEADERS cannot add betas


@pytest.mark.parametrize(
    ("custom_headers", "listed", "absent"),
    [
        ("x-extra: 1\nX-Two: 2", [("x-extra", "1"), ("X-Two", "2")], ["x-extra", "X-Two"]),
        ("Content-Type: text/plain", [("Content-Type", "text/plain")], []),
        ("X-Stainless-Raw-Response: stream", [("X-Stainless-Raw-Response", "stream")], []),
        (
            "x-api-key: sk-env\nanthropic-version: 2023-06-01\nAuthorization: Bearer sk-env",
            [
                ("x-api-key", "sk-env"),
                ("anthropic-version", "2023-06-01"),
                ("Authorization", "Bearer sk-env"),
            ],
            [],
        ),
        (
            "\N{IDEOGRAPHIC SPACE}x-api-key\N{NO-BREAK SPACE}: sk-env\n"
            " anthropic-version\t: 2023-06-01\n"
            "\N{NO-BREAK SPACE}Authorization\t: Bearer sk-env",
            [
                ("x-api-key", "sk-env"),
                ("anthropic-version", "2023-06-01"),
                ("Authorization", "Bearer sk-env"),
            ],
            [],
        ),
        (
            "x-api-key: a\nX-Api-Key: sk-env\nx-api-key: b\n"
            "anthropic-version: a\nAnthropic-Version: 2023-06-01\n"
            "Authorization: a\nauthorization: Bearer sk-env\nanthropic-beta: a\nAnthropic-Beta: b",
            [
                ("x-api-key", "a"),
                ("X-Api-Key", "sk-env"),
                ("x-api-key", "b"),
                ("anthropic-version", "a"),
                ("Anthropic-Version", "2023-06-01"),
                ("Authorization", "a"),
                ("authorization", "Bearer sk-env"),
                ("anthropic-beta", "a"),
                ("Anthropic-Beta", "b"),
            ],
            [],
        ),
        (
            " \N{IDEOGRAPHIC SPACE}x-pad\N{NO-BREAK SPACE} : a:b\nx-empty:",
            [("x-pad", "a:b"), ("x-empty", "")],
            ["x-pad", "x-empty"],
        ),
        (
            "x-extra: 1\nX-Extra: 2\nx-extra: 3",
            [("x-extra", "1"), ("X-Extra", "2"), ("x-extra", "3")],
            ["x-extra"],
        ),
        ("x-cr\r: 1\n: v", [("x-cr", "1"), ("", "v")], ["x-cr"]),
    ],
    ids=[
        "extra-names",
        "sdk-owned-name",
        "raw-response-mode",
        "pinned-names",
        "padded-pinned-names",
        "pinned-names-in-two-spellings",
        "padded-name-colon-in-value-empty-value",
        "one-name-in-two-spellings",
        "carriage-return-and-empty-name",
    ],
)
async def test_no_value_from_anthropic_custom_headers_reaches_the_wire(
    custom_headers: str,
    listed: list[tuple[str, str]],
    absent: list[str],
    monkeypatch: pytest.MonkeyPatch,
    mock_anthropic: MockAnthropic,
) -> None:
    # W2.1 (H1a, criterion C-value): the SDK merges this variable into every request's headers.
    # For each name it lists, no header of that name may carry the variable's value, on the
    # adapter call AND the pre-flight; extra names are absent. What a name the SDK sets itself
    # carries instead (dropped, or an httpx2 default) is a residual, deliberately not pinned here.
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", custom_headers)
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 7})
        return httpx2.Response(
            200,
            content=_provider_response(
                [{"type": "tool_use", "id": "t", "name": "needs_help", "input": {}}]
            ),
        )

    async with mock_anthropic(handler) as client:
        brain_for = adapter_brain_for(client=client, model=SMOKE_MODEL, registry=_sniff_registry())
        await brain_for("acme")([], _drop)
        counted = await hunt_smoke.count_tokens_preflight(client, _sniff_registry())

    assert counted == 7  # the pre-flight still returns its count (`Raw-Response: stream` broke it)
    assert [str(r.url) for r in seen] == [MESSAGES_URL, f"{MESSAGES_URL}/count_tokens"]
    for req in seen:
        for name, value in listed:
            assert value not in req.headers.get_list(name)  # the variable's value never on the wire
        for name in absent:
            assert name not in req.headers  # an extra name never reaches the wire at all
        assert req.headers.get_list("authorization") == ["Bearer sk-test"]  # the pins keep winning
        assert req.headers.get_list("anthropic-version") == ["2026-10-21"]
        assert "x-api-key" not in req.headers and "anthropic-beta" not in req.headers


async def test_a_line_without_a_colon_lists_no_header(
    monkeypatch: pytest.MonkeyPatch, mock_anthropic: MockAnthropic
) -> None:
    # The SDK skips a line with no colon, so it lists no name, and the factory must not cover one:
    # the SDK's own Content-Type stays on both calls (it is dropped only when the name is listed).
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "Content-Type")
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 7})
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

    assert [r.headers.get_list("content-type") for r in seen] == [["application/json"]] * 2


# ── the pins are the factory's, not the SDK's defaults (W2.1b, W1a) ────────────


def _x_api_key_auth(self: anthropic.AsyncAnthropic) -> dict[str, str]:
    """A stand-in for a later SDK that authenticates an api_key with x-api-key by default and
    switches a beta on by default."""
    return {"X-Api-Key": self.api_key or "", "anthropic-beta": "a-default-beta"}


async def test_the_pins_hold_when_the_sdk_changes_its_defaults(
    monkeypatch: pytest.MonkeyPatch, mock_anthropic: MockAnthropic
) -> None:
    # Today the pinned wire equals the SDK 2.6.0 defaults, so with ANTHROPIC_CUSTOM_HEADERS unset
    # the checks above cannot tell a pin from no pin. Give the SDK other defaults (another API
    # version, x-api-key auth, a default beta): the wire must not move. Both targets are strings,
    # so pyright's private-use check stays clean; an SDK bump that renames them turns this red,
    # loudly, and the canary at the end turns it red if the patches stop moving the SDK's wire.
    monkeypatch.setattr("anthropic._client.ANTHROPIC_API_VERSION", "1999-01-01")
    monkeypatch.setattr("anthropic.AsyncAnthropic._api_key_auth", property(_x_api_key_auth))
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
        assert req.headers.get_list("authorization") == ["Bearer sk-test"]
        assert req.headers.get_list("anthropic-version") == ["2026-10-21"]
        assert "x-api-key" not in req.headers
        assert "anthropic-beta" not in req.headers

    # Canary: under the same patches a bare SDK client (no pins) does move, so the patches act on
    # what the SDK sends; without this, a patch that did nothing would leave the test green.
    async with anthropic.AsyncAnthropic(
        api_key="sk-test",
        base_url=brain.ANTHROPIC_BASE_URL,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    ) as bare:
        await hunt_smoke.count_tokens_preflight(bare, _sniff_registry())
    assert seen[-1].headers.get_list("x-api-key") == ["sk-test"]
    assert seen[-1].headers.get_list("anthropic-version") == ["1999-01-01"]
    assert seen[-1].headers.get_list("anthropic-beta") == ["a-default-beta"]
