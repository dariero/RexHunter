"""P5 Unit 2b — offline guards for the paid smoke (NO spend, no network).

Two locks the paid call depends on:
  1. schema-clean — `build_tools` strips every `title` from the derived `input_schema`s; the
     `@rex_tool` args-models still emit `additionalProperties: false` (kept — Anthropic accepts it);
  2. Sonnet 5 request guards — `build_request_body` omits the params that 400 on Sonnet 5
     (`temperature`/`top_p`/`top_k`/`thinking`) and never ends `messages` on an assistant prefill.

Since W2.0 the smoke runs on the Anthropic SDK (raw mode), so a third lock: 3. the smoke's HTTP
path, driven offline over an httpx2 `MockTransport` with `fixture_dir=tmp_path` passed explicitly
(it has no default, so the real tests/fixtures is never written): one free pre-flight, exactly ONE
paid call (the SDK's max_retries=0 is the hard one-call guard), the raw bytes persisted before
parsing. And since W2.1b (F8) a fourth: 4. each paid smoke writes a NEW file, reserved before any
call, so a recorded fixture such as smoke_sonnet5.json is never overwritten.
"""

import asyncio
import hashlib
import inspect
import json
import sys
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import anthropic
import httpx2
import pytest

from rexhunter import brain, smoke
from rexhunter.brain import SMOKE_MODEL, build_request_body, build_tools
from rexhunter.events import BrainParseError
from rexhunter.loop import Decision, ToolCallDecision
from rexhunter.stub import build_registry

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]
SMOKE_FIXTURE = Path(__file__).parent / "fixtures" / "smoke_sonnet5.json"  # read-only here
FIXTURES = Path(__file__).parent / "fixtures"  # the real fixture tree, read-only here
GOLDEN_SHA256 = "bcbdbbab3549e7a77c848d9c96e2c5ccf9e1162c4ea463092c025ff6fbea12a5"  # smoke_sonnet5
NOW = datetime(2026, 11, 14, 0, 0, 0, tzinfo=UTC)  # a pinned clock, so the file name is known
FIXTURE_NAME = "smoke_claude-sonnet-5_20261114T000000000000Z.json"  # what run_smoke writes at NOW


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


def _names(directory: Path) -> list[str]:
    """The sorted file names in `directory` (sync, so the async tests do no blocking Path call)."""
    return sorted(p.name for p in directory.iterdir())


@pytest.fixture(autouse=True)
def _module_fixture_dir_is_a_decoy(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Every test here points smoke.FIXTURE_DIR at a decoy outside tmp_path: a run_smoke that wrote
    under the module directory instead of (or as well as) its fixture_dir= fails at teardown, and
    never reaches the repo's tests/fixtures."""
    decoy = tmp_path_factory.mktemp("module_fixture_dir") / "fixtures"
    monkeypatch.setattr(smoke, "FIXTURE_DIR", decoy)
    yield
    assert not decoy.exists(), f"run_smoke wrote under the module FIXTURE_DIR: {decoy}"


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
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    out = tmp_path / FIXTURE_NAME
    paths: list[str] = []
    async with mock_anthropic(_smoke_handler(paths)) as client:
        raw, decision = await smoke.run_smoke(client, fixture_dir=tmp_path, now=NOW)
    assert paths == ["/v1/messages/count_tokens", "/v1/messages"]  # free pre-flight, ONE paid call
    assert raw == SMOKE_FIXTURE.read_bytes()  # the raw provider bytes, unchanged by the SDK
    assert out.read_bytes() == raw  # persisted (invariant 6)
    assert isinstance(decision, ToolCallDecision) and decision.tool == "sniff"


@pytest.mark.anyio
async def test_run_smoke_never_retries_a_failed_paid_call(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # The hard one-call guard on the SDK: a 503 is NOT retried (max_retries=0), the smoke stops,
    # and no error body is written over the fixture.
    out = tmp_path / FIXTURE_NAME
    paths: list[str] = []
    async with mock_anthropic(_smoke_handler(paths, messages_status=503)) as client:
        with pytest.raises(SystemExit):
            await smoke.run_smoke(client, fixture_dir=tmp_path, now=NOW)
    assert paths.count("/v1/messages") == 1
    assert not out.exists()
    assert _names(tmp_path) == []  # the reservation is dropped: no paid bytes were written


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
    mock_anthropic: MockAnthropic,
) -> None:
    # A paid 200 is persisted FIRST (invariant 6), even when the stop_reason check or
    # parse_decision then fails: the money-analog of "make the deliverable durable first".
    out = tmp_path / FIXTURE_NAME
    body = json.dumps(
        {"model": SMOKE_MODEL, "stop_reason": stop_reason, "content": content}
    ).encode()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 1})
        return httpx2.Response(200, content=body)

    async with mock_anthropic(handler) as client:
        with pytest.raises(raised):
            await smoke.run_smoke(client, fixture_dir=tmp_path, now=NOW)
    assert out.read_bytes() == body


# ── 4. F8 (W2.1b): a paid smoke writes a NEW file, never over a recorded one ──────


def _serving(paths: list[str], body: bytes) -> Callable[[httpx2.Request], httpx2.Response]:
    """count_tokens → 200; messages → `body` (bytes that differ from any seeded fixture)."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 1234})
        return httpx2.Response(200, content=body)

    return handler


@pytest.mark.anyio
async def test_run_smoke_writes_a_new_file_and_leaves_the_recorded_fixture_alone(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # tmp_path stands in for tests/fixtures, seeded with the recorded smoke fixture. The mock serves
    # DIFFERENT bytes (the same JSON re-serialised), so an overwrite would change the seed's sha256.
    golden = SMOKE_FIXTURE.read_bytes()
    assert hashlib.sha256(golden).hexdigest() == GOLDEN_SHA256  # the seed is the real recording
    seed = tmp_path / "smoke_sonnet5.json"
    seed.write_bytes(golden)
    served = json.dumps(json.loads(golden), indent=1).encode()
    assert served != golden
    paths: list[str] = []
    async with mock_anthropic(_serving(paths, served)) as client:
        raw, decision = await smoke.run_smoke(client, fixture_dir=tmp_path, now=NOW)
    assert hashlib.sha256(seed.read_bytes()).hexdigest() == GOLDEN_SHA256  # never overwritten
    assert _names(tmp_path) == sorted([seed.name, FIXTURE_NAME])
    assert (tmp_path / FIXTURE_NAME).read_bytes() == raw == served  # the new file holds the reply
    assert isinstance(decision, ToolCallDecision) and decision.tool == "sniff"


@pytest.mark.anyio
async def test_run_smoke_refuses_an_existing_target_before_any_call(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # The name run_smoke would write already exists: it refuses before the pre-flight (so at $0)
    # and leaves that file as it was.
    target = tmp_path / FIXTURE_NAME
    target.write_bytes(b"recorded")
    paths: list[str] = []
    async with mock_anthropic(_serving(paths, b"{}")) as client:
        with pytest.raises(SystemExit):
            await smoke.run_smoke(client, fixture_dir=tmp_path, now=NOW)
    assert paths == []  # no request at all: not the free pre-flight, not the paid call
    assert target.read_bytes() == b"recorded"


def _tree_digest(root: Path) -> dict[str, str]:
    """{relative path: sha256} of every file under `root`."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.mark.anyio
async def test_a_redirected_smoke_run_leaves_the_real_fixture_tree_unchanged(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # Backstop for the repo itself: after a run with fixture_dir=tmp_path the real tree is
    # identical. The mock serves bytes unlike any recording, so a stray write would change a digest.
    before = _tree_digest(FIXTURES)
    served = json.dumps(json.loads(SMOKE_FIXTURE.read_bytes()), indent=1).encode()
    paths: list[str] = []
    async with mock_anthropic(_serving(paths, served)) as client:
        await smoke.run_smoke(client, fixture_dir=tmp_path, now=NOW)
    assert _tree_digest(FIXTURES) == before
    assert _names(tmp_path) == [FIXTURE_NAME]


def test_the_fixture_name_is_one_path_component_in_utc() -> None:
    # Any model id reduces to one path component, and any aware clock is written in UTC to the
    # microsecond, so two runs in one second get two names.
    aedt = timezone(timedelta(hours=11))
    when = datetime(2026, 11, 14, 10, 0, 0, 7, tzinfo=aedt)
    path = smoke.fixture_path(Path("d"), "us.anthropic/claude:0 [1m]", when)
    assert path == Path("d") / "smoke_us.anthropic_claude_0__1m__20261113T230000000007Z.json"


def test_a_naive_clock_is_refused() -> None:
    # Python reads a naive datetime as local time; the name would silently carry a non-UTC stamp.
    with pytest.raises(ValueError, match="aware"):
        smoke.fixture_path(Path("d"), "m", datetime(2026, 11, 14))


def test_run_smoke_has_no_default_for_fixture_dir_or_now() -> None:
    # A forgotten argument must be a TypeError, never a write into the repo: no default to fall
    # back on (checked on the signature, so no test ever calls run_smoke without them).
    params = inspect.signature(smoke.run_smoke).parameters
    for name in ("fixture_dir", "now"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is inspect.Parameter.empty


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("failure", "raised"),
    [
        (KeyboardInterrupt(), KeyboardInterrupt),
        (asyncio.CancelledError(), asyncio.CancelledError),
        (httpx2.ConnectError("down"), anthropic.APIConnectionError),
    ],
    ids=["ctrl-c", "cancel", "transport-error"],
)
async def test_any_failure_before_a_paid_reply_drops_the_reservation(
    failure: BaseException,
    raised: type[BaseException],
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
) -> None:
    # Not only a non-200: Ctrl-C, a cancel or a transport error during the paid call leaves no
    # empty file behind, and the failure surfaces unchanged.
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 1})
        raise failure

    async with mock_anthropic(handler) as client:
        with pytest.raises(raised):
            await smoke.run_smoke(client, fixture_dir=tmp_path, now=NOW)
    assert _names(tmp_path) == []


@pytest.mark.anyio
async def test_the_paid_reply_is_on_disk_before_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_anthropic: MockAnthropic
) -> None:
    # Persist-first (invariant 6), checked at the moment it matters: when parse_decision runs, the
    # file already holds the whole reply (written, flushed and closed).
    seen: list[bytes] = []
    real = brain.parse_decision

    def spy(raw: bytes) -> Decision:
        seen.append((tmp_path / FIXTURE_NAME).read_bytes())
        return real(raw)

    monkeypatch.setattr(brain, "parse_decision", spy)
    paths: list[str] = []
    async with mock_anthropic(_smoke_handler(paths)) as client:
        raw, _decision = await smoke.run_smoke(client, fixture_dir=tmp_path, now=NOW)
    assert seen == [raw]


def test_main_passes_the_module_fixture_dir_and_an_aware_utc_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # main() is run_smoke's only production caller. smoke.FIXTURE_DIR is the autouse decoy here,
    # run_smoke is a spy, and the client is the factory's over a MockTransport: nothing is sent,
    # nothing is written.
    seen: dict[str, Any] = {}
    factory = brain.anthropic_client  # the real one, captured before it is patched below

    async def spy(
        client: anthropic.AsyncAnthropic, *, fixture_dir: Path, now: datetime
    ) -> tuple[bytes, Decision]:
        seen.update(fixture_dir=fixture_dir, now=now)
        return b"", ToolCallDecision(tool="sniff", args={})

    def offline_client(api_key: str, *, timeout: float) -> anthropic.AsyncAnthropic:
        return factory(api_key, transport=httpx2.MockTransport(_smoke_handler([])), timeout=timeout)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setattr(sys, "argv", ["smoke"])
    monkeypatch.setattr(brain, "anthropic_client", offline_client)
    monkeypatch.setattr(smoke, "run_smoke", spy)
    smoke.main()
    assert seen["fixture_dir"] is smoke.FIXTURE_DIR
    assert seen["now"].utcoffset() == timedelta(0)
