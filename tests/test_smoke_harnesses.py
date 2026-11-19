"""The smoke harnesses' capture path (W3.4b), offline: every response is built here and answered by
an httpx2 MockTransport with a dummy key, so nothing reaches the network and nothing is spent.

W3.1-K1: the capturing tee read each body to its end before the adapter saw a byte, so a stream
could not stop at its first error frame (Z-stop, W3.1) and the thinking relay arrived all at once.
W3.2-X17: the tee rebuilt each response with two headers only, so a 429's `retry-after` never
reached the loop's backoff.
Fixtures: both harnesses wrote what they captured only after the hunt and the reporter, so a
Ctrl-C mid-hunt or a reporter that raised lost every paid response. Each harness here runs with
its fixtures and its run DB under `tmp_path`, never tests/fixtures or the system temp dir.
"""

import asyncio
import contextlib
import functools
import gzip
import hashlib
import json
import tempfile
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from rexhunter import brain, db, hunt_smoke, hunt_smoke_thinking, stub
from rexhunter.loop import run_hunt

pytestmark = pytest.mark.anyio

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]


def _frame(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


MESSAGE_START = _frame(
    "message_start",
    {
        "type": "message_start",
        "message": {
            "model": brain.SMOKE_MODEL,
            "content": [],
            "usage": {"input_tokens": 100, "output_tokens": 1},
        },
    },
)
ERROR_FRAME = _frame(
    "error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
)
NEEDS_HELP_REPLY = json.dumps(
    {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": brain.SMOKE_MODEL,
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "needs_help", "input": {}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
).encode()


async def _one_hunt(
    tee: hunt_smoke.CapturingTransport,
    mock_anthropic: MockAnthropic,
    db_path: Path,
    *,
    stream: bool,
    retry_budget: int,
) -> None:
    """One mock-gym hunt on the adapter over `tee`, the way the harnesses drive it."""
    registry = stub.build_registry()
    async with mock_anthropic(tee) as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            model=brain.SMOKE_MODEL,
            registry=registry,
            max_tokens=brain.HUNT_MAX_TOKENS,
            thinking=brain.HUNT_THINKING if stream else None,
            stream=stream,
        )
        conn = await db.connect(db_path)
        try:
            await run_hunt(
                conn,
                territory="mock-gym",
                brain=brain_for("mock-gym"),
                registry=registry,
                retry_budget=retry_budget,
                max_iterations=1,
            )
        finally:
            await conn.close()


async def test_a_stream_behind_the_tee_stops_at_its_first_error_frame(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # W3.1-K1: the body's second chunk is pulled only if something reads past the error frame.
    pulled: list[int] = []

    class Body(httpx2.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            pulled.append(1)
            yield MESSAGE_START + ERROR_FRAME
            pulled.append(2)
            yield _frame("ping", {"type": "ping"})

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=Body())

    tee = hunt_smoke.CapturingTransport(httpx2.MockTransport(handler), brain.MESSAGES_URL)
    await _one_hunt(tee, mock_anthropic, tmp_path / "rex.db", stream=True, retry_budget=0)
    assert pulled == [1]
    assert tee.captured == [MESSAGE_START + ERROR_FRAME]  # captured as far as it was read


async def test_the_tee_keeps_retry_after(
    tmp_path: Path, mock_anthropic: MockAnthropic, retry_waits: list[float]
) -> None:
    # W3.2-X17: a 429 asking for 3 s; the first retry's wait is drawn from [0, 0.5] s otherwise.
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            429,
            headers={"retry-after": "3"},
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}},
        )

    tee = hunt_smoke.CapturingTransport(httpx2.MockTransport(handler), brain.MESSAGES_URL)
    await _one_hunt(tee, mock_anthropic, tmp_path / "rex.db", stream=False, retry_budget=1)
    assert retry_waits == [3.0]
    assert len(tee.captured) == 2  # both attempts teed


async def test_a_compressed_reply_passes_the_tee_decoded(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # The reason the old tee rebuilt its responses: a gzip `content-encoding` header left over
    # decoded bytes makes the adapter's read fail. The tee drops that header, keeps the rest.
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"content-encoding": "gzip", "content-type": "application/json"},
            content=gzip.compress(NEEDS_HELP_REPLY),
        )

    tee = hunt_smoke.CapturingTransport(httpx2.MockTransport(handler), brain.MESSAGES_URL)
    db_path = tmp_path / "rex.db"
    await _one_hunt(tee, mock_anthropic, db_path, stream=False, retry_budget=0)
    assert tee.captured == [NEEDS_HELP_REPLY]
    conn = await db.connect(db_path)
    try:
        async with conn.execute("SELECT outcome FROM runs") as cur:
            rows = [tuple(r) for r in await cur.fetchall()]
    finally:
        await conn.close()
    assert rows == [("needs_help",)]  # the decoded reply parsed


# ── Fixtures: a failed hunt still writes what it captured ─────────────────────

FIXTURES = Path(__file__).parent / "fixtures"  # the recorded tree, read-only here
SNIFF_REPLY = json.dumps(
    {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": brain.SMOKE_MODEL,
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "sniff", "input": {"prey": "mock-gym"}}
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
).encode()
SNIFF_STREAM = b"".join(
    [
        MESSAGE_START,
        _frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "sniff",
                    "input": {},
                },
            },
        ),
        _frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"prey": "mock-gym"}'},
            },
        ),
        _frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 20},
            },
        ),
        _frame("message_stop", {"type": "message_stop"}),
    ]
)


def _tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            h.update(str(path.relative_to(root)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def _written(fixture_dir: Path) -> list[bytes]:
    return [p.read_bytes() for p in sorted(fixture_dir.glob("hunt_*/call_*.json"))]


def _answer_once_then_hang(first: bytes, second: asyncio.Event) -> Callable[..., Any]:
    """The free pre-flight, one brain reply, then a brain call that never answers: the moment an
    operator presses Ctrl-C."""
    calls = {"n": 0}

    async def handler(request: httpx2.Request) -> httpx2.Response:
        if str(request.url) == brain.COUNT_TOKENS_URL:
            return httpx2.Response(200, json={"input_tokens": 1})
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx2.Response(200, content=first)
        second.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable: the hunt is cancelled while this call waits")

    return handler


def _route_plain_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, handler: Callable[..., Any]
) -> Path:
    """hunt_smoke.run_live_hunt has no transport or directory seam, and write_fixtures' default
    directory is tests/fixtures, bound when it was defined, so each is routed here by hand: the
    live transport to `handler`, the fixtures to the returned directory, the run DB under
    tmp_path."""
    fixture_dir = tmp_path / "fx"
    monkeypatch.setattr(brain, "live_transport", lambda: httpx2.MockTransport(handler))
    monkeypatch.setattr(
        hunt_smoke,
        "write_fixtures",
        functools.partial(hunt_smoke.write_fixtures, fixture_dir=fixture_dir),
    )
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(stub, "SNIFF_INTERVAL", 0.0)  # the stub sniff's 5 s beat proves nothing
    return fixture_dir


async def test_a_cancelled_plain_hunt_still_writes_what_it_captured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _tree_digest(FIXTURES)
    second = asyncio.Event()
    fixture_dir = _route_plain_harness(
        monkeypatch, tmp_path, _answer_once_then_hang(SNIFF_REPLY, second)
    )
    task = asyncio.create_task(hunt_smoke.run_live_hunt("offline-key"))
    await second.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _written(fixture_dir) == [SNIFF_REPLY]
    assert _tree_digest(FIXTURES) == before


async def test_the_plain_harness_keeps_its_fixtures_when_a_reply_is_not_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A 502 from a proxy, in HTML: the hunt ends typed after its retries, then the reporter's
    # json.loads raises on the body. The fixtures are already on disk by then.
    html = b"<html><body>502 Bad Gateway</body></html>"

    def handler(request: httpx2.Request) -> httpx2.Response:
        if str(request.url) == brain.COUNT_TOKENS_URL:
            return httpx2.Response(200, json={"input_tokens": 1})
        return httpx2.Response(502, headers={"content-type": "text/html"}, content=html)

    before = _tree_digest(FIXTURES)
    fixture_dir = _route_plain_harness(monkeypatch, tmp_path, handler)
    with contextlib.suppress(json.JSONDecodeError):
        await hunt_smoke.run_live_hunt("offline-key")
    assert _written(fixture_dir) == [html] * 3  # the first attempt and its two retries
    assert _tree_digest(FIXTURES) == before


async def test_a_cancelled_thinking_hunt_still_writes_what_it_captured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # the run DB
    monkeypatch.setattr(stub, "SNIFF_INTERVAL", 0.0)
    second = asyncio.Event()
    fixture_dir = tmp_path / "fx"
    task = asyncio.create_task(
        hunt_smoke_thinking.run_live_hunt(
            "offline-key",
            inner=httpx2.MockTransport(_answer_once_then_hang(SNIFF_STREAM, second)),
            fixture_dir=fixture_dir,
        )
    )
    await second.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _written(fixture_dir) == [SNIFF_STREAM]


async def test_the_thinking_harness_keeps_its_fixtures_when_its_reporter_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # W3.1-O1's body, a `data:` line that is not JSON: since W3.4 the hunt ends typed, but the
    # reporter re-reads each captured stream with iter_sse_events, which raises on that line.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # the run DB
    body = (
        MESSAGE_START
        + b"event: content_block_delta\ndata: {not json\n\n"
        + SNIFF_STREAM[len(MESSAGE_START) :]
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        if str(request.url) == brain.COUNT_TOKENS_URL:
            return httpx2.Response(200, json={"input_tokens": 1})
        return httpx2.Response(200, content=body)

    fixture_dir = tmp_path / "fx"
    with contextlib.suppress(ValueError):
        await hunt_smoke_thinking.run_live_hunt(
            "offline-key", inner=httpx2.MockTransport(handler), fixture_dir=fixture_dir
        )
    assert _written(fixture_dir) == [body]
