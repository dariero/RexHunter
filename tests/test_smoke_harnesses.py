"""The smoke harnesses' capture path (W3.4b), offline: every response is built here and answered by
an httpx2 MockTransport with a dummy key, so nothing reaches the network and nothing is spent.

W3.1-K1: the capturing tee read each body to its end before the adapter saw a byte, so a stream
could not stop at its first error frame (Z-stop, W3.1) and the thinking relay arrived all at once.
W3.2-X17: the tee rebuilt each response with two headers only, so a 429's `retry-after` never
reached the loop's backoff.
"""

import gzip
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from rexhunter import brain, db, hunt_smoke, stub
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
