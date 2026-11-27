"""The prompt cache's token counts, read (`P5.1`, Part A1).

A response counts the tokens written to and read from the prompt cache apart from `input_tokens`.
`brain._read_usage`, the one reader, reads both on the plain and the streamed path alike, under the
check it gives the other two counts, and a null or absent cache count reads 0; the UsageEvent the
loop stores carries them.

Offline throughout: an httpx2 MockTransport answers every call under a dummy key, and every
response is built here from literals.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from rexhunter import brain, db, loop, stub
from rexhunter.events import BrainParseError, UsageEvent

pytestmark = pytest.mark.anyio

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]
MODEL = brain.SMOKE_MODEL
DUMMY_KEY = "sk-test-not-real"

NEEDS_HELP: dict[str, Any] = {
    "type": "tool_use",
    "id": "toolu_cache",
    "name": "needs_help",
    "input": {},
}
# What a cached call reports: 100 tokens after the breakpoint, 300 written to the cache, 700 read
# from it; 1 output token at message_start and 20 by the end.
CACHED_START: dict[str, Any] = {
    "input_tokens": 100,
    "cache_creation_input_tokens": 300,
    "cache_read_input_tokens": 700,
    "output_tokens": 1,
}
CACHED_END: dict[str, Any] = {**CACHED_START, "output_tokens": 20}
CACHED_USAGE = {"input_tokens": 100, "output_tokens": 20, "cache_creation": 300, "cache_read": 700}


def _plain(content: list[dict[str, Any]], *, usage: dict[str, Any] | None = None) -> bytes:
    return json.dumps(
        {
            "id": "msg_cache",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "content": content,
            "usage": {"input_tokens": 10, "output_tokens": 2} if usage is None else usage,
        }
    ).encode()


def _frame(data: dict[str, Any]) -> bytes:
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()


def _stream(*, start: dict[str, Any] | None = None, end: dict[str, Any] | None = None) -> bytes:
    return b"".join(
        [
            _frame(
                {
                    "type": "message_start",
                    "message": {
                        "model": MODEL,
                        "content": [],
                        "usage": {"input_tokens": 10} if start is None else start,
                    },
                }
            ),
            _frame({"type": "content_block_start", "index": 0, "content_block": NEEDS_HELP}),
            _frame({"type": "content_block_stop", "index": 0}),
            _frame(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 2} if end is None else end,
                }
            ),
            _frame({"type": "message_stop"}),
        ]
    )


async def _drop(_text: str) -> None:
    return None


def _serving(replies: list[bytes], sent: list[httpx2.Request]) -> httpx2.MockTransport:
    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, content=replies[min(len(sent), len(replies)) - 1])

    return httpx2.MockTransport(handler)


async def _logged_usage(
    tmp_path: Path, mock_anthropic: MockAnthropic, reply: bytes, *, stream: bool
) -> tuple[UsageEvent, dict[str, Any]]:
    """One needs_help hunt answered by `reply`: its UsageEvent, and that event's stored payload."""
    sent: list[httpx2.Request] = []
    registry = stub.build_registry()
    async with mock_anthropic(_serving([reply], sent), api_key=DUMMY_KEY) as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            model=MODEL,
            registry=registry,
            thinking=brain.HUNT_THINKING if stream else None,
            stream=stream,
        )
        conn = await db.connect(tmp_path / "rex.db")
        try:
            run_id = await loop.run_hunt(
                conn, territory="mock-gym", brain=brain_for("mock-gym"), registry=registry
            )
            (usage,) = [e for e in await db.read_events(conn, run_id) if isinstance(e, UsageEvent)]
            rows = await conn.execute_fetchall(
                "SELECT payload FROM trajectory_events WHERE type = 'usage'"
            )
        finally:
            await conn.close()
    ((payload,),) = rows
    return usage, json.loads(str(payload))


def _counts(usage: UsageEvent) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation": usage.cache_creation_input_tokens,
        "cache_read": usage.cache_read_input_tokens,
    }


async def test_the_stream_carries_the_cache_counts(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    usage, payload = await _logged_usage(
        tmp_path, mock_anthropic, _stream(start=CACHED_START, end=CACHED_END), stream=True
    )
    assert _counts(usage) == CACHED_USAGE
    assert (payload["cache_creation_input_tokens"], payload["cache_read_input_tokens"]) == (
        300,
        700,
    )


async def test_the_plain_turn_carries_the_cache_counts(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    usage, payload = await _logged_usage(
        tmp_path, mock_anthropic, _plain([NEEDS_HELP], usage=CACHED_END), stream=False
    )
    assert _counts(usage) == CACHED_USAGE
    assert (payload["cache_creation_input_tokens"], payload["cache_read_input_tokens"]) == (
        300,
        700,
    )


async def _plain_turn(mock_anthropic: MockAnthropic, usage: dict[str, Any]) -> loop.Decision:
    sent: list[httpx2.Request] = []
    async with mock_anthropic(
        _serving([_plain([NEEDS_HELP], usage=usage)], sent), api_key=DUMMY_KEY
    ) as client:
        brain_for = brain.adapter_brain_for(
            client=client, model=MODEL, registry=stub.build_registry()
        )
        return await brain_for("mock-gym")([], _drop)


@pytest.mark.parametrize("key", ["cache_creation_input_tokens", "cache_read_input_tokens"])
async def test_a_null_cache_count_reads_as_zero(key: str, mock_anthropic: MockAnthropic) -> None:
    decision = await _plain_turn(
        mock_anthropic, {"input_tokens": 100, "output_tokens": 20, key: None}
    )
    assert decision.usage is not None
    assert getattr(decision.usage, key) == 0


@pytest.mark.parametrize("value", ["x", -1, True, 1.0, brain.MAX_TOKEN_COUNT + 1, [7]])
@pytest.mark.parametrize("key", ["cache_creation_input_tokens", "cache_read_input_tokens"])
async def test_a_cache_count_that_is_not_a_token_count_refuses_the_turn(
    key: str, value: object, mock_anthropic: MockAnthropic
) -> None:
    with pytest.raises(BrainParseError, match="usage not counted"):
        await _plain_turn(mock_anthropic, {"input_tokens": 100, "output_tokens": 20, key: value})
