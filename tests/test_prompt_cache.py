"""Prompt caching on the live brain, and the cache's token counts read (`P5.1`, Part A1).

A response counts the tokens written to and read from the prompt cache apart from `input_tokens`.
`brain._read_usage`, the one reader, reads both on the plain and the streamed path alike, under the
check it gives the other two counts, and a null or absent cache count reads 0; the UsageEvent the
loop stores carries them.

The live brain (`brain.select_brain_for`'s live branch: the daemon and `eval_wave`'s live wave)
asks for automatic caching with a top-level `cache_control`, and the wave's count pre-flight counts
that same first body. Every other path sends the bytes it sent before: the wire check's bodies 1 to
9, pinned here by digest.

Offline throughout: an httpx2 MockTransport answers every call under a dummy key, and every
response is built here from literals, or is a recorded fixture's bytes, read only.
"""

import hashlib
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from rexhunter import brain, db, hunt_smoke, loop, stub
from rexhunter.board.catalogue import EVAL_BOARDS, EVAL_CATALOGUE, GYM_BOARDS, GYM_CATALOGUE
from rexhunter.board.client import BoardClient
from rexhunter.board.gym import build_gym
from rexhunter.board.tools import GYM_SYSTEM_PROMPT, build_registry, gym_seed
from rexhunter.events import BrainParseError, UsageEvent
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]
MODEL = brain.SMOKE_MODEL
DUMMY_KEY = "sk-test-not-real"
FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "hunt_5b62bc48-7577-4a52-b003-bdabf592061d"
    / "call_01.json"
)

# The wire check's bodies: 1 to 9 as every path without the cache setting sends them, and their
# successors with it, 10 (body 7, the live first request on gym-greenhouse) and 12 (body 8, the
# count body), each the older body with the one field added (11, body 9's, is pinned in
# tests/test_eval_wave.py).
BODY_1 = "9a55cae7ac0258b353538d248e7e6af868ac4aa9bf93363fdd3be94ffa185a88"
BODY_2 = "761995a6b42d31ab4bb7e4d49cae1ac52dcf727737f09d29f045fa1a9b57b3ef"
BODY_3 = "95bd65977e3897fb9db0617abd947285bccc7146e83a98c4219f5fbc301f34d1"
BODY_4 = "bb8870fe7fcf3651898bd99046b9e1fca0a41fa396cd91d6ad3b0179af4af8e4"
BODY_6 = "5f41494171ac3c0473db068cbfc3fdae6d552d7f9a6fdc4e6cb76f4e3efae7a8"
BODY_7 = "6410c3d0206e90a4ec1899fde35f6cabc54635039dab560b3a55e99cd6cfcca0"
BODY_8 = "16d7595dabcf1c23de57f375d2122df15b29f656ea79e8d7c16893ebe521c5bf"
BODY_9 = "d17fc81ee94efc1374c8a07ce6ffb047365fbd23cac351df0c46695ef39bbc98"
BODY_10 = "2ae0821a905cd9dd9d8ac35446765d99bd394b4a153d97ae55dc3c55f6c42492"
BODY_12 = "457ddb6676681959b28f0e324f4e10cddec9648cfe0070d96212676962cc8ab6"

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


def _stream(
    *, start: dict[str, Any] | None = None, end: dict[str, Any] | None = None, cut: bool = False
) -> bytes:
    frames = [
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
    return b"".join(frames[:-1] if cut else frames)  # `cut`: no message_stop, a retried attempt


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


def _digests(sent: list[httpx2.Request]) -> list[str]:
    return [hashlib.sha256(request.content).hexdigest() for request in sent]


@asynccontextmanager
async def _gym() -> AsyncGenerator[tuple[ToolRegistry, ToolRegistry]]:
    # The daemon's registry (the six gym boards) and the wave's (those and gym-mimic), over one
    # in-process gym: nothing reaches a host but the gym.
    gym = httpx2.ASGITransport(app=build_gym(GYM_CATALOGUE | EVAL_CATALOGUE))
    async with BoardClient(gym) as board:
        yield stub.gym_registry(board), build_registry(board, GYM_BOARDS | EVAL_BOARDS)


async def test_the_live_brain_asks_for_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    # The daemon's selection, as the lifespan makes it (server.py), offline: the live brain's
    # first request is body 7 with the one top-level field added, nothing else moved.
    sent: list[httpx2.Request] = []
    factory = brain.anthropic_client

    def offline(api_key: str, *, timeout: float) -> anthropic.AsyncAnthropic:
        return factory(api_key, transport=_serving([_stream()], sent), timeout=timeout)

    monkeypatch.setenv("REXHUNTER_BRAIN", "live")
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_KEY)
    monkeypatch.setattr(brain, "anthropic_client", offline)
    async with _gym() as (daemon_registry, _wave_registry):
        brain_for, client = brain.select_brain_for(
            daemon_registry, default=stub.stub_brain_for, system=GYM_SYSTEM_PROMPT, seed=gym_seed
        )
        assert client is not None
        try:
            await brain_for("gym-greenhouse")([], _drop)
        finally:
            await client.close()
        uncached: list[httpx2.Request] = []
        async with factory(DUMMY_KEY, transport=_serving([_stream()], uncached)) as plain:
            await brain.adapter_brain_for(
                client=plain,
                model=MODEL,
                registry=daemon_registry,
                max_tokens=brain.HUNT_MAX_TOKENS,
                thinking=brain.HUNT_THINKING,
                stream=True,
                system=GYM_SYSTEM_PROMPT,
                seed=gym_seed,
            )("gym-greenhouse")([], _drop)

    (request,) = sent
    body = json.loads(request.content)
    assert body["cache_control"] == {"type": "ephemeral"}
    assert {k: v for k, v in body.items() if k != "cache_control"} == json.loads(
        uncached[0].content
    )
    assert _digests(sent) == [BODY_10]


async def test_the_count_preflight_with_the_setting_counts_the_cached_body(
    mock_anthropic: MockAnthropic,
) -> None:
    sent: list[httpx2.Request] = []
    async with _gym() as (_daemon_registry, wave_registry):
        async with mock_anthropic(_serving([b'{"input_tokens": 1}'], sent)) as client:
            await hunt_smoke.count_tokens_preflight(
                client,
                wave_registry,
                territory="gym-greenhouse",
                thinking=brain.HUNT_THINKING,
                system=GYM_SYSTEM_PROMPT,
                seed=gym_seed,
                prompt_cache=True,
            )
    assert json.loads(sent[0].content)["cache_control"] == {"type": "ephemeral"}
    assert _digests(sent) == [BODY_12]


type Drive = Callable[[MockAnthropic, list[httpx2.Request]], Awaitable[None]]


async def _plain_first_call(mock_anthropic: MockAnthropic, sent: list[httpx2.Request]) -> None:
    async with mock_anthropic(_serving([_plain([NEEDS_HELP])], sent)) as client:
        b = brain.adapter_brain_for(client=client, model=MODEL, registry=stub.build_registry())
        await b("mock-gym")([], _drop)


async def _streamed_first_call(mock_anthropic: MockAnthropic, sent: list[httpx2.Request]) -> None:
    async with mock_anthropic(_serving([_stream()], sent)) as client:
        b = brain.adapter_brain_for(
            client=client,
            model=MODEL,
            registry=stub.build_registry(),
            max_tokens=brain.HUNT_MAX_TOKENS,
            thinking=brain.HUNT_THINKING,
            stream=True,
        )
        await b("mock-gym")([], _drop)


async def _sniff_count(mock_anthropic: MockAnthropic, sent: list[httpx2.Request]) -> None:
    async with mock_anthropic(_serving([b'{"input_tokens": 1}'], sent)) as client:
        await hunt_smoke.count_tokens_preflight(client, stub.build_registry())


async def _hunt(
    mock_anthropic: MockAnthropic, sent: list[httpx2.Request], replies: list[bytes], *, stream: bool
) -> None:
    registry = stub.build_registry()
    async with mock_anthropic(_serving(replies, sent)) as client:
        b = brain.adapter_brain_for(
            client=client,
            model=MODEL,
            registry=registry,
            max_tokens=brain.HUNT_MAX_TOKENS if stream else 1024,
            thinking=brain.HUNT_THINKING if stream else None,
            stream=stream,
        )
        conn = await db.connect(":memory:")
        try:
            await loop.run_hunt(
                conn, territory="mock-gym", brain=b("mock-gym"), registry=registry, max_iterations=2
            )
        finally:
            await conn.close()


async def _two_turn_plain_hunt(mock_anthropic: MockAnthropic, sent: list[httpx2.Request]) -> None:
    await _hunt(mock_anthropic, sent, [FIXTURE.read_bytes(), _plain([NEEDS_HELP])], stream=False)
    del sent[0]  # body 4 is the hunt's second request


async def _cut_then_retried(mock_anthropic: MockAnthropic, sent: list[httpx2.Request]) -> None:
    await _hunt(mock_anthropic, sent, [_stream(cut=True), _stream()], stream=True)


def _gym_turn(territory: str, *, stream: bool, wave: bool) -> Drive:
    async def drive(mock_anthropic: MockAnthropic, sent: list[httpx2.Request]) -> None:
        async with _gym() as (daemon_registry, wave_registry):
            async with mock_anthropic(
                _serving([_stream() if stream else _plain([NEEDS_HELP])], sent)
            ) as client:
                b = brain.adapter_brain_for(
                    client=client,
                    model=MODEL,
                    registry=wave_registry if wave else daemon_registry,
                    max_tokens=brain.HUNT_MAX_TOKENS,
                    thinking=brain.HUNT_THINKING if stream else None,
                    stream=stream,
                    system=GYM_SYSTEM_PROMPT,
                    seed=gym_seed,
                )
                await b(territory)([], _drop)

    return drive


async def _gym_count(mock_anthropic: MockAnthropic, sent: list[httpx2.Request]) -> None:
    async with _gym() as (_daemon_registry, wave_registry):
        async with mock_anthropic(_serving([b'{"input_tokens": 1}'], sent)) as client:
            await hunt_smoke.count_tokens_preflight(
                client,
                wave_registry,
                territory="gym-greenhouse",
                thinking=brain.HUNT_THINKING,
                system=GYM_SYSTEM_PROMPT,
                seed=gym_seed,
            )


@pytest.mark.parametrize(
    ("drive", "digests"),
    [
        (_plain_first_call, [BODY_1]),
        (_streamed_first_call, [BODY_2]),
        (_sniff_count, [BODY_3]),
        (_two_turn_plain_hunt, [BODY_4]),
        (_cut_then_retried, [BODY_2, BODY_2]),  # body 5: the retry sends body 2 again
        (_gym_turn("gym-greenhouse", stream=False, wave=False), [BODY_6]),
        (_gym_turn("gym-greenhouse", stream=True, wave=False), [BODY_7]),
        (_gym_count, [BODY_8]),
        (_gym_turn("gym-mimic", stream=True, wave=True), [BODY_9]),
    ],
    ids=[f"body-{n}" for n in range(1, 10)],
)
async def test_paths_without_the_setting_send_todays_bytes(
    drive: Drive,
    digests: list[str],
    mock_anthropic: MockAnthropic,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stub, "SNIFF_INTERVAL", 0.0)  # body 4's sniff, without its 5 s beat
    sent: list[httpx2.Request] = []
    await drive(mock_anthropic, sent)
    assert all("cache_control" not in json.loads(request.content) for request in sent)
    assert _digests(sent) == digests
