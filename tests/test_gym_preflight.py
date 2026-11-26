"""The free pre-flight counts a gym hunt's first body (`P2.5`): `hunt_smoke.count_tokens_preflight`
takes the hunt's prompt and seed, and without them still sends the sniff hunt's body.

Offline: a MockTransport answers each count with a literal, under a dummy key; nothing is sent.
"""

import hashlib
import json
from collections.abc import Callable

import anthropic
import httpx2
import pytest

from rexhunter import brain, hunt_smoke, stub
from rexhunter.board.catalogue import EVAL_BOARDS, EVAL_CATALOGUE, GYM_BOARDS, GYM_CATALOGUE
from rexhunter.board.client import BoardClient
from rexhunter.board.gym import build_gym
from rexhunter.board.tools import GYM_SYSTEM_PROMPT, build_registry, gym_seed

pytestmark = pytest.mark.anyio

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]

# The wire check's body 3, the sniff hunt's count body, and body 8, gym-greenhouse's count body with
# thinking on: body 7 (the daemon's live first request) without `max_tokens` and `stream`.
BODY_3 = "95bd65977e3897fb9db0617abd947285bccc7146e83a98c4219f5fbc301f34d1"
BODY_8 = "16d7595dabcf1c23de57f375d2122df15b29f656ea79e8d7c16893ebe521c5bf"


def _counting(sent: list[httpx2.Request]) -> Callable[[httpx2.Request], httpx2.Response]:
    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, json={"input_tokens": 1234})

    return handler


async def test_the_preflight_counts_the_gym_first_body(mock_anthropic: MockAnthropic) -> None:
    sent: list[httpx2.Request] = []
    gym = httpx2.ASGITransport(app=build_gym(GYM_CATALOGUE | EVAL_CATALOGUE))
    async with BoardClient(gym) as board, mock_anthropic(_counting(sent)) as client:
        registry = build_registry(board, GYM_BOARDS | EVAL_BOARDS)
        counted = await hunt_smoke.count_tokens_preflight(
            client,
            registry,
            territory="gym-greenhouse",
            thinking=brain.HUNT_THINKING,
            system=GYM_SYSTEM_PROMPT,
            seed=gym_seed,
        )

    (request,) = sent
    body = json.loads(request.content)
    assert counted == 1234
    assert str(request.url) == brain.COUNT_TOKENS_URL
    assert body["system"] == GYM_SYSTEM_PROMPT
    assert body["messages"] == [{"role": "user", "content": gym_seed("gym-greenhouse")}]
    assert body["thinking"] == brain.HUNT_THINKING
    assert "max_tokens" not in body and "stream" not in body
    assert hashlib.sha256(request.content).hexdigest() == BODY_8


async def test_the_preflight_without_a_prompt_sends_the_sniff_body(
    mock_anthropic: MockAnthropic,
) -> None:
    sent: list[httpx2.Request] = []
    async with mock_anthropic(_counting(sent)) as client:
        await hunt_smoke.count_tokens_preflight(client, stub.build_registry())

    (request,) = sent
    assert json.loads(request.content)["system"] == brain.HUNT_SYSTEM_PROMPT
    assert hashlib.sha256(request.content).hexdigest() == BODY_3
