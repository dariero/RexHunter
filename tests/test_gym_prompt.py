"""The gym prompt and seed reach the adapter (`P2.4`, fork `profile`).

`adapter_brain_for` takes the system prompt and the seed directive it sends; their defaults are
the sniff hunt's, so a caller that passes neither sends the bytes it sent before. A board hunt
passes the gym's, which names the board tools. Offline: the SDK client runs over an httpx2
`MockTransport`, with a dummy key.
"""

import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
import pytest

from rexhunter import brain, stub
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]

NEEDS_HELP = json.dumps(
    {
        "id": "msg_gym",
        "type": "message",
        "role": "assistant",
        "model": brain.SMOKE_MODEL,
        "stop_reason": "tool_use",
        "content": [{"type": "tool_use", "id": "toolu_gym", "name": "needs_help", "input": {}}],
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
).encode()


async def _first_request(
    mock_anthropic: MockAnthropic, registry: ToolRegistry, territory: str, **kwargs: Any
) -> dict[str, Any]:
    sent: list[bytes] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.content)
        return httpx2.Response(200, content=NEEDS_HELP)

    async with mock_anthropic(handler) as client:
        brain_for = brain.adapter_brain_for(
            client=client, model=brain.SMOKE_MODEL, registry=registry, **kwargs
        )

        async def sink(_text: str) -> None:
            return None

        await brain_for(territory)([], sink)
    (body,) = sent
    return json.loads(body)


async def test_the_adapter_sends_the_gym_prompt_seed_and_tools(
    mock_anthropic: MockAnthropic,
) -> None:
    # Looked up here, so on a tree without them this test fails on its own line.
    from rexhunter.board.catalogue import GYM_BOARDS, GYM_CATALOGUE
    from rexhunter.board.client import BoardClient
    from rexhunter.board.gym import build_gym
    from rexhunter.board.tools import GYM_SYSTEM_PROMPT, build_registry, gym_seed

    async with BoardClient(httpx2.ASGITransport(app=build_gym(GYM_CATALOGUE))) as board:
        body = await _first_request(
            mock_anthropic,
            build_registry(board, GYM_BOARDS),
            "gym-greenhouse",
            max_tokens=brain.HUNT_MAX_TOKENS,
            system=GYM_SYSTEM_PROMPT,
            seed=gym_seed,
        )

    assert body["system"] == GYM_SYSTEM_PROMPT
    assert body["messages"] == [{"role": "user", "content": gym_seed("gym-greenhouse")}]
    assert [t["name"] for t in body["tools"]] == [
        "list_postings",
        "read_posting",
        "hunt_complete",
        "needs_help",
    ]
    list_schema = body["tools"][0]["input_schema"]
    assert list_schema == {
        "additionalProperties": False,
        "properties": {
            "board": {"type": "string"},
            "page": {"default": 1, "maximum": 10, "minimum": 1, "type": "integer"},
        },
        "required": ["board"],
        "type": "object",
    }
    assert body["max_tokens"] == brain.HUNT_MAX_TOKENS
    assert body["thinking"] == {"type": "between_tools"}


async def test_the_adapter_defaults_keep_the_hunt_prompt_and_seed(
    mock_anthropic: MockAnthropic,
) -> None:
    body = await _first_request(mock_anthropic, stub.build_registry(), "mock-gym")

    assert body["system"] == brain.HUNT_SYSTEM_PROMPT
    assert body["messages"] == [
        {
            "role": "user",
            "content": "Hunt mock-gym for AI-engineering job postings. Call sniff to investigate; "
            "call hunt_complete with your catch when done, or needs_help if you are stuck.",
        }
    ]
