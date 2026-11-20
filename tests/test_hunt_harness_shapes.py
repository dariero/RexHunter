"""Each gated hunt harness's requests and its count-only path.

Per harness: the free pre-flight is the hunt's first request less `max_tokens` and `stream`, both
carry that harness's thinking value, its client has its own read timeout, and `--count-only` sends
the pre-flight alone. Every response is built here and every client runs over a MockTransport, so
nothing reaches the network.
"""

import functools
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn

import anthropic
import httpx2
import pytest

from rexhunter import brain, hunt_smoke, hunt_smoke_thinking, stub

NEEDS_HELP: dict[str, Any] = {
    "type": "tool_use",
    "id": "toolu_1",
    "name": "needs_help",
    "input": {},
}


def _plain_reply() -> bytes:
    return json.dumps(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": brain.SMOKE_MODEL,
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "content": [NEEDS_HELP],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
    ).encode()


def _frame(data: dict[str, Any]) -> str:
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n"


def _stream_reply() -> bytes:
    start: dict[str, Any] = {
        "model": brain.SMOKE_MODEL,
        "content": [],
        "usage": {"input_tokens": 10},
    }
    frames = [
        _frame({"type": "message_start", "message": start}),
        _frame({"type": "content_block_start", "index": 0, "content_block": NEEDS_HELP}),
        _frame({"type": "content_block_stop", "index": 0}),
        _frame(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 2},
            }
        ),
        _frame({"type": "message_stop"}),
    ]
    return "".join(frames).encode()


class _Recorder:
    def __init__(self, reply: bytes) -> None:
        self.requests: list[httpx2.Request] = []
        self._reply = reply

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if str(request.url) == brain.COUNT_TOKENS_URL:
            return httpx2.Response(200, json={"input_tokens": 7})
        return httpx2.Response(200, content=self._reply)


def _reached(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise AssertionError("--count-only reached the paid hunt")


def _thinking_of(module: ModuleType) -> dict[str, Any]:
    if module is hunt_smoke_thinking:
        return hunt_smoke_thinking.THINKING
    return brain.thinking_floor(brain.SMOKE_MODEL)  # the plain hunt sends the model's floor


HARNESSES = [
    pytest.param(hunt_smoke, 120.0, id="hunt_smoke"),
    pytest.param(hunt_smoke_thinking, 180.0, id="hunt_smoke_thinking"),
]


@pytest.mark.anyio
@pytest.mark.parametrize(("module", "read_timeout"), HARNESSES)
async def test_each_harness_sends_its_own_request_shape(
    module: ModuleType, read_timeout: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    streamed = module is hunt_smoke_thinking
    record = _Recorder(_stream_reply() if streamed else _plain_reply())
    fixture_dir = tmp_path / "fx"
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(stub, "SNIFF_INTERVAL", 0.0)
    if streamed:
        await hunt_smoke_thinking.run_live_hunt(
            "offline-key", inner=httpx2.MockTransport(record), fixture_dir=fixture_dir
        )
    else:
        monkeypatch.setattr(brain, "live_transport", lambda: httpx2.MockTransport(record))
        monkeypatch.setattr(
            hunt_smoke,
            "write_fixtures",
            functools.partial(hunt_smoke.write_fixtures, fixture_dir=fixture_dir),
        )
        await hunt_smoke.run_live_hunt("offline-key")

    preflight, first = record.requests  # the hunt asks for help on its first call
    assert str(preflight.url) == brain.COUNT_TOKENS_URL
    assert str(first.url) == brain.MESSAGES_URL
    sent = json.loads(first.content)
    unsent = ("max_tokens", "stream")
    assert json.loads(preflight.content) == {k: v for k, v in sent.items() if k not in unsent}
    assert sent["thinking"] == _thinking_of(module)
    assert sent["max_tokens"] == module.HUNT_MAX_TOKENS
    assert sent.get("stream", False) is streamed
    assert [r.extensions["timeout"]["read"] for r in record.requests] == [read_timeout] * 2


@pytest.mark.parametrize(
    "module", [hunt_smoke, hunt_smoke_thinking], ids=["hunt_smoke", "hunt_smoke_thinking"]
)
def test_count_only_sends_the_preflight_alone(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _Recorder(b"")
    factory = brain.anthropic_client  # the real one, captured before it is patched below

    def offline_client(
        api_key: str, *, transport: object = None, timeout: float = 120.0
    ) -> anthropic.AsyncAnthropic:
        # Whatever transport the caller asked for, the requests go to the recorder.
        return factory(api_key, transport=httpx2.MockTransport(record), timeout=timeout)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.delenv("REXHUNTER_BRAIN", raising=False)
    monkeypatch.setattr(sys, "argv", [module.__name__, "--count-only"])
    monkeypatch.setattr(brain, "anthropic_client", offline_client)
    monkeypatch.setattr(brain, "live_transport", _reached)
    monkeypatch.setattr(module, "run_live_hunt", _reached)
    module.main()

    (request,) = record.requests
    assert str(request.url) == brain.COUNT_TOKENS_URL
    body = json.loads(request.content)
    assert body["thinking"] == _thinking_of(module)
    assert "max_tokens" not in body
    assert request.extensions["timeout"]["read"] == 120.0  # the count-only client, both harnesses
    out = capsys.readouterr().out
    assert out.startswith("[count-only] input_tokens=7 ")
    assert out.endswith(", NO paid call made.\n") and out.count("\n") == 1
