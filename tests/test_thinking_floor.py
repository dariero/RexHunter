"""W2.2 (F5) — the lowest thinking setting follows the model (offline, no spend, no network).

A hunt that wants thinking off sends the model's lowest setting, read from one table keyed by API
model id (`brain.THINKING_FLOOR`, fork F1): `claude-sonnet-5` turns thinking off with
`{"type": "disabled"}`; `claude-sonnet-5-5` rejects that with a 400, and its lowest setting is
`{"type": "between_tools"}`, which takes no other field. Both thinking-off paths read the table:
the adapter's default (`adapter_brain_for` without `thinking`) and the free pre-flight's body
(`hunt_smoke._hunt_body`). A model with no row fails closed, before any request is built.

The models here are literals, never `brain.SMOKE_MODEL`, so these tests mean the same thing
whichever model is live; the pre-flight tests monkeypatch `SMOKE_MODEL`, the one input
`_hunt_body` reads.
"""

import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
import pytest

from rexhunter import brain, hunt_smoke, stub

pytestmark = pytest.mark.anyio

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]
NO_ROW_MODEL = "claude-opus-5-5"  # rejects both off values: no lowest setting a hunt can use
_HUNT_COMPLETE: dict[str, Any] = {
    "type": "tool_use",
    "id": "t",
    "name": "hunt_complete",
    "input": {"catch": []},
}


async def _drop(_text: str) -> None:
    """A ThinkingSink that discards deltas (the plain path relays none)."""


async def _adapter_body(mock_anthropic: MockAnthropic, model: str) -> dict[str, Any]:
    """Drive one adapter turn on `model` with no `thinking` and return the body it sent."""
    sent: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json={"content": [_HUNT_COMPLETE]})

    async with mock_anthropic(handler) as client:
        brain_for = brain.adapter_brain_for(
            client=client, model=model, registry=stub.build_registry()
        )
        await brain_for("mock-gym")([], _drop)
    assert len(sent) == 1
    return sent[0]


def test_the_table_has_a_row_for_each_model_a_hunt_runs_with_thinking_off() -> None:
    # Two rows, each the whole thinking value: `between_tools` carries the one key `type`
    # (`display`, `budget_tokens` or `block_binding` beside it is a 400).
    assert brain.THINKING_FLOOR == {
        "claude-sonnet-5": {"type": "disabled"},
        "claude-sonnet-5-5": {"type": "between_tools"},
    }
    assert list(brain.THINKING_FLOOR["claude-sonnet-5-5"]) == ["type"]


def test_the_lookup_returns_a_fresh_copy_of_the_row() -> None:
    floor = brain.thinking_floor("claude-sonnet-5-5")
    assert floor == {"type": "between_tools"}
    floor["display"] = "summarized"  # a caller's edit must not reach the table
    assert brain.THINKING_FLOOR["claude-sonnet-5-5"] == {"type": "between_tools"}


def test_the_lookup_refuses_a_model_with_no_row() -> None:
    with pytest.raises(ValueError, match=NO_ROW_MODEL) as raised:
        brain.thinking_floor(NO_ROW_MODEL)
    assert "THINKING_FLOOR" in str(raised.value)


async def test_the_adapter_on_sonnet_5_5_without_thinking_sends_between_tools(
    mock_anthropic: MockAnthropic,
) -> None:
    body = await _adapter_body(mock_anthropic, "claude-sonnet-5-5")
    assert body["model"] == "claude-sonnet-5-5"
    assert body["thinking"] == {"type": "between_tools"}


async def test_the_adapter_on_sonnet_5_without_thinking_still_sends_disabled(
    mock_anthropic: MockAnthropic,
) -> None:
    # The Sonnet 5 wire is unchanged by W2.2: the recorded runs replay against it.
    body = await _adapter_body(mock_anthropic, "claude-sonnet-5")
    assert body["model"] == "claude-sonnet-5"
    assert body["thinking"] == {"type": "disabled"}


async def test_the_adapter_refuses_a_model_with_no_row_before_any_request(
    mock_anthropic: MockAnthropic,
) -> None:
    calls: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)  # a request here means the check came too late
        return httpx2.Response(200, json={"content": [_HUNT_COMPLETE]})

    async with mock_anthropic(handler) as client:
        with pytest.raises(ValueError, match=NO_ROW_MODEL):
            brain.adapter_brain_for(
                client=client, model=NO_ROW_MODEL, registry=stub.build_registry()
            )
    assert calls == []


async def test_the_preflight_body_follows_the_model(
    mock_anthropic: MockAnthropic, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(brain, "SMOKE_MODEL", "claude-sonnet-5-5")
    sent: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json={"input_tokens": 1})

    async with mock_anthropic(handler) as client:
        await hunt_smoke.count_tokens_preflight(client, stub.build_registry())
    assert len(sent) == 1
    assert sent[0]["model"] == "claude-sonnet-5-5"
    assert sent[0]["thinking"] == {"type": "between_tools"}


async def test_the_preflight_refuses_a_model_with_no_row_before_any_request(
    mock_anthropic: MockAnthropic, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(brain, "SMOKE_MODEL", NO_ROW_MODEL)
    calls: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)  # the free call must not be sent either
        return httpx2.Response(200, json={"input_tokens": 1})

    async with mock_anthropic(handler) as client:
        with pytest.raises(ValueError, match=NO_ROW_MODEL):
            await hunt_smoke.count_tokens_preflight(client, stub.build_registry())
    assert calls == []
