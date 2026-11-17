"""W3.4 Part A: the spend and the boundary, one section per commit.

Every response is built here from literals; nothing under tests/fixtures/ is read or written. The
request model is `brain.SMOKE_MODEL`. Zero network: a `MockTransport` answers every call. A test
whose name starts `test_guard_` passed before W3.4 too: it pins what must not change.
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import anthropic
import httpx2
import pytest

from rexhunter import brain, cost, db
from rexhunter.events import BrainExchangeEvent, ErrorEvent, TrajectoryEvent, UsageEvent
from rexhunter.loop import run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]
Respond = Callable[[httpx2.Request], httpx2.Response]

MODEL = brain.SMOKE_MODEL
RETRY_BUDGET = 2  # run_hunt's default, passed explicitly: a retried call makes 3 requests
MALFORMED = ("aborted", "malformed provider response")
E, U, ERR = "brain_exchange", "usage", "error"

SNIFF: dict[str, Any] = {
    "type": "tool_use",
    "id": "toolu_w34",
    "name": "sniff",
    "input": {"prey": "mock-gym"},
}
# What a whole literal stream reports: 100 input tokens; 1 output token at message_start and 20 by
# the end (message_delta's cumulative count).
START_USAGE: dict[str, Any] = {"input_tokens": 100, "output_tokens": 1}


async def sniff(prey: str) -> str:
    """Sniff a territory for postings."""
    return f"posting:{prey}"


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.tool(sniff)
    return reg


def _plain(stop_reason: str, content: list[dict[str, Any]], usage: dict[str, Any]) -> bytes:
    """A whole Messages-API response, as the plain path reads it."""
    return json.dumps(
        {
            "id": "msg_w34",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "content": content,
            "usage": usage,
        }
    ).encode()


def _frame(data: dict[str, Any]) -> bytes:
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()


def _message_start(usage: dict[str, Any]) -> bytes:
    return _frame(
        {"type": "message_start", "message": {"model": MODEL, "content": [], "usage": usage}}
    )


def _message_delta(usage: dict[str, Any]) -> bytes:
    return _frame(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": usage,
        }
    )


# A whole streamed sniff turn, frame by frame.
SNIFF_STREAM = [
    _message_start(START_USAGE),
    _frame({"type": "content_block_start", "index": 0, "content_block": {**SNIFF, "input": {}}}),
    _frame(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(SNIFF["input"])},
        }
    ),
    _frame({"type": "content_block_stop", "index": 0}),
    _message_delta({"output_tokens": 20}),
    _frame({"type": "message_stop"}),
]


def _ok(body: bytes) -> Respond:
    return lambda _request: httpx2.Response(200, content=body)


class Hunt(NamedTuple):
    events: list[TrajectoryEvent]  # every event, BrainExchangeEvents included
    row: tuple[str, str | None]
    requests: int


async def _hunt(
    tmp_path: Path, mock_anthropic: MockAnthropic, respond: Respond, *, stream: bool
) -> Hunt:
    """One run_hunt whose every brain request is answered by `respond`."""
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return respond(request)

    reg = _registry()
    async with mock_anthropic(handler, api_key="offline") as client:
        brain_for = brain.adapter_brain_for(
            client=client,
            model=MODEL,
            registry=reg,
            thinking=brain.HUNT_THINKING if stream else None,
            stream=stream,
        )
        conn = await db.connect(tmp_path / "rex.db")
        try:
            run_id = await run_hunt(
                conn,
                territory="mock-gym",
                brain=brain_for("mock-gym"),
                registry=reg,
                retry_budget=RETRY_BUDGET,
                max_iterations=3,
                cost_ceiling_usd=1000.0,
                tool_timeout_s=5.0,
            )
            events = await db.read_events(conn, run_id)
            async with conn.execute(
                "SELECT outcome, abort_reason FROM runs WHERE id = ?", (run_id,)
            ) as cur:
                row = await cur.fetchone()
        finally:
            await conn.close()
    assert row is not None
    return Hunt(events, (str(row[0]), row[1]), len(requests))


def _types(events: Sequence[TrajectoryEvent]) -> list[str]:
    return [e.type for e in events]


def _errors(events: Sequence[TrajectoryEvent]) -> list[ErrorEvent]:
    return [e for e in events if isinstance(e, ErrorEvent)]


def _usages(events: Sequence[TrajectoryEvent]) -> list[tuple[int, int]]:
    return [(e.input_tokens, e.output_tokens) for e in events if isinstance(e, UsageEvent)]


def _assert_malformed(hunt: Hunt, *, types: list[str], cause: str) -> ErrorEvent:
    """One attempt, refused "malformed provider response" with its bytes and its exchange first."""
    assert _types(hunt.events) == types
    assert hunt.row == MALFORMED
    assert hunt.requests == 1  # fatal: never retried
    [error] = _errors(hunt.events)
    assert (error.tool, error.error, error.retryable) == (
        "<brain>",
        "malformed provider response",
        False,
    )
    assert error.raw_response  # the bytes (invariant 6)
    assert cause in (error.detail or "")
    [exchange] = [e for e in hunt.events if isinstance(e, BrainExchangeEvent)]
    assert exchange.raw_response  # the attempt's bytes as received (W3.3)
    return error


# ── counts: a token count crosses the boundary only as a bounded non-negative int ──────────
# W3.2-RV12 (a negative count lowered the folds), the count types (a bool, a numeric string or a
# float read as a number), W3.2-RV1b (a 315 to 4300 digit count on an accepted turn was written and
# a later fold raised) and W3.1-O5 (an accepted turn's count that is not a number reached the
# `<loop>` catch-all without its bytes). `brain.MAX_TOKEN_COUNT` is 10,000,000: one past it is
# refused on both paths, the bound itself is read.

BAD_COUNTS = [True, "100", 100.0, -1, 10_000_001, 10**400, "x", None, [1]]
BAD_IDS = [
    "bool",
    "numeric-string",
    "float",
    "negative",
    "bound+1",
    "401-digits",
    "word",
    "null",
    "list",
]


@pytest.mark.parametrize("bad", BAD_COUNTS, ids=BAD_IDS)
async def test_a_refused_turn_whose_count_is_not_a_token_count_is_not_counted(
    bad: object, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # A turn refused for its stop reason. Before W3.4 `int()` read a bool, a numeric string, a
    # float, a negative count and one past the bound as tokens; a word, null and a list went
    # uncounted already, and so did a 401-digit count (the pricing check W3.4 drops).
    body = _plain("max_tokens", [SNIFF], {"input_tokens": bad, "output_tokens": 20})
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=False)
    error = _assert_malformed(hunt, types=[E, ERR], cause="stop_reason='max_tokens'")
    assert "usage not counted" in (error.detail or "")


async def test_a_negative_count_never_lowers_the_fold(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # W3.2-RV12: a refused turn's negative count was counted, so the spend folds went down.
    body = _plain("max_tokens", [SNIFF], {"input_tokens": -5_000_000, "output_tokens": 20})
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=False)
    assert cost.fold_cost(hunt.events) == 0.0
    assert _usages(hunt.events) == []


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
@pytest.mark.parametrize("bad", BAD_COUNTS, ids=BAD_IDS)
async def test_an_accepted_turn_whose_count_is_not_a_token_count_is_refused_with_its_bytes(
    bad: object, stream: bool, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # W3.1-O5 (a word, null, a list: `int()` raised, so the run ended at the `<loop>` catch-all
    # without its bytes), the count types and a negative count or one past the bound (the turn was
    # dispatched and the count written), and W3.2-RV1b (401 digits: written, then the next
    # attempt's fold raised). Now the turn is refused with its bytes and writes no UsageEvent.
    usage = {"input_tokens": bad, "output_tokens": 20}
    if stream:
        body = b"".join([_message_start(usage), *SNIFF_STREAM[1:]])
    else:
        body = _plain("tool_use", [SNIFF], usage)
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=stream)
    error = _assert_malformed(hunt, types=[E, ERR], cause="usage not counted")
    if not stream:
        assert error.raw_response == body


async def test_a_huge_count_on_an_accepted_turn_is_never_written(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # W3.2-RV1b: a 401-digit count on an accepted turn was written; the next attempt's fold raised
    # OverflowError and the run ended at the `<loop>` catch-all.
    body = _plain("tool_use", [SNIFF], {"input_tokens": 10**400, "output_tokens": 20})
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=False)
    assert hunt.row == MALFORMED
    assert _usages(hunt.events) == []
    assert cost.fold_cost(hunt.events) == 0.0  # the fold never raises


@pytest.mark.parametrize("stop_reason", ["tool_use", "max_tokens"], ids=["accepted", "refused"])
@pytest.mark.parametrize("digits", [4301, 5000])
async def test_guard_a_count_over_4300_digits_is_refused_at_the_envelope(
    digits: int, stop_reason: str, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # Past Python's default `int_max_str_digits` (4300), `repr` of an int raises ValueError, so a
    # refusal whose message showed such a count would raise inside itself. pydantic's JSON parser
    # refuses the literal first ("number out of range"): the body fails the envelope, refused with
    # its bytes and its exchange, not counted. `json.dumps` cannot write the literal, so it is put
    # into the body's bytes here.
    marker = "COUNT_LITERAL"
    body = _plain(stop_reason, [SNIFF], {"input_tokens": marker, "output_tokens": 20})
    body = body.replace(f'"{marker}"'.encode(), b"9" * digits)
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=False)
    error = _assert_malformed(hunt, types=[E, ERR], cause="number out of range")
    assert error.raw_response == body


@pytest.mark.parametrize(
    ("usage", "counted"),
    [
        ({"input_tokens": 10_000_000, "output_tokens": 10_000_000}, (10_000_000, 10_000_000)),
        ({"input_tokens": 100}, (100, 0)),  # an absent count reads as 0, as before W3.4
        ({}, (0, 0)),
        (
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": None,
                "service_tier": "standard",
                "server_tool_use": {"web_search_requests": 1},
            },
            (100, 20),
        ),
    ],
    ids=["the-bound", "absent-output", "empty", "other-keys"],
)
async def test_guard_a_token_count_that_reads_is_counted_as_before(
    usage: dict[str, Any],
    counted: tuple[int, int],
    tmp_path: Path,
    mock_anthropic: MockAnthropic,
) -> None:
    # Only `input_tokens` and `output_tokens` are read; the other keys of `usage` are not.
    body = _plain("tool_use", [{**SNIFF, "name": "needs_help", "input": {}}], usage)
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=False)
    assert hunt.row == ("needs_help", None)
    assert _usages(hunt.events) == [counted]


# ── N1: a null count in message_delta keeps the earlier count, as the SDK does (W3.2-N1) ─────
# The assembler wrote every key of message_delta's `usage` over the counts so far, a null one too,
# so a null `input_tokens` replaced message_start's count. The SDK's accumulator keeps the earlier
# count when the new one is null (anthropic 2.6.0 lib/streaming/_messages.py:547-556). A count that
# is not null still replaces the earlier one: they are cumulative (the streaming page, :334).

NULL_COUNTS_DELTA = _message_delta(
    {"input_tokens": None, "cache_read_input_tokens": None, "output_tokens": 20}
)


async def test_a_null_count_in_message_delta_keeps_message_starts(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    body = b"".join([*SNIFF_STREAM[:4], NULL_COUNTS_DELTA, SNIFF_STREAM[5]])
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    assert _usages(hunt.events)[:1] == [(100, 20)]
    assert _types(hunt.events)[:4] == [E, U, "tool_call", "tool_result"]


async def test_a_null_count_in_message_delta_keeps_message_starts_on_a_refused_stream(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # A stream cut before message_stop (W3.1-T, retried): each attempt is counted with
    # message_start's input count, not left uncounted for a null.
    body = b"".join([*SNIFF_STREAM[:4], NULL_COUNTS_DELTA])
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    assert _types(hunt.events) == [E, U, ERR] * (RETRY_BUDGET + 1)
    assert _usages(hunt.events) == [(100, 20)] * (RETRY_BUDGET + 1)


async def test_guard_a_count_in_message_delta_still_replaces_message_starts(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    delta = _message_delta({"input_tokens": 150, "output_tokens": 20})
    body = b"".join([*SNIFF_STREAM[:4], delta, SNIFF_STREAM[5]])
    hunt = await _hunt(tmp_path, mock_anthropic, _ok(body), stream=True)
    assert _usages(hunt.events)[:1] == [(150, 20)]
