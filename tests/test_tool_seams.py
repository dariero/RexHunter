"""The tool seams a board tool builds on: a tool's view, its error bytes, its `retry-after`.

A tool may register a view, and `brain.project_messages` renders each of its results through that
view, again from the stored bytes on every brain call; a tool that registers none renders as
before, and a result that is not UTF-8 renders escaped instead of ending the run at the `<loop>`
backstop. A ToolError carries the response bytes the tool got, stored on its ErrorEvent
(invariant 6), and a RetryableToolError's `retry_after` sets the least wait before its retry, cut
to RETRY_AFTER_CAP_S.

The tool errors are looked up on `loop` inside each test, so on a tree without them each test
fails on its own line. Every response is built here from literals; nothing under tests/fixtures/
is read or written. Zero network: a MockTransport answers every brain call.
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from rexhunter import brain, db, loop
from rexhunter.events import ErrorEvent, ToolResultEvent, TrajectoryEvent
from rexhunter.loop import Brain, Decision, HuntComplete, ToolCallDecision, run_hunt
from rexhunter.tools import ToolFn, ToolRegistry

pytestmark = pytest.mark.anyio

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]
BrainFactory = Callable[[Sequence[Decision]], Brain]
type Row = tuple[str, str | None]

MODEL = brain.SMOKE_MODEL
POSTING = '{"title": "Café engineer ✓"}'.encode()
NOT_UTF8 = b'{"title": "Caf\xe9 engineer"}'  # a lone 0xE9: Latin-1, not UTF-8


def _tool_use(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": f"toolu_{name}", "name": name, "input": tool_input}


def _plain(block: dict[str, Any]) -> bytes:
    return json.dumps(
        {
            "id": "msg_seams",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "content": [block],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
    ).encode()


async def _row(conn: Any, run_id: str) -> Row:
    async with conn.execute(
        "SELECT outcome, abort_reason FROM runs WHERE id = ?", (run_id,)
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    return str(row[0]), row[1]


async def _adapter_hunt(
    tmp_path: Path, mock_anthropic: MockAnthropic, reg: ToolRegistry, tool_input: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[TrajectoryEvent], Row]:
    """Call one asks for the registry's one tool, call two for needs_help. Returns each request
    body as sent, the run's events and its outcome."""
    (tool,) = reg.registered()
    responses = [_plain(_tool_use(tool.name, tool_input)), _plain(_tool_use("needs_help", {}))]
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, content=responses[min(len(bodies), len(responses)) - 1])

    conn = await db.connect(tmp_path / "rex.db")
    try:
        async with mock_anthropic(handler, api_key="offline") as client:
            brain_for = brain.adapter_brain_for(client=client, model=MODEL, registry=reg)
            run_id = await run_hunt(
                conn, territory="seams", brain=brain_for("seams"), registry=reg, max_iterations=3
            )
        events = await db.read_events(conn, run_id)
        row = await _row(conn, run_id)
    finally:
        await conn.close()
    return bodies, events, row


def _result_content(body: dict[str, Any]) -> Any:
    """The content of the tool_result the request's last turn carries."""
    (block,) = body["messages"][-1]["content"]
    assert block["type"] == "tool_result"
    return block["content"]


def _results(events: Sequence[TrajectoryEvent]) -> list[bytes]:
    return [e.raw_response for e in events if isinstance(e, ToolResultEvent)]


async def test_a_tool_without_a_view_projects_as_today(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    # The guard: a UTF-8 body reaches the model as its decoded text, before and after the view
    # seam, so every existing tool and recorded run renders unchanged.
    reg = ToolRegistry()

    async def scout(board: str) -> bytes:
        """Scout a board."""
        return POSTING

    reg.tool(scout)
    bodies, events, row = await _adapter_hunt(tmp_path, mock_anthropic, reg, {"board": "gh"})
    assert row == ("needs_help", None)
    assert len(bodies) == 2
    assert _result_content(bodies[1]) == POSTING.decode()
    assert _results(events) == [POSTING]


async def test_a_tool_result_that_is_not_utf8_does_not_end_the_run(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    reg = ToolRegistry()

    async def scout(board: str) -> bytes:
        """Scout a board."""
        return NOT_UTF8

    reg.tool(scout)
    bodies, events, row = await _adapter_hunt(tmp_path, mock_anthropic, reg, {"board": "gh"})
    assert row == ("needs_help", None)
    assert [e for e in events if isinstance(e, ErrorEvent)] == []
    assert len(bodies) == 2
    assert _result_content(bodies[1]) == '{"title": "Caf\\xe9 engineer"}'  # the byte escaped
    assert _results(events) == [NOT_UTF8]  # stored as it arrived (invariant 6)


async def test_a_registered_view_renders_the_tool_result(
    tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    reg = ToolRegistry()

    async def scout(board: str) -> bytes:
        """Scout a board."""
        return POSTING

    def headline(raw_request: bytes, raw_response: bytes) -> str:
        return f"{json.loads(raw_request)['board']}: {json.loads(raw_response)['title']}"

    reg.tool(scout, view=headline)
    bodies, events, row = await _adapter_hunt(tmp_path, mock_anthropic, reg, {"board": "gh"})
    assert row == ("needs_help", None)
    assert len(bodies) == 2
    assert _result_content(bodies[1]) == "gh: Café engineer ✓"  # the view, from the stored args
    assert _results(events) == [POSTING]  # the log keeps the bytes, not the view


async def _tool_hunt(
    tmp_path: Path, scripted_brain: BrainFactory, tool: ToolFn, *, retry_budget: int
) -> tuple[list[TrajectoryEvent], Row]:
    reg = ToolRegistry()
    reg.tool(tool)
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await run_hunt(
            conn,
            territory="seams",
            brain=scripted_brain([ToolCallDecision(tool=tool.__name__, args={}), HuntComplete()]),
            registry=reg,
            retry_budget=retry_budget,
        )
        events = await db.read_events(conn, run_id)
        row = await _row(conn, run_id)
    finally:
        await conn.close()
    return events, row


async def test_a_tool_retry_waits_at_least_its_retry_after(
    tmp_path: Path, scripted_brain: BrainFactory, retry_waits: list[float]
) -> None:
    attempts = 0

    async def board() -> str:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise loop.RetryableToolError("busy", retry_after=3.0)
        return "ok"

    _, row = await _tool_hunt(tmp_path, scripted_brain, board, retry_budget=2)
    assert retry_waits == [3.0, 3.0]  # above both jitter bounds (0.5 s, 1 s), under the cap
    assert row == ("completed", None)


async def test_a_tool_retry_after_is_cut_to_the_cap(
    tmp_path: Path, scripted_brain: BrainFactory, retry_waits: list[float]
) -> None:
    async def board() -> str:
        raise loop.RetryableToolError("busy", retry_after=600.0)

    _, row = await _tool_hunt(tmp_path, scripted_brain, board, retry_budget=2)
    assert retry_waits == [loop.RETRY_AFTER_CAP_S] * 2
    assert row == ("aborted", "tool failure")


@pytest.mark.parametrize(
    ("name", "retryable"),
    [("ToolError", False), ("FatalToolError", False), ("RetryableToolError", True)],
)
async def test_a_tool_error_keeps_its_response_bytes(
    name: str, retryable: bool, tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    body = b"\xe9 an error body that is not UTF-8"
    error_type: type[loop.ToolError] = getattr(loop, name)  # not evaluated at run time

    async def board() -> str:
        raise error_type("the board said no", raw_response=body)

    events, row = await _tool_hunt(tmp_path, scripted_brain, board, retry_budget=1)
    errors = [(e.retryable, e.raw_response) for e in events if isinstance(e, ErrorEvent)]
    assert errors == [(retryable, body)] * (2 if retryable else 1)  # each attempt, base64 both ways
    assert row == ("aborted", "tool failure")
