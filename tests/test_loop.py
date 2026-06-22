"""P2.2 · Unit D — run_hunt's non-failure behaviours.

The gate (test_stage2_gate.py) covers failure injection; here: the happy path, the NeedsHelp
terminal (no event, outcome="needs_help" - per the slice's scope decision (d)), the retry
iteration (each failed attempt is its OWN event, read straight from the log), and the
max-iteration breaker that stops a brain which never finishes.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import aiosqlite
import pytest

from rexhunter import db
from rexhunter.events import ErrorEvent, ToolCallEvent, ToolResultEvent, TrajectoryEvent
from rexhunter.loop import (
    Brain,
    Decision,
    HuntComplete,
    NeedsHelp,
    RetryableToolError,
    ToolCallDecision,
    run_hunt,
)
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

BrainFactory = Callable[[Sequence[Decision]], Brain]


async def outcome_of(conn: aiosqlite.Connection, run_id: str) -> str:
    async with conn.execute("SELECT outcome FROM runs WHERE id = ?", (run_id,)) as cur:
        row = await cur.fetchone()
    assert row is not None and row[0] is not None
    return str(row[0])


async def test_happy_path_completes(tmp_path: Path, scripted_brain: BrainFactory) -> None:
    reg = ToolRegistry()

    @reg.tool
    async def fetch(board: str) -> str:
        return f"posting@{board}"

    conn = await db.connect(tmp_path / "rex.db")
    try:
        brain = scripted_brain(
            [ToolCallDecision(tool=fetch.__name__, args={"board": "greenhouse"}), HuntComplete()]
        )
        run_id = await run_hunt(conn, territory="gate", brain=brain, registry=reg)

        events = await db.read_events(conn, run_id)
        assert [type(e).__name__ for e in events] == ["ToolCallEvent", "ToolResultEvent"]
        [result] = [e for e in events if isinstance(e, ToolResultEvent)]
        assert result.raw_request == b'{"board":"greenhouse"}'
        assert result.raw_response == b'"posting@greenhouse"'  # the return value, serialised
        assert await outcome_of(conn, run_id) == "completed"
    finally:
        await conn.close()


async def test_needs_help_ends_with_no_event(tmp_path: Path, scripted_brain: BrainFactory) -> None:
    reg = ToolRegistry()
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await run_hunt(
            conn, territory="gate", brain=scripted_brain([NeedsHelp()]), registry=reg
        )

        # terminal decisions append NO event - the outcome lives in runs (decision (d)).
        assert await db.read_events(conn, run_id) == []
        assert await outcome_of(conn, run_id) == "needs_help"
    finally:
        await conn.close()


async def test_retry_iteration_records_each_attempt(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    reg = ToolRegistry()
    attempts = 0

    @reg.tool
    async def flaky(n: int) -> str:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise RetryableToolError(f"transient #{attempts}")
        return f"ok after {attempts}"

    conn = await db.connect(tmp_path / "rex.db")
    try:
        brain = scripted_brain(
            [ToolCallDecision(tool=flaky.__name__, args={"n": 1}), HuntComplete()]
        )
        run_id = await run_hunt(conn, territory="gate", brain=brain, registry=reg, retry_budget=3)

        events = await db.read_events(conn, run_id)
        calls = [e for e in events if isinstance(e, ToolCallEvent)]
        errors = [e for e in events if isinstance(e, ErrorEvent)]
        results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(calls) == 1  # one dispatch
        assert len(errors) == 2  # two retryable failures, each its own event
        assert all(e.retryable is True for e in errors)
        assert len(results) == 1 and results[0].raw_response == b'"ok after 3"'
        assert await outcome_of(conn, run_id) == "completed"
    finally:
        await conn.close()


async def test_invalid_tool_args_abort_before_dispatch(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    # invariant 3 at the tool boundary: args that fail the tool's model never execute. A fatal
    # ErrorEvent is logged and the run aborts - with NO ToolCallEvent (validation precedes it).
    reg = ToolRegistry()

    @reg.tool
    async def needs_int(n: int) -> int:
        return n * 2

    conn = await db.connect(tmp_path / "rex.db")
    try:
        brain = scripted_brain(
            [ToolCallDecision(tool=needs_int.__name__, args={"n": "not-an-int"})]
        )
        run_id = await run_hunt(conn, territory="gate", brain=brain, registry=reg)

        events = await db.read_events(conn, run_id)
        errors = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(errors) == 1 and errors[0].retryable is False
        assert "invalid tool args" in errors[0].error
        assert not [e for e in events if isinstance(e, ToolCallEvent)]  # validation precedes it
        assert await outcome_of(conn, run_id) == "aborted"
    finally:
        await conn.close()


async def test_max_iterations_breaker_aborts(tmp_path: Path) -> None:
    reg = ToolRegistry()

    @reg.tool
    async def noop() -> str:
        return "tick"

    async def always_call(_context: list[TrajectoryEvent]) -> Decision:
        return ToolCallDecision(tool=noop.__name__, args={})

    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await run_hunt(
            conn, territory="gate", brain=always_call, registry=reg, max_iterations=3
        )

        results = [e for e in await db.read_events(conn, run_id) if isinstance(e, ToolResultEvent)]
        assert len(results) == 3  # exactly max_iterations successful calls, then the breaker
        assert await outcome_of(conn, run_id) == "aborted"
    finally:
        await conn.close()
