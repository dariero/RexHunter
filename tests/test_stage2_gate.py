"""Stage 2 gate — ADR definition-of-done #2.

Three failure-injection scenarios, first-class: a tool that RAISES, a tool that HANGS past
its timeout, and a (stub) brain naming an UNKNOWN tool. Each must produce typed events and a
clean run outcome - never an unhandled exception escaping run_hunt.

The hang test is the slice's sharpest hazard: tiny timeout + budget and an asyncio.sleep
(cancellable) tool make the deadline fire in milliseconds, so the test cannot wedge the suite.
"""

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path

import aiosqlite
import pytest

from rexhunter import db
from rexhunter.events import ErrorEvent, ToolCallEvent, ToolResultEvent, TrajectoryEvent
from rexhunter.loop import Brain, Decision, ToolCallDecision, run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

BrainFactory = Callable[[Sequence[Decision]], Brain]


async def outcome_of(conn: aiosqlite.Connection, run_id: str) -> str:
    async with conn.execute("SELECT outcome FROM runs WHERE id = ?", (run_id,)) as cur:
        row = await cur.fetchone()
    assert row is not None and row[0] is not None
    return str(row[0])


async def test_a_tool_that_raises_aborts_cleanly(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    reg = ToolRegistry()

    @reg.tool
    async def boom(target: str) -> None:
        raise ValueError(f"cannot reach {target}")

    conn = await db.connect(tmp_path / "rex.db")
    try:
        brain = scripted_brain([ToolCallDecision(tool=boom.__name__, args={"target": "acme"})])
        run_id = await run_hunt(conn, territory="gate", brain=brain, registry=reg)

        events = await db.read_events(conn, run_id)
        errors = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(errors) == 1
        assert errors[0].retryable is False  # a raise is fatal, not retried
        assert "ValueError" in errors[0].error
        assert errors[0].raw_request == b'{"target":"acme"}'  # invariant 6: the request bytes
        assert not [e for e in events if isinstance(e, ToolResultEvent)]
        assert await outcome_of(conn, run_id) == "aborted"
    finally:
        await conn.close()


async def test_a_tool_that_hangs_times_out_and_aborts(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    reg = ToolRegistry()

    @reg.tool
    async def hang() -> None:
        await asyncio.sleep(10)  # cancellable; the deadline fires, the suite does NOT hang

    conn = await db.connect(tmp_path / "rex.db")
    try:
        brain = scripted_brain([ToolCallDecision(tool=hang.__name__, args={})])
        run_id = await run_hunt(
            conn,
            territory="gate",
            brain=brain,
            registry=reg,
            tool_timeout_s=0.05,
            retry_budget=1,  # -> 2 attempts, each a timeout
        )

        events = await db.read_events(conn, run_id)
        errors = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(errors) == 2
        assert all("TimeoutError" in e.error for e in errors)
        assert all(e.retryable is True for e in errors)  # a timeout is classified retryable
        assert errors[0].raw_request == b"{}"  # invariant 6: the request bytes, even on timeout
        assert not [e for e in events if isinstance(e, ToolResultEvent)]
        assert await outcome_of(conn, run_id) == "aborted"
    finally:
        await conn.close()


async def test_an_unknown_tool_name_aborts_cleanly(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    reg = ToolRegistry()  # empty: "ghost" is not registered

    conn = await db.connect(tmp_path / "rex.db")
    try:
        brain = scripted_brain([ToolCallDecision(tool="ghost", args={"q": "AI"})])
        run_id = await run_hunt(conn, territory="gate", brain=brain, registry=reg)

        events = await db.read_events(conn, run_id)
        errors = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(errors) == 1
        assert errors[0].retryable is False
        assert "unknown tool" in errors[0].error.lower()
        assert errors[0].raw_request == b'{"q": "AI"}'  # the attempted args (invariant 6)
        assert not [e for e in events if isinstance(e, ToolCallEvent)]  # never dispatched
        assert await outcome_of(conn, run_id) == "aborted"
    finally:
        await conn.close()


async def test_an_unforeseen_exception_is_caught_by_the_backstop(tmp_path: Path) -> None:
    # The three scenarios above are handled inside the tool harness; this proves the GENERAL
    # clause - "never an unhandled exception escaping the loop" - via run_hunt's backstop, by
    # making the brain itself explode (a path the per-tool handling never reaches).
    reg = ToolRegistry()

    async def exploding_brain(_context: list[TrajectoryEvent]) -> Decision:
        raise RuntimeError("brain bug")

    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await run_hunt(conn, territory="gate", brain=exploding_brain, registry=reg)

        errors = [e for e in await db.read_events(conn, run_id) if isinstance(e, ErrorEvent)]
        assert len(errors) == 1
        assert errors[0].tool == "<loop>"  # a loop-level failure, not a tool's
        assert "RuntimeError" in errors[0].error
        assert await outcome_of(conn, run_id) == "aborted"  # clean outcome, run_hunt returned
    finally:
        await conn.close()
