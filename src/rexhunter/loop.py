"""The hand-rolled agent loop & tool harness (ADR pillar 2).

The loop is the observability surface: because every state transition is code we own, emitting
its trajectory event is one append at the site of the transition (invariant 1, write-ahead).
Built on one file across two `P2.2` units - the error taxonomy + single-attempt execution
wrapper (Unit C), then the decision union and the `run_hunt` loop that drives them (Unit D).

`run_hunt` is where ADR DoD #2 lives: a tool that raises, one that hangs past its timeout, and
a brain naming an unknown tool each become typed events and a clean run outcome - never an
unhandled exception escaping the loop.
"""

import asyncio
import json
import traceback
from collections.abc import Awaitable, Callable
from typing import Any, assert_never

import aiosqlite
from pydantic import BaseModel, ValidationError

from rexhunter import db, events
from rexhunter.tools import ToolFn, ToolRegistry

# ── Error taxonomy + single-attempt execution (Unit C) ───────────────────────


class RetryableToolError(Exception):
    """A tool failure the loop may re-try within budget (a transient blip, a 429, a 5xx).

    Tools opt INTO retry by raising this (or a subclass). The taxonomy's default is fatal: a
    plain exception is assumed not to fix itself on a second attempt.
    """


def classify(exc: BaseException) -> bool:
    """The retryable-vs-fatal taxonomy. ``True`` = retryable (re-try within budget).

    Retryable: a timeout (the tool may simply have been slow this once) and any
    RetryableToolError. Fatal: everything else - a ValidationError (bad args never become good
    on retry), a bug in the tool, an unknown-tool lookup. Retrying a fatal error only burns
    budget without changing the outcome.
    """
    return isinstance(exc, TimeoutError | RetryableToolError)


def _to_bytes(result: object) -> bytes:
    """Serialise a tool's return value to the raw response bytes the log stores (invariant 6).

    A Pydantic model dumps to its JSON; raw bytes pass straight through (a live adapter's HTTP
    body is already bytes); anything else is JSON-encoded. The bytes are what a ghost replay
    re-feeds, so they must be the exact thing the tool produced.
    """
    if isinstance(result, BaseModel):
        return result.model_dump_json().encode()
    if isinstance(result, bytes):
        return result
    return json.dumps(result).encode()


async def execute_once(fn: ToolFn, kwargs: dict[str, Any], *, timeout_s: float) -> bytes:
    """Run ONE tool attempt under a deadline; return the raw response bytes (invariant 6).

    Single attempt by design - the retry *iteration* lives in `run_hunt` so every attempt
    appends its own event (single writer, invariant 7). A tool that overruns ``timeout_s`` is
    cancelled at the deadline and surfaces as ``TimeoutError`` (asyncio.timeout raises it on
    expiry), which `classify` treats as retryable. The wrapper does not swallow: a tool's own
    exception propagates for the loop to classify and log.
    """
    async with asyncio.timeout(timeout_s):
        result = await fn(**kwargs)
    return _to_bytes(result)


# ── Decisions: the brain's output, the loop's input (Unit D) ─────────────────


class ToolCallDecision(BaseModel):
    """Run a tool with these args. In P5 the brain validates the provider's tool_use block
    into this; in P2.2 the stub brain constructs it directly (no model, no spend)."""

    tool: str
    args: dict[str, Any]


class HuntComplete(BaseModel):
    """Terminal: the hunt is done. Appends no event - the outcome lives in runs (like P1)."""


class NeedsHelp(BaseModel):
    """Terminal: Rex is stuck and wants a human. P2.2 ends the run cleanly with
    outcome="needs_help"; the durable-pause machinery (awaiting_verdict, park-and-persist)
    is P4. needs_help collapses to aborted trivially later; aborted can't be split back."""


type Decision = ToolCallDecision | HuntComplete | NeedsHelp
type Context = list[events.TrajectoryEvent]
type Brain = Callable[[Context], Awaitable[Decision]]


# ── The loop ─────────────────────────────────────────────────────────────────


async def _run_tool(
    conn: aiosqlite.Connection,
    run_id: str,
    registry: ToolRegistry,
    decision: ToolCallDecision,
    *,
    timeout_s: float,
    retry_budget: int,
) -> bool:
    """Dispatch one ToolCallDecision. ``True`` = continue the hunt; ``False`` = abort (fatal).

    resolve -> validate -> ToolCallEvent (write-ahead) -> attempt loop. A success appends a
    ToolResultEvent; each retryable failure appends ErrorEvent(retryable=True) and re-tries
    within budget; a fatal failure or an exhausted budget appends a final ErrorEvent and
    aborts. Every append happens here, in the owning task (single writer, invariant 7). The raw
    request rides on every event so a dead attempt is a pytest fixture for free (invariant 6).
    """
    tool_name, args = decision.tool, decision.args

    try:
        tool = registry.get(tool_name)
    except KeyError as exc:
        await db.append_event(
            conn,
            run_id,
            events.ErrorEvent(
                tool=tool_name,
                retryable=False,
                error=f"unknown tool: {tool_name!r}",
                raw_request=json.dumps(args).encode(),
                detail=repr(exc),
            ),
        )
        return False

    try:
        validated = tool.validate(args)
    except ValidationError as exc:
        await db.append_event(
            conn,
            run_id,
            events.ErrorEvent(
                tool=tool_name,
                retryable=False,
                error="invalid tool args",
                raw_request=json.dumps(args).encode(),
                detail=str(exc),
            ),
        )
        return False

    raw_request = validated.model_dump_json().encode()
    kwargs = validated.model_dump()
    await db.append_event(
        conn, run_id, events.ToolCallEvent(tool=tool_name, raw_request=raw_request)
    )

    attempts = retry_budget + 1  # retry_budget is the number of RETRIES after the first try
    for attempt in range(attempts):
        try:
            raw_response = await execute_once(tool.fn, kwargs, timeout_s=timeout_s)
        except Exception as exc:
            retryable = classify(exc)
            await db.append_event(
                conn,
                run_id,
                events.ErrorEvent(
                    tool=tool_name,
                    retryable=retryable,
                    error=repr(exc),
                    raw_request=raw_request,
                    detail=traceback.format_exc(),
                ),
            )
            if retryable and attempt < attempts - 1:
                continue  # re-try within budget
            return False  # fatal, or retryable budget exhausted -> abort
        else:
            await db.append_event(
                conn,
                run_id,
                events.ToolResultEvent(
                    tool=tool_name, raw_request=raw_request, raw_response=raw_response
                ),
            )
            return True  # success -> continue the hunt
    return False  # defensive: range(attempts >= 1) always returns inside the loop


async def _drive(
    conn: aiosqlite.Connection,
    run_id: str,
    *,
    brain: Brain,
    registry: ToolRegistry,
    timeout_s: float,
    retry_budget: int,
    max_iterations: int,
) -> tuple[str, str | None]:
    """plan -> act -> observe until a terminal decision or the iteration breaker.

    Returns (outcome, abort_reason). HuntComplete / NeedsHelp are terminal and append no event
    (terminal decisions are recorded by runs.outcome, not the log - matching P1). A tool
    failure aborts; exceeding max_iterations trips the breaker so a stub brain that never
    finishes cannot loop forever.
    """
    for _ in range(max_iterations):
        context = await db.read_events(conn, run_id)  # minimal window; real assembly is P5
        decision = await brain(context)
        match decision:
            case ToolCallDecision():
                ok = await _run_tool(
                    conn, run_id, registry, decision, timeout_s=timeout_s, retry_budget=retry_budget
                )
                if not ok:
                    return "aborted", "tool failure"
            case HuntComplete():
                return "completed", None
            case NeedsHelp():
                return "needs_help", None
            case _:  # pragma: no cover - the union is exhaustive; this proves it to pyright
                assert_never(decision)
    return "aborted", "max iterations"


async def run_hunt(
    conn: aiosqlite.Connection,
    *,
    territory: str,
    brain: Brain,
    registry: ToolRegistry,
    tool_timeout_s: float = 30.0,
    retry_budget: int = 2,
    max_iterations: int = 50,
) -> str:
    """Drive one hunt to a typed outcome; return its run_id (the durable handle).

    The DoD #2 backstop lives here: NO exception escapes run_hunt. The known taxonomy
    (raise / timeout / unknown tool / bad args) is handled inside `_run_tool`; the try/except
    catches only the genuinely unforeseen, records it as a loop-level ErrorEvent, and aborts.
    Cancellation is BaseException (not Exception), so it still propagates - the daemon's
    shutdown path closes the run (P2.3), exactly as the prototype sniff loop does today.
    """
    run_id = await db.start_run(conn, territory=territory)
    try:
        outcome, abort_reason = await _drive(
            conn,
            run_id,
            brain=brain,
            registry=registry,
            timeout_s=tool_timeout_s,
            retry_budget=retry_budget,
            max_iterations=max_iterations,
        )
    except asyncio.CancelledError as cancel:
        # Graceful shutdown (P2.3) closes the run HERE, not via the boot crash-sweep - that
        # conflates a clean stop with a kill -9 (DoD #1). The mark-aborted write must complete
        # even if a SECOND cancel lands mid-flight (aggressive shutdown, a double cancel from the
        # task group): we shield it and drive it to done before re-raising, so the run can never
        # be left outcome IS NULL. CancelledError still propagates - the group tears down on
        # purpose. (Tested adversarially in test_cancelled_cleanup_survives_a_second_cancel.)
        cleanup = asyncio.ensure_future(
            db.finish_run(conn, run_id, outcome="aborted", abort_reason="daemon shutdown")
        )
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                pass  # a re-cancel hit our frame; the shielded write runs on - wait it out
        raise cancel
    except Exception as exc:
        await db.append_event(
            conn,
            run_id,
            events.ErrorEvent(
                tool="<loop>",
                retryable=False,
                error=repr(exc),
                raw_request=b"",
                detail=traceback.format_exc(),
            ),
        )
        outcome, abort_reason = "aborted", "unhandled exception in loop"
    await db.finish_run(conn, run_id, outcome=outcome, abort_reason=abort_reason)
    return run_id
