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
import httpx
from pydantic import BaseModel, ValidationError

from rexhunter import cost, db, events, verdicts
from rexhunter.tools import ToolFn, ToolRegistry

# The default per-run spend guard (USD). The stub/scripted brains carry no usage (cost folds to 0),
# so this never bites them; the paid smoke (`P5` Unit 2c.2) overrides it low. Enforced in `_drive`.
COST_CEILING_USD = 1.0

# ── Error taxonomy + single-attempt execution (Unit C) ───────────────────────


class RetryableToolError(Exception):
    """A tool failure the loop may re-try within budget (a transient blip, a 429, a 5xx).

    Tools opt INTO retry by raising this (or a subclass). The taxonomy's default is fatal: a
    plain exception is assumed not to fix itself on a second attempt.
    """


def classify(exc: BaseException) -> bool:
    """The retryable-vs-fatal taxonomy. ``True`` = retryable (re-try within budget).

    Retryable: a timeout (the tool may simply have been slow this once), any RetryableToolError,
    and — since `P5` Unit 2c — transient httpx failures from the brain's HTTP call: any transport
    error (connect/read/write/pool timeouts, connection resets) and a 429 / 5xx HTTP status.
    Fatal: everything else - a 4xx client error, a BrainParseError / ValidationError (bad bytes
    never become good on retry), a bug in the tool, an unknown-tool lookup. Retrying a fatal error
    only burns budget without changing the outcome. httpx is the transport, not a vendor SDK, so
    the loop knowing its exception shapes does not couple it to a provider (ADR §What point 1).
    """
    if isinstance(exc, TimeoutError | RetryableToolError | httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or 500 <= code < 600
    return False


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
    tool_use_id: str = ""  # the provider's correlation key (P5); "" for the stub brain
    usage: events.UsageEvent | None = None  # per-call token cost (P5 Unit 2c); None = no spend
    thinking: bytes = b""  # the turn's VERBATIM signed thinking block (P5 Unit 3c); "" = none


class HuntComplete(BaseModel):
    """Terminal: the hunt is done. The outcome lives in runs (like P1) - no terminal event.

    `catch` is the postings the hunt judged worth a human verdict; on completion each is written
    to the prey pen (status='awaiting_verdict') BEFORE the run closes (P4). Capturing is an
    internal state write, not an externally visible side effect - Tiny Arms (invariant 4) is
    untouched: the human verdict, not Rex, is what later moves a captured row."""

    catch: list[str] = []
    usage: events.UsageEvent | None = None  # per-call token cost (P5 Unit 2c); None = no spend


class NeedsHelp(BaseModel):
    """Terminal: Rex is stuck and wants a human. P2.2 ends the run cleanly with
    outcome="needs_help"; the durable-pause machinery (awaiting_verdict, park-and-persist)
    is P4. needs_help collapses to aborted trivially later; aborted can't be split back."""

    usage: events.UsageEvent | None = None  # per-call token cost (P5 Unit 2c); None = no spend


type Decision = ToolCallDecision | HuntComplete | NeedsHelp
type Context = list[events.TrajectoryEvent]

# The live-reasoning relay sink (`P5` Unit 3b): the brain calls it with each streamed thinking
# delta; the loop's closure appends a ThinkingDelta write-ahead (inv 1) then the hub broadcasts it.
# Threaded to `brain()` at CALL time (the loop owns conn/run_id/publish), so the scheduler and
# `select_brain_for` are untouched. A non-streaming brain (stub/scripted) simply never calls it.
type ThinkingSink = Callable[[str], Awaitable[None]]
type Brain = Callable[[Context, ThinkingSink], Awaitable[Decision]]

# The run-finished pulse hook (frontend Step 2a): an id-less NOTIFICATION of the already-committed
# runs.outcome, fanned to live viewers post-commit (inv 1) — hub.notify's shape, never publish (no
# id: the trajectory resume cursor must not move). NOT a trajectory event — terminal decisions emit
# none (the settled ADR reconciliation); this relays a runs-table fact, as slice C relays verdicts.
type NotifyFn = Callable[[str], None]


# ── The loop ─────────────────────────────────────────────────────────────────


async def _run_tool(
    conn: aiosqlite.Connection,
    run_id: str,
    registry: ToolRegistry,
    decision: ToolCallDecision,
    *,
    timeout_s: float,
    retry_budget: int,
    publish: db.PublishFn | None = None,
) -> bool:
    """Dispatch one ToolCallDecision. ``True`` = continue the hunt; ``False`` = abort (fatal).

    resolve -> validate -> ToolCallEvent (write-ahead) -> attempt loop. A success appends a
    ToolResultEvent; each retryable failure appends ErrorEvent(retryable=True) and re-tries
    within budget; a fatal failure or an exhausted budget appends a final ErrorEvent and
    aborts. Every append happens here, in the owning task (single writer, invariant 7). The raw
    request rides on every event so a dead attempt is a pytest fixture for free (invariant 6).
    """
    tool_name, args, tool_use_id = decision.tool, decision.args, decision.tool_use_id

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
            publish=publish,
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
            publish=publish,
        )
        return False

    raw_request = validated.model_dump_json().encode()
    kwargs = validated.model_dump()
    await db.append_event(
        conn,
        run_id,
        events.ToolCallEvent(
            tool=tool_name,
            raw_request=raw_request,
            tool_use_id=tool_use_id,
            thinking=decision.thinking,  # the turn's signed block, echoed verbatim on replay (3c)
        ),
        publish=publish,
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
                publish=publish,
            )
            if retryable and attempt < attempts - 1:
                continue  # re-try within budget
            return False  # fatal, or retryable budget exhausted -> abort
        else:
            await db.append_event(
                conn,
                run_id,
                events.ToolResultEvent(
                    tool=tool_name,
                    raw_request=raw_request,
                    raw_response=raw_response,
                    tool_use_id=tool_use_id,
                ),
                publish=publish,
            )
            return True  # success -> continue the hunt
    return False  # defensive: range(attempts >= 1) always returns inside the loop


async def _call_brain(
    conn: aiosqlite.Connection,
    run_id: str,
    brain: Brain,
    context: list[events.TrajectoryEvent],
    *,
    retry_budget: int,
    publish: db.PublishFn | None = None,
) -> Decision | None:
    """One brain call, retrying the brain's HTTP transport failures; ``None`` = abort.

    Owns exactly the errors this unit adds: httpx transport + status failures. A transient blip
    (`classify` True: timeout / 429 / 5xx) is re-tried within budget, each attempt appending its own
    ErrorEvent(tool="<brain>") in the owning task (single writer, invariant 7); a fatal one (a 4xx)
    logs once and aborts. Everything else propagates unchanged: BrainParseError to run_hunt's clause
    (invariant 6, raw bytes preserved) and any genuinely unforeseen exception to run_hunt's `<loop>`
    backstop (DoD #2) - `_call_brain` narrows to httpx so it never swallows a brain bug.
    """

    async def sink(text: str) -> None:
        # The live relay (Unit 3b): each streamed thinking delta committed write-ahead (inv 1) then
        # broadcast by the hub (publish) — Rex's reasoning as the live feed. Awaited inline in the
        # owning task, never a spawned one, so the single-writer invariant (7) holds.
        await db.append_event(conn, run_id, events.ThinkingDelta(text=text), publish=publish)

    attempts = retry_budget + 1
    for attempt in range(attempts):
        try:
            return await brain(context, sink)
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            retryable = classify(exc)
            raw_response = exc.response.content if isinstance(exc, httpx.HTTPStatusError) else None
            await db.append_event(
                conn,
                run_id,
                events.ErrorEvent(
                    tool="<brain>",
                    retryable=retryable,
                    error=repr(exc),
                    raw_request=b"",
                    raw_response=raw_response,
                    detail=traceback.format_exc(),
                ),
                publish=publish,
            )
            if retryable and attempt < attempts - 1:
                continue  # transient blip -> re-try within budget
            return None  # fatal 4xx, or retryable budget exhausted -> abort
    return None  # pragma: no cover - range(attempts >= 1) always returns inside the loop


async def _drive(
    conn: aiosqlite.Connection,
    run_id: str,
    *,
    territory: str,
    brain: Brain,
    registry: ToolRegistry,
    timeout_s: float,
    retry_budget: int,
    max_iterations: int,
    cost_ceiling_usd: float,
    publish: db.PublishFn | None = None,
) -> tuple[str, str | None]:
    """plan -> act -> observe until a terminal decision, a breaker, or a brain failure.

    Returns (outcome, abort_reason). Two circuit breakers guard the loop (ADR §Budget guards):
    max_iterations bounds the turn count, and the cost ceiling - folded from the run's UsageEvents
    (invariant 5, no stored counter) - aborts BEFORE the next paid call once spend crosses it. Each
    iteration projects the log into the brain's context, records the call's UsageEvent (if the brain
    reported one), then acts. NeedsHelp is terminal and appends no event; HuntComplete captures its
    catch into the prey pen (each a PreyCapturedEvent + row, P4) THEN closes via runs.outcome. A
    tool failure or an exhausted brain retry budget aborts.
    """
    for _ in range(max_iterations):
        context = await db.read_events(conn, run_id)  # the window the brain projects into messages
        if cost.fold_cost(context) >= cost_ceiling_usd:
            return "aborted", "cost ceiling"
        decision = await _call_brain(
            conn, run_id, brain, context, retry_budget=retry_budget, publish=publish
        )
        if decision is None:
            return "aborted", "brain call failed"
        if decision.usage is not None:
            # per-call spend, folded next iter
            await db.append_event(conn, run_id, decision.usage, publish=publish)
        match decision:
            case ToolCallDecision():
                ok = await _run_tool(
                    conn,
                    run_id,
                    registry,
                    decision,
                    timeout_s=timeout_s,
                    retry_budget=retry_budget,
                    publish=publish,
                )
                if not ok:
                    return "aborted", "tool failure"
            case HuntComplete():
                for posting in decision.catch:
                    await verdicts.capture_prey(
                        conn, run_id, territory=territory, posting=posting, publish=publish
                    )
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
    cost_ceiling_usd: float = COST_CEILING_USD,
    publish: db.PublishFn | None = None,
    notify: NotifyFn | None = None,
) -> str:
    """Drive one hunt to a typed outcome; return its run_id (the durable handle).

    The DoD #2 backstop lives here: NO exception escapes run_hunt. The known taxonomy
    (raise / timeout / unknown tool / bad args) is handled inside `_run_tool`; the try/except
    catches only the genuinely unforeseen, records it as a loop-level ErrorEvent, and aborts.
    Cancellation is BaseException (not Exception), so it still propagates - the daemon's
    shutdown path closes the run (P2.3), exactly as the prototype sniff loop does today.
    `notify` (Step 2a) fires one id-less run-finished pulse after the closing commit — LIVE
    closures only (the cancel path re-raises before the pulse; boot's sweep never sees it).
    """
    # Record the caps this run runs under (4a): the same two values _drive enforces, snapshotted
    # write-once onto the runs row — the frontend's per-run HP/stamina denominators.
    run_id = await db.start_run(
        conn,
        territory=territory,
        cost_ceiling_usd=cost_ceiling_usd,
        max_iterations=max_iterations,
    )
    try:
        outcome, abort_reason = await _drive(
            conn,
            run_id,
            territory=territory,
            brain=brain,
            registry=registry,
            timeout_s=tool_timeout_s,
            retry_budget=retry_budget,
            max_iterations=max_iterations,
            cost_ceiling_usd=cost_ceiling_usd,
            publish=publish,
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
    except events.BrainParseError as exc:
        # The provider->Decision boundary (P5) rejected a payload. The generic backstop below
        # would drop the bytes (raw_request=b""); this specific clause, placed first, preserves
        # the raw payload (invariant 6) so the malformed response is a ghost-replay fixture, then
        # ends the run in a typed outcome - never an unhandled escape (DoD #5).
        await db.append_event(
            conn,
            run_id,
            events.ErrorEvent(
                tool="<brain>",
                retryable=False,
                error="malformed provider response",
                raw_request=b"",
                raw_response=exc.raw,
                detail=exc.detail,
            ),
            publish=publish,
        )
        outcome, abort_reason = "aborted", "malformed provider response"
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
            publish=publish,
        )
        outcome, abort_reason = "aborted", "unhandled exception in loop"
    await db.finish_run(conn, run_id, outcome=outcome, abort_reason=abort_reason)
    if notify is not None:
        # The live-completion pulse (Step 2a): strictly post-commit (inv 1 — runs.outcome is truth
        # before any viewer hears of it), id-less (never publish — the resume cursor must not
        # move), and it appends nothing (inv 7). The CancelledError clause above re-raises before
        # reaching this line, so a shutdown-abort never pulses (its viewers are tearing down too).
        notify(
            json.dumps(
                {"type": "run_finished", "run_id": run_id, "outcome": outcome},
                separators=(",", ":"),  # compact, byte-consistent with the Pydantic frames
            )
        )
    return run_id
