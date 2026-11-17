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
import logging
import random
import traceback
from collections.abc import Awaitable, Callable
from typing import Any, assert_never

import aiosqlite
from pydantic import BaseModel, ValidationError
from pydantic.json_schema import SkipJsonSchema

from rexhunter import cost, db, events, verdicts
from rexhunter.tools import ToolFn, ToolRegistry

logger = logging.getLogger("rexhunter")

# The default per-run spend guard (USD). The stub/scripted brains carry no usage (cost folds to 0),
# so this never bites them; the paid smoke (`P5` Unit 2c.2) overrides it low. Enforced in
# `_call_brain`, before every brain attempt (W3.2-C1).
COST_CEILING_USD = 1.0

# ── Error taxonomy + single-attempt execution (Unit C) ───────────────────────


class RetryableToolError(Exception):
    """A tool failure the loop may re-try within budget (a transient blip, a 429, a 5xx).

    Tools opt INTO retry by raising this (or a subclass). The taxonomy's default is fatal: a
    plain exception is assumed not to fix itself on a second attempt.
    """


class BrainCallError(Exception):
    """A brain call failed at the provider/transport layer: the ONE provider-neutral shape (W2.0
    D1a) the adapter translates its SDK / transport errors into, so the loop never imports a vendor
    SDK. `retryable` is the adapter's verdict (a transport blip, a 429 or a 5xx is transient);
    `status` is the HTTP status, or for an SSE error frame on a 200 stream the status its type
    stands for (`brain.STREAM_ERROR_STATUS`, None for a type with no row; W3.1), and None for a
    transport failure; `raw_response` is the provider's error body (invariant 6): the frame's own
    `data` bytes for an SSE error frame; for a stream whose body ended before `message_stop`, the
    bytes assembled from it (W3.1); None when no response arrived (a reset mid-body included).
    `usage` (W3.2-U2a) is the token count a 200 stream reported before the adapter refused it (an
    error frame, a body cut before `message_stop`, and since W3.4 a reset mid-stream, W3.2-U2b),
    when it reads cleanly: `_call_brain` appends it as a UsageEvent before the ErrorEvent, once per
    attempt. None for a status error (its body carries no count) and for a transport failure before
    the response.
    `retry_after` (W3.2-B2) is a status response's `retry-after` header in seconds, None when it
    was absent or not a finite, non-negative number of seconds; the retry waits at least that
    long, cut to RETRY_AFTER_CAP_S, 60 s (`backoff_delay`, W3.4 Cap60). `exchange` (W3.3) is the
    attempt's raw request and response (the bytes as received, a stream's partial body included),
    which `_call_brain` appends before the usage and the ErrorEvent; `raw_response` stays what it
    was. None from a brain that reports no bytes."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status: int | None = None,
        raw_response: bytes | None = None,
        usage: events.UsageEvent | None = None,
        retry_after: float | None = None,
        exchange: events.BrainExchangeEvent | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.raw_response = raw_response
        self.usage = usage
        self.retry_after = retry_after
        self.exchange = exchange


def classify(exc: BaseException) -> bool:
    """The retryable-vs-fatal taxonomy. ``True`` = retryable (re-try within budget).

    Retryable: a timeout (the tool may simply have been slow this once), any RetryableToolError,
    and a BrainCallError the adapter marked retryable. Since W2.0 (D1a) the brain's transport
    failures reach the loop ONLY as that provider-neutral BrainCallError, whose `retryable` flag
    IS the verdict: the adapter (`brain.brain_call_error`) applies the taxonomy `P5` Unit 2c
    taught this function (a transport error, a 429 or a 5xx is transient; any other status is
    fatal), so the loop imports no vendor SDK and no transport library.
    Fatal: everything else - a 4xx client error, a BrainParseError / ValidationError (bad bytes
    never become good on retry), a bug in the tool, an unknown-tool lookup. Retrying a fatal error
    only burns budget without changing the outcome.
    """
    if isinstance(exc, BrainCallError):
        return exc.retryable
    return isinstance(exc, TimeoutError | RetryableToolError)


# ── Retry backoff (W3.2-B2, fragility #2) ────────────────────────────────────
# A retried brain call or tool waits before it runs again. The SDK's own retries are off (D1b,
# `max_retries=0`), so the loop owns the wait, shaped like the one it replaces: a 0.5 s base that
# doubles per retry up to an 8 s cap (`INITIAL_RETRY_DELAY`, `MAX_RETRY_DELAY`, anthropic 2.6.0
# `_constants.py`), drawn with full jitter. A brain retry after a status response that carried
# `retry-after` waits at least that many seconds, up to RETRY_AFTER_CAP_S, 60 s (W3.4 Cap60: the
# rate-limits page says to wait that many seconds, and "Earlier retries might fail"; W3.2 cut it at
# the jitter's 8 s). The jitter keeps its 8 s cap, and a tool retry carries no `retry-after`.
BACKOFF_BASE_S = 0.5
BACKOFF_CAP_S = 8.0
RETRY_AFTER_CAP_S = 60.0

# The two seams the tests replace (W3.2-Y1), looked up at call time: the wait itself and the
# jitter's random source. tests/conftest.py records every wait instead of sleeping.
retry_wait: Callable[[float], Awaitable[None]] = asyncio.sleep
jitter_source = random.Random()


def backoff_delay(retry: int, *, retry_after: float | None = None) -> float:
    """Seconds to wait before retry number `retry` (1 = the first retry): full jitter, a uniform
    draw in [0, min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2 ** (retry - 1))], and at least `retry_after`
    (cut to RETRY_AFTER_CAP_S, 60 s; W3.4 Cap60) when the failed response carried one."""
    bound = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2.0 ** min(retry - 1, 16))  # no float overflow
    delay = jitter_source.uniform(0.0, bound)
    if retry_after is not None:
        delay = max(delay, min(retry_after, RETRY_AFTER_CAP_S))
    return delay


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
    assistant_content: bytes = b""  # the WHOLE turn: the response's `content` array as JSON bytes
    # (W2.3, D7), set by `brain.parse_decision` on both paths; b"" for the stub/scripted brains
    # the attempt's raw I/O (W3.3); None = none. Kept out of the JSON schema: HuntComplete and
    # NeedsHelp derive the terminal tools' `input_schema` the model sees (`brain.build_tools`).
    exchange: SkipJsonSchema[events.BrainExchangeEvent | None] = None


class HuntComplete(BaseModel):
    """Terminal: the hunt is done. The outcome lives in runs (like P1) - no terminal event.

    `catch` is the postings the hunt judged worth a human verdict; on completion each is written
    to the prey pen (status='awaiting_verdict') BEFORE the run closes (P4). Capturing is an
    internal state write, not an externally visible side effect - Tiny Arms (invariant 4) is
    untouched: the human verdict, not Rex, is what later moves a captured row."""

    catch: list[str] = []
    # per-call token cost (P5 Unit 2c); None = no spend. Kept out of the JSON schema like
    # `exchange` (W3.4, W3.3-F1): the model is never shown a `usage` input.
    usage: SkipJsonSchema[events.UsageEvent | None] = None
    # the attempt's raw I/O (W3.3); None = none. Kept out of the JSON schema: HuntComplete and
    # NeedsHelp derive the terminal tools' `input_schema` the model sees (`brain.build_tools`).
    exchange: SkipJsonSchema[events.BrainExchangeEvent | None] = None


class NeedsHelp(BaseModel):
    """Terminal: Rex is stuck and wants a human. P2.2 ends the run cleanly with
    outcome="needs_help"; the durable-pause machinery (awaiting_verdict, park-and-persist)
    is P4. needs_help collapses to aborted trivially later; aborted can't be split back."""

    # per-call token cost (P5 Unit 2c); None = no spend. Kept out of the JSON schema like
    # `exchange` (W3.4, W3.3-F1): the model is never shown a `usage` input.
    usage: SkipJsonSchema[events.UsageEvent | None] = None
    # the attempt's raw I/O (W3.3); None = none. Kept out of the JSON schema: HuntComplete and
    # NeedsHelp derive the terminal tools' `input_schema` the model sees (`brain.build_tools`).
    exchange: SkipJsonSchema[events.BrainExchangeEvent | None] = None


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
    ToolResultEvent; each retryable failure appends ErrorEvent(retryable=True), waits a backoff
    (`backoff_delay`, W3.2-B2) and re-tries within budget; a fatal failure or an exhausted budget
    appends a final ErrorEvent and aborts. Every append happens here, in the owning task (single
    writer, invariant 7). The raw request rides on every event so a dead attempt is a pytest
    fixture for free (invariant 6).
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
            assistant_content=decision.assistant_content,  # the whole turn, echoed on replay
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
                # back off once the ErrorEvent is committed (inv 1), then re-try within budget
                await retry_wait(backoff_delay(attempt + 1))
                continue
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
    *,
    retry_budget: int,
    cost_ceiling_usd: float,
    daemon_spend_ceiling_usd: float | None,
    publish: db.PublishFn | None = None,
) -> Decision | str:
    """One brain call, retrying the brain's HTTP transport failures; returns the Decision, or the
    abort reason ("cost ceiling", "daemon spend ceiling", "brain call failed").

    Both spend ceilings are checked before EVERY attempt, a retry included (W3.2-C1, fragility #4),
    each a fold over the log as it stands at that moment (invariant 5, never a counter): this run's
    UsageEvents against `cost_ceiling_usd` first, then every run's (`db.read_usage`, the fold
    `scheduler.daemon_spend_usd` makes) against `daemon_spend_ceiling_usd` (None = ungated). So a
    run, or the daemon, goes past its ceiling by at most the one call each hunt has in flight, of
    the calls the log counts. The context the brain projects is read at the same point, and only
    the event types the call uses (`db.read_context`, W3.3-W: the conversation's turns and the
    usage the per-run fold reads; the whole conversation is still sent): what a failed attempt left
    in the log (its ThinkingDeltas, UsageEvent and `<brain>` ErrorEvent) is not conversation, so a
    retry sends the same request.

    Owns exactly one error shape: the provider-neutral BrainCallError (W2.0 D1a), which the adapter
    raises for every transport + status failure, so the loop knows no vendor or transport types. A
    transient blip (`classify` True: timeout / 429 / 5xx) is re-tried within budget after a backoff
    wait (`backoff_delay`: at least its `retry-after`, up to 60 s; W3.2-B2, W3.4), each attempt
    appending its own ErrorEvent(tool="<brain>") in the owning task (single writer, invariant 7)
    before the wait; a fatal one (a 4xx) logs once and aborts. Everything else propagates
    unchanged: BrainParseError to run_hunt's clause (invariant 6, raw bytes preserved) and any
    genuinely unforeseen exception to run_hunt's `<loop>` backstop (DoD #2) - `_call_brain`
    narrows to BrainCallError so it never swallows a brain bug. An attempt the adapter refused
    after a 200 (W3.2-U2a) can still be billed: when its error carries a usage, that usage is
    appended as a UsageEvent before the ErrorEvent, so the spend folds count every attempt whose
    error carries one (since W3.4 a transport failure mid-stream carries one too, W3.2-U2b). Before
    either, an attempt whose brain reports its bytes (W3.3, the adapter's `exchange`, on its
    Decision or its BrainCallError) appends its BrainExchangeEvent, committed and never published
    (fork X2): the request it sent and the response it got (invariant 6).
    """

    async def sink(text: str) -> None:
        # The live relay (Unit 3b): each streamed thinking delta committed write-ahead (inv 1) then
        # broadcast by the hub (publish) — Rex's reasoning as the live feed. Awaited inline in the
        # owning task, never a spawned one, so the single-writer invariant (7) holds.
        await db.append_event(conn, run_id, events.ThinkingDelta(text=text), publish=publish)

    attempts = retry_budget + 1
    for attempt in range(attempts):
        context = await db.read_context(conn, run_id)  # what the call uses (W3.3-W), every turn
        if cost.fold_cost(context) >= cost_ceiling_usd:
            return "cost ceiling"
        if (
            daemon_spend_ceiling_usd is not None
            and cost.fold_cost(await db.read_usage(conn)) >= daemon_spend_ceiling_usd
        ):
            return "daemon spend ceiling"
        try:
            decision = await brain(context, sink)
        except BrainCallError as exc:
            retryable = classify(exc)
            if exc.exchange is not None:
                # the attempt's raw I/O first (inv 6, W3.3): committed, never published (X2)
                await db.append_event(conn, run_id, exc.exchange)
            if exc.usage is not None:
                # the refused 200's usage, counted write-ahead before its error (inv 1, 5; W3.2-U2a)
                await db.append_event(conn, run_id, exc.usage, publish=publish)
            # the provider's error body, or the bytes assembled before a stream ended short (inv 6);
            # None if none came
            raw_response = exc.raw_response
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
                # A transient blip: back off once the UsageEvent and ErrorEvent are committed
                # (inv 1), at least `retry-after` (up to 60 s) if the response carried one, then
                # re-try within budget; the ceilings are checked again at the top, after the wait.
                await retry_wait(backoff_delay(attempt + 1, retry_after=exc.retry_after))
                continue
            return "brain call failed"  # fatal 4xx, or retryable budget exhausted -> abort
        else:
            if decision.exchange is not None:
                # the attempt's raw I/O, before its UsageEvent (`_drive`) and what it does (inv 6,
                # W3.3): committed, never published (X2)
                await db.append_event(conn, run_id, decision.exchange)
            return decision
    return "brain call failed"  # pragma: no cover - range(attempts >= 1) returns inside the loop


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
    daemon_spend_ceiling_usd: float | None = None,
    publish: db.PublishFn | None = None,
) -> tuple[str, str | None]:
    """plan -> act -> observe until a terminal decision, a breaker, or a brain failure.

    Returns (outcome, abort_reason). Three circuit breakers guard the loop: the two of ADR §Budget
    guards (ADR:262), max_iterations bounding the turn count and the per-run cost ceiling, and the
    daemon spend ceiling the scheduler injects (W.1). Both spend ceilings - each folded from
    UsageEvents (invariant 5, no stored counter) - abort BEFORE the next paid attempt once spend
    crosses them (`_call_brain`, W3.2-C1). Each iteration calls the brain on the log projected
    into its context, records the call's UsageEvent (if the brain reported one), then acts.
    NeedsHelp is terminal and appends no event; HuntComplete captures its catch into the prey pen
    (each a PreyCapturedEvent + row, P4) THEN closes via runs.outcome. A tool failure or an
    exhausted brain retry budget aborts.
    """
    for _ in range(max_iterations):
        decision = await _call_brain(
            conn,
            run_id,
            brain,
            retry_budget=retry_budget,
            cost_ceiling_usd=cost_ceiling_usd,
            daemon_spend_ceiling_usd=daemon_spend_ceiling_usd,
            publish=publish,
        )
        if isinstance(decision, str):
            return "aborted", decision  # a spend ceiling, or the brain call failed
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
    daemon_spend_ceiling_usd: float | None = None,
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
    `daemon_spend_ceiling_usd` (W3.2-C1) is the scheduler's id-scoped ceiling, checked before
    every brain attempt alongside the per-run one; None (the default) leaves the run ungated by it.
    """
    # Record the run's own caps (4a): the per-run ceiling and max_iterations _drive enforces,
    # snapshotted write-once onto the runs row — the frontend's per-run HP/stamina denominators.
    # The daemon ceiling is the scheduler's, injected (4b), never recorded here.
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
            daemon_spend_ceiling_usd=daemon_spend_ceiling_usd,
            publish=publish,
        )
    except asyncio.CancelledError as cancel:
        # Graceful shutdown (P2.3) closes the run HERE, not via the boot crash-sweep - that
        # conflates a clean stop with a kill -9 (DoD #1). The mark-aborted write must complete
        # even if a SECOND cancel lands mid-flight (aggressive shutdown, a double cancel from the
        # task group): we shield it and drive it to done before re-raising, so the run is not
        # left outcome IS NULL unless the write itself fails (a lock held past busy_timeout, an
        # I/O error; R2-RV1). Then the failure is logged, the cancel still propagates, and the
        # boot sweep closes the run 'crashed' (DoD #1). CancelledError always propagates - the
        # group tears down on purpose. (Tested adversarially in
        # test_cancelled_cleanup_survives_a_second_cancel and tests/test_retries_and_ceilings.py.)
        cleanup = asyncio.ensure_future(
            db.finish_run(conn, run_id, outcome="aborted", abort_reason="daemon shutdown")
        )
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                pass  # a re-cancel hit our frame; the shielded write runs on - wait it out
            except Exception:
                break  # the write itself failed (R2-RV1): logged below, and the cancel goes on
        if not cleanup.cancelled() and (failure := cleanup.exception()) is not None:
            logger.error(
                "run %s: the shutdown write failed (%r); the boot sweep will close it",
                run_id,
                failure,
            )
        raise cancel  # always: the daemon's teardown must not stop at a failed write
    except events.BrainParseError as exc:
        # The provider->Decision boundary (P5) rejected a payload. The generic backstop below
        # would drop the bytes (raw_request=b""); this specific clause, placed first, preserves
        # the raw payload (invariant 6) so the malformed response is a ghost-replay fixture, then
        # ends the run in a typed outcome - never an unhandled escape (DoD #5). A refused 200 can
        # still be billed: its usage, when the error carries one, is counted first (inv 1, 5;
        # W3.2-U2a). Before both, the attempt's raw request and response (inv 6, W3.3), committed
        # and never published (X2).
        if exc.exchange is not None:
            await db.append_event(conn, run_id, exc.exchange)
        if exc.usage is not None:
            await db.append_event(conn, run_id, exc.usage, publish=publish)
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
