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
import math
import random
import traceback
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import datetime
from typing import Any, Literal, assert_never

import aiosqlite
from pydantic import BaseModel, Field, ValidationError
from pydantic.json_schema import SkipJsonSchema

from rexhunter import cost, db, events, verdicts
from rexhunter.tools import ToolFn, ToolRegistry

logger = logging.getLogger("rexhunter")

# The default per-run spend guard (USD). The stub/scripted brains carry no usage (cost folds to 0),
# so this never bites them; the paid smoke (`P5` Unit 2c.2) overrides it low. Enforced in
# `_call_brain`, before every brain attempt (W3.2-C1).
COST_CEILING_USD = 1.0

# The wrap-up notice's text (`P5.1`), as Darie approved it on 27 Nov 2026.
WRAP_UP_TEXT = (
    "Budget: this is your last call before this hunt's spend limit. Call hunt_complete now with"
    " the postings you have read that are worth a human's verdict, or needs_help."
)
# The rule's multiple of the last call's cost: the coming call and one more like it, so the
# notice lands on the last call that fits under the run's ceiling.
WRAP_UP_FACTOR = 2

# ── Error taxonomy + single-attempt execution (Unit C) ───────────────────────


class ToolError(Exception):
    """A tool failure that carries the response it got: `raw_response` is the body the tool read
    before it failed, None when none arrived. The loop stores it on the attempt's ErrorEvent
    (invariant 6), so a board's error body is in the log, not only its status. Fatal unless it is
    a RetryableToolError (`classify`)."""

    def __init__(self, *args: object, raw_response: bytes | None = None) -> None:
        super().__init__(*args)
        self.raw_response = raw_response


class RetryableToolError(ToolError):
    """A tool failure the loop may re-try within budget (a transient blip, a 429, a 5xx).

    Tools opt INTO retry by raising this (or a subclass). The taxonomy's default is fatal: a
    plain exception is assumed not to fix itself on a second attempt. `retry_after` is the wait
    the source asked for, in seconds (a board's `retry-after` header, read by
    `parse_retry_after`); the retry waits at least that long, cut to RETRY_AFTER_CAP_S.
    """

    def __init__(
        self, *args: object, raw_response: bytes | None = None, retry_after: float | None = None
    ) -> None:
        super().__init__(*args, raw_response=raw_response)
        self.retry_after = retry_after


class FatalToolError(ToolError):
    """A tool failure no retry can fix (a 404, a body that fails the tool's own boundary), with
    the response bytes it got."""


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
    never become good on retry), a ToolError that is not retryable (a FatalToolError included), a
    bug in the tool, an unknown-tool lookup. Retrying a fatal error only burns budget without
    changing the outcome.
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
# the jitter's 8 s). The jitter keeps its 8 s cap. A tool retry waits at least the `retry_after`
# its RetryableToolError carries (a board's own header), cut the same way.
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


def parse_retry_after(value: str | None) -> float | None:
    """A `retry-after` header as seconds, or None: absent, or not a finite number of seconds that
    is not negative (an HTTP-date included), the header is ignored. One reader for every source
    that sends the header: the brain's provider and a board."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


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

    `catch` names the postings the hunt judged worth a human verdict, each by reference (board,
    posting_id); on completion each reference this run's own posting-source results hold is
    written to the prey pen (status='awaiting_verdict') with the board's bytes, BEFORE the run
    closes (P4; `_capture_catch`). Capturing is an internal state write, not an externally visible
    side effect - Tiny Arms (invariant 4) is untouched: the human verdict, not Rex, is what later
    moves a captured row."""

    catch: list[events.PostingRef] = Field(
        default=[],
        description=(
            "A reference to each posting you read with read_posting in this hunt and judge worth "
            "a human's verdict."
        ),
    )
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


async def _attempts[T, E: Exception](
    retry_budget: int,
    errors: type[E],
    attempt: Callable[[], Awaitable[T]],
    failed: Callable[[E], Awaitable[tuple[bool, float | None]]],
) -> T | None:
    # The one retry loop `_run_tool` and `_call_brain` share. `failed` logs a failure of type
    # `errors` and says whether it is retryable and any `retry-after`; a retryable one waits a
    # backoff once its events are committed (inv 1) and tries again within budget. None: a fatal
    # failure, or the budget spent. Any other exception propagates unchanged.
    attempts = retry_budget + 1  # retry_budget is the number of RETRIES after the first try
    for n in range(attempts):
        try:
            result = await attempt()
        except errors as exc:
            retryable, retry_after = await failed(exc)
            if retryable and n < attempts - 1:
                await retry_wait(backoff_delay(n + 1, retry_after=retry_after))
                continue
            return None
        return result
    return None  # range(attempts >= 1) returns inside the loop


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
    (`backoff_delay`, W3.2-B2; at least a RetryableToolError's `retry_after`) and re-tries within
    budget; a fatal failure or an exhausted budget appends a final ErrorEvent and aborts. Every
    append happens here, in the owning task (single writer, invariant 7). The raw request rides on
    every event, and a ToolError's response bytes on its ErrorEvent, so a dead attempt is a pytest
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

    async def failed(exc: Exception) -> tuple[bool, float | None]:
        retryable = classify(exc)
        await db.append_event(
            conn,
            run_id,
            events.ErrorEvent(
                tool=tool_name,
                retryable=retryable,
                error=repr(exc),
                raw_request=raw_request,
                # the body the tool got before it failed (invariant 6); None when none came
                raw_response=exc.raw_response if isinstance(exc, ToolError) else None,
                detail=traceback.format_exc(),
            ),
            publish=publish,
        )
        # a retry waits at least the source's `retry-after`, cut to RETRY_AFTER_CAP_S
        return retryable, exc.retry_after if isinstance(exc, RetryableToolError) else None

    raw_response = await _attempts(
        retry_budget, Exception, lambda: execute_once(tool.fn, kwargs, timeout_s=timeout_s), failed
    )
    if raw_response is None:
        return False  # fatal, or retryable budget exhausted -> abort
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


async def _call_brain(
    conn: aiosqlite.Connection,
    run_id: str,
    brain: Brain,
    *,
    retry_budget: int,
    cost_ceiling_usd: float,
    daemon_spend_ceiling_usd: float | None,
    daemon_spend_since: Callable[[], datetime] | None = None,
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
    retry sends the same request. The daemon's fold starts at the instant `daemon_spend_since`
    gives at that attempt (the scheduler's rolling day, `P5.1`; None = the whole log).

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

    async def attempt() -> Decision | str:
        # Both ceilings before every attempt, a retry included, on the log as it stands then.
        context = await db.read_context(conn, run_id)  # what the call uses (W3.3-W), every turn
        if cost.fold_cost(context) >= cost_ceiling_usd:
            return "cost ceiling"
        since = None if daemon_spend_since is None else daemon_spend_since()
        if (
            daemon_spend_ceiling_usd is not None
            and cost.fold_cost(await db.read_usage(conn, since=since)) >= daemon_spend_ceiling_usd
        ):
            return "daemon spend ceiling"
        return await brain(context, sink)

    async def failed(exc: BrainCallError) -> tuple[bool, float | None]:
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
        # A transient blip waits at least `retry-after` (up to 60 s) if the response carried one.
        return retryable, exc.retry_after

    result = await _attempts(retry_budget, BrainCallError, attempt, failed)
    if result is None:
        return "brain call failed"  # fatal 4xx, or retryable budget exhausted -> abort
    if not isinstance(result, str) and result.exchange is not None:
        # the attempt's raw I/O, before its UsageEvent (`_drive`) and what it does (inv 6, W3.3):
        # committed, never published (X2)
        await db.append_event(conn, run_id, result.exchange)
    return result


async def _capture_catch(
    conn: aiosqlite.Connection,
    run_id: str,
    *,
    territory: str,
    registry: ToolRegistry,
    catch: list[events.PostingRef],
    publish: db.PublishFn | None = None,
) -> None:
    """Pen each distinct reference in the catch, in order, with the board's bytes for it.

    A reference resolves only against this run's own results of a posting-source tool whose
    stored request is that reference (the models compared, not the bytes); the latest such read
    gives the bytes. A reference the run never read appends a `<capture>` ErrorEvent and is not
    penned; a posting already in the pen, in any status, is not penned again (`capture_prey`).
    Either way the run completes. Every write here is the owning task's (invariant 7)."""
    read: dict[events.PostingRef, bytes] = {}
    for event in await db.read_context(conn, run_id):
        if not isinstance(event, events.ToolResultEvent) or event.tool not in registry:
            continue
        if not registry.get(event.tool).posting_source:
            continue
        try:
            ref = events.PostingRef.model_validate_json(event.raw_request)
        except ValidationError:
            continue  # a stored request that is no reference cannot be the posting one names
        read[ref] = event.raw_response
    for ref in dict.fromkeys(catch):
        raw_posting = read.get(ref)
        if raw_posting is None:
            await db.append_event(
                conn,
                run_id,
                events.ErrorEvent(
                    tool="<capture>",
                    retryable=False,
                    error="unresolved catch reference",
                    raw_request=ref.model_dump_json().encode(),
                ),
                publish=publish,
            )
            continue
        await verdicts.capture_prey(
            conn,
            run_id,
            territory=territory,
            raw_posting=raw_posting,
            board=ref.board,
            posting_id=ref.posting_id,
            publish=publish,
        )


def _wrap_up_reason(
    context: Context, *, cost_ceiling_usd: float, last_iteration: bool
) -> Literal["cost", "iterations"] | None:
    """Why the coming brain call gets the wrap-up notice (`P5.1`), or None. Folded from the run's
    own context (invariant 5): never a second notice, since the first is in the log; then the cost
    rule, the run's spend plus `WRAP_UP_FACTOR` times its last call's cost against its ceiling;
    then the last iteration."""
    if any(isinstance(event, events.WrapUpNoticeEvent) for event in context):
        return None
    usage = [event for event in context if isinstance(event, events.UsageEvent)]
    last_call = cost.cost_of(usage[-1]) if usage else 0.0
    if cost.fold_cost(context) + WRAP_UP_FACTOR * last_call >= cost_ceiling_usd:
        return "cost"
    return "iterations" if last_iteration else None


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
    daemon_spend_since: Callable[[], datetime] | None = None,
    wrap_up: bool = False,
    publish: db.PublishFn | None = None,
) -> tuple[str, str | None]:
    """plan -> act -> observe until a terminal decision, a breaker, or a brain failure.

    Returns (outcome, abort_reason). Three circuit breakers guard the loop: the two of ADR §Budget
    guards (ADR:262), max_iterations bounding the turn count and the per-run cost ceiling, and the
    daemon spend ceiling the scheduler injects (W.1). Both spend ceilings - each folded from
    UsageEvents (invariant 5, no stored counter) - abort BEFORE the next paid attempt once spend
    crosses them (`_call_brain`, W3.2-C1). Each iteration calls the brain on the log projected
    into its context, records the call's UsageEvent (if the brain reported one), then acts.
    NeedsHelp is terminal and appends no event; HuntComplete captures the references of its catch
    that the run read into the prey pen (each a PreyCapturedEvent + row, P4; `_capture_catch`)
    THEN closes via runs.outcome. A tool failure or an exhausted brain retry budget aborts.
    With `wrap_up` (`P5.1`) the loop appends the wrap-up notice before the call
    `_wrap_up_reason` picks, so the projection puts it in that call's request.
    """
    for iteration in range(max_iterations):
        if wrap_up:
            reason = _wrap_up_reason(
                await db.read_context(conn, run_id),
                cost_ceiling_usd=cost_ceiling_usd,
                last_iteration=iteration == max_iterations - 1,
            )
            if reason is not None:
                notice = events.WrapUpNoticeEvent(text=WRAP_UP_TEXT, reason=reason)
                await db.append_event(conn, run_id, notice, publish=publish)
        decision = await _call_brain(
            conn,
            run_id,
            brain,
            retry_budget=retry_budget,
            cost_ceiling_usd=cost_ceiling_usd,
            daemon_spend_ceiling_usd=daemon_spend_ceiling_usd,
            daemon_spend_since=daemon_spend_since,
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
                await _capture_catch(
                    conn,
                    run_id,
                    territory=territory,
                    registry=registry,
                    catch=decision.catch,
                    publish=publish,
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
    daemon_spend_since: Callable[[], datetime] | None = None,
    wrap_up: bool = False,
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
    `daemon_spend_since` is the scheduler's window, the one its launch gate folds (`P5.1`); None
    (the default) folds the whole log. `wrap_up` (`P5.1`, off by default) has the loop give the
    wrap-up notice before the run's last call, and is recorded on the run's row.
    """
    # Record the run's own caps (4a): the per-run ceiling and max_iterations _drive enforces,
    # snapshotted write-once onto the runs row — the frontend's per-run HP/stamina denominators.
    # The daemon ceiling is the scheduler's, injected (4b), never recorded here.
    run_id = await db.start_run(
        conn,
        territory=territory,
        cost_ceiling_usd=cost_ceiling_usd,
        max_iterations=max_iterations,
        wrap_up=wrap_up,
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
            daemon_spend_since=daemon_spend_since,
            wrap_up=wrap_up,
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
        # A re-cancel is waited out (`db.drive_to_done`); a failed write ends the wait: it is
        # logged below, and the cancel goes on.
        with suppress(Exception):
            await db.drive_to_done(cleanup)
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
