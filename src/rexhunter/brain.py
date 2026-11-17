"""The brain socket — Pillar 5. Unit 1: the edge boundary. Unit 2a/2b: the provider adapter.

`parse_decision` is the ONE line (ADR §The What point 3) where a raw provider response becomes
a typed `Decision`. LLM output is untrusted input (invariant 3) — treated with the same
suspicion as scraped HTML — so raw bytes cross exactly one Pydantic boundary here and typed
objects only exist inward of it. A payload that does not validate raises `BrainParseError`
carrying the raw bytes (invariant 6); `run_hunt` records it as an `ErrorEvent` and ends the run
in a typed outcome, never a mystery string three functions deep.

Native tool calling, no text parsing (§The What point 2): the actionable signal is a structured
`tool_use` block (name, `input`, `id`). Reserved block names are the terminal decisions — the
model *calls* `hunt_complete` / `needs_help` rather than us parsing prose — so a structured
`catch` survives without ever reading free text. The `tool_use_id` rides onto `ToolCallDecision`
so the loop can stamp it on both the call and result events. Exactly one `tool_use` block is
actionable per turn: zero (nothing to do) or more than one (ambiguous — the loop runs one tool
per iteration) is a `BrainParseError`, not a silent first-wins.

Unit 2a adds the provider adapter (`adapter_brain_for`) over **httpx** — still no vendor SDK (the
loop stays SDK-free, ADR §The What point 1). The adapter POSTs through an INJECTED client, so this
module opens no connection of its own; a `MockTransport` gives the offline tests zero network.
Unit 2b splits out `build_request_body` + `request_headers` (shared with the paid smoke, one
request shape, no drift) and strips `title` from the derived tool schemas. Streaming and budget
accounting are later units.

W2.0 moves the transport onto the Anthropic SDK in RAW mode (ADR Pillar 5, the logged D1
exception): `anthropic_client` builds the one client (max_retries=0, the pinned wire), the SSE
assembler and `parse_decision` stay hand-rolled and still see the provider's own bytes, and SDK /
transport errors become the loop's provider-neutral `BrainCallError`; the loop imports no vendor
SDK. `request_headers` is gone: the factory owns the headers.
"""

import json
import math
import os
import ssl
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any, cast

import anthropic
import certifi
import httpx2
from anthropic import AsyncAPIResponse
from anthropic.types import Message
from pydantic import BaseModel, ConfigDict, ValidationError

from rexhunter import cost
from rexhunter.events import (
    BrainExchangeEvent,
    BrainParseError,
    ErrorEvent,
    ToolCallEvent,
    ToolResultEvent,
    TrajectoryEvent,
    UsageEvent,
)
from rexhunter.loop import (
    Brain,
    BrainCallError,
    Context,
    Decision,
    HuntComplete,
    NeedsHelp,
    ThinkingSink,
    ToolCallDecision,
)
from rexhunter.tools import ToolRegistry

# Reserved tool names that map to terminal decisions rather than a registry dispatch. Unit 2a's
# adapter presents these to the model as tools (the D1 cross-unit commitment, honoured here).
HUNT_COMPLETE = "hunt_complete"
NEEDS_HELP = "needs_help"

_TOOL_USE = "tool_use"  # the Anthropic content-block type we act on

# The Anthropic Messages API surface (from the claude-api reference). The adapter/smoke target the
# full URLs rather than a client base_url, so an injected client needs no configuration.
MESSAGES_URL = "https://api.anthropic.com/v1/messages"
COUNT_TOKENS_URL = "https://api.anthropic.com/v1/messages/count_tokens"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2026-10-21"
_DEFAULT_MAX_TOKENS = 1024


def tls_context() -> ssl.SSLContext:
    """The trust store every real brain transport verifies against: HEAD's, kept (W2.0). It mirrors
    httpx 0.28's `create_ssl_context`: `SSL_CERT_FILE`, then `SSL_CERT_DIR`, else certifi's bundle.
    httpx2's own default on macOS and Windows is the system store (truststore), a wider trust set;
    moving to it is a separate decision, not part of a behaviour-preserving transport swap."""
    if cafile := os.environ.get("SSL_CERT_FILE"):
        return ssl.create_default_context(cafile=cafile)
    if capath := os.environ.get("SSL_CERT_DIR"):
        return ssl.create_default_context(capath=capath)
    return ssl.create_default_context(cafile=certifi.where())


def live_transport() -> httpx2.AsyncHTTPTransport:
    """A real network transport on `tls_context`, for the entrypoints that wrap one in a
    capturing tee (the factory builds its own when no transport is injected)."""
    return httpx2.AsyncHTTPTransport(verify=tls_context())


def anthropic_client(
    api_key: str,
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
    timeout: float = 120.0,
) -> anthropic.AsyncAnthropic:
    """The ONE place an Anthropic SDK client is built (W2.0): the daemon, the gated entrypoints and
    the tests all come through here, so the request shape cannot drift between them.

    - `max_retries=0` (D1b): the SDK never retries; the loop owns every retry and logs each attempt
      as its own ErrorEvent. It is also the paid smoke's hard one-call guard.
    - The wire is PINNED explicitly (W2.1b): `Authorization: Bearer <key>` and
      `anthropic-version: 2026-10-21`, with `x-api-key` and `anthropic-beta` omitted. Today the
      pins equal the SDK 2.6.0 defaults; they keep these four headers fixed if a later SDK changes
      its defaults (pinned by tests/test_brain_adapter.py). Fixtures recorded before W2.1b used
      `x-api-key` and 2023-06-01; they hold response bytes only, so they replay unchanged.
    - Explicit `api_key` and `base_url`: the SDK otherwise resolves credentials from the
      environment / profiles on disk, and `ANTHROPIC_BASE_URL` would redirect the request and the
      key.
    - `ANTHROPIC_CUSTOM_HEADERS` (W2.1): the SDK merges that variable into every request's headers
      (anthropic/_client.py `__init__`). Every name it lists gets an entry here, parsed the way the
      SDK parses it (lines split on newlines, the name before the first colon, stripped): a pinned
      name, in any spelling, gets its pinned value, and any other name is set to `omit`. So no
      value from the variable reaches the wire, and it cannot switch betas on. The residual: a
      listed name the SDK sets itself (the client-level defaults Accept, Content-Type, User-Agent,
      X-Stainless-Async and the X-Stainless-* platform headers, and the per-request X-Stainless
      timeout, read-timeout and retry-count headers) loses its value on the wire, dropped or
      downgraded to the httpx2 default, so the wire differs from the unset case; whether the API
      accepts a POST without Content-Type is unverified. Host, Content-Length, Connection and
      Accept-Encoding (httpx2 sets them per request), X-Stainless-Raw-Response and the pinned names
      keep their values (a listed spelling of a pinned name can become the header's name on the
      wire; the value stays pinned). A listed name that is not valid UTF-8 makes every request
      fail before it is sent, as it did before W2.1. A client `copy()` or `with_options()` keeps
      these entries but re-reads the variable, so a name or spelling added to it after the factory
      ran is not covered on the copy and can reach the wire, a second spelling of a pinned name
      included; src/ never copies the client.
    - No redirects: the SDK's default http client follows them, cross-origin, and on a 307 or 308
      would resend the paid POST body (httpx2 strips `Authorization` on a cross-origin redirect, so
      the Bearer key would not follow; an `x-api-key` would have). HEAD's httpx client never
      followed one, so a 3xx stays a fatal status error. The factory builds the http client itself
      (the SDK's keep-alive defaults, `follow_redirects=False`); callers inject only a `transport`
      (a capturing tee, a `MockTransport` in tests), so every caller gets the same client.
    - HEAD's trust store: the client verifies against `tls_context()` (certifi), not httpx2's
      system-store default.

    `default_headers` is typed `Mapping[str, str]`, but the SDK drops `omit`-valued headers at
    runtime (anthropic/_base_client.py `build_headers`), hence the one cast below; the wire test
    (tests/test_brain_adapter.py) pins the result."""
    pinned: dict[str, str | anthropic.Omit] = {
        "Authorization": f"Bearer {api_key}",
        "anthropic-version": _ANTHROPIC_VERSION,
        "x-api-key": anthropic.omit,
        "anthropic-beta": anthropic.omit,
    }
    # Every name ANTHROPIC_CUSTOM_HEADERS lists gets an entry of its own, in its own spelling: the
    # pinned value for a pinned name (any case, so a second spelling listed after the first cannot
    # win the case-insensitive fold), `omit` for any other. Omitting a pinned name instead would
    # drop our own key and send the request unauthenticated.
    by_lower = {name.lower(): value for name, value in pinned.items()}
    for line in os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "").split("\n"):
        name, colon, _value = line.partition(":")
        if colon:
            pinned[name.strip()] = by_lower.get(name.strip().lower(), anthropic.omit)
    # `transport` is passed only when given: the SDK builds its keep-alive transport when the key
    # is absent, and `transport=None` would skip that.
    client_options: dict[str, Any] = {"follow_redirects": False, "verify": tls_context()}
    if transport is not None:
        client_options["transport"] = transport
    return anthropic.AsyncAnthropic(
        api_key=api_key,
        base_url=ANTHROPIC_BASE_URL,
        max_retries=0,
        timeout=timeout,
        http_client=anthropic.DefaultAsyncHttpxClient(**client_options),
        default_headers=cast("dict[str, str]", pinned),
    )


# The live model (D2, W2.5), named for the Unit 2b smoke call: every gated entrypoint and the
# daemon's `select_brain_for` run on it. Claude Sonnet 5.5: a request with no `thinking` field
# runs adaptive thinking (so the response carries `thinking` blocks), and sampling params, thinking
# budgets, assistant prefill, forced tool choice and `thinking: {type: "disabled"}` are all
# rejected with a 400. build_request_body sends no sampling params and adds no prefill, no caller
# passes a thinking budget, HUNT_TOOL_CHOICE is `auto`, and a thinking-off hunt sends
# `thinking_floor(SMOKE_MODEL)`. The recorded Sonnet 5 runs replay on their own model
# (RECORDED_MODEL in each replay test).
SMOKE_MODEL = "claude-sonnet-5-5"

# The hunt directive as a stable system prompt (the thing the P2.2 stub brain never needed). It
# tells the model what sniff/hunt_complete/needs_help mean AND encodes Tiny Arms (invariant 4) in
# prose — Rex scouts and captures, a human decides. Kept module-level + frozen so the request prefix
# is byte-stable (prompt-cache friendly). Threaded via build_request_body(system=...).
HUNT_SYSTEM_PROMPT = (
    "You are Rex, an autonomous agent scouting AI-engineering job postings. Investigate a "
    "territory by calling the sniff tool. When you have gathered postings worth a human's verdict, "
    "call hunt_complete with them as your catch. If the territory is unworkable or you are stuck, "
    "call needs_help. You cannot apply, submit, message, or contact anyone — you only scout and "
    "capture; a human alone decides what happens to each catch."
)

# One tool per iteration. `disable_parallel_tool_use` caps the model to a single `tool_use`
# block, so a parallel-tool reply can't trip the 2a multi-`tool_use` reject and abort the hunt.
# `type:"auto"` still lets the model choose or terminate. Threaded via build_request_body; the
# 2b smoke omits it (unforced) and keeps its shape.
HUNT_TOOL_CHOICE: dict[str, Any] = {"type": "auto", "disable_parallel_tool_use": True}

# The thinking-on hunt shape (`P5` Unit 3, reused by the daemon live-wiring): adaptive thinking with
# summarized display, streamed as SSE. Defined ONCE here (the "one request shape" discipline) — the
# gated entrypoint (`hunt_smoke_thinking`) and the daemon `select_brain_for(live)` both read it, so
# the daemon streams reasoning through the SAME brain the harness proved, no drift.
HUNT_THINKING: dict[str, Any] = {"type": "adaptive", "display": "summarized"}
# Thinking tokens AND the tool_use share `max_tokens`, so a tight 1024 truncates into
# stop_reason=max_tokens (an incomplete tool_use the assembler can't finalize). 4096 clears both.
HUNT_MAX_TOKENS = 4096

# The lowest thinking setting per model (W2.2, F5, fork F1): what a thinking-off hunt sends, keyed
# by API model id like `cost.PRICE_PER_MTOK`, one row per model a hunt runs with thinking off.
# Claude Sonnet 5 turns thinking off with `disabled`. On Claude Sonnet 5.5, "`disabled` returns a
# 400 `invalid_request_error`"; its lowest setting is `between_tools` (up-front thinking off;
# progress updates between tool calls still come back as thinking blocks), and "`between_tools`
# takes no other field: `display`, `budget_tokens`, or `block_binding` sent with it returns a 400
# error". A model with no row fails closed in `thinking_floor`, before any request is built: it
# has no off setting (Opus 5.5 and the Fable models reject both values), or its row is not added
# yet. Read 14 Nov 2026:
#   https://platform.claude.com/docs/en/build-with-claude/thinking ("Configuring thinking" table,
#   "Turning thinking off", #progress-updates)
#   https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide#thinking-on-by-default
#   https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide#turn-off-up-front-thinking
THINKING_FLOOR: dict[str, dict[str, str]] = {
    "claude-sonnet-5": {"type": "disabled"},
    "claude-sonnet-5-5": {"type": "between_tools"},
}


def thinking_floor(model: str) -> dict[str, Any]:
    """The model's lowest thinking setting, as a fresh copy of its `THINKING_FLOOR` row (so no
    caller can edit the table). A model with no row raises ValueError: fail closed, never a guess
    that the API would answer with a 400."""
    row = THINKING_FLOOR.get(model)
    if row is None:
        raise ValueError(
            f"no lowest thinking setting for model {model!r} in brain.THINKING_FLOOR: add its row "
            "from the thinking docs before running it with thinking off"
        )
    return dict(row)


class _ToolUseBlock(BaseModel):
    """The one content block we consume, validated strictly on the fields we read. Extra keys are
    tolerated (a provider may add fields); `input` stays an untyped object — the per-tool schema
    validates it at dispatch (`tool.validate`, the real second boundary from `P2.2`)."""

    type: str
    id: str
    name: str
    input: dict[str, Any]


class _ProviderResponse(BaseModel):
    """The Anthropic Messages-API envelope, validated for THE SHAPE WE CONSUME only. It is
    untrusted provider I/O, so — unlike our own event models (`extra="forbid"`) — the envelope is
    lenient on unmodeled fields (`id`/`model`/`role`/`usage`/…): we validate `content` is a list
    and pull the actionable `tool_use` block out of it, tolerating sibling blocks (text, thinking)
    without letting them reject the payload. Only that block is validated further; the whole
    `content` list is otherwise kept as received and stored as the turn (W2.3)."""

    model_config = ConfigDict(extra="ignore")

    stop_reason: str | None = None
    content: list[dict[str, Any]]
    model: str | None = None  # names the priced model (P5 Unit 2c cost accounting)
    usage: dict[str, Any] | None = None  # {input_tokens, output_tokens, …}; absent on hand fixtures


def _extract_usage(response: _ProviderResponse) -> UsageEvent | None:
    """Read the envelope's token counts into a typed `UsageEvent` (P5 Unit 2c). None when the
    payload carries no `usage` (hand-authored fixtures) — the loop then records no spend for that
    call. Reads the SAME validated envelope as the decision, not a second inv-3 crossing. Used for
    an accepted turn only (W3.2-U2a): a refused one reads its count with `_refused_usage`."""
    if response.usage is None:
        return None
    return UsageEvent(
        model=response.model or "",
        input_tokens=int(response.usage.get("input_tokens", 0)),
        output_tokens=int(response.usage.get("output_tokens", 0)),
    )


def _refused_usage(model: object, usage: object) -> tuple[UsageEvent | None, str]:
    """The token count of a 200 the boundary refuses (W3.2-U2a), and a note for the error.

    Read so it never raises: the refusal must still end typed, with its bytes. Returns (None, "")
    when the response carries no count (no `usage`, or an empty one: a stream cut before its
    `message_start`), and (None, a note) when a count does not read as tokens or cannot be priced
    (a float overflow in `cost.cost_of`); that call is not counted, and the note says so in the
    ErrorEvent. A model that is not a string prices at the
    fallback (`cost.py`), the safe direction for a guard. The accepted path keeps
    `_extract_usage`, which raises on such a count (W3.1-O5)."""
    if not isinstance(usage, dict) or not usage:
        return None, ""
    counts = cast("dict[str, Any]", usage)
    try:
        counted = UsageEvent(
            model=model if isinstance(model, str) else "",
            input_tokens=int(counts.get("input_tokens", 0)),
            output_tokens=int(counts.get("output_tokens", 0)),
        )
        # Priced once here: a count int() reads but the fold cannot price (its float overflows)
        # would make every later fold of the log raise, so it is not counted either.
        cost.cost_of(counted)
    except TypeError, ValueError, OverflowError:
        return None, f"; usage not counted, it does not read as a priceable token count: {counts!r}"
    return counted, ""


def _refusal(raw: bytes, detail: str, *, model: object, usage: object) -> BrainParseError:
    """A BrainParseError for a 200 the boundary refuses, carrying its usage (W3.2-U2a)."""
    counted, note = _refused_usage(model, usage)
    return BrainParseError(raw=raw, detail=detail + note, usage=counted)


def parse_decision(raw: bytes) -> Decision:
    """Raw provider bytes -> typed `Decision`, or `BrainParseError` (invariants 3 + 6). Pure: no
    clock, no randomness, no network — the same bytes always yield the same decision. The parsed
    `UsageEvent` (token cost, P5 Unit 2c) rides onto the Decision so the loop can fold spend from
    the log (invariant 5) without a second crossing of the raw bytes. A `ToolCallDecision` also
    carries the whole turn (W2.3, D7): the validated envelope's `content` array, every block in
    order, as JSON bytes, which the loop stores on the ToolCallEvent for the next call's echo. Both
    paths come through here, so the plain and the streaming turn are stored the same way. A turn
    is acted on only when it stopped for its tool call (W3.1, fork S1): any other `stop_reason`,
    `None` included, is a BrainParseError, so a turn that stopped for another reason (a
    `max_tokens` cut, a refusal, a stream cut before its `message_delta`) is never dispatched or
    stored. A refused turn can still be billed (W3.2-U2a): every BrainParseError raised once the
    envelope validates carries the envelope's usage (`_refusal`), which the loop counts before the
    ErrorEvent, the safe direction for a guard; the accepted turn's usage is read last, so a
    refusal never fails on its count."""
    try:
        response = _ProviderResponse.model_validate_json(raw)
    except ValidationError as exc:
        raise BrainParseError(raw=raw, detail=str(exc)) from exc

    # Act only on a turn that stopped for its tool call (W3.1, S1). The canonical loop runs tools
    # "while `stop_reason == "tool_use"`" and exits on any other stop reason, and a `max_tokens`
    # cut can leave an incomplete tool use that needs a retry with a higher `max_tokens`. A turn
    # whose tool use is whole but which stopped on `max_tokens` after it is declined too: the
    # thinking docs' continue rule for it (a `tool_result` for each `tool_use`) is fork S2, not
    # taken. Read 15 Nov 2026:
    #   https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works
    #   https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons
    #   https://platform.claude.com/docs/en/build-with-claude/thinking (progress updates)
    # `None` too: "The `stop_reason` field is part of every successful Messages API response", and
    # a stream sets it in `message_delta`, so a stream cut before that frame carries none. Checked
    # before the usage is read, so such a turn ends here whatever its usage holds.
    # Every refusal from here on is a 200 that can still be billed (W3.2-U2a): it carries the
    # envelope's usage, read by `_refused_usage`, which never raises, so the loop counts it first.
    if response.stop_reason != _TOOL_USE:
        raise _refusal(
            raw,
            (
                f"stop_reason={response.stop_reason!r}: the loop acts only on a turn that "
                "stopped for its tool call"
            ),
            model=response.model,
            usage=response.usage,
        )

    # Exactly one actionable block. We count `tool_use`-typed blocks only, so a `thinking` (or
    # other) sibling doesn't count against the total — zero means nothing to act on, more than one
    # is ambiguous for a one-tool-per-iteration loop; neither is a silent first-wins.
    tool_use_blocks = [b for b in response.content if b.get("type") == _TOOL_USE]
    if len(tool_use_blocks) != 1:
        raise _refusal(
            raw,
            (
                f"expected exactly one tool_use block, found {len(tool_use_blocks)}: the loop acts "
                "on one native tool call per iteration, never on free text or a batch"
            ),
            model=response.model,
            usage=response.usage,
        )
    try:
        block = _ToolUseBlock.model_validate(tool_use_blocks[0])
    except ValidationError as exc:
        raise _refusal(raw, str(exc), model=response.model, usage=response.usage) from exc

    # Reserved names are terminal decisions (they have no registry handler, so this parser is
    # their only boundary — hence `catch` IS validated here, unlike regular-tool args).
    decision: Decision
    if block.name == HUNT_COMPLETE:
        try:
            decision = HuntComplete.model_validate(block.input)
        except ValidationError as exc:
            raise _refusal(raw, str(exc), model=response.model, usage=response.usage) from exc
    elif block.name == NEEDS_HELP:
        decision = NeedsHelp()
    else:
        decision = ToolCallDecision(
            tool=block.name,
            args=block.input,
            tool_use_id=block.id,
            # The whole turn, from the SAME validated envelope (not a second crossing of `raw`):
            # every block, in order, with every key the provider sent. Parsed JSON is what the API
            # compares, so re-serialising it here keeps the turn exactly as returned.
            assistant_content=json.dumps(response.content).encode(),
        )
    # every Decision carries the call's cost (None for a no-usage payload); read only once the turn
    # is accepted, with `_extract_usage`, so a count that is not a number still raises (W3.1-O5)
    decision.usage = _extract_usage(response)
    return decision


# ── The streaming transport (Unit 3a): hand-rolled SSE, fed the SDK's raw stream since W2.0 ────
# The API streams its response as SSE events. We assemble them into the SAME `_ProviderResponse`
# shape parse_decision consumes — no vendor SDK (ADR §What point 1), and the raw signed thinking
# block stays in hand (invariants 3 + 6), which the SDK's typed-delta layer would hide. (W2.0 kept
# this assembler and puts only the transport on the SDK, raw mode: the D9 probe confirmed the typed
# layer re-serialises the signed block.) P3's own outbound SSE parser is this same protocol inbound;
# the system is an SSE relay end to end.


def iter_sse_events(raw: bytes) -> Iterator[dict[str, Any]]:
    """Parse an Anthropic SSE response body into its `data:` JSON events (offline, pure).

    Frames are blank-line separated; the `event:` label is ignored (the payload's own `type` is
    authoritative), and comment/keep-alive lines and trailing blank frames fold to nothing. LLM
    output is untrusted (invariant 3): a `data:` line that is not valid JSON raises here, at the
    boundary, never leaking downstream.
    """
    for frame in raw.split(b"\n\n"):
        data = b"".join(
            line.removeprefix(b"data:").strip()
            for line in frame.splitlines()
            if line.startswith(b"data:")
        )
        if data:
            yield json.loads(data)


class StreamAssembler:
    """Fold the SSE event stream into an assembled Messages-API response (the `_ProviderResponse`
    shape parse_decision consumes). Stateful + incremental so the live path can relay thinking
    deltas as they arrive; pure and offline-testable by feeding fixture events.

    Display streams, execution waits (ADR §What point 5): a `tool_use` block's `input` is parsed
    from its accumulated `input_json_delta` only at `content_block_stop`; until then its start
    `input` (`{}`) stands. A partial stream is un-dispatchable because `_drive_stream` hands a turn
    on only when the stream is `finished` (W3.1, fork T3): every started block got its
    `content_block_stop`, a `message_delta` carried a stop reason, and `message_stop` arrived.
    """

    def __init__(self) -> None:
        self._blocks: dict[int, dict[str, Any]] = {}
        self._json_buf: dict[int, str] = {}
        self._model: str | None = None
        self._stop_reason: str | None = None
        self._usage: dict[str, Any] = {}
        self._stopped: set[int] = set()  # the block indexes that got their content_block_stop
        self._message_stopped = False  # message_stop arrived (W3.1)
        self._failed_inputs: dict[int, str] = {}  # tool_use input that was not JSON (W3.1)
        self._error_frame: dict[str, Any] | None = None  # the first SSE `error` event (W3.1)
        self.complete = False  # True once message_stop / a stop_reason has arrived

    def feed(self, event: dict[str, Any]) -> str | None:
        """Fold one SSE event. Returns the thinking-delta text to relay (the caller appends it
        write-ahead then broadcasts, invariant 1), or None for every other event."""
        etype = event.get("type")
        if etype == "message_start":
            message = event.get("message", {})
            self._model = message.get("model")
            self._usage.update(message.get("usage", {}))  # input_tokens (+ an initial output count)
        elif etype == "content_block_start":
            self._blocks[int(event["index"])] = dict(event["content_block"])
        elif etype == "content_block_delta":
            return self._apply_delta(int(event["index"]), event["delta"])
        elif etype == "content_block_stop":
            index = int(event["index"])
            self._stopped.add(index)
            self._finalize(index)
        elif etype == "message_delta":
            delta = event.get("delta", {})
            if delta.get("stop_reason") is not None:
                self._stop_reason = delta["stop_reason"]
                self.complete = True
            self._usage.update(event.get("usage", {}))  # cumulative output_tokens
        elif etype == "message_stop":
            self.complete = True
            self._message_stopped = True
        elif etype == "error" and self._error_frame is None:
            # Recorded, never raised (W3.1, fork Z1): `_drive_stream` turns it into the call's
            # verdict, and an offline reader of the bytes (the smoke reporters) folds on.
            self._error_frame = event
        return None

    @property
    def error_frame(self) -> dict[str, Any] | None:
        """The first SSE `error` event the stream carried (its parsed `data`), or None."""
        return self._error_frame

    @property
    def stop_reason(self) -> str | None:
        """The stop reason the stream's `message_delta` carried, or None if none arrived."""
        return self._stop_reason

    @property
    def ended(self) -> bool:
        """`message_stop` arrived: the provider sent the whole body (W3.1, T3). A stream that did
        not end was cut on the way (a transport failure)."""
        return self._message_stopped

    @property
    def open_blocks(self) -> list[int]:
        """The started blocks that never got their `content_block_stop`, in index order."""
        return sorted(set(self._blocks) - self._stopped)

    @property
    def finished(self) -> bool:
        """The stream is a whole turn (W3.1, T3): every started content block got its
        `content_block_stop`, a `message_delta` carried a stop reason, and `message_stop` arrived
        (the event flow in https://platform.claude.com/docs/en/build-with-claude/streaming, read
        15 Nov 2026). `complete` is looser: a stop reason alone sets it."""
        return self.ended and not self.open_blocks and self._stop_reason is not None

    def _apply_delta(self, index: int, delta: dict[str, Any]) -> str | None:
        block = self._blocks.get(index)
        if (
            block is None
        ):  # a delta before its block start — defensive; never in a well-formed stream
            return None
        dtype = delta.get("type")
        if dtype == "thinking_delta":
            text = str(delta.get("thinking", ""))
            block["thinking"] = str(block.get("thinking", "")) + text
            return text  # the one event that feeds the live relay (Unit 3b)
        if dtype == "signature_delta":  # opaque; concatenated verbatim, never interpreted (inv 3)
            block["signature"] = str(block.get("signature", "")) + str(delta.get("signature", ""))
        elif dtype == "text_delta":
            block["text"] = str(block.get("text", "")) + str(delta.get("text", ""))
        elif dtype == "input_json_delta":
            self._json_buf[index] = self._json_buf.get(index, "") + str(
                delta.get("partial_json", "")
            )
        return None

    def _finalize(self, index: int) -> None:
        block = self._blocks.get(index)
        if block is not None and block.get("type") == _TOOL_USE:
            buf = self._json_buf.get(index, "")
            # Parsed only now; the tool's own schema validates it at dispatch (`tool.validate`).
            # Parsing the tool input never raises here (W3.1, fork P1): input that does not parse
            # keeps the block's start `input` and is recorded in `failed_inputs`, which
            # `_drive_stream` refuses, so a reader of the captured bytes (hunt_smoke_thinking's
            # reporter) still gets to `write_fixtures`. ValueError, not only JSONDecodeError (its
            # subclass): `json.loads` raises a plain ValueError for an integer over 4300 digits.
            try:
                block["input"] = json.loads(buf) if buf else {}
            except ValueError:
                self._failed_inputs[index] = buf

    @property
    def failed_inputs(self) -> dict[int, str]:
        """The tool_use blocks whose input did not parse as JSON at their `content_block_stop`:
        index -> the accumulated `input_json_delta` text, as the model sent it."""
        return dict(self._failed_inputs)

    def assembled(self) -> dict[str, Any]:
        """The assembled response in `_ProviderResponse` shape. Blocks in content-index order; an
        unfinalized tool_use keeps its empty start `input` (execution waits)."""
        return {
            "type": "message",
            "role": "assistant",
            "model": self._model,
            "stop_reason": self._stop_reason,
            "content": [self._blocks[i] for i in sorted(self._blocks)],
            "usage": self._usage,
        }


# ── The provider adapter (Unit 2a) + the shared request shape (Unit 2b): the SDK in raw mode ──

# Concise, model-facing descriptions for the reserved terminal tools. These are prose (how the
# model should use the signal), NOT the schema — the schema is still DERIVED from the typed model
# (`model_json_schema()`), so ADR §What point 1 ("never hand-written schemas") holds.
_TERMINAL_TOOLS: tuple[tuple[str, type[BaseModel], str], ...] = (
    (HUNT_COMPLETE, HuntComplete, "End the hunt. `catch` is the list of postings worth a verdict."),
    (NEEDS_HELP, NeedsHelp, "End the hunt and hand off to a human — Rex is stuck."),
)


def _strip_titles(schema: Any) -> Any:
    """Recursively drop `title` keys from a JSON schema. Pydantic's `model_json_schema()` emits a
    `title` at the object root and on every property; Anthropic's `input_schema` neither needs nor
    wants them, so we present the model a cleaner, provider-neutral schema. `additionalProperties`
    (emitted by the `@rex_tool` args-models' `extra="forbid"`) is KEPT — Anthropic accepts it."""
    if isinstance(schema, dict):
        items = cast("dict[str, Any]", schema).items()
        return {key: _strip_titles(value) for key, value in items if key != "title"}
    if isinstance(schema, list):
        return [_strip_titles(item) for item in cast("list[Any]", schema)]
    return schema


def build_request_body(
    *,
    model: str,
    max_tokens: int,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    system: str | None = None,
    thinking: dict[str, Any] | None = None,
    tool_choice: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The Anthropic Messages request body — shared by the adapter and the Unit 2b smoke so there
    is ONE request shape (no drift). Core keys are model / max_tokens / messages / tools; `system`,
    `thinking`, and `tool_choice` are OPTIONAL and only added when passed, so the Unit 2b smoke
    (passing none) keeps its captured shape and its guard (tests/test_smoke_offline.py) stays green.
    The always-omitted guards hold by construction, on Sonnet 5 and Sonnet 5.5 alike: never
    `temperature`/`top_p`/`top_k` (→ 400 at any value under `anthropic-version: 2026-10-21`), never
    MANUAL thinking `{type:"enabled",budget_tokens}` (→ 400), and `messages` is passed through
    unchanged (no assistant prefill → 400). The thinking-off value is per model (`thinking_floor`):
    `{type:"disabled"}` on Sonnet 5; Sonnet 5.5 400s it, its lowest is `{type:"between_tools"}`."""
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "tools": tools,
    }
    if system is not None:
        body["system"] = system
    if thinking is not None:
        body["thinking"] = thinking
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    return body


def build_tools(registry: ToolRegistry) -> list[dict[str, Any]]:
    """The Anthropic `tools` array: every registered tool plus the two reserved terminal tools.
    Each `input_schema` is DERIVED from a typed signature — the registry tool's args-model, or the
    terminal decision's Pydantic model — never hand-written (ADR §What point 1), then `title`-
    stripped (`_strip_titles`) for a clean provider-neutral schema. The terminal tools have no
    registry handler; the model calls them to signal completion (the D1 commitment)."""
    tools: list[dict[str, Any]] = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": _strip_titles(tool.json_schema),
        }
        for tool in registry.registered()
    ]
    tools.extend(
        {
            "name": name,
            "description": description,
            "input_schema": _strip_titles(model.model_json_schema()),
        }
        for name, model, description in _TERMINAL_TOOLS
    )
    return tools


def _seed_directive(territory: str) -> str:
    """The opening user turn: name the territory and the tools it can call."""
    return (
        f"Hunt {territory} for AI-engineering job postings. Call sniff to investigate; "
        "call hunt_complete with your catch when done, or needs_help if you are stuck."
    )


def project_messages(territory: str, context: Sequence[TrajectoryEvent]) -> list[dict[str, Any]]:
    """Fold the trajectory log into a valid Anthropic `messages` array (P5 Unit 2c). This is a
    PROJECTION of the log (invariant 2), derived per brain call — not a second stored chat history.
    The seed directive opens it; then each event maps to its conversational turn:

      - ToolCallEvent    -> the assistant turn. Since W2.3 (D7) the event stores the whole turn
        (`assistant_content`, the response's `content` array) and it is echoed EXACTLY as stored,
        every block in order: thinking and redacted_thinking blocks, text, the tool_use with its
        `caller` and the model's own `input`. An event written before W2.3 gets the old shape: its
        signed thinking block (if any), then a `tool_use` rebuilt from id/name/`raw_request`;
      - ToolResultEvent  -> a user turn with the matching `tool_result` (paired by `tool_use_id`);
      - a FATAL tool ErrorEvent -> an `is_error` tool_result paired to the open tool_use (ErrorEvent
        carries no id, so we pair it with the last-opened call).

    Skipped (not conversation): retryable ErrorEvents (a retry follows, superseded by its
    ToolResultEvent), `<brain>`/`<loop>` failures (transport/loop errors, not turns), and
    SniffEvent / PreyCapturedEvent / UsageEvent / ThinkingDelta (the stored turn, not its partial
    deltas, is what goes back).
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": _seed_directive(territory)}]
    open_tool_use_id = ""  # the id of the tool_use awaiting its result (for id-less ErrorEvents)
    for event in context:
        if isinstance(event, ToolCallEvent) and event.assistant_content:
            # The whole turn, echoed exactly as the API returned it (W2.3, D7, fork E1): nothing is
            # rebuilt, so the conversation stays append-only. A thinking block that is reordered,
            # edited or dropped is a 400, and on models with preserved thinking the prefix check
            # compares every earlier message, the tool_use included.
            turn: list[dict[str, Any]] = json.loads(event.assistant_content)
            messages.append({"role": "assistant", "content": turn})
            open_tool_use_id = event.tool_use_id
        elif isinstance(event, ToolCallEvent):
            content: list[dict[str, Any]] = []
            if event.thinking:
                # Lead with the model's VERBATIM signed thinking block (`P5` Unit 3c, invariant 6):
                # echo the exact bytes captured off the stream — NEVER rebuild. The signature is
                # bound to the block's content, so a re-serialised/normalised block would 400
                # ("thinking blocks cannot be modified") and abort the reconstructed turn.
                content.append(json.loads(event.thinking))
            content.append(
                {
                    "type": "tool_use",
                    "id": event.tool_use_id,
                    "name": event.tool,
                    "input": json.loads(event.raw_request),
                }
            )
            messages.append({"role": "assistant", "content": content})
            open_tool_use_id = event.tool_use_id
        elif isinstance(event, ToolResultEvent):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": event.tool_use_id,
                            "content": event.raw_response.decode(),
                        }
                    ],
                }
            )
            open_tool_use_id = ""
        elif isinstance(event, ErrorEvent):
            if event.retryable or event.tool in ("<brain>", "<loop>"):
                continue  # a retry follows / not a conversational turn
            # A fatal TOOL error — only reachable for a tool that already emitted a ToolCallEvent,
            # so open_tool_use_id is its id. (unknown-tool / bad-args errors abort the run before
            # any re-projection, so this branch never pairs against a stale/empty id.)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": open_tool_use_id,
                            "content": event.error,
                            "is_error": True,
                        }
                    ],
                }
            )
            open_tool_use_id = ""
    return messages


# The Messages API's error types and the HTTP status each stands for (W3.1, fork Z1), so an SSE
# `error` frame on a 200 stream gets the verdict its status would: "an `overloaded_error`, which
# would normally correspond to an HTTP 529 in a non-streaming context". The 11 rows of "HTTP
# errors", in the page's order. The mapping is not one-to-one on the HTTP side (the API "might also
# use" `invalid_request_error` "for other 4XX status codes", and returns `timeout_error` "with
# other status codes, such as a 500"), and "new `type` values can appear": a type with no row
# fails closed (fatal, status None) until it gets one. Read 15 Nov 2026:
#   https://platform.claude.com/docs/en/api/errors
#   https://platform.claude.com/docs/en/build-with-claude/streaming (#error-events)
STREAM_ERROR_STATUS: dict[str, int] = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "conflict_error": 409,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "timeout_error": 504,
    "overloaded_error": 529,
}


def _retryable_status(code: int) -> bool:
    """The taxonomy's rule for a status, kept in one place: a 429 or a 5xx is transient (retried
    within budget), any other status fatal. `brain_call_error` and the stream's error frame both
    apply it."""
    return code == 429 or 500 <= code < 600


def _stream_error(
    frame: dict[str, Any], data: str, *, usage: UsageEvent | None, note: str
) -> BrainCallError:
    """The verdict on a stream that carried an SSE `error` frame (W3.1, Z1): the status its type
    stands for (`STREAM_ERROR_STATUS`, None for a type with no row), retryable by the status rule,
    and the frame's own `data` text as the raw response (invariant 6). `usage` and `note` are what
    the stream reported before the frame (W3.2-U2a, `_refused_usage`)."""
    error = frame.get("error")
    error_type = cast("dict[str, Any]", error).get("type") if isinstance(error, dict) else None
    status = STREAM_ERROR_STATUS.get(error_type) if isinstance(error_type, str) else None
    return BrainCallError(
        f"stream error frame: {error_type!r}{note}",
        retryable=status is not None and _retryable_status(status),
        status=status,
        raw_response=data.encode(),
        usage=usage,
    )


async def _drive_stream(lines: AsyncIterator[str], sink: ThinkingSink) -> bytes:
    """Consume a streaming Messages response: relay each thinking delta via `sink` (write-ahead,
    live — Unit 3b), assemble the blocks, and return the assembled response bytes for
    `parse_decision`. `lines` is the provider's raw SSE body, line by line (since W3.3 the adapter
    frames them with `_received_lines` from the SDK's raw-mode `iter_bytes`, keeping the bytes, with
    the framing `iter_lines` had; W2.0 to W3.2 passed `iter_lines`); a non-2xx never gets here, the
    SDK raises it as a status error first. The `data:` framing mirrors `iter_sse_events` over the
    live line iterator; the assembler (`feed`) is the shared logic. An SSE `error` frame ends the
    read and raises a BrainCallError with the verdict its type stands for and the frame's own bytes
    (W3.1, Z1), before any other check. Only a `finished` stream is returned (W3.1, T3): a body
    that ends on a frame boundary before `message_stop` raises a retryable BrainCallError, and a
    body that reached `message_stop` but is not whole raises a BrainParseError, each carrying the
    assembled bytes; so does a whole stream whose closed tool_use input does not parse as JSON
    (W3.1, P1). Each of these errors also carries the usage the stream reported before it ended
    (W3.2-U2a), for the loop to count. (A body cut inside a `data:` line still fails `json.loads`
    and reaches the loop's `<loop>` backstop: O1, not in W3.1; a transport failure mid-stream
    carries no usage, W3.2-U2b.)"""
    assembler = StreamAssembler()
    data_buf: list[str] = []
    error_data: str | None = None  # the first SSE error frame's own `data` text (W3.1, Z1)

    def fold(data: str) -> str | None:
        # One frame path for the in-loop and the final frame, so an error frame is seen on both.
        nonlocal error_data
        text = assembler.feed(json.loads(data))
        if error_data is None and assembler.error_frame is not None:
            error_data = data
        return text

    async for line in lines:
        if line.startswith("data:"):
            data_buf.append(line.removeprefix("data:").strip())
        elif not line.strip():  # a blank line closes the frame
            if data_buf:
                text = fold("".join(data_buf))
                data_buf = []
                if error_data is not None:
                    # Z-stop: the first error frame ends the read. Nothing after it can make the
                    # turn usable, and what follows one is undocumented; reading on would let a
                    # later transport failure replace the frame's verdict. It holds where `lines`
                    # is the live body: a transport that buffers the whole response first (the
                    # smoke harness's CapturingTransport) has finished its read before this runs.
                    break
                if text:
                    await sink(text)  # inline await — single writer (inv 7), live (inv 1)
    if data_buf:  # a final frame not terminated by a trailing blank line
        fold("".join(data_buf))
    assembled = assembler.assembled()
    raw = json.dumps(assembled).encode()
    # What the stream reported it spent before it ended (W3.2-U2a): `message_start`'s counts, as
    # updated by any `message_delta`. Each refusal below carries it, so the loop counts the refused
    # 200 (it can still be billed) before its ErrorEvent; `_refused_usage` never raises.
    usage, note = _refused_usage(assembled["model"], assembled["usage"])
    # An SSE error frame, wherever it sat, is the call's verdict (W3.1, Z1): it explains why the
    # stream is short, so it is checked before T.
    if error_data is not None and assembler.error_frame is not None:
        raise _stream_error(assembler.error_frame, error_data, usage=usage, note=note)
    # A stream is a turn only when it is whole (W3.1, T3). The body ended before `message_stop`:
    # the connection was cut on the way, the class a reset mid-stream is, so the loop retries it
    # (unless a spend ceiling stops the run first, W3.2-C1).
    if not assembler.ended:
        raise BrainCallError(
            f"stream cut before message_stop (stop_reason={assembler.stop_reason!r}, open "
            f"blocks {assembler.open_blocks}){note}",
            retryable=True,
            raw_response=raw,
            usage=usage,
        )
    # `message_stop` arrived, so the provider sent the whole body, yet a block was left open or no
    # stop reason came: a malformed response (fatal, ADR Pillar 2 taxonomy), not a blip to retry.
    if not assembler.finished:
        raise BrainParseError(
            raw=raw,
            detail=(
                f"stream ended without a whole turn: open blocks {assembler.open_blocks}, "
                f"stop_reason={assembler.stop_reason!r}{note}"
            ),
            usage=usage,
        )
    # A whole stream whose closed tool_use input does not parse (W3.1, P1): a `max_tokens` cut can
    # end a parameter midway, or the response is malformed. Never run, never guessed at: fatal,
    # with the model's own text in `detail` (the block in `raw` keeps its start input).
    if assembler.failed_inputs:
        raise BrainParseError(
            raw=raw,
            detail=(
                f"tool_use input does not parse as JSON at content_block_stop: "
                f"{assembler.failed_inputs!r}, "
                f"stop_reason={assembler.stop_reason!r}{note}"
            ),
            usage=usage,
        )
    return raw


async def _tee(chunks: AsyncIterator[bytes], received: list[bytes]) -> AsyncIterator[bytes]:
    """Pass the stream's body through unchanged, keeping every chunk as it arrives (W3.3, S1)."""
    async for chunk in chunks:
        received.append(chunk)
        yield chunk


def _received_lines(
    chunks: AsyncIterator[bytes], received: list[bytes], *, encoding: str
) -> AsyncIterator[str]:
    """The stream's body as lines for `_drive_stream`, framed from the chunks `iter_bytes` gives,
    each kept in `received` (W3.3, fork S1): the attempt stores the body as received, not a
    re-encoded assembly. The framing is httpx2's own: a local `httpx2.Response` over the tee runs
    the same TextDecoder (the response's encoding, `errors="replace"`) and LineDecoder that the
    SDK's `iter_lines` ran before W3.3, so the lines are unchanged (pinned against `aiter_lines` at
    every chunk size in tests/test_brain_exchange.py). The local response has no Content-Encoding,
    so nothing is decoded twice: `iter_bytes` already gave the content-decoded body."""
    framer = httpx2.Response(200, content=_tee(chunks, received), default_encoding=encoding)
    return framer.aiter_lines()


def brain_call_error(
    exc: anthropic.APIStatusError | anthropic.APIConnectionError | httpx2.TransportError,
    *,
    exchange: BrainExchangeEvent | None = None,
) -> BrainCallError:
    """Translate an SDK or transport failure into the loop's provider-neutral BrainCallError (W2.0
    D1a), so the loop never sees a vendor or transport type. The verdict is the taxonomy the loop
    always applied: a transport failure before the response (an `APIConnectionError` whose cause
    is an `httpx2.TransportError`; the adapter re-raises any other cause before calling this), a
    transport error while the body streams (the SDK does not wrap those, so it arrives as a bare
    `httpx2.TransportError`, timeouts included) and a 429 / 5xx status are transient; any other
    status is fatal (`_retryable_status`, the rule an SSE error frame's status gets too, W3.1).
    The error body rides along (invariant 6). The message names the real cause, not the SDK's
    generic wrapper text, as HEAD's ErrorEvent did. The stream checks in `_drive_stream` raise
    their BrainCallError directly; this translates only what the SDK and the transport raise. A
    status response's `retry-after` header rides along too (W3.2-B2, `_retry_after`), for the
    loop's backoff. `exchange` is the attempt's raw I/O (W3.3), passed through for the loop."""
    if isinstance(exc, anthropic.APIStatusError):
        code = exc.status_code
        return BrainCallError(
            repr(exc),
            retryable=_retryable_status(code),
            status=code,
            raw_response=exc.response.content,
            retry_after=_retry_after(exc.response.headers.get("retry-after")),
            exchange=exchange,
        )
    return BrainCallError(repr(exc.__cause__ or exc), retryable=True, exchange=exchange)


def _retry_after(value: str | None) -> float | None:
    """A `retry-after` header as seconds (W3.2-B2), or None: absent, or not a finite number of
    seconds that is not negative (an HTTP-date included), the header is ignored."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _is_transport_failure(exc: anthropic.APIConnectionError) -> bool:
    """The SDK wraps EVERY exception raised while it sends as an APIConnectionError: a refused
    connection or a timeout, but also a bug in a transport or a body that will not decode. Only the
    first kind is a transient blip; HEAD sent the others to the loop's `<loop>` backstop."""
    return isinstance(exc.__cause__, httpx2.TransportError)


def adapter_brain_for(
    *,
    client: anthropic.AsyncAnthropic,
    model: str,
    registry: ToolRegistry,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    thinking: dict[str, Any] | None = None,
    stream: bool = False,
) -> Callable[[str], Brain]:
    """A `brain_for` factory (the scheduler's seam, `scheduler.py`) backed by the Anthropic Messages
    API over the INJECTED SDK client (built by `anthropic_client`). `brain()` never opens or closes
    the client: its lifecycle belongs to whoever constructed it (the daemon/lifespan for a live
    call, a `MockTransport` in tests), the same single-owner discipline the run connections follow.

    Each turn (P5 Unit 2c): project the log into `messages` (`project_messages`, inv 2), thread the
    `HUNT_SYSTEM_PROMPT`. `thinking` defaults to the model's lowest setting, `thinking_floor(model)`
    (W2.2: `{type:"disabled"}` on Sonnet 5, `{type:"between_tools"}` on Sonnet 5.5; a model with no
    row raises here, before any request is built), 2c's thinking-off hunt (since W2.3 its echoed
    turn is the whole stored turn, e.g. text then the tool_use); `P5` Unit 3 passes
    `{type:"adaptive", display:"summarized"}` + `stream=True`. The SDK runs in
    RAW mode (W2.0, D9a): on the streaming path `with_streaming_response` hands the SSE body line by
    line to our assembler, which relays the deltas live via the `sink`; on the plain path
    `with_raw_response` hands over the response bytes. Either way the provider's own bytes cross the
    single boundary at `parse_decision` (invariant 3). SDK and transport errors become one
    BrainCallError the loop's `classify` sorts (429/5xx and transport retryable, other statuses
    fatal); the parsed `UsageEvent` rides onto the Decision (invariant 5). Since W3.3 every attempt
    that sent a request also reports its raw I/O (invariant 6, fork E2): a `BrainExchangeEvent` with
    the body httpx2 sent and the body as received (on the stream path every chunk `iter_bytes`
    gave, framed into lines by `_received_lines`), on the Decision, the BrainParseError or the
    BrainCallError, for the loop to append. SDK and transport errors are translated by the clauses
    below; anything else (a bug, a body that is not JSON: W3.1-O1 to O5) propagates without one.
    """
    tools = build_tools(registry)  # derived once — stable across the run
    # Fail closed at factory time (W2.2): a model with no floor row raises here, before any run.
    thinking = thinking_floor(model) if thinking is None else thinking

    def brain_for(territory: str) -> Brain:
        async def brain(context: Context, sink: ThinkingSink) -> Decision:
            body = build_request_body(
                model=model,
                max_tokens=max_tokens,
                messages=project_messages(territory, context),
                tools=tools,
                system=HUNT_SYSTEM_PROMPT,
                thinking=thinking,
                tool_choice=HUNT_TOOL_CHOICE,
            )
            # The attempt's raw I/O (W3.3, invariant 6): the request httpx2 sent (fork R1) and the
            # body as received (fork S1). The stream path keeps every chunk here, so a body cut by a
            # transport failure keeps what arrived before the cut.
            received: list[bytes] = []
            sent: bytes | None = None
            status: int | None = None
            try:
                if not stream:
                    # Cast, not inferred: `**body` matches several `create` overloads, so pyright
                    # cannot pick one. Passing `stream=False` would add it to the request JSON.
                    plain = cast(
                        AsyncAPIResponse[Message],
                        await client.messages.with_raw_response.create(**body),
                    )
                    raw = await plain.read()  # the provider's bytes, as they came
                    exchange = BrainExchangeEvent(
                        raw_request=plain.http_request.content,
                        raw_response=raw,
                        status=plain.status_code,
                    )
                else:
                    async with client.messages.with_streaming_response.create(
                        **body, stream=True
                    ) as streamed:
                        sent, status = streamed.http_request.content, streamed.status_code
                        lines = _received_lines(
                            streamed.iter_bytes(),
                            received,
                            encoding=streamed.http_response.encoding or "utf-8",
                        )
                        try:
                            raw = await _drive_stream(lines, sink)
                        except (BrainParseError, BrainCallError) as exc:
                            # a stream the guard refused (W3.1): its bytes as received ride on
                            # the error; `raw` / `raw_response` keep the assembled bytes
                            exc.exchange = BrainExchangeEvent(
                                raw_request=sent, raw_response=b"".join(received), status=status
                            )
                            raise
                    exchange = BrainExchangeEvent(
                        raw_request=sent, raw_response=b"".join(received), status=status
                    )
            except anthropic.APIConnectionError as exc:
                if not _is_transport_failure(exc):
                    # Not the network: re-raise the real error for the <loop> backstop, as at HEAD
                    # (from None: the SDK's wrapper says nothing the cause does not).
                    raise (exc.__cause__ or exc) from None
                # no response arrived; the SDK error carries the request it built (W3.3, R1)
                raise brain_call_error(
                    exc, exchange=BrainExchangeEvent(raw_request=exc.request.content)
                ) from exc
            except anthropic.APIStatusError as exc:
                raise brain_call_error(
                    exc,
                    exchange=BrainExchangeEvent(
                        raw_request=exc.request.content,
                        raw_response=exc.response.content,
                        status=exc.status_code,
                    ),
                ) from exc
            except httpx2.TransportError as exc:
                # a failure while the body streams (a reset mid-frame) is a transport blip too; the
                # attempt keeps the bytes received before it (W3.3; its usage is not counted,
                # W3.2-U2b). The plain path never gets here: the SDK reads its body in `create`.
                exchange = None
                if sent is not None:
                    exchange = BrainExchangeEvent(
                        raw_request=sent, raw_response=b"".join(received), status=status
                    )
                raise brain_call_error(exc, exchange=exchange) from exc
            # The one boundary (invariant 3). A ToolCallDecision carries the whole turn from here
            # on both paths (W2.3), so the next call echoes it exactly as returned. The attempt's
            # exchange rides on the decision, or on the boundary's refusal (W3.3, fork E2).
            try:
                decision = parse_decision(raw)
            except BrainParseError as exc:
                exc.exchange = exchange
                raise
            decision.exchange = exchange
            return decision

        return brain

    return brain_for


def select_brain_for(
    registry: ToolRegistry,
    *,
    default: Callable[[str], Brain] | None = None,
) -> tuple[Callable[[str], Brain], anthropic.AsyncAnthropic | None]:
    """The autonomous-spender containment (P5 Unit 2c): the daemon's `brain_for`, selected by the
    `REXHUNTER_BRAIN` env var. Default `"stub"` returns a no-spend brain and no client — a default
    start constructs nothing that could hit the network; `default` (the daemon's injectable stub
    seam, `daemon_config`'s brain) is returned when given, else the module `stub_brain_for`.
    `"live"` is the single opt-in that arms spending: it builds the STREAMING/THINKING adapter (`P5`
    Unit 3 — `stream=True` + adaptive thinking, so Rex's reasoning relays through the hub) over a
    real SDK client (`anthropic_client`, max_retries=0 + the pinned wire) and RETURNS that client so
    the caller (lifespan / entrypoints) owns its close.
    Any other value is a hard error, never a silent fall-through to spending.
    """
    mode = os.environ.get("REXHUNTER_BRAIN", "stub")
    if mode == "stub":
        if default is not None:
            return default, None
        from rexhunter import stub  # lazy: the stub path pulls in no adapter/client machinery

        return stub.stub_brain_for, None
    if mode == "live":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("REXHUNTER_BRAIN=live requires ANTHROPIC_API_KEY to be set")
        client = anthropic_client(api_key, timeout=120.0)
        brain_for = adapter_brain_for(
            client=client,
            model=SMOKE_MODEL,
            registry=registry,
            max_tokens=HUNT_MAX_TOKENS,
            thinking=HUNT_THINKING,
            stream=True,  # the daemon streams reasoning to /events — the Unit-3 brain, not 2c's
        )
        return brain_for, client
    raise ValueError(f"unknown REXHUNTER_BRAIN={mode!r} (expected 'stub' or 'live')")
