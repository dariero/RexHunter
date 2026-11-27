"""Typed trajectory-event model + the validation boundary (invariant 3).

Stage 2 replaces Stage 1's raw-string payload with a typed Pydantic v2 event. The
boundary has two halves: the write side is a compile-time guarantee (you cannot pass a
raw string to db.append_event), and the read side is the runtime crossing below
(decode_event) - raw bytes from a durable, possibly stale or hand-edited log are
rejected here, never surfaced as an untyped string. LLM output (Stage 4) crosses the
same line and earns the same suspicion.

`P2.2` grows the union from one member to four: the loop emits ToolCallEvent /
ToolResultEvent / ErrorEvent as it dispatches tools, so the discriminated union promised
in `P2.1` goes live. Tool/error events carry the raw request and response *bytes* they
operated on (invariant 6) - which is why the shared base encodes bytes as base64.
"""

import json
from enum import StrEnum
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _Event(BaseModel):
    """Shared base so boundary strictness is structural, not per-model remembered.

    - extra="forbid": unknown fields are rejected at the boundary (invariant 3).
    - base64 bytes: raw I/O fields (invariant 6) are `bytes`; Pydantic's JSON default encodes
      bytes as UTF-8 and *raises* on non-UTF-8 content. base64 makes binary round-trip
      losslessly (a scraped page or a provider blob is not guaranteed to be valid UTF-8).
    """

    model_config = ConfigDict(
        extra="forbid",
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )


class SniffEvent(_Event):
    type: Literal["sniff"] = "sniff"  # discriminator; mirrors the DB `type` column
    prey: str  # the structured scent (the flavour string is projection, not stored)


class ToolCallEvent(_Event):
    """The dispatch: a tool is about to run. Written before execution (invariant 1)."""

    type: Literal["tool_call"] = "tool_call"
    tool: str
    raw_request: bytes  # the JSON-serialised validated args (invariant 6)
    tool_use_id: str = ""  # the provider's correlation key (P5); "" for stub/pre-P5 dispatches
    assistant_content: bytes = b""  # the WHOLE assistant turn (W2.3, D7, invariant 6): the JSON
    # bytes of the response's `content` array, every block in order (thinking, redacted_thinking,
    # text, the tool_use with its `caller` and the model's own `input`). project_messages echoes it
    # exactly as stored, never rebuilt: the docs say to send each assistant turn back exactly as the
    # API returned it, and a reordered, edited or dropped thinking block is a 400. Empty for
    # stub/scripted turns and every event written before W2.3.
    thinking: bytes = b""  # the pre-W2.3 form (`P5` Unit 3, invariant 6): the turn's FIRST signed
    # thinking block, `{type:"thinking", thinking, signature}`, kept on streaming turns only. Events
    # written since W2.3 leave it empty and store the whole turn in `assistant_content`; it is kept
    # so older logs and fixtures still decode and replay. project_messages reads it only when
    # `assistant_content` is empty: it leads the reconstructed turn — echoed, never rebuilt: the
    # signature is bound to the exact block content, so a rebuild would 400 ("thinking blocks cannot
    # be modified"). Empty for stub/scripted turns and every pre-Unit-3 event/fixture.


class ToolResultEvent(_Event):
    """A tool returned. Carries the raw request it ran on + the raw response (invariant 6)."""

    type: Literal["tool_result"] = "tool_result"
    tool: str
    raw_request: bytes
    raw_response: bytes
    tool_use_id: str = ""  # mirrors the ToolCallEvent's key -> the log's pairing key (P5)


class PreyCapturedEvent(_Event):
    """A hunt caught a posting worth a human verdict. Run-scoped (the owning hunt writes it,
    invariant 7); carries the raw posting bytes (invariant 6). The `prey` projection row is
    inserted in the SAME transaction as this event (see verdicts.capture_prey), so the pen is
    rebuildable from the log: this event is the capture half, VerdictEvent the lifecycle half.

    `board` and `posting_id` are the reference the catch named; `raw_posting` is then the board's
    own bytes for it, the response of the run's read of that posting. Both are "" on a capture
    from before catch by reference, so every older log still decodes."""

    type: Literal["prey_captured"] = "prey_captured"
    prey_id: str  # the pen row this event creates; makes the rebuild deterministic
    territory: str
    raw_posting: bytes
    board: str = ""
    posting_id: str = ""

    @property
    def label(self) -> str:
        """The pen card's text, derived from the stored bytes, never stored apart from them
        (invariant 5): "title \N{EN DASH} location (board)". A capture from before catch by
        reference (`board == ""`), or one whose bytes carry no title, shows its bytes decoded,
        as a capture did before."""
        title = _posting_title(self.raw_posting) if self.board else None
        if title is None:
            return self.raw_posting.decode(errors="replace")
        return f"{title} ({self.board})"


def _text_at(body: object, *path: str) -> str | None:
    for key in path:
        if not isinstance(body, dict):
            return None
        body = cast("dict[str, object]", body).get(key)
    return body if isinstance(body, str) else None


def _posting_title(raw_posting: bytes) -> str | None:
    # The title, then the location after an en dash, from either documented board shape
    # (Greenhouse: `title`, `location.name`; Lever: `text`, `categories.location`), or None when
    # the bytes carry no title. Never raises: the pen renders whatever bytes it holds.
    try:
        body: object = json.loads(raw_posting.decode("utf-8-sig"))
    except ValueError, RecursionError:
        return None
    title = _text_at(body, "title") or _text_at(body, "text")
    if title is None:
        return None
    location = _text_at(body, "location", "name") or _text_at(body, "categories", "location")
    return f"{title} \N{EN DASH} {location}" if location else title


class ErrorEvent(_Event):
    """A tool attempt failed: raised, timed out, unknown name, or invalid args.

    `retryable` mirrors the loop's taxonomy (a retryable attempt is re-tried within budget, unless
    a spend ceiling stops the run first, W3.2-C1; a fatal one ends the run). `raw_response` is
    optional - an unknown tool or a timeout never produced one. `detail` carries the traceback /
    classifier message; the raw payload makes a dead run a pytest fixture for free (invariant 6).
    """

    type: Literal["error"] = "error"
    tool: str
    retryable: bool
    error: str
    raw_request: bytes
    raw_response: bytes | None = None
    detail: str | None = None


class ThinkingDelta(_Event):
    """A streamed fragment of the model's reasoning (ADR pillar 5, `P5` Unit 3). The live
    consciousness feed: appended write-ahead (invariant 1) as it arrives off the SSE stream, then
    broadcast by the P3 hub — rendering Rex's chain of thought as the hunt runs.

    Relay-only telemetry — NOT part of the messages projection (like UsageEvent, `project_messages`
    skips it). A reconstructed assistant turn must echo the model's VERBATIM signed thinking block
    (invariant 6), never a rebuild folded from these partial deltas: the signature is bound to the
    exact block content, so a fold would drift the bytes and the API would 400 ("thinking blocks
    cannot be modified"). The durable replay artifact is the whole turn on ToolCallEvent
    (`assistant_content`, W2.3; the single signed block in `thinking` before it); these deltas are
    the live feed — a separate consumer of the same log (the two-cursor design)."""

    type: Literal["thinking_delta"] = "thinking_delta"
    text: str


def _is_zero(count: int) -> bool:
    return count == 0


class UsageEvent(_Event):
    """Per-brain-call token accounting (ADR pillar 5, `P5`). Run-scoped telemetry (invariant 7)
    the cost breaker folds into a running spend (invariant 5 — cost is DERIVED from these, never a
    stored counter). NOT part of the messages projection: it is cost metadata, not conversation, so
    the projection (brain.py) skips it. `model` names the priced model; the fold lives in cost.py.
    """

    type: Literal["usage"] = "usage"
    model: str
    input_tokens: int
    output_tokens: int
    # The tokens written to and read from the prompt cache (`P5.1`), which the response counts apart
    # from `input_tokens`. At 0 a field is left out of the stored payload, so a call that used no
    # cache stores the bytes it stored before `P5.1`, and the recorded logs' digests hold.
    cache_creation_input_tokens: int = Field(default=0, exclude_if=_is_zero)
    cache_read_input_tokens: int = Field(default=0, exclude_if=_is_zero)


class BrainExchangeEvent(_Event):
    """One brain attempt's raw I/O (W3.3, fragility #3, invariant 6): the request bytes it sent
    and the response bytes it got, so a dead run is a pytest fixture from its own log (ADR:30) and
    rebuilds offline to identical events (DoD #5). Written by the owning task before the attempt's
    other events (its ThinkingDeltas, written during the call, come first), only when the brain
    reports bytes: the adapter attaches one to what it returns or raises (fork E2), so the stub and
    scripted brains write none. Never published to the hub, and left out of the SSE backfill and
    the projection's rows (fork X2): audit data, up to a few MB per run, that the UI never shows.

    - `raw_request`: the body httpx2 put on the wire (`http_request.content`, or the SDK error's
      `request`; fork R1), not a re-serialisation of the request dict.
    - `raw_response`: the body as received, content-decoded (gzip and the like undone by httpx2):
      `read()` on the plain path, every chunk `iter_bytes` gave on the stream path, joined (fork
      S1), so a stream that was cut keeps what arrived before the cut. None when no response
      arrived (a connection failure).
    - `status`: the HTTP status the response came with (200 for a stream that then failed), None
      when none arrived, so a non-200 attempt (a 429 or 529 retry) replays as one."""

    type: Literal["brain_exchange"] = "brain_exchange"
    raw_request: bytes
    raw_response: bytes | None = None
    status: int | None = None


# The trajectory-event union, now multi-member and discriminated by `type`: Pydantic routes a
# payload straight to the model whose Literal tag matches, and an unknown/absent tag raises at
# the boundary. Adding a member is one line here; the read crossing below never changes.
type TrajectoryEvent = Annotated[
    SniffEvent
    | ToolCallEvent
    | ToolResultEvent
    | ErrorEvent
    | PreyCapturedEvent
    | UsageEvent
    | ThinkingDelta
    | BrainExchangeEvent,
    Field(discriminator="type"),
]

# Explicit generic: pyright cannot infer T from a PEP 695 alias passed by value, so we
# state it. Pydantic resolves the alias at runtime regardless.
_EVENT_ADAPTER: TypeAdapter[TrajectoryEvent] = TypeAdapter(TrajectoryEvent)


def decode_event(payload: str) -> TrajectoryEvent:
    """Invariant-3 read crossing: raw bytes -> typed, or ValidationError. The ONE line."""
    return _EVENT_ADAPTER.validate_json(payload)


class BrainParseError(Exception):
    """The provider->Decision boundary (P5) rejected a payload: LLM output is untrusted input
    (invariant 3), and this raw payload is not a turn the loop may act on: it does not validate
    into the decision union, or (W3.1) a streamed response ended without a whole turn or closed
    a tool_use whose input does not parse as JSON. Carries the raw bytes (invariant 6) so the
    loop can attach them to an ErrorEvent - a malformed response is diagnosable from the ghost
    replay, never a mystery string. Raised by brain.parse_decision and by brain's stream reader
    (`_drive_stream`); caught by run_hunt, which records it and ends the run in a typed
    outcome. `usage` (W3.2-U2a) is the response's own token count when the refusal came after the
    envelope validated and the count reads cleanly: a refused 200 can still be billed, so run_hunt
    appends it as a UsageEvent before the ErrorEvent (invariants 1 and 5), the safe direction for a
    guard. None otherwise: no count, a count that does not read, or a body that fails the
    envelope. `exchange` (W3.3) is the attempt's raw request and response, which the adapter
    attaches and run_hunt appends before the usage (invariant 6); `raw` stays what it was (the
    assembled bytes on the stream path). None from a brain that reports no bytes."""

    def __init__(
        self,
        raw: bytes,
        detail: str,
        *,
        usage: UsageEvent | None = None,
        exchange: BrainExchangeEvent | None = None,
    ) -> None:
        super().__init__(detail)
        self.raw = raw
        self.detail = detail
        self.usage = usage
        self.exchange = exchange


# ── The pen log: verdict events (ADR pillar 4) ────────────────────────────────
# A second log, NOT the trajectory union: a verdict arrives after the run has exited, from the
# POST handler, so it is not a run-scoped trajectory event (invariant 7). It crosses the SAME
# validation boundary (invariant 3) on read, via decode_verdict_event below.


class PostingRef(BaseModel):
    """One posting on one board: the arguments a posting-source tool (`read_posting`) took. A
    catch names its postings this way, and capture resolves each reference against the run's own
    posting-source results, so only a posting the run read reaches the pen. Frozen, so equal
    references are one key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    board: str
    posting_id: str


class Verdict(StrEnum):
    """The human verdicts on a penned posting. FEAST/RELEASE/AMBER act on an awaiting row;
    REENTER returns an ambered row to the pen. The string value is what the log stores."""

    FEAST = "feast"
    RELEASE = "release"
    AMBER = "amber"
    REENTER = "reenter"


class VerdictEvent(_Event):
    """A human verdict, appended to pen_events. `prey.status`/`reason`/`provenance` are the
    projection of these (invariant 2); the kind is the transition applied. RELEASE carries the
    rejection reason (labelled data); AMBER carries provenance."""

    type: Literal["verdict"] = "verdict"
    prey_id: str
    verdict: Verdict
    reason: str | None = None
    provenance: str | None = None


def decode_verdict_event(payload: str) -> VerdictEvent:
    """Invariant-3 read crossing for the pen log — the sibling of decode_event for the trajectory
    log. Raw bytes from a durable, possibly stale row -> typed VerdictEvent, or ValidationError."""
    return VerdictEvent.model_validate_json(payload)
