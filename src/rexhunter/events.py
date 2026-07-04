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

from enum import StrEnum
from typing import Annotated, Literal

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
    rebuildable from the log: this event is the capture half, VerdictEvent the lifecycle half."""

    type: Literal["prey_captured"] = "prey_captured"
    prey_id: str  # the pen row this event creates; makes the rebuild deterministic
    territory: str
    raw_posting: bytes


class ErrorEvent(_Event):
    """A tool attempt failed: raised, timed out, unknown name, or invalid args.

    `retryable` mirrors the loop's taxonomy (a retryable attempt is re-tried within budget; a
    fatal one ends the run). `raw_response` is optional - an unknown tool or a timeout never
    produced one. `detail` carries the traceback / classifier message; the raw payload makes a
    dead run a pytest fixture for free (invariant 6).
    """

    type: Literal["error"] = "error"
    tool: str
    retryable: bool
    error: str
    raw_request: bytes
    raw_response: bytes | None = None
    detail: str | None = None


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


# The trajectory-event union, now multi-member and discriminated by `type`: Pydantic routes a
# payload straight to the model whose Literal tag matches, and an unknown/absent tag raises at
# the boundary. Adding a member is one line here; the read crossing below never changes.
type TrajectoryEvent = Annotated[
    SniffEvent | ToolCallEvent | ToolResultEvent | ErrorEvent | PreyCapturedEvent | UsageEvent,
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
    (invariant 3), and this raw payload does not validate into the decision union. Carries the
    raw bytes (invariant 6) so the loop can attach them to an ErrorEvent - a malformed response
    is diagnosable from the ghost replay, never a mystery string. Raised by brain.parse_decision;
    caught by run_hunt, which records it and ends the run in a typed outcome."""

    def __init__(self, raw: bytes, detail: str) -> None:
        super().__init__(detail)
        self.raw = raw
        self.detail = detail


# ── The pen log: verdict events (ADR pillar 4) ────────────────────────────────
# A second log, NOT the trajectory union: a verdict arrives after the run has exited, from the
# POST handler, so it is not a run-scoped trajectory event (invariant 7). It crosses the SAME
# validation boundary (invariant 3) on read, via decode_verdict_event below.


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
