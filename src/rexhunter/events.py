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


class ToolResultEvent(_Event):
    """A tool returned. Carries the raw request it ran on + the raw response (invariant 6)."""

    type: Literal["tool_result"] = "tool_result"
    tool: str
    raw_request: bytes
    raw_response: bytes


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


# The trajectory-event union, now multi-member and discriminated by `type`: Pydantic routes a
# payload straight to the model whose Literal tag matches, and an unknown/absent tag raises at
# the boundary. Adding member #5 is one line here; the read crossing below never changes.
type TrajectoryEvent = Annotated[
    SniffEvent | ToolCallEvent | ToolResultEvent | ErrorEvent,
    Field(discriminator="type"),
]

# Explicit generic: pyright cannot infer T from a PEP 695 alias passed by value, so we
# state it. Pydantic resolves the alias at runtime regardless.
_EVENT_ADAPTER: TypeAdapter[TrajectoryEvent] = TypeAdapter(TrajectoryEvent)


def decode_event(payload: str) -> TrajectoryEvent:
    """Invariant-3 read crossing: raw bytes -> typed, or ValidationError. The ONE line."""
    return _EVENT_ADAPTER.validate_json(payload)
