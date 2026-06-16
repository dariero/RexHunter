"""Typed trajectory-event model + the validation boundary (invariant 3).

Stage 2 replaces Stage 1's raw-string payload with a typed Pydantic v2 event. The
boundary has two halves: the write side is a compile-time guarantee (you cannot pass a
raw string to db.append_event), and the read side is the runtime crossing below
(decode_event) - raw bytes from a durable, possibly stale or hand-edited log are
rejected here, never surfaced as an untyped string. LLM output (Stage 4) crosses the
same line and earns the same suspicion.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter


class SniffEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown fields rejected at the boundary
    type: Literal["sniff"] = "sniff"  # discriminator; mirrors the DB `type` column
    prey: str  # the structured scent (the flavour string is projection, not stored)


# The trajectory-event union. ONE member today, written union-shaped so member #2 costs a
# single line:
#   type TrajectoryEvent = Annotated[SniffEvent | ToolCallEvent, Field(discriminator="type")]
# Pydantic v2 cannot form a *discriminated* union from one member, so the discriminator is
# latent today - the `type: Literal["sniff"]` field alone enforces the tag (an unknown
# `type` value raises). Nothing below changes when the union grows; the crossing is written
# once, only this alias does.
type TrajectoryEvent = SniffEvent

# Explicit generic: pyright cannot infer T from a PEP 695 alias passed by value, so we
# state it. Pydantic resolves the alias at runtime regardless.
_EVENT_ADAPTER: TypeAdapter[TrajectoryEvent] = TypeAdapter(TrajectoryEvent)


def decode_event(payload: str) -> TrajectoryEvent:
    """Invariant-3 read crossing: raw bytes -> typed, or ValidationError. The ONE line."""
    return _EVENT_ADAPTER.validate_json(payload)
