"""The brain socket — Pillar 5. Unit 1: the edge boundary (offline, no vendor SDK).

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
so the loop can stamp it on both the call and result events (the provider's correlation key
doubles as the log's pairing key).

This module imports NO vendor SDK and opens NO network connection. The provider adapter (where
the SDK lives and the first paid call happens), streaming, and budget accounting are later `P5`
units — none of them gate DoD #5.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from rexhunter.events import BrainParseError
from rexhunter.loop import Decision, HuntComplete, NeedsHelp, ToolCallDecision

# Reserved tool names that map to terminal decisions rather than a registry dispatch. Unit 2's
# adapter must present these to the model as tools (a cross-unit commitment, recorded here).
HUNT_COMPLETE = "hunt_complete"
NEEDS_HELP = "needs_help"

_TOOL_USE = "tool_use"  # the Anthropic content-block type we act on


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
    without letting them reject the payload."""

    model_config = ConfigDict(extra="ignore")

    stop_reason: str | None = None
    content: list[dict[str, Any]]


def parse_decision(raw: bytes) -> Decision:
    """Raw provider bytes -> typed `Decision`, or `BrainParseError` (invariants 3 + 6). Pure: no
    clock, no randomness, no network — the same bytes always yield the same decision."""
    try:
        response = _ProviderResponse.model_validate_json(raw)
    except ValidationError as exc:
        raise BrainParseError(raw=raw, detail=str(exc)) from exc

    block_raw = next((b for b in response.content if b.get("type") == _TOOL_USE), None)
    if block_raw is None:
        raise BrainParseError(
            raw=raw,
            detail="no tool_use block: the loop acts on native tool calls, never on free text",
        )
    try:
        block = _ToolUseBlock.model_validate(block_raw)
    except ValidationError as exc:
        raise BrainParseError(raw=raw, detail=str(exc)) from exc

    # Reserved names are terminal decisions (they have no registry handler, so this parser is
    # their only boundary — hence `catch` IS validated here, unlike regular-tool args).
    if block.name == HUNT_COMPLETE:
        try:
            return HuntComplete.model_validate(block.input)
        except ValidationError as exc:
            raise BrainParseError(raw=raw, detail=str(exc)) from exc
    if block.name == NEEDS_HELP:
        return NeedsHelp()
    return ToolCallDecision(tool=block.name, args=block.input, tool_use_id=block.id)
