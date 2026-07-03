"""The brain socket — Pillar 5. Unit 1: the edge boundary. Unit 2a: the provider adapter.

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
module opens no connection of its own; a `MockTransport` gives the offline tests zero network. The
live smoke call is Unit 2b (paid, cost-quoted first); streaming and budget accounting are later
units.
"""

from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from rexhunter.events import BrainParseError
from rexhunter.loop import Brain, Context, Decision, HuntComplete, NeedsHelp, ToolCallDecision
from rexhunter.tools import ToolRegistry

# Reserved tool names that map to terminal decisions rather than a registry dispatch. Unit 2a's
# adapter presents these to the model as tools (the D1 cross-unit commitment, honoured here).
HUNT_COMPLETE = "hunt_complete"
NEEDS_HELP = "needs_help"

_TOOL_USE = "tool_use"  # the Anthropic content-block type we act on

# The Anthropic Messages API surface (from the claude-api reference). The adapter targets the full
# URL rather than the client's base_url, so the injected client needs no configuration.
_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 1024

# Reserved for the Unit 2b paid smoke call; UNUSED in this offline unit (set now so 2b inherits
# it). A cheap tier is appropriate for a single one-call smoke test.
SMOKE_MODEL = "claude-haiku-4-5-20251001"


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

    # Exactly one actionable block. We count `tool_use`-typed blocks only, so a `thinking` (or
    # other) sibling doesn't count against the total — zero means nothing to act on, more than one
    # is ambiguous for a one-tool-per-iteration loop; neither is a silent first-wins.
    tool_use_blocks = [b for b in response.content if b.get("type") == _TOOL_USE]
    if len(tool_use_blocks) != 1:
        raise BrainParseError(
            raw=raw,
            detail=(
                f"expected exactly one tool_use block, found {len(tool_use_blocks)}: the loop acts "
                "on one native tool call per iteration, never on free text or a batch"
            ),
        )
    try:
        block = _ToolUseBlock.model_validate(tool_use_blocks[0])
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


# ── The provider adapter (Unit 2a): httpx, no vendor SDK ─────────────────────

# Concise, model-facing descriptions for the reserved terminal tools. These are prose (how the
# model should use the signal), NOT the schema — the schema is still DERIVED from the typed model
# (`model_json_schema()`), so ADR §What point 1 ("never hand-written schemas") holds.
_TERMINAL_TOOLS: tuple[tuple[str, type[BaseModel], str], ...] = (
    (HUNT_COMPLETE, HuntComplete, "End the hunt. `catch` is the list of postings worth a verdict."),
    (NEEDS_HELP, NeedsHelp, "End the hunt and hand off to a human — Rex is stuck."),
)


def build_tools(registry: ToolRegistry) -> list[dict[str, Any]]:
    """The Anthropic `tools` array: every registered tool plus the two reserved terminal tools.
    Each `input_schema` is DERIVED from a typed signature — the registry tool's args-model, or the
    terminal decision's Pydantic model — never hand-written (ADR §What point 1). The terminal tools
    have no registry handler; the model calls them to signal completion (the D1 commitment)."""
    tools: list[dict[str, Any]] = [
        {"name": tool.name, "description": tool.description, "input_schema": tool.json_schema}
        for tool in registry.registered()
    ]
    tools.extend(
        {"name": name, "description": description, "input_schema": model.model_json_schema()}
        for name, model, description in _TERMINAL_TOOLS
    )
    return tools


def _seed_messages(territory: str) -> list[dict[str, Any]]:
    """A minimal valid `messages` array — one seed user turn per territory. Full multi-turn
    threading of prior ToolCallEvent/ToolResultEvent (paired by `tool_use_id`) is DEFERRED to a
    later unit (the loop flags this same seam, loop.py:220 "real assembly is P5"); the offline
    tests inject a transport that ignores the body, and `max_iterations` bounds any repeat-the-seed
    behaviour, so this does not block the Unit 2b smoke call."""
    return [
        {
            "role": "user",
            "content": (
                f"Hunt {territory} for AI-engineering job postings. Call sniff to investigate; "
                "call hunt_complete with your catch when done, or needs_help if you are stuck."
            ),
        }
    ]


def adapter_brain_for(
    *,
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    registry: ToolRegistry,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> Callable[[str], Brain]:
    """A `brain_for` factory (the scheduler's seam, `scheduler.py`) backed by the Anthropic Messages
    API over the INJECTED httpx client. `brain()` never opens or closes the client — its lifecycle
    belongs to whoever constructed it (the daemon/lifespan for a live call, a `MockTransport` in
    tests), the same single-owner discipline the run connections follow.

    Deferred, each named (not silent), each a later unit:
      - `system` prompt — the model isn't yet told sniff=investigate / hunt_complete=terminate.
      - non-200 handling — the raw error body flows straight into `parse_decision` and becomes a
        `BrainParseError` carrying it (invariant 6); retryable-vs-fatal HTTP classification (429 /
        5xx) is a later unit. There is no `raise_for_status`: the response bytes cross one boundary,
        at `parse_decision` (the user's spec + invariant 3).
    """
    tools = build_tools(registry)  # derived once — stable across the run
    headers = {"x-api-key": api_key, "anthropic-version": _ANTHROPIC_VERSION}

    def brain_for(territory: str) -> Brain:
        async def brain(_context: Context) -> Decision:
            body: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": _seed_messages(territory),
                "tools": tools,
            }
            response = await client.post(_MESSAGES_URL, headers=headers, json=body)
            return parse_decision(response.content)  # one boundary (invariant 3)

        return brain

    return brain_for
