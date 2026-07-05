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
"""

import json
import os
from collections.abc import Callable, Iterator, Sequence
from typing import Any, cast

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from rexhunter.events import (
    BrainParseError,
    ErrorEvent,
    ToolCallEvent,
    ToolResultEvent,
    TrajectoryEvent,
    UsageEvent,
)
from rexhunter.loop import (
    Brain,
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
_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 1024

# The model for the Unit 2b single live smoke call (the first paid call). Sonnet 5: adaptive
# thinking runs automatically (so the response carries `thinking` sibling blocks), and sampling
# params / manual thinking / assistant prefill are all rejected with a 400 — build_request_body
# holds those guards by construction.
SMOKE_MODEL = "claude-sonnet-5"

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
    model: str | None = None  # names the priced model (P5 Unit 2c cost accounting)
    usage: dict[str, Any] | None = None  # {input_tokens, output_tokens, …}; absent on hand fixtures


def _extract_usage(response: _ProviderResponse) -> UsageEvent | None:
    """Read the envelope's token counts into a typed `UsageEvent` (P5 Unit 2c). None when the
    payload carries no `usage` (hand-authored fixtures) — the loop then records no spend for that
    call. Reads the SAME validated envelope as the decision, not a second inv-3 crossing."""
    if response.usage is None:
        return None
    return UsageEvent(
        model=response.model or "",
        input_tokens=int(response.usage.get("input_tokens", 0)),
        output_tokens=int(response.usage.get("output_tokens", 0)),
    )


def parse_decision(raw: bytes) -> Decision:
    """Raw provider bytes -> typed `Decision`, or `BrainParseError` (invariants 3 + 6). Pure: no
    clock, no randomness, no network — the same bytes always yield the same decision. The parsed
    `UsageEvent` (token cost, P5 Unit 2c) rides onto the Decision so the loop can fold spend from
    the log (invariant 5) without a second crossing of the raw bytes."""
    try:
        response = _ProviderResponse.model_validate_json(raw)
    except ValidationError as exc:
        raise BrainParseError(raw=raw, detail=str(exc)) from exc

    usage = _extract_usage(response)

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
    decision: Decision
    if block.name == HUNT_COMPLETE:
        try:
            decision = HuntComplete.model_validate(block.input)
        except ValidationError as exc:
            raise BrainParseError(raw=raw, detail=str(exc)) from exc
    elif block.name == NEEDS_HELP:
        decision = NeedsHelp()
    else:
        decision = ToolCallDecision(tool=block.name, args=block.input, tool_use_id=block.id)
    decision.usage = usage  # every Decision carries the call's cost (None for a no-usage payload)
    return decision


# ── The streaming transport (Unit 3a): hand-rolled SSE over httpx, no vendor SDK ───────────────
# Sonnet 5 streams its response as SSE events. We assemble them into the SAME `_ProviderResponse`
# shape parse_decision consumes — no vendor SDK (ADR §What point 1), and the raw signed thinking
# block stays in hand (invariants 3 + 6), which the SDK's typed-delta layer would hide. P3's own
# outbound SSE parser is this same protocol inbound; the system is an SSE relay end to end.


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
    from its accumulated `input_json_delta` only at `content_block_stop`, so a partial stream leaves
    the block un-dispatchable (its start `input` — `{}` — stands until the block is complete).
    """

    def __init__(self) -> None:
        self._blocks: dict[int, dict[str, Any]] = {}
        self._json_buf: dict[int, str] = {}
        self._model: str | None = None
        self._stop_reason: str | None = None
        self._usage: dict[str, Any] = {}
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
            self._finalize(int(event["index"]))
        elif etype == "message_delta":
            delta = event.get("delta", {})
            if delta.get("stop_reason") is not None:
                self._stop_reason = delta["stop_reason"]
                self.complete = True
            self._usage.update(event.get("usage", {}))  # cumulative output_tokens
        elif etype == "message_stop":
            self.complete = True
        return None

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
            block["input"] = json.loads(buf) if buf else {}  # complete + validated only now

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


# ── The provider adapter (Unit 2a) + the shared request shape (Unit 2b): httpx, no vendor SDK ──

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


def request_headers(api_key: str) -> dict[str, str]:
    """The Anthropic auth + version headers. Shared by the adapter and the Unit 2b smoke."""
    return {"x-api-key": api_key, "anthropic-version": _ANTHROPIC_VERSION}


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
    The always-omitted Sonnet 5 guards still hold by construction: never `temperature`/
    `top_p`/`top_k` (any non-default → 400), never MANUAL thinking `{type:"enabled",budget_tokens}`
    (→ 400 — but `{type:"disabled"}` IS accepted on Sonnet 5, which the loop path uses), and
    `messages` is passed through unchanged (no assistant prefill → 400)."""
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

      - ToolCallEvent    -> an assistant turn with the native `tool_use` block (id/name/input);
      - ToolResultEvent  -> a user turn with the matching `tool_result` (paired by `tool_use_id`);
      - a FATAL tool ErrorEvent -> an `is_error` tool_result paired to the open tool_use (ErrorEvent
        carries no id, so we pair it with the last-opened call).

    Skipped (not conversation): retryable ErrorEvents (a retry follows, superseded by its
    ToolResultEvent), `<brain>`/`<loop>` failures (transport/loop errors, not turns), and
    SniffEvent / PreyCapturedEvent / UsageEvent. Assistant *thinking* is NOT reconstructed — it is
    not captured until Unit 3, which is why the loop path disables thinking (bare tool_use replay).
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": _seed_directive(territory)}]
    open_tool_use_id = ""  # the id of the tool_use awaiting its result (for id-less ErrorEvents)
    for event in context:
        if isinstance(event, ToolCallEvent):
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


async def _drive_stream(response: httpx.Response, sink: ThinkingSink) -> bytes:
    """Consume a streaming Messages response: relay each thinking delta via `sink` (write-ahead,
    live — Unit 3b), assemble the blocks, and return the assembled response bytes for
    `parse_decision`. A non-2xx body is the error JSON, not SSE, so `raise_for_status` first turns
    it into an `HTTPStatusError` the loop classifies. The `data:` framing mirrors `iter_sse_events`
    but over the live line iterator (`aiter_lines`) — the assembler (`feed`) is the shared logic."""
    if response.status_code != 200:
        await response.aread()  # a streaming body must be read before raise_for_status
        response.raise_for_status()
    assembler = StreamAssembler()
    data_buf: list[str] = []
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            data_buf.append(line.removeprefix("data:").strip())
        elif not line.strip():  # a blank line closes the frame
            if data_buf:
                text = assembler.feed(json.loads("".join(data_buf)))
                data_buf = []
                if text:
                    await sink(text)  # inline await — single writer (inv 7), live (inv 1)
    if data_buf:  # a final frame not terminated by a trailing blank line
        assembler.feed(json.loads("".join(data_buf)))
    return json.dumps(assembler.assembled()).encode()


def _extract_thinking_block(raw: bytes) -> bytes:
    """The turn's VERBATIM signed thinking block from an assembled response, for replay (inv 6):
    its raw JSON bytes, or b"" if the turn carried no thinking. `project_messages` echoes these
    bytes onto the reconstructed assistant turn — never rebuilds them: the signature is bound to the
    exact block content (a rebuild would 400 "thinking blocks cannot be modified")."""
    for block in json.loads(raw).get("content", []):
        if block.get("type") == "thinking":
            return json.dumps(block).encode()
    return b""


def adapter_brain_for(
    *,
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    registry: ToolRegistry,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    thinking: dict[str, Any] | None = None,
    stream: bool = False,
) -> Callable[[str], Brain]:
    """A `brain_for` factory (the scheduler's seam, `scheduler.py`) backed by the Anthropic Messages
    API over the INJECTED httpx client. `brain()` never opens or closes the client — its lifecycle
    belongs to whoever constructed it (the daemon/lifespan for a live call, a `MockTransport` in
    tests), the same single-owner discipline the run connections follow.

    Each turn (P5 Unit 2c): project the log into `messages` (`project_messages`, inv 2), thread the
    `HUNT_SYSTEM_PROMPT`. `thinking` defaults to `{type:"disabled"}` (2c's bare tool_use replay);
    `P5` Unit 3 passes `{type:"adaptive", display:"summarized"}` + `stream=True`. On the streaming
    path (Unit 3b) the SSE deltas are relayed live via the `sink` and assembled before crossing the
    single boundary at `parse_decision`; on the non-streaming path a plain POST does the same.
    `raise_for_status()` turns a non-2xx into an `httpx.HTTPStatusError` the loop's `classify` sorts
    (429/5xx retryable, 4xx fatal); the parsed `UsageEvent` rides onto the Decision (invariant 5).
    """
    tools = build_tools(registry)  # derived once — stable across the run
    headers = request_headers(api_key)
    thinking = {"type": "disabled"} if thinking is None else thinking

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
            if not stream:
                response = await client.post(MESSAGES_URL, headers=headers, json=body)
                response.raise_for_status()  # non-2xx -> HTTPStatusError, classified by the loop
                return parse_decision(response.content)  # one boundary (invariant 3)
            body["stream"] = True
            async with client.stream("POST", MESSAGES_URL, headers=headers, json=body) as response:
                raw = await _drive_stream(response, sink)
            decision = parse_decision(raw)  # the assembled stream crosses the one boundary (inv 3)
            if isinstance(decision, ToolCallDecision):
                # Carry the turn's VERBATIM signed thinking block so the NEXT call's reconstructed
                # assistant turn leads with it (Unit 3c, invariant 6) — echoed, never rebuilt.
                decision.thinking = _extract_thinking_block(raw)
            return decision

        return brain

    return brain_for


def select_brain_for(
    registry: ToolRegistry,
    *,
    default: Callable[[str], Brain] | None = None,
) -> tuple[Callable[[str], Brain], httpx.AsyncClient | None]:
    """The autonomous-spender containment (P5 Unit 2c): the daemon's `brain_for`, selected by the
    `REXHUNTER_BRAIN` env var. Default `"stub"` returns a no-spend brain and no client — a default
    start constructs nothing that could hit the network; `default` (the daemon's injectable stub
    seam, `daemon_config`'s brain) is returned when given, else the module `stub_brain_for`.
    `"live"` is the single opt-in that arms spending: it builds the STREAMING/THINKING adapter (`P5`
    Unit 3 — `stream=True` + adaptive thinking, so Rex's reasoning relays through the hub) over a
    real httpx client and RETURNS that client so the caller (lifespan / entrypoints) owns its close.
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
        client = httpx.AsyncClient(timeout=120.0)
        brain_for = adapter_brain_for(
            client=client,
            api_key=api_key,
            model=SMOKE_MODEL,
            registry=registry,
            max_tokens=HUNT_MAX_TOKENS,
            thinking=HUNT_THINKING,
            stream=True,  # the daemon streams reasoning to /events — the Unit-3 brain, not 2c's
        )
        return brain_for, client
    raise ValueError(f"unknown REXHUNTER_BRAIN={mode!r} (expected 'stub' or 'live')")
