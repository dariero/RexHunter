"""Stage 5 gate — ADR definition-of-done #5 (the edge boundary, offline).

`P5` Unit 1 is `brain.py`'s `parse_decision`: raw provider bytes cross ONE Pydantic boundary
(invariant 3 — LLM output is untrusted input, same suspicion as scraped HTML) and become a
typed `Decision`. There is no vendor SDK, no network, no paid call anywhere in this slice.

Four assertions, mapping 1:1 to DoD #5:
  1. a clean Anthropic-shaped ``tool_use`` payload  -> ``ToolCallDecision`` (name, args, id);
     a ``hunt_complete``-shaped payload             -> ``HuntComplete`` (structured catch).
  2. the same ``tool_use_id`` lands on BOTH ``ToolCallEvent`` and ``ToolResultEvent``
     (the provider's correlation key doubles as the log's pairing key — §What point 2).
  3. a malformed payload -> a typed ``ErrorEvent`` carrying the raw payload (invariant 6);
     the run ends in a typed outcome, never an unhandled exception escaping the loop.
  4. replaying the golden fixtures through ``parse_decision`` twice reproduces byte-identical
     events — no clock, no randomness, no network.

The fixtures are hand-authored raw bytes in the shape the Anthropic Messages API returns; the
gate is agnostic to our internal reserved-name constants — it authors the wire, not the seam.
"""

import json
from pathlib import Path

import pytest

from rexhunter import db
from rexhunter.brain import parse_decision
from rexhunter.events import BrainParseError, ErrorEvent, ToolCallEvent, ToolResultEvent
from rexhunter.loop import HuntComplete, NeedsHelp, ToolCallDecision, run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

GOLDEN_FIXTURES = 4  # knob: clean tool_use, HuntComplete-shaped, NeedsHelp-shaped, malformed


def _provider_response(stop_reason: str, content: list[dict[str, object]]) -> bytes:
    """An Anthropic Messages-API response envelope. The envelope carries fields we do NOT model
    (id/model/role/usage/stop_sequence) — a real payload the boundary must tolerate while still
    validating strictly the sub-shape we consume."""
    return json.dumps(
        {
            "id": "msg_01FIXTURE",
            "type": "message",
            "role": "assistant",
            "model": "claude-fixture",
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 34},
            "content": content,
        }
    ).encode()


# ── The three golden fixtures (raw bytes, hand-authored) ──────────────────────

# 1. a clean tool_use: a text block AND a tool_use block; parse must pick the tool_use one.
CLEAN_TOOL_USE = _provider_response(
    "tool_use",
    [
        {"type": "text", "text": "Let me sniff acme."},
        {"type": "tool_use", "id": "toolu_01ABC", "name": "sniff", "input": {"prey": "acme"}},
    ],
)

# 2. HuntComplete-shaped: a reserved-name terminal tool carrying the structured catch.
HUNT_COMPLETE_SHAPED = _provider_response(
    "tool_use",
    [
        {
            "type": "tool_use",
            "id": "toolu_01DONE",
            "name": "hunt_complete",
            "input": {"catch": ["posting:acme", "posting:globex"]},
        }
    ],
)

# 3. NeedsHelp-shaped: the other reserved-name terminal tool (D1 commits the adapter to it too;
#    this closes that dispatch branch rather than leaving it uncovered).
NEEDS_HELP_SHAPED = _provider_response(
    "tool_use",
    [{"type": "tool_use", "id": "toolu_01HELP", "name": "needs_help", "input": {}}],
)

# 4. malformed: fails on a MODELED field — `input` is a string, not the object we require.
#    (An extra="ignore" envelope would swallow an unmodeled field; this genuinely rejects.)
MALFORMED = _provider_response(
    "tool_use",
    [{"type": "tool_use", "id": "toolu_01BAD", "name": "sniff", "input": "not-an-object"}],
)

FIXTURES = {
    "tool_use": CLEAN_TOOL_USE,
    "hunt_complete": HUNT_COMPLETE_SHAPED,
    "needs_help": NEEDS_HELP_SHAPED,
    "malformed": MALFORMED,
}


def test_golden_fixture_count_matches_the_knob() -> None:
    assert len(FIXTURES) == GOLDEN_FIXTURES


# ── DoD #5 · assertion 1 — the edge boundary maps bytes to a typed Decision ───


def test_clean_tool_use_parses_to_a_toolcall_decision() -> None:
    decision = parse_decision(CLEAN_TOOL_USE)
    assert isinstance(decision, ToolCallDecision)
    assert decision.tool == "sniff"
    assert decision.args == {"prey": "acme"}  # structured args, not text-parsed
    assert decision.tool_use_id == "toolu_01ABC"  # the provider's correlation key survives


def test_hunt_complete_shaped_parses_to_hunt_complete() -> None:
    decision = parse_decision(HUNT_COMPLETE_SHAPED)
    assert isinstance(decision, HuntComplete)
    assert decision.catch == ["posting:acme", "posting:globex"]  # structured catch, validated


def test_needs_help_shaped_parses_to_needs_help() -> None:
    decision = parse_decision(NEEDS_HELP_SHAPED)
    assert isinstance(decision, NeedsHelp)  # the second reserved terminal name (D1)


# ── DoD #5 · assertion 2 — tool_use_id pairs ToolCallEvent <-> ToolResultEvent ─


async def test_tool_use_id_correlates_call_and_result_events(tmp_path: Path) -> None:
    reg = ToolRegistry()

    @reg.tool
    async def sniff(prey: str) -> str:
        return f"posting:{prey}"

    known_id = "toolu_01CORRELATE"

    async def brain(_context: list[object]) -> object:
        brain.calls = getattr(brain, "calls", 0) + 1  # type: ignore[attr-defined]
        if brain.calls == 1:  # type: ignore[attr-defined]
            return ToolCallDecision(
                tool=sniff.__name__, args={"prey": "acme"}, tool_use_id=known_id
            )
        return HuntComplete()  # terminal: close the run cleanly after the one tool call

    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await run_hunt(conn, territory="gate", brain=brain, registry=reg)  # type: ignore[arg-type]

        events = await db.read_events(conn, run_id)
        calls = [e for e in events if isinstance(e, ToolCallEvent)]
        results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(calls) == 1 and len(results) == 1
        assert calls[0].tool_use_id == known_id
        assert results[0].tool_use_id == known_id  # same key on both -> the pairing key
    finally:
        await conn.close()


# ── DoD #5 · assertion 3 — malformed -> ErrorEvent(raw), typed outcome, no escape ─


async def test_malformed_payload_becomes_an_error_event_carrying_raw(tmp_path: Path) -> None:
    reg = ToolRegistry()

    async def brain(_context: list[object]) -> object:
        # A stand-in for the (deferred) adapter: the provider handed back a malformed payload;
        # parse_decision rejects it at the boundary. run_hunt must catch this, not let it escape.
        return parse_decision(MALFORMED)

    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await run_hunt(conn, territory="gate", brain=brain, registry=reg)  # type: ignore[arg-type]

        events = await db.read_events(conn, run_id)
        errors = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(errors) == 1
        assert errors[0].raw_response == MALFORMED  # invariant 6: the raw payload is preserved
        assert errors[0].retryable is False  # a bad payload does not become good on retry

        async with conn.execute("SELECT outcome FROM runs WHERE id = ?", (run_id,)) as cur:
            row = await cur.fetchone()
        assert row is not None and row[0] == "aborted"  # a typed outcome, run_hunt returned
    finally:
        await conn.close()


def test_malformed_parse_raises_a_typed_boundary_error() -> None:
    # The parser itself is pure: it RAISES the typed boundary error carrying the raw bytes; the
    # loop (above) is what turns that into an ErrorEvent. Prove the parser half in isolation.
    with pytest.raises(BrainParseError) as excinfo:
        parse_decision(MALFORMED)
    assert excinfo.value.raw == MALFORMED


# ── DoD #5 · assertion 4 — determinism: two passes, byte-identical, no network ─


def test_replay_is_byte_identical_across_two_passes() -> None:
    for raw in (CLEAN_TOOL_USE, HUNT_COMPLETE_SHAPED, NEEDS_HELP_SHAPED):
        first, second = parse_decision(raw), parse_decision(raw)
        assert first.model_dump_json() == second.model_dump_json()  # type: ignore[union-attr]

    # the malformed path is deterministic too: same raw preserved on both rejections
    with pytest.raises(BrainParseError) as a:
        parse_decision(MALFORMED)
    with pytest.raises(BrainParseError) as b:
        parse_decision(MALFORMED)
    assert a.value.raw == b.value.raw == MALFORMED
