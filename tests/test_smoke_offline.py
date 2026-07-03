"""P5 Unit 2b — offline guards for the paid smoke (NO spend, no network).

Two locks the paid call depends on:
  1. schema-clean — `build_tools` strips every `title` from the derived `input_schema`s; the
     `@rex_tool` args-models still emit `additionalProperties: false` (kept — Anthropic accepts it);
  2. Sonnet 5 request guards — `build_request_body` omits the params that 400 on Sonnet 5
     (`temperature`/`top_p`/`top_k`/`thinking`) and never ends `messages` on an assistant prefill.
"""

from typing import Any, cast

from rexhunter.brain import SMOKE_MODEL, build_request_body, build_tools
from rexhunter.stub import build_registry


def _has_title(node: Any) -> bool:
    """True if a `title` key appears anywhere in the (possibly nested) schema."""
    if isinstance(node, dict):
        return "title" in node or any(_has_title(v) for v in cast("dict[str, Any]", node).values())
    if isinstance(node, list):
        return any(_has_title(v) for v in cast("list[Any]", node))
    return False


# ── 1. schema-clean ──────────────────────────────────────────────────────────


def test_build_tools_strips_title_from_every_schema() -> None:
    for tool in build_tools(build_registry()):
        assert not _has_title(tool["input_schema"]), f"`title` leaked into {tool['name']}'s schema"


def test_rex_tool_schema_keeps_additionalproperties_false() -> None:
    # @rex_tool args-models set extra="forbid" → additionalProperties:false IS emitted, and is KEPT
    # (Anthropic accepts it). The terminal-tool models are plain BaseModels and don't emit it —
    # non-strict tool use, which is fine (strict mode is out of scope for 2b).
    by_name = {t["name"]: t for t in build_tools(build_registry())}
    assert by_name["sniff"]["input_schema"]["additionalProperties"] is False
    assert "additionalProperties" not in by_name["hunt_complete"]["input_schema"]


# ── 2. Sonnet 5 request guards ───────────────────────────────────────────────


def test_smoke_request_body_omits_sonnet5_forbidden_params() -> None:
    messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
    body = build_request_body(
        model=SMOKE_MODEL, max_tokens=4096, messages=messages, tools=build_tools(build_registry())
    )
    assert body["model"] == "claude-sonnet-5"
    assert body["max_tokens"] >= 1024  # room for adaptive-thinking tokens + the tool_use block
    for forbidden in ("temperature", "top_p", "top_k", "thinking"):
        assert forbidden not in body  # any of these → 400 on Sonnet 5
    assert body["messages"][-1]["role"] != "assistant"  # no assistant prefill (→ 400)
