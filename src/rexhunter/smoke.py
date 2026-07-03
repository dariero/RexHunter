"""P5 Unit 2b — the single live smoke call (the FIRST PAID call). Spends on exactly one call.

NOT a test — pytest must never spend. Run manually once, with the key exported:

    ANTHROPIC_API_KEY=... uv run python -m rexhunter.smoke

Sequence (persist-first, one call, no retry — the hard one-call guard):
  1. build the title-stripped tools + a directive message (bias toward a real `sniff` tool_use,
     while `tool_choice` stays AUTO so adaptive thinking still runs → the response carries
     `thinking` sibling blocks the lenient scan must skip);
  2. FREE `count_tokens` pre-flight against the EXACT payload — gate on 200 (validates egress,
     auth, and that Anthropic accepts our schema keys); a non-200 STOPS before any paid call;
  3. ONE `messages` call on `SMOKE_MODEL` (Sonnet 5) via the shared adapter request shape;
  4. on 200, persist the raw response bytes as the golden fixture (inv 6) BEFORE parse/assert can
     throw — the money-analog of "make the deliverable durable first";
  5. branch on `stop_reason` (`max_tokens` → truncation, report it, don't auto-retry), then
     `parse_decision`.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from rexhunter import brain
from rexhunter.loop import Decision, ToolCallDecision
from rexhunter.stub import build_registry

SMOKE_MAX_TOKENS = 4096  # generous so adaptive-thinking tokens don't starve the tool_use block
FIXTURE_PATH = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "smoke_sonnet5.json"


def _smoke_messages(territory: str) -> list[dict[str, Any]]:
    """A directive prompt biasing Sonnet 5 toward a real `sniff` tool_use (a ToolCallDecision).
    `tool_choice` is left AUTO (never forced) so adaptive thinking still runs and the response
    carries `thinking` siblings — forcing tool_choice would suppress thinking, and then the fixture
    would not exercise the 2a lenient scan, which is the whole reason for a real fixture."""
    return [
        {
            "role": "user",
            "content": (
                f"You are hunting {territory} for AI-engineering job postings. "
                "Begin by calling the sniff tool to investigate the territory. "
                "Do not call hunt_complete or needs_help yet."
            ),
        }
    ]


async def count_tokens(
    client: httpx.AsyncClient,
    api_key: str,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> int:
    """FREE pre-flight: count input tokens for the EXACT payload. A non-200 STOPS before any paid
    call — it means egress / auth / schema is wrong; fix it and spend nothing."""
    body: dict[str, Any] = {"model": model, "messages": messages, "tools": tools}
    resp = await client.post(
        brain.COUNT_TOKENS_URL, headers=brain.request_headers(api_key), json=body
    )
    if resp.status_code != 200:
        raise SystemExit(
            f"[preflight http {resp.status_code}] count_tokens failed — fix before spending.\n"
            f"{resp.text[:2000]}"
        )
    return int(resp.json()["input_tokens"])


async def run_smoke(
    client: httpx.AsyncClient, api_key: str, *, territory: str = "mock-gym"
) -> tuple[bytes, Decision]:
    """One free pre-flight + ONE paid call. Persists the raw 200 response BEFORE parsing (inv 6)."""
    tools = brain.build_tools(build_registry())
    messages = _smoke_messages(territory)

    n_in = await count_tokens(
        client, api_key, model=brain.SMOKE_MODEL, messages=messages, tools=tools
    )
    print(f"[preflight] count_tokens OK — input_tokens={n_in}")

    body = brain.build_request_body(
        model=brain.SMOKE_MODEL, max_tokens=SMOKE_MAX_TOKENS, messages=messages, tools=tools
    )
    resp = await client.post(brain.MESSAGES_URL, headers=brain.request_headers(api_key), json=body)
    raw = resp.content
    if resp.status_code != 200:
        # Don't overwrite the golden fixture with an error body; surface it. (A pre-output error is
        # not billed; count_tokens already validated the shape, so this should be rare.)
        raise SystemExit(
            f"[http {resp.status_code}] paid call failed:\n{raw.decode(errors='replace')[:2000]}"
        )

    # 200 → persist the paid artifact FIRST, before stop_reason / parse / assert can throw (inv 6).
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_bytes(raw)
    print(f"[fixture] wrote {len(raw)} bytes -> {FIXTURE_PATH}")

    if resp.json().get("stop_reason") == "max_tokens":
        raise SystemExit(
            f"[truncation] stop_reason=max_tokens at max_tokens={SMOKE_MAX_TOKENS} — adaptive "
            "thinking starved the tool_use block. Bump the budget and re-run (a fresh paid call)."
        )
    decision = brain.parse_decision(raw)
    return raw, decision


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set — export it before the smoke call.")
    count_only = "--count-only" in sys.argv  # FREE pre-flight only; spends NOTHING.

    async def _go() -> None:
        # httpx does not retry by default — exactly one messages call (the hard one-call guard).
        async with httpx.AsyncClient(timeout=120.0) as client:
            if count_only:
                tools = brain.build_tools(build_registry())
                messages = _smoke_messages("mock-gym")
                n_in = await count_tokens(
                    client, api_key, model=brain.SMOKE_MODEL, messages=messages, tools=tools
                )
                print(f"[count-only] input_tokens={n_in} — validated, NO paid call made.")
                return
            _raw, decision = await run_smoke(client, api_key)
            assert isinstance(decision, ToolCallDecision), (
                f"expected ToolCallDecision, got {decision!r}"
            )
            print(f"[ok] tool={decision.tool} tool_use_id={decision.tool_use_id}")

    asyncio.run(_go())


if __name__ == "__main__":
    main()
