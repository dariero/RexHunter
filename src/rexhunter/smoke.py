"""P5 Unit 2b — the single live smoke call (the FIRST PAID call). Spends on exactly one call.

NOT a test — pytest must never spend. Run manually once, with the key exported:

    ANTHROPIC_API_KEY=... PYTHONPATH=src uv run python -m rexhunter.smoke

Sequence (persist-first, one call, no retry — the hard one-call guard):
  1. build the title-stripped tools + a directive message (bias toward a real `sniff` tool_use,
     while `tool_choice` stays AUTO so adaptive thinking still runs → the response carries
     `thinking` sibling blocks the lenient scan must skip);
  2. FREE `count_tokens` pre-flight against the EXACT payload — gate on 200 (validates egress,
     auth, and that Anthropic accepts our schema keys); a non-200 STOPS before any paid call;
  3. ONE `messages` call on `SMOKE_MODEL` (Sonnet 5) via the shared adapter request shape;
  4. on 200, persist the raw response bytes (inv 6) BEFORE parse/assert can throw, the money-analog
     of "make the deliverable durable first", into a NEW `smoke_<model>_<UTC time>.json` reserved
     with "xb" before step 2 (F8: an existing name refuses at $0; smoke_sonnet5.json stays the
     recorded fixture);
  5. branch on `stop_reason` (`max_tokens` → truncation, report it, don't auto-retry), then
     `parse_decision`.
"""

import asyncio
import json
import os
import re
import sys
from datetime import UTC, datetime
from io import BufferedWriter
from pathlib import Path
from typing import Any, cast

import anthropic
from anthropic import AsyncAPIResponse
from anthropic.types import Message

from rexhunter import brain
from rexhunter.loop import Decision, ToolCallDecision
from rexhunter.stub import build_registry

SMOKE_MAX_TOKENS = 4096  # generous so adaptive-thinking tokens don't starve the tool_use block
# A paid smoke writes a NEW file here per run (F8, W2.1b); smoke_sonnet5.json stays the recording.
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")  # any other character in a model id becomes "_"


def fixture_path(fixture_dir: Path, model: str, now: datetime) -> Path:
    """The file one paid smoke writes: `smoke_<model>_<UTC time to the microsecond>Z.json` under
    `fixture_dir`. The model id is reduced to one path component, and `now` (an aware datetime) is
    converted to UTC, so two runs in one second still get two names. A naive `now` is refused:
    Python would read it as local time without a word."""
    if now.utcoffset() is None:
        raise ValueError(f"fixture_path needs an aware datetime, got naive {now!r}")
    stamp = now.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return fixture_dir / f"smoke_{_UNSAFE_NAME.sub('_', model)}_{stamp}.json"


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
    client: anthropic.AsyncAnthropic,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> int:
    """FREE pre-flight: count input tokens for the EXACT payload. A non-200 STOPS before any paid
    call — it means egress / auth / schema is wrong; fix it and spend nothing."""
    body: dict[str, Any] = {"model": model, "messages": messages, "tools": tools}
    try:
        result = await client.messages.count_tokens(**body)
    except anthropic.APIStatusError as exc:
        raise SystemExit(
            f"[preflight http {exc.status_code}] count_tokens failed: fix before spending.\n"
            f"{exc.response.text[:2000]}"
        ) from exc
    return result.input_tokens


async def _preflight_and_pay(
    client: anthropic.AsyncAnthropic, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> bytes:
    """The free pre-flight, then the ONE paid call: returns the raw 200 body (a non-200 stops)."""
    n_in = await count_tokens(client, model=brain.SMOKE_MODEL, messages=messages, tools=tools)
    print(f"[preflight] count_tokens OK — input_tokens={n_in}")

    body = brain.build_request_body(
        model=brain.SMOKE_MODEL, max_tokens=SMOKE_MAX_TOKENS, messages=messages, tools=tools
    )
    try:
        # Raw mode: the provider's bytes as they came (a cast: `**body` matches several overloads).
        resp = cast(
            AsyncAPIResponse[Message], await client.messages.with_raw_response.create(**body)
        )
    except anthropic.APIStatusError as exc:
        # No error body is ever persisted (run_smoke drops the reservation); surface it. (A
        # pre-output error is not billed; count_tokens already validated the shape, so this should
        # be rare.)
        raise SystemExit(
            f"[http {exc.status_code}] paid call failed:\n"
            f"{exc.response.content.decode(errors='replace')[:2000]}"
        ) from exc
    return await resp.read()


def _reserve(fixture_dir: Path, now: datetime) -> tuple[Path, BufferedWriter]:
    """Create this run's fixture file BEFORE any call (sync, like hunt_smoke.write_fixtures).
    "xb" refuses an existing name, so a collision costs $0 and a recorded fixture is never
    overwritten."""
    path = fixture_path(fixture_dir, brain.SMOKE_MODEL, now)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    try:
        return path, path.open("xb")
    except FileExistsError as exc:
        raise SystemExit(f"[fixture] {path} exists: refusing before any call.") from exc


async def run_smoke(
    client: anthropic.AsyncAnthropic,
    *,
    fixture_dir: Path,
    now: datetime,
    territory: str = "mock-gym",
) -> tuple[bytes, Decision]:
    """One free pre-flight + ONE paid call (the client's max_retries=0 is the hard one-call guard).
    Persists the raw 200 response BEFORE parsing (inv 6), into a NEW file reserved before any call
    (F8). `fixture_dir` and `now` have no defaults: a caller that forgets them gets a TypeError,
    never a write under tests/fixtures."""
    tools = brain.build_tools(build_registry())
    messages = _smoke_messages(territory)

    path, reserved = _reserve(fixture_dir, now)
    paid = False
    try:
        with reserved:
            raw = await _preflight_and_pay(client, messages=messages, tools=tools)
            # 200 → persist the paid artifact FIRST, before stop_reason / parse / assert can throw
            # (inv 6): written, flushed and closed before either runs. From here on the file may
            # hold paid bytes, so it is kept even if the write itself fails part way.
            paid = True
            reserved.write(raw)
            reserved.flush()
    finally:
        # Drop the reservation only while no paid reply has reached it (a non-200, a transport
        # error, Ctrl-C, a cancel); never once one has (persist-first).
        if not paid:
            path.unlink(missing_ok=True)
    print(f"[fixture] wrote {len(raw)} bytes -> {path}")

    if json.loads(raw).get("stop_reason") == "max_tokens":
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
        # max_retries=0 (brain.anthropic_client): exactly one messages call, the hard one-call
        # guard.
        async with brain.anthropic_client(api_key, timeout=120.0) as client:
            if count_only:
                tools = brain.build_tools(build_registry())
                messages = _smoke_messages("mock-gym")
                n_in = await count_tokens(
                    client, model=brain.SMOKE_MODEL, messages=messages, tools=tools
                )
                print(f"[count-only] input_tokens={n_in} — validated, NO paid call made.")
                return
            _raw, decision = await run_smoke(client, fixture_dir=FIXTURE_DIR, now=datetime.now(UTC))
            assert isinstance(decision, ToolCallDecision), (
                f"expected ToolCallDecision, got {decision!r}"
            )
            print(f"[ok] tool={decision.tool} tool_use_id={decision.tool_use_id}")

    asyncio.run(_go())


if __name__ == "__main__":
    main()
