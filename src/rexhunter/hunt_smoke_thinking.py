"""P5 Unit 3c — the gated THINKING-ON live hunt entrypoint. SPENDS on the paid path.

NOT a test — pytest must never spend. Like `hunt_smoke` (2c.2) but with thinking ON and STREAMING:

    ANTHROPIC_API_KEY=... uv run python -m rexhunter.hunt_smoke_thinking --count-only   # FREE $0
    ANTHROPIC_API_KEY=... REXHUNTER_BRAIN=live uv run python -m rexhunter.hunt_smoke_thinking  # $

Sonnet 5 runs adaptive thinking (summarized), streamed as SSE. Two differences from 2c.2 that are
load-bearing at spend time:

  - `HUNT_MAX_TOKENS` is RAISED to 4096: thinking tokens AND the tool_use both count against
    `max_tokens`, so a tight 1024 budget truncates into `stop_reason=max_tokens` (an incomplete
    tool_use the assembler can't finish). Streaming removes the HTTP-timeout reason to keep it low.
  - `COST_CEILING_USD` is RAISED to 0.20: the gate is CALL TWO accepted (the reconstructed assistant
    turn leads with the verbatim signed block), so the breaker must clear TWO thinking-inflated
    calls — a ceiling that aborts after call one would spend and prove nothing.

The captured raw SSE stream of each brain call becomes a golden fixture (invariant 6); the offline
replay test re-drives them through the streaming assembler with zero spend.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx

from rexhunter import brain, cost, db, stub
from rexhunter.hunt_smoke import CapturingTransport, write_fixtures
from rexhunter.loop import run_hunt
from rexhunter.tools import ToolRegistry

TERRITORY = "mock-gym"
HUNT_MAX_TOKENS = 4096  # thinking ON → thinking + tool_use share the output budget; 1024 truncates
COST_CEILING_USD = 0.20  # must clear TWO thinking-inflated calls (the gate is call-two-accepted)
MAX_ITERATIONS = 3  # a few turns — enough to reach call two, the reconstructed-signed-block gate
THINKING: dict[str, Any] = {"type": "adaptive", "display": "summarized"}
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _hunt_body(registry: ToolRegistry, *, territory: str = TERRITORY) -> dict[str, Any]:
    """The iteration-1 body — the EXACT shape the streaming adapter's first call sends, so the free
    pre-flight prices precisely what the paid call spends on (minus `stream`, which count_tokens and
    the body builder don't carry)."""
    return brain.build_request_body(
        model=brain.SMOKE_MODEL,
        max_tokens=HUNT_MAX_TOKENS,
        messages=brain.project_messages(territory, []),
        tools=brain.build_tools(registry),
        system=brain.HUNT_SYSTEM_PROMPT,
        thinking=THINKING,
        tool_choice=brain.HUNT_TOOL_CHOICE,
    )


async def count_tokens_preflight(
    client: httpx.AsyncClient, api_key: str, registry: ToolRegistry, *, territory: str = TERRITORY
) -> int:
    """FREE pre-flight: count input tokens for the EXACT thinking-on payload. A non-200 STOPS before
    any paid call — auth / egress / a thinking param count_tokens rejects surfaces here for $0."""
    body = _hunt_body(registry, territory=territory)
    count_body: dict[str, Any] = {k: v for k, v in body.items() if k != "max_tokens"}
    resp = await client.post(
        brain.COUNT_TOKENS_URL, headers=brain.request_headers(api_key), json=count_body
    )
    if resp.status_code != 200:
        raise SystemExit(
            f"[preflight http {resp.status_code}] count_tokens rejected the exact payload — "
            f"fix before spending.\n{resp.text[:2000]}"
        )
    return int(resp.json()["input_tokens"])


def _report_findings(captured: list[bytes]) -> None:
    """Streaming-aware (unlike 2c.2's JSON-body reporter): ASSEMBLE each captured SSE stream, then
    surface, never paper over — a parallel-tool response despite the flag is a real finding; a
    `max_tokens` stop is truncation (thinking-on makes it the likely first failure — bump the
    budget), not a parse error."""
    for i, raw in enumerate(captured, start=1):
        assembler = brain.StreamAssembler()
        for event in brain.iter_sse_events(raw):
            assembler.feed(event)
        message = assembler.assembled()
        n_tool_use = sum(1 for b in message.get("content", []) if b.get("type") == "tool_use")
        if n_tool_use > 1:
            print(
                f"[FINDING] call {i}: {n_tool_use} tool_use blocks despite "
                "disable_parallel_tool_use — surface it, do not paper over"
            )
        if message.get("stop_reason") == "max_tokens":
            print(
                f"[truncation] call {i}: stop_reason=max_tokens — bump HUNT_MAX_TOKENS and re-run "
                "(a fresh paid call); thinking + tool_use share the output budget"
            )


async def run_live_hunt(
    api_key: str,
    *,
    inner: httpx.AsyncBaseTransport | None = None,
    fixture_dir: Path = FIXTURE_DIR,
) -> dict[str, Any]:
    """Free pre-flight → one capped STREAMING hunt on the teed client → fixtures + report.

    `inner`/`fixture_dir` are injection seams for the OFFLINE harness test (a `MockTransport` + a
    tmp dir), so the whole path is proven with zero spend before the real run. Default `inner` is a
    live `AsyncHTTPTransport`; the paid path leaves both at their defaults.
    """
    registry = stub.build_registry()
    tee = CapturingTransport(inner or httpx.AsyncHTTPTransport(), brain.MESSAGES_URL)
    async with httpx.AsyncClient(transport=tee, timeout=180.0) as client:
        n_in = await count_tokens_preflight(client, api_key, registry)
        print(f"[preflight] count_tokens OK — input_tokens={n_in}")
        brain_for = brain.adapter_brain_for(
            client=client,
            api_key=api_key,
            model=brain.SMOKE_MODEL,
            registry=registry,
            max_tokens=HUNT_MAX_TOKENS,
            thinking=THINKING,
            stream=True,
        )
        db_path = Path(tempfile.mkdtemp(prefix="rexhunter-think-")) / "rex.db"
        conn = await db.connect(db_path)
        try:
            run_id = await run_hunt(
                conn,
                territory=TERRITORY,
                brain=brain_for(TERRITORY),
                registry=registry,
                cost_ceiling_usd=COST_CEILING_USD,
                max_iterations=MAX_ITERATIONS,
                tool_timeout_s=60.0,
            )
            events = await db.read_events(conn, run_id)
            spend = cost.fold_cost(events)
            async with conn.execute(
                "SELECT outcome, abort_reason FROM runs WHERE id = ?", (run_id,)
            ) as cur:
                row = await cur.fetchone()
            outcome, reason = (str(row[0]), row[1]) if row else ("unknown", None)
        finally:
            await conn.close()

    print(
        f"[hunt] run={run_id} outcome={outcome} reason={reason} brain_calls={len(tee.captured)} "
        f"spend=${spend:.4f} (ceiling ${COST_CEILING_USD})"
    )
    _report_findings(tee.captured)
    paths = write_fixtures(tee.captured, run_id, fixture_dir=fixture_dir)
    print(f"[fixtures] wrote {len(paths)} streaming fixture(s) under hunt_{run_id}/")
    return {
        "run_id": run_id,
        "outcome": outcome,
        "reason": reason,
        "brain_calls": len(tee.captured),
        "spend": spend,
        "fixtures": paths,
    }


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set — export it before the pre-flight or hunt.")

    async def _go() -> None:
        if "--count-only" in sys.argv:
            async with httpx.AsyncClient(timeout=120.0) as client:
                n = await count_tokens_preflight(client, api_key, stub.build_registry())
            print(f"[count-only] input_tokens={n} — validated, NO paid call made.")
            return
        if os.environ.get("REXHUNTER_BRAIN") != "live":
            raise SystemExit(
                "refusing to spend: set REXHUNTER_BRAIN=live to run the paid hunt (opt-in only)."
            )
        await run_live_hunt(api_key)

    asyncio.run(_go())


if __name__ == "__main__":
    main()
