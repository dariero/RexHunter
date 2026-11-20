"""P5 Unit 3c — the gated THINKING-ON live hunt entrypoint. SPENDS on the paid path.

NOT a test — pytest must never spend. Like `hunt_smoke` (2c.2) but with thinking ON and STREAMING:

    ANTHROPIC_API_KEY=... PYTHONPATH=src \
        uv run python -m rexhunter.hunt_smoke_thinking --count-only   # FREE $0
    ANTHROPIC_API_KEY=... REXHUNTER_BRAIN=live PYTHONPATH=src \
        uv run python -m rexhunter.hunt_smoke_thinking  # $

The live model runs adaptive thinking (summarized), streamed as SSE. Two differences from 2c.2
that are load-bearing at spend time:

  - `HUNT_MAX_TOKENS` is RAISED to 4096: thinking tokens AND the tool_use both count against
    `max_tokens`, so a tight 1024 budget truncates into `stop_reason=max_tokens` (an incomplete
    tool_use the assembler can't finish). Streaming removes the HTTP-timeout reason to keep it low.
  - `COST_CEILING_USD` is RAISED to 0.20: the gate is CALL TWO accepted (the reconstructed assistant
    turn leads with the verbatim signed block), so the breaker must clear TWO thinking-inflated
    calls — a ceiling that aborts after call one would spend and prove nothing.

The captured raw SSE stream of each brain call becomes a golden fixture (invariant 6); the offline
replay test re-drives them through the streaming assembler with zero spend. The harness is
`hunt_smoke`'s, run with this module's shape.
"""

from pathlib import Path
from typing import Any

import anthropic
import httpx2

from rexhunter import brain, hunt_smoke
from rexhunter.tools import ToolRegistry

TERRITORY = "mock-gym"
# One definition, no drift: the thinking-on shape lives in brain.py (the "one request shape"
# discipline) — the daemon's `select_brain_for(live)` reads the SAME constants, so this gated
# entrypoint and the daemon stream through the identical brain.
HUNT_MAX_TOKENS = (
    brain.HUNT_MAX_TOKENS
)  # thinking + tool_use share the output budget; 1024 truncates
THINKING: dict[str, Any] = brain.HUNT_THINKING  # adaptive, summarized
COST_CEILING_USD = 0.20  # must clear TWO thinking-inflated calls (the gate is call-two-accepted)
MAX_ITERATIONS = 3  # a few turns — enough to reach call two, the reconstructed-signed-block gate
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


async def count_tokens_preflight(
    client: anthropic.AsyncAnthropic, registry: ToolRegistry, *, territory: str = TERRITORY
) -> int:
    """FREE pre-flight: count input tokens for the EXACT thinking-on payload. A non-200 STOPS before
    any paid call — auth / egress / a thinking param count_tokens rejects surfaces here for $0."""
    return await hunt_smoke.count_tokens_preflight(
        client, registry, territory=territory, thinking=THINKING
    )


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
    inner: httpx2.AsyncBaseTransport | None = None,
    fixture_dir: Path = FIXTURE_DIR,
    db_dir: Path | None = None,
) -> dict[str, Any]:
    """Free pre-flight → one capped STREAMING hunt on the teed client → fixtures + report.

    `inner`/`fixture_dir`/`db_dir` are injection seams for the OFFLINE harness test (a
    `MockTransport` + tmp dirs), so the whole path is proven with zero spend before the real run.
    Default `inner` is a live `AsyncHTTPTransport`; the paid path leaves all three at their
    defaults, so its run DB lands in a fresh directory under the system temp dir that nothing
    removes: since W3.3 that DB holds every attempt's request and response bytes.
    """
    return await hunt_smoke.run_capped_hunt(
        api_key,
        territory=TERRITORY,
        thinking=THINKING,
        stream=True,
        max_tokens=HUNT_MAX_TOKENS,
        cost_ceiling_usd=COST_CEILING_USD,
        max_iterations=MAX_ITERATIONS,
        client_timeout_s=180.0,
        db_prefix="rexhunter-think-",
        fixture_kind="streaming",
        preflight=count_tokens_preflight,
        report=_report_findings,
        inner=inner,
        fixture_dir=fixture_dir,
        db_dir=db_dir,
    )


def main() -> None:
    hunt_smoke.run_main(preflight=count_tokens_preflight, run_live=run_live_hunt)


if __name__ == "__main__":
    main()
