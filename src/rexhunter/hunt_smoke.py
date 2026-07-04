"""P5 Unit 2c.2 — the gated live hunt entrypoint. SPENDS on the paid path.

NOT a test — pytest must never spend. Two modes:

    ANTHROPIC_API_KEY=... uv run python -m rexhunter.hunt_smoke --count-only   # FREE pre-flight, $0
    ANTHROPIC_API_KEY=... REXHUNTER_BRAIN=live uv run python -m rexhunter.hunt_smoke   # paid hunt

`--count-only` runs just the free `count_tokens` pre-flight against the EXACT iteration-1 payload.
The default paid path is opt-in twice over — it refuses to spend unless `REXHUNTER_BRAIN=live` and a
key are both set — and drives ONE budget-capped multi-iteration hunt on `SMOKE_MODEL` against the
deterministic mock-gym `sniff` board. It tees each brain call's raw response bytes into golden
fixtures (invariant 6 — the brain's raw response is not otherwise in the log), then reports the
outcome, spend, and any findings (a `max_tokens` truncation, or parallel tool calls past the flag).
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx

from rexhunter import brain, cost, db, stub
from rexhunter.loop import run_hunt
from rexhunter.tools import ToolRegistry

TERRITORY = "mock-gym"  # the deterministic mock board (stub `sniff`) — repeatable, no external dep
HUNT_MAX_TOKENS = 1024  # thinking off → tiny output; the live request-shape floor (>= 1024)
COST_CEILING_USD = (
    0.05  # the load-bearing spend guard for the smoke (cost.py over-meters at sticker)
)
MAX_ITERATIONS = 3  # a few turns: enough to reach call two (the reconstructed-turn gate)
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _hunt_body(registry: ToolRegistry, *, territory: str = TERRITORY) -> dict[str, Any]:
    """The iteration-1 messages body — the EXACT shape the adapter's first call sends (same builder,
    same constants), so the pre-flight prices precisely what the paid call will spend on."""
    return brain.build_request_body(
        model=brain.SMOKE_MODEL,
        max_tokens=HUNT_MAX_TOKENS,
        messages=brain.project_messages(territory, []),
        tools=brain.build_tools(registry),
        system=brain.HUNT_SYSTEM_PROMPT,
        thinking={"type": "disabled"},
        tool_choice=brain.HUNT_TOOL_CHOICE,
    )


async def count_tokens_preflight(
    client: httpx.AsyncClient, api_key: str, registry: ToolRegistry, *, territory: str = TERRITORY
) -> int:
    """FREE pre-flight: count input tokens for the EXACT hunt payload (minus `max_tokens`, which the
    count_tokens endpoint does not take). A non-200 STOPS before any paid call — egress / auth /
    schema is wrong; the free call is where that surfaces, for $0."""
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


class CapturingTransport(httpx.AsyncBaseTransport):
    """Tees every response for `capture_url` into `captured` (invariant 6 fixture capture) without
    touching the adapter. Wraps any inner transport — a live `AsyncHTTPTransport`, or a
    `MockTransport` in tests. Reads the body once, returns a fresh buffered response, same bytes."""

    def __init__(self, inner: httpx.AsyncBaseTransport, capture_url: str) -> None:
        self._inner = inner
        self._capture_url = capture_url
        self.captured: list[bytes] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        body = await response.aread()  # DECODED bytes (httpx requests gzip by default)
        if str(request.url) == self._capture_url:
            self.captured.append(body)
        # Reconstruct with status + body ONLY. Re-attaching the inner headers would carry a stale
        # `Content-Encoding: gzip` / `Content-Length` over already-decoded bytes and make the
        # adapter's `.content` read error on a live gzipped response. The adapter reads only the
        # status (`raise_for_status`) and body (`parse_decision`), never a header — so dropping them
        # is safe and lets httpx recompute `Content-Length` from `content=`.
        return httpx.Response(response.status_code, content=body, request=request)


def _write_fixtures(captured: list[bytes]) -> list[Path]:
    """Persist each captured brain response as an ordered golden fixture (invariant 6)."""
    paths: list[Path] = []
    for i, raw in enumerate(captured, start=1):
        path = FIXTURE_DIR / f"hunt_smoke_call_{i:02d}.json"
        path.write_bytes(raw)
        paths.append(path)
    return paths


def _report_findings(captured: list[bytes]) -> None:
    """Surface, never paper over: a parallel-tool response despite the flag is a real finding; a
    `max_tokens` stop is truncation (bump the budget), not a parse error."""
    for i, raw in enumerate(captured, start=1):
        payload = json.loads(raw)
        n_tool_use = sum(1 for b in payload.get("content", []) if b.get("type") == "tool_use")
        if n_tool_use > 1:
            print(
                f"[FINDING] call {i}: {n_tool_use} tool_use blocks despite "
                "disable_parallel_tool_use — surface it, do not paper over"
            )
        if payload.get("stop_reason") == "max_tokens":
            print(
                f"[truncation] call {i}: stop_reason=max_tokens — bump HUNT_MAX_TOKENS and re-run "
                "(a fresh paid call); this is truncation, not a parse error"
            )


async def run_live_hunt(api_key: str) -> None:
    """The paid path: free pre-flight → one capped hunt on the teed client → fixtures + report."""
    registry = stub.build_registry()
    tee = CapturingTransport(httpx.AsyncHTTPTransport(), brain.MESSAGES_URL)
    async with httpx.AsyncClient(transport=tee, timeout=120.0) as client:
        n_in = await count_tokens_preflight(client, api_key, registry)
        print(f"[preflight] count_tokens OK — input_tokens={n_in}")
        brain_for = brain.adapter_brain_for(
            client=client,
            api_key=api_key,
            model=brain.SMOKE_MODEL,
            registry=registry,
            max_tokens=HUNT_MAX_TOKENS,
        )
        db_path = Path(tempfile.mkdtemp(prefix="rexhunter-hunt-")) / "rex.db"
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
    paths = _write_fixtures(tee.captured)
    print(f"[fixtures] wrote {len(paths)} golden fixture(s): {', '.join(p.name for p in paths)}")


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
