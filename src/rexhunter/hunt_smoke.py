"""P5 Unit 2c.2 — the gated live hunt entrypoint. SPENDS on the paid path.

NOT a test — pytest must never spend. Two modes:

    ANTHROPIC_API_KEY=... PYTHONPATH=src \
        uv run python -m rexhunter.hunt_smoke --count-only   # FREE pre-flight, $0
    ANTHROPIC_API_KEY=... REXHUNTER_BRAIN=live PYTHONPATH=src \
        uv run python -m rexhunter.hunt_smoke   # paid hunt

`--count-only` runs just the free `count_tokens` pre-flight against the EXACT iteration-1 payload.
The default paid path is opt-in twice over — it refuses to spend unless `REXHUNTER_BRAIN=live` and a
key are both set — and drives ONE budget-capped multi-iteration hunt on `SMOKE_MODEL` against the
deterministic mock-gym `sniff` board. It tees each brain call's raw response bytes into golden
fixtures (invariant 6; since W3.3 the log keeps every attempt's bytes too), then reports the
outcome, spend, and any findings (a `max_tokens` truncation, or parallel tool calls past the flag).

The harness itself (the pre-flight, the tee, the capped hunt, the fixture write and the entrypoint
gate) is shared with `hunt_smoke_thinking`, which passes its own shape to `run_capped_hunt`.
"""

import asyncio
import json
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import aiosqlite
import anthropic
import httpx2

from rexhunter import brain, cost, db, stub
from rexhunter.loop import run_hunt
from rexhunter.tools import ToolRegistry

TERRITORY = "mock-gym"  # the deterministic mock board (stub `sniff`) — repeatable, no external dep
HUNT_MAX_TOKENS = 1024  # thinking off → tiny output; the live request-shape floor (>= 1024)
COST_CEILING_USD = (
    0.05  # the load-bearing spend guard for the smoke (cost.py meters at the $2/$10 sticker, D6)
)
MAX_ITERATIONS = 3  # a few turns: enough to reach call two (the reconstructed-turn gate)
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

type Preflight = Callable[[anthropic.AsyncAnthropic, ToolRegistry], Awaitable[int]]


def _hunt_body(
    registry: ToolRegistry,
    *,
    territory: str = TERRITORY,
    thinking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The iteration-1 messages body — the EXACT shape the adapter's first call sends (same builder,
    same constants), so the pre-flight prices precisely what the paid call will spend on.
    `thinking` None is the adapter's default, the model's floor, by the same lookup; the
    pre-flight drops `max_tokens`, so the thinking hunt's larger budget needs no parameter."""
    return brain.build_request_body(
        model=brain.SMOKE_MODEL,
        max_tokens=HUNT_MAX_TOKENS,
        messages=brain.project_messages(territory, []),
        tools=brain.build_tools(registry),
        system=brain.HUNT_SYSTEM_PROMPT,
        thinking=brain.thinking_floor(brain.SMOKE_MODEL) if thinking is None else thinking,
        tool_choice=brain.HUNT_TOOL_CHOICE,
    )


async def count_tokens_preflight(
    client: anthropic.AsyncAnthropic,
    registry: ToolRegistry,
    *,
    territory: str = TERRITORY,
    thinking: dict[str, Any] | None = None,
) -> int:
    """FREE pre-flight: count input tokens for the EXACT hunt payload (minus `max_tokens`, which the
    count_tokens endpoint does not take). A non-200 STOPS before any paid call — egress / auth /
    schema is wrong; the free call is where that surfaces, for $0."""
    body = _hunt_body(registry, territory=territory, thinking=thinking)
    count_body: dict[str, Any] = {k: v for k, v in body.items() if k != "max_tokens"}
    try:
        result = await client.messages.count_tokens(**count_body)
    except anthropic.APIStatusError as exc:
        raise SystemExit(
            f"[preflight http {exc.status_code}] count_tokens rejected the exact payload: "
            f"fix before spending.\n{exc.response.text[:2000]}"
        ) from exc
    return result.input_tokens


# The headers that describe the encoded body. The tee hands on DECODED bytes (httpx2 asks for
# compression by default), so a stale `Content-Encoding: gzip` would make the adapter's read fail
# and a `Content-Length` would count the wrong bytes. Every other header passes: `retry-after`
# (W3.2-X17) reaches the loop's backoff, `request-id` still names a failed request.
_ENCODED_BODY_HEADERS = frozenset({"content-encoding", "content-length"})


class _TeeStream(httpx2.AsyncByteStream):
    """The inner response's decoded body, handed on chunk by chunk as it arrives and kept in `sink`
    on the way, so the reader sees each chunk when it lands and can stop early."""

    def __init__(self, response: httpx2.Response, sink: bytearray) -> None:
        self._response = response
        self._sink = sink

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._response.aiter_bytes():
            self._sink.extend(chunk)
            yield chunk

    async def aclose(self) -> None:
        await self._response.aclose()


class CapturingTransport(httpx2.AsyncBaseTransport):
    """Tees every response for `capture_url` into `captured` (invariant 6 fixture capture) without
    touching the adapter. Wraps any inner transport: a live `AsyncHTTPTransport`, or a
    `MockTransport` in tests. The body passes through as it arrives (W3.1-K1), so a stream the
    adapter stops reading at its first error frame (Z-stop) is captured up to there and never read
    to its end first, and the thinking relay streams as it does without the tee. Any other URL,
    the count_tokens pre-flight among them, is returned untouched."""

    def __init__(self, inner: httpx2.AsyncBaseTransport, capture_url: str) -> None:
        self._inner = inner
        self._capture_url = capture_url
        self._sinks: list[bytearray] = []

    @property
    def captured(self) -> list[bytes]:
        """Each captured response's decoded body, in order, as far as the adapter read it."""
        return [bytes(sink) for sink in self._sinks]

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        response = await self._inner.handle_async_request(request)
        if str(request.url) != self._capture_url:
            return response
        sink = bytearray()
        self._sinks.append(sink)
        headers = [
            (name, value)
            for name, value in response.headers.multi_items()
            if name.lower() not in _ENCODED_BODY_HEADERS
        ]
        return httpx2.Response(
            response.status_code,
            headers=headers,
            stream=_TeeStream(response, sink),
            request=request,
            extensions=response.extensions,
        )


def write_fixtures(
    captured: list[bytes], run_id: str, *, fixture_dir: Path = FIXTURE_DIR
) -> list[Path]:
    """Persist each captured brain response as an ordered golden fixture (invariant 6), namespaced
    UNDER the run's id (`hunt_<run_id>/call_NN.json`). Run-id scoping means a re-run writes a fresh
    directory instead of clobbering a prior run's fixtures (and the replay test pins one run, so it
    can never mix `call_NN` bytes across two different hunts). `fixture_dir` is an injection seam
    for the offline harness test (a tmp dir), so it never writes into the repo without spending."""
    run_dir = fixture_dir / f"hunt_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, raw in enumerate(captured, start=1):
        path = run_dir / f"call_{i:02d}.json"
        path.write_bytes(raw)
        paths.append(path)
    return paths


async def fresh_db_run_id(conn: aiosqlite.Connection) -> str:
    """The id of the one run a harness's fresh DB holds: the fixture directory's name when the hunt
    raised before it returned its id (a cancel, Ctrl-C). A DB with no run gives "unknown"."""
    async with conn.execute("SELECT id FROM runs ORDER BY started_at LIMIT 1") as cur:
        row = await cur.fetchone()
    return str(row[0]) if row is not None else "unknown"


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


async def run_capped_hunt(
    api_key: str,
    *,
    territory: str,
    thinking: dict[str, Any] | None,
    stream: bool,
    max_tokens: int,
    cost_ceiling_usd: float,
    max_iterations: int,
    client_timeout_s: float,
    db_prefix: str,
    fixture_kind: str,
    preflight: Preflight,
    report: Callable[[list[bytes]], None],
    inner: httpx2.AsyncBaseTransport | None = None,
    fixture_dir: Path | None = None,
    db_dir: Path | None = None,
) -> dict[str, Any]:
    """The paid path both gated hunts share: free pre-flight → one capped hunt on the teed client →
    fixtures + report. `thinking` None is the model's floor, the adapter's default. Each module
    passes its own `preflight` and `report`, so the names a caller patches there still run.

    `inner`, `fixture_dir` and `db_dir` are the offline seams. The paid path leaves them unset, so
    its run DB lands in a fresh directory under the system temp dir that nothing removes: since
    W3.3 that DB holds every attempt's request and response bytes.
    """
    registry = stub.build_registry()
    tee = CapturingTransport(inner or brain.live_transport(), brain.MESSAGES_URL)
    async with brain.anthropic_client(api_key, transport=tee, timeout=client_timeout_s) as client:
        n_in = await preflight(client, registry)
        print(f"[preflight] count_tokens OK — input_tokens={n_in}")
        brain_for = brain.adapter_brain_for(
            client=client,
            model=brain.SMOKE_MODEL,
            registry=registry,
            max_tokens=max_tokens,
            thinking=thinking,
            stream=stream,
        )
        db_path = Path(tempfile.mkdtemp(prefix=db_prefix, dir=db_dir)) / "rex.db"
        conn = await db.connect(db_path)
        run_id: str | None = None
        paths: list[Path] = []
        try:
            run_id = await run_hunt(
                conn,
                territory=territory,
                brain=brain_for(territory),
                registry=registry,
                cost_ceiling_usd=cost_ceiling_usd,
                max_iterations=max_iterations,
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
            # Persist first (invariant 6): every captured response is on disk before anything
            # else can fail (a cancel, a raising read above) and before the reporter re-reads it.
            if tee.captured:
                named = run_id if run_id is not None else await fresh_db_run_id(conn)
                # With no directory given, write_fixtures keeps its own default: a caller's
                # keyword would override one bound into write_fixtures itself.
                if fixture_dir is None:
                    paths = write_fixtures(tee.captured, named)
                else:
                    paths = write_fixtures(tee.captured, named, fixture_dir=fixture_dir)
                print(
                    f"[fixtures] wrote {len(paths)} {fixture_kind} fixture(s) under hunt_{named}/"
                )
            await conn.close()

    print(
        f"[hunt] run={run_id} outcome={outcome} reason={reason} brain_calls={len(tee.captured)} "
        f"spend=${spend:.4f} (ceiling ${cost_ceiling_usd})"
    )
    report(tee.captured)
    return {
        "run_id": run_id,
        "outcome": outcome,
        "reason": reason,
        "brain_calls": len(tee.captured),
        "spend": spend,
        "fixtures": paths,
    }


async def run_live_hunt(api_key: str) -> None:
    """The paid path: free pre-flight → one capped hunt on the teed client → fixtures + report."""
    await run_capped_hunt(
        api_key,
        territory=TERRITORY,
        thinking=None,
        stream=False,
        max_tokens=HUNT_MAX_TOKENS,
        cost_ceiling_usd=COST_CEILING_USD,
        max_iterations=MAX_ITERATIONS,
        client_timeout_s=120.0,
        db_prefix="rexhunter-hunt-",
        fixture_kind="golden",
        preflight=count_tokens_preflight,
        report=_report_findings,
    )


def run_main(*, preflight: Preflight, run_live: Callable[[str], Awaitable[object]]) -> None:
    """The entrypoint both gated hunts share: the paid-entry gate, then the free pre-flight alone
    on `--count-only`, else `run_live`."""
    api_key, count_only = brain.paid_entry(before="the pre-flight or hunt", paid="hunt")

    async def _go() -> None:
        if count_only:
            async with brain.anthropic_client(api_key, timeout=120.0) as client:
                n = await preflight(client, stub.build_registry())
            print(f"[count-only] input_tokens={n} — validated, NO paid call made.")
            return
        await run_live(api_key)

    asyncio.run(_go())


def main() -> None:
    run_main(preflight=count_tokens_preflight, run_live=run_live_hunt)


if __name__ == "__main__":
    main()
