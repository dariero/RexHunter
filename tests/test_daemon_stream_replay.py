"""P5 Daemon live-wiring · W.3 — offline replay of the ONE gated live daemon hunt (the gate).

`tests/fixtures/daemon_hunt_<run_id>.sse` is the daemon's OUTBOUND `/events` stream for the single
gated live hunt (run `4688398b…`, `claude-sonnet-5`, thinking-on + streaming, driven by the real
lifespan, $0.0224). It is the canonical stream rebuilt from the log via `server.catch_up` +
`Envelope.sse` (the envelope carries the stored payload, so this equals the live frames) so it is
free of the curl-reconnect duplicates the raw capture had; one id-less heartbeat is appended,
exactly as the daemon emits. Its 15 data frames were VERIFIED byte-identical to the raw curl capture
(deduped by id) — so this is the daemon's actual outbound bytes (invariant 6), not a rebuild that
merely resembles them.

This is W.3's novel surface — P3's `/events` smoke, now over a REAL streaming brain in the daemon
(vs the buffered stub). Re-driving the frames offline (no network, no spend) certifies the gate:

  - Rex's reasoning streamed as `thinking_delta` frames, committed DURING each brain call and
    interleaved with the tool events — the first delta's id precedes the turn's `tool_call` id
    (incremental, not batched — the live-feed property W.2 proves structurally, here end to end);
  - the `tool_call` carries its VERBATIM signed thinking block (Unit 3c, invariant 6);
  - `tool_call` / `tool_result` pair by `tool_use_id`;
  - the `usage` frames FOLD (via `cost`) to the run's reported spend — the id-scoped daemon budget
    (`daemon_spend_usd` is exactly this fold) reconciles to $0.0224;
  - a heartbeat frame carries no id (never advances the cursor);
  - ids are strictly monotonic.

This hunt ended `needs_help` (the model judged the mock board unworkable), so there is no
`prey_captured` in THIS stream — that event type's live streaming through the daemon is covered by
the free stub smoke (P3's frames: tool_call / tool_result / prey_captured / heartbeat).
"""

import json
from pathlib import Path

import pytest

from rexhunter import cost
from rexhunter.events import (
    ThinkingDelta,
    ToolCallEvent,
    ToolResultEvent,
    TrajectoryEvent,
    UsageEvent,
    decode_event,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
RUN_ID = "4688398b-f24d-42ee-be11-715aa9a22fb5"  # THE gated live daemon hunt; the fixture pins it
FIXTURE = FIXTURE_DIR / f"daemon_hunt_{RUN_ID}.sse"
LIVE_SPEND_USD = 0.0224  # the [hunt] report; fold_cost over the two UsageEvents must reproduce it


def _parse_stream(raw: str) -> tuple[list[tuple[int, TrajectoryEvent]], int]:
    """Parse the daemon's `/events` SSE into (id, decoded event) pairs + a heartbeat count.

    Each data frame is `id: N` + `data: <stored payload>`; the payload crosses the SAME validation
    boundary the log read uses (`decode_event`, invariant 3). A heartbeat is a lone `: keep-alive`
    frame — asserted here to carry NO id (it must never advance the client's Last-Event-ID).
    """
    events: list[tuple[int, TrajectoryEvent]] = []
    heartbeats = 0
    for frame in raw.split("\n\n"):
        if not frame.strip():
            continue
        lines = frame.split("\n")
        if any(line.startswith(": keep-alive") for line in lines):
            assert not any(line.startswith("id:") for line in lines)  # heartbeat: no id
            heartbeats += 1
            continue
        fid = next(int(line[4:]) for line in lines if line.startswith("id: "))
        data = next(line[6:] for line in lines if line.startswith("data: "))
        events.append((fid, decode_event(data)))
    return events, heartbeats


def _load() -> tuple[list[int], list[TrajectoryEvent], int]:
    ids_events, heartbeats = _parse_stream(FIXTURE.read_text())
    return [i for i, _ in ids_events], [e for _, e in ids_events], heartbeats


def test_fixture_exists_and_ids_are_strictly_monotonic() -> None:
    ids, events, _ = _load()
    assert events, "the daemon stream fixture is empty"
    assert ids == sorted(ids) and len(set(ids)) == len(ids)  # strictly monotonic, no dupes


def test_reasoning_streamed_incrementally_before_the_tool_dispatched() -> None:
    """The live-feed property: `thinking_delta` frames arrive as they stream, and the FIRST one's id
    precedes the turn's `tool_call` id — the deltas were committed DURING the brain call, before the
    decision returned and the tool dispatched (incremental, not buffered)."""
    ids, events, _ = _load()
    pairs = list(zip(ids, events, strict=True))
    deltas = [(i, e) for i, e in pairs if isinstance(e, ThinkingDelta)]
    calls = [(i, e) for i, e in pairs if isinstance(e, ToolCallEvent)]
    assert deltas and calls
    first_delta_id = deltas[0][0]
    first_call_id = calls[0][0]
    assert first_delta_id < first_call_id  # reasoning streamed BEFORE the tool dispatched
    # and the streamed text is real, non-empty reasoning (not blank deltas).
    assert "".join(e.text for _, e in deltas).strip()


def test_tool_call_carries_its_verbatim_signed_thinking_block_and_pairs_with_its_result() -> None:
    _, events, _ = _load()
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(calls) == 1 and len(results) == 1
    call, result = calls[0], results[0]
    assert call.tool == "sniff"
    assert call.tool_use_id and call.tool_use_id == result.tool_use_id  # paired, correlation intact
    # the VERBATIM signed thinking block rides on the tool_call (Unit 3c, invariant 6).
    block = json.loads(call.thinking)
    assert block["type"] == "thinking" and len(block.get("signature", "")) > 100


def test_usage_frames_fold_to_the_reported_daemon_spend() -> None:
    """Budget reconciliation: the stream's `usage` frames fold (via `cost`) to the run's spend. This
    fold IS `scheduler.daemon_spend_usd` (fold_cost over the log's UsageEvents), so the id-scoped
    daemon budget reconciles to the hunt's actual $0.0224."""
    _, events, _ = _load()
    usage = [e for e in events if isinstance(e, UsageEvent)]
    assert len(usage) == 2  # two brain calls
    assert cost.fold_cost(events) == pytest.approx(LIVE_SPEND_USD, abs=1e-4)


def test_stream_carries_a_heartbeat_and_ends_needs_help_without_prey() -> None:
    """A heartbeat frame was present (id-less — the parser asserts it never carries an id). The
    hunt ended `needs_help`, so the stream carries NO `prey_captured` — that event's live stream is
    covered by the free stub smoke (P3's tool_call / tool_result / prey_captured frames)."""
    _, events, heartbeats = _load()
    assert heartbeats >= 1
    assert not any(type(e).__name__ == "PreyCapturedEvent" for e in events)
