"""Property-based DoD for the projection reducer (ADR invariant 2 — the pillar the whole system
exists to serve).

``view.project(rows, clock)`` is a PURE, TOTAL fold of the trajectory log into a ``ViewState`` the
retro-game renderer draws. The live renderer and the ghost replayer are ONE reducer fed by two
cursors (inv 2). Four guards form the testable core; each mirrors a proven P3 streaming guarantee at
the view layer:

  (a) incremental == batch  — prefix+tail == whole  [P3 catch-up+splice, test_stage3_gate.py:104]
  (b) idempotent replay     — a row id <= high-water is a no-op       [P3 id dedup,        :144]
  (c) determinism           — same (log, clock) -> same ViewState     [P3 byte-identical,  :81]
  two-cursor (inv 2)        — one run read by id-cursor == by seq-cursor

The strategy generates a GLOBALLY id-monotonic interleaved multi-run stream — the SSE delivery
reality (``catch_up`` and the live queue are both ``ORDER BY id``, server.py:120,162). It never
generates arbitrary id disorder: the high-water guard is correct BECAUSE real delivery is monotonic,
so a disordered stream would test a condition that cannot occur and yield false failures.

view.py is pure — imports only ``events`` + ``cost`` (mirrors cost.py's "below the loop" rule); the
purity itself is asserted below (``test_view_imports_only_events_and_cost``).
"""

import ast
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from rexhunter import events, view
from rexhunter.view import LogRow, RunView, ViewState

# A fixed injected clock (inv 5): noon UTC -> DAY. project() never reads a wall clock, so every test
# passes the instant in. created_at on rows is inert here (the reducer's fold is clock-free — the
# clock enters only finalize), so a single constant suffices.
_CLOCK = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
_CREATED_AT = "2026-07-06T00:00:00+00:00"

_TOOLS = ["sniff", "fetch", "parse"]
_MODELS = ["claude-sonnet-5", "unknown-model"]  # unknown -> cost.py high fallback (still a fold)
_TERRITORIES = ["mock-gym", "greenhouse"]
_RUN_IDS = ["run-a", "run-b", "run-c"]

_small_bytes = st.binary(max_size=6)  # inv-6 raw I/O may be non-utf8 binary — the reducer must cope
_small_text = st.text(max_size=8)
# "" is the STUB reality (loop.py:198,233): the reducer must pair tool call/result by run-scoped
# position, NOT tool_use_id equality (which collides at "").
_tool_use_ids = st.sampled_from(["", "tu_1", "tu_2"])


@st.composite
def _producible_event(draw: st.DrawFn) -> events.TrajectoryEvent:
    """One event of a type the system can actually emit (events.py producers). SniffEvent is
    DELIBERATELY excluded — it has no producer in src/ (a P2.1 vestige), so a live stream never
    carries it; its totality is checked separately."""
    kind = draw(st.integers(min_value=0, max_value=5))
    if kind == 0:
        return events.ToolCallEvent(
            tool=draw(st.sampled_from(_TOOLS)),
            raw_request=draw(_small_bytes),
            tool_use_id=draw(_tool_use_ids),
            thinking=draw(_small_bytes),
        )
    if kind == 1:
        return events.ToolResultEvent(
            tool=draw(st.sampled_from(_TOOLS)),
            raw_request=draw(_small_bytes),
            raw_response=draw(_small_bytes),
            tool_use_id=draw(_tool_use_ids),
        )
    if kind == 2:
        return events.PreyCapturedEvent(
            prey_id=draw(st.uuids().map(str)),
            territory=draw(st.sampled_from(_TERRITORIES)),
            raw_posting=draw(_small_bytes),
        )
    if kind == 3:
        return events.ErrorEvent(
            tool=draw(st.sampled_from([*_TOOLS, "<brain>", "<loop>"])),
            retryable=draw(st.booleans()),
            error=draw(_small_text),
            raw_request=draw(_small_bytes),
        )
    if kind == 4:
        return events.ThinkingDelta(text=draw(_small_text))
    return events.UsageEvent(
        model=draw(st.sampled_from(_MODELS)),
        input_tokens=draw(st.integers(min_value=0, max_value=5000)),
        output_tokens=draw(st.integers(min_value=0, max_value=5000)),
    )


@st.composite
def _monotonic_log(draw: st.DrawFn, *, run_ids: Sequence[str] | None = None) -> list[LogRow]:
    """A globally id-monotonic, per-run seq-monotonic interleaved log — the SSE delivery reality.

    The global ``id`` strictly increases across the whole interleaved stream (position + 1); each
    run's ``seq`` increases independently. This is exactly what single-writer monotonic append
    guarantees (inv 7, db.py:112-121). No arbitrary id disorder is produced — see the module
    docstring for why that would be an unsound test.
    """
    rids = (
        list(run_ids)
        if run_ids is not None
        else draw(st.lists(st.sampled_from(_RUN_IDS), min_size=1, max_size=3, unique=True))
    )
    n = draw(st.integers(min_value=0, max_value=25))
    seq_of: dict[str, int] = {}
    rows: list[LogRow] = []
    for i in range(n):
        rid = draw(st.sampled_from(rids))
        event = draw(_producible_event())
        seq = seq_of.get(rid, 0)
        seq_of[rid] = seq + 1
        rows.append(LogRow(id=i + 1, seq=seq, run_id=rid, created_at=_CREATED_AT, event=event))
    return rows


def _run_view(state: ViewState, run_id: str) -> RunView | None:
    return next((rv for rv in state.runs if rv.run_id == run_id), None)


# ── (a) incremental == batch — the catch-up==live-splice guarantee at the view layer ──────────────


@settings(deadline=None)
@given(rows=_monotonic_log(), data=st.data())
def test_incremental_equals_batch(rows: list[LogRow], data: st.DataObject) -> None:
    """Folding the whole log == folding a prefix then folding the tail onto that accumulator, at
    ANY split. Stated on the clock-free ``fold`` (not project). Mirrors P3 reconnect: a viewer that
    replays only the missed tail onto its snapshot ends identical to one never disconnected
    (test_stage3_gate.py:104)."""
    i = data.draw(st.integers(min_value=0, max_value=len(rows)))
    whole = view.fold(rows)
    stepwise = view.fold_from(view.fold(rows[:i]), rows[i:])
    assert whole == stepwise


# ── (b) idempotent replay — the monotonic-id dedup at the view layer ──────────────────────────────


@settings(deadline=None)
@given(rows=_monotonic_log(), data=st.data())
def test_idempotent_replay_is_a_noop(rows: list[LogRow], data: st.DataObject) -> None:
    """Any row already folded in (id <= high-water) re-applied is a no-op — on the accumulator AND
    on the finalized ViewState. Dedup is on the GLOBAL id, never seq. Mirrors the P3 splice dropping
    ``id <= high-water`` so a reconnect double-delivers nothing (test_stage3_gate.py:144)."""
    acc = view.fold(rows)
    if not rows:
        return  # nothing < high-water to replay
    # Every row in the log has id <= acc.high_water (== the last id), so all are replays.
    replays = data.draw(st.lists(st.sampled_from(rows), max_size=2 * len(rows)))
    replayed = view.fold_from(acc, replays)
    assert replayed == acc
    assert view.finalize(replayed, _CLOCK) == view.finalize(acc, _CLOCK)


# ── (c) determinism — the byte-identical-feeds guarantee at the view layer ────────────────────────


@settings(deadline=None)
@given(rows=_monotonic_log())
def test_determinism_same_input_same_viewstate(rows: list[LogRow]) -> None:
    """Same (log, clock) -> identical ViewState, byte-for-byte, with no now()/random inside. Mirrors
    P3 byte-identical feeds from the stored-payload envelope (test_stage3_gate.py:81)."""
    first = view.project(rows, _CLOCK)
    second = view.project(rows, _CLOCK)
    assert first == second
    assert repr(first) == repr(second)  # the byte-level check: no nondeterministic ordering


# ── two-cursor (inv 2): "one renderer, two cursors" — both CONTRACT LOCKS, not discriminators ─────


@settings(deadline=None)
@given(rows=_monotonic_log(run_ids=["run-solo"]))
def test_two_cursor_single_run_is_the_literal_inv2_claim(rows: list[LogRow]) -> None:
    """CONTRACT LOCK (not a discriminating property). Within ONE run, single-writer monotonic append
    (inv 7) makes the id cursor (``ORDER BY id``, catch_up server.py:120) and the seq cursor
    (``ORDER BY seq``, read_events db.py:159) the SAME row list, so project() is trivially equal.
    Its job is to PIN the inv-2 "live renderer and ghost replayer are one renderer" contract: it
    fails only if a future reducer starts depending on which cursor field it reads."""
    by_id = sorted(rows, key=lambda r: r.id)
    by_seq = sorted(rows, key=lambda r: r.seq)
    assert view.project(by_id, _CLOCK) == view.project(by_seq, _CLOCK)


@settings(deadline=None)
@given(rows=_monotonic_log(run_ids=_RUN_IDS))
def test_two_cursor_multi_run_tripwire(rows: list[LogRow]) -> None:
    """CONTRACT LOCK / regression tripwire. For each run, its GHOST projection (that run's rows read
    by the per-run seq cursor, ``WHERE run_id=? ORDER BY seq``) must yield the SAME per-run RunView
    as the LIVE projection of the whole globally-id-ordered interleaved stream. Catches a reducer
    that buckets or dedups on ``seq`` (which resets per run) instead of the global ``id``, or that
    lets one run's events bleed into another's card. Sound under the id-monotonic delivery premise
    (both feeds are monotonic within their scope). Global fields (high_water, total spend, pen)
    legitimately differ between one-run and all-runs and are deliberately not compared."""
    live = view.project(sorted(rows, key=lambda r: r.id), _CLOCK)
    for rid in {r.run_id for r in rows}:
        ghost_rows = sorted((r for r in rows if r.run_id == rid), key=lambda r: r.seq)
        ghost = view.project(ghost_rows, _CLOCK)
        assert _run_view(ghost, rid) == _run_view(live, rid)


# ── totality + a grounded example ─────────────────────────────────────────────────────────────────


def test_reducer_is_total_over_the_non_producible_sniff_event() -> None:
    """The reducer is TOTAL over the WHOLE union, including SniffEvent (no producer in src/, so the
    strategy never generates it) — a legacy or hand-edited log row must not crash it."""
    row = LogRow(id=1, seq=0, run_id="r", created_at=_CREATED_AT, event=events.SniffEvent(prey="x"))
    state = view.project([row], _CLOCK)
    assert state.high_water == 1  # applied (advanced high-water), folds to nothing visible


def test_projects_a_real_stub_hunt_trajectory() -> None:
    """The §2 finding, pinned: the default stub hunt emits tool_call -> tool_result -> prey_captured
    (loop.py:195,229; verdicts.py:72). The reducer projects one penned prey and a closed tool."""
    rows = [
        LogRow(
            id=1,
            seq=0,
            run_id="run-1",
            created_at=_CREATED_AT,
            event=events.ToolCallEvent(tool="sniff", raw_request=b"{}"),
        ),
        LogRow(
            id=2,
            seq=1,
            run_id="run-1",
            created_at=_CREATED_AT,
            event=events.ToolResultEvent(
                tool="sniff", raw_request=b"{}", raw_response=b"posting:mock-gym"
            ),
        ),
        LogRow(
            id=3,
            seq=2,
            run_id="run-1",
            created_at=_CREATED_AT,
            event=events.PreyCapturedEvent(
                prey_id="p1", territory="mock-gym", raw_posting=b"posting:mock-gym"
            ),
        ),
    ]
    state = view.project(rows, _CLOCK)
    assert state.high_water == 3
    assert len(state.pen) == 1
    assert state.pen[0].territory == "mock-gym"
    assert state.pen[0].posting == "posting:mock-gym"
    # The pure (trajectory) tier knows only the CAPTURE-time status; the real current status is the
    # ⊕ pen_events tier, overlaid by the assembler OUTSIDE this reducer (viewstate.build_viewstate).
    assert state.pen[0].status == "awaiting_verdict"
    (run,) = state.runs
    assert run.current_tool is None  # the tool_result closed the sniff call (run-scoped pairing)
    assert run.prey_count == 1


# ── purity DoD: view.py imports ONLY events + cost (no db/hub/server/verdicts/loop/scheduler) ─────


def test_view_imports_only_events_and_cost() -> None:
    """DoD: view.py is a pure projection — no I/O module in its import graph, so it is
    Hypothesis-testable in isolation. Asserts the rexhunter-internal imports are EXACTLY
    {events, cost} (importing verdicts/db/etc. would transitively pull in aiosqlite)."""
    tree = ast.parse(Path(view.__file__).read_text())
    internal: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("rexhunter"):
            # `from rexhunter import cost, events` -> the NAMES are the submodules; `from
            # rexhunter.events import X` -> the module itself is the dependency.
            if node.module == "rexhunter":
                internal.update(f"rexhunter.{a.name}" for a in node.names)
            else:
                internal.add(node.module)
        elif isinstance(node, ast.Import):
            internal.update(a.name for a in node.names if a.name.startswith("rexhunter"))
    assert internal == {"rexhunter.events", "rexhunter.cost"}
