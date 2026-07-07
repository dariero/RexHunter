"""Slice B · the dumb renderer.

``render(ViewState) -> HTML`` is a PURE, TOTAL, deterministic projection-to-markup (ADR invariant 2,
the last presentation hop). LLM output and scraped postings are untrusted input (invariant 3), so
every ViewState string is ``html.escape``-d — a posting carrying ``<script>`` renders inert. No
framework/template (frugality inverts: rendering is not a learning objective). The ``/viewstate``
endpoint smoke proves the server-rendered board reflects a seeded log.
"""

from pathlib import Path

import pytest

from rexhunter import db, events, render, server, verdicts
from rexhunter.view import Phase, PreyCard, RunView, TerritoryView, ViewState

_A_RUN = RunView(
    run_id="run-1",
    current_tool="sniff",
    thinking="scanning the gym…",
    spent_usd=0.0149,
    prey_count=1,
    error_count=0,
)
_A_PREY = PreyCard(prey_id="p1", territory="mock-gym", posting="posting:mock-gym", status="feasted")
_A_STATE = ViewState(
    high_water=3, runs=(_A_RUN,), pen=(_A_PREY,), spent_usd=0.0149, phase=Phase.DAY
)


def test_render_reflects_the_viewstate() -> None:
    """The dumb renderer draws exactly what is in the ViewState — run id, current tool, the penned
    posting, and the status (as both a CSS-class hook and a badge)."""
    html = render.render(_A_STATE)
    assert 'class="board"' in html
    assert "run-1" in html
    assert "sniff" in html  # current_tool
    assert "posting:mock-gym" in html  # the posting
    assert "status-feasted" in html  # the status drives a CSS class
    assert "FEASTED" in html  # the badge


def test_render_is_deterministic() -> None:
    """Same ViewState -> byte-identical HTML (no now()/random inside) — the render analogue of the
    reducer's determinism law."""
    assert render.render(_A_STATE) == render.render(_A_STATE)


def test_render_is_total_over_an_empty_viewstate() -> None:
    """A board renders from an empty ViewState — no run, no prey — without crashing."""
    empty = ViewState(high_water=0, runs=(), pen=(), spent_usd=0.0, phase=Phase.DAY)
    assert 'class="board"' in render.render(empty)


def test_render_escapes_an_untrusted_posting() -> None:
    """Invariant 3: a scraped/LLM posting is untrusted — a ``<script>`` payload must render inert
    (escaped), never as live markup that could execute."""
    hostile = PreyCard(
        prey_id="p",
        territory="mock-gym",
        posting="<script>alert('xss')</script>",
        status="awaiting_verdict",
    )
    state = ViewState(high_water=1, runs=(), pen=(hostile,), spent_usd=0.0, phase=Phase.NIGHT)
    html = render.render(state)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_render_escapes_untrusted_thinking() -> None:
    """Thinking is raw LLM output (invariant 3) — an injected tag in the reasoning stream is escaped
    too, not just the posting."""
    run = RunView(
        run_id="r",
        current_tool=None,
        thinking="<img src=x onerror=alert(1)>",
        spent_usd=0.0,
        prey_count=0,
        error_count=0,
    )
    state = ViewState(high_water=1, runs=(run,), pen=(), spent_usd=0.0, phase=Phase.DAY)
    html = render.render(state)
    assert "<img" not in html
    assert "&lt;img" in html


# ── B2: verdict-action buttons, conditional on the state machine ──────────────────────────────────

_AWAITING = PreyCard(prey_id="pa", territory="mock-gym", posting="job-a", status="awaiting_verdict")
_AMBERED = PreyCard(prey_id="pb", territory="mock-gym", posting="job-b", status="ambered")


def _pen(*prey: PreyCard) -> ViewState:
    return ViewState(high_water=1, runs=(), pen=prey, spent_usd=0.0, phase=Phase.DAY)


def test_render_awaiting_prey_offers_the_three_verdict_buttons() -> None:
    """An awaiting row exposes the awaiting→{feast,release,amber} transitions (verdicts.py:45-47),
    each carrying its prey_id, plus the reason/provenance note input."""
    html = render.render(_pen(_AWAITING))
    for verdict in ("feast", "release", "amber"):
        assert f'data-verdict="{verdict}"' in html
    assert 'data-prey-id="pa"' in html
    assert 'class="reason"' in html


def test_render_gives_a_terminal_prey_no_verdict_buttons() -> None:
    """feasted/released are terminal (no outgoing transition) — a resolved row is not re-votable, so
    it renders no action button. _A_PREY is feasted."""
    assert "data-verdict" not in render.render(_pen(_A_PREY))


def test_render_offers_only_reenter_on_an_ambered_prey() -> None:
    """ambered's one transition is REENTER→awaiting (verdicts.py:48): a Reenter button and none of
    the awaiting-only verdicts."""
    html = render.render(_pen(_AMBERED))
    assert 'data-verdict="reenter"' in html
    assert 'data-verdict="feast"' not in html


# ── 2b: closed-run cards prune into the territory tile row ───────────────────────────────────────

_OPEN_RUN = RunView(
    run_id="run-open",
    current_tool="sniff",
    thinking="",
    spent_usd=0.0,
    prey_count=0,
    error_count=0,
    territory="mock-gym",
    outcome=None,
)
_CLOSED_RUN = RunView(
    run_id="run-closed",
    current_tool=None,
    thinking="",
    spent_usd=0.0,
    prey_count=1,
    error_count=0,
    territory="mock-gym",
    outcome="completed",
)


def _board(
    runs: tuple[RunView, ...] = (), territories: tuple[TerritoryView, ...] = ()
) -> ViewState:
    return ViewState(
        high_water=1, runs=runs, pen=(), spent_usd=0.0, phase=Phase.DAY, territories=territories
    )


def _tile(territory: str = "mock-gym", latest_outcome: str | None = "completed") -> TerritoryView:
    return TerritoryView(
        territory=territory,
        latest_outcome=latest_outcome,
        last_started_at="2026-07-01T00:00:00+00:00",
    )


def test_render_shows_only_open_runs_as_cards() -> None:
    """A closed run (outcome set) leaves the card list — its story lives in its territory's tile
    (the 2b card→tile handoff). Exactly the open run renders as a card."""
    html = render.render(_board(runs=(_OPEN_RUN, _CLOSED_RUN)))
    assert html.count('<article class="run"') == 1
    assert "run-open" in html
    assert "run-closed" not in html


def test_render_closed_runs_do_not_render_cards() -> None:
    """All-closed → zero run cards and the 'no active hunts' placeholder — an empty section,
    never a crash (render stays total)."""
    html = render.render(_board(runs=(_CLOSED_RUN,)))
    assert '<article class="run"' not in html
    assert "no active hunts" in html


def test_render_draws_a_territory_tile_per_territory() -> None:
    """N TerritoryViews → N tiles inside <section class="territories">, each naming its
    territory."""
    html = render.render(
        _board(territories=(_tile("greenhouse", "completed"), _tile("mock-gym", None)))
    )
    assert 'class="territories"' in html
    assert html.count('<article class="tile"') == 2
    assert "greenhouse" in html
    assert "mock-gym" in html


def test_render_tile_state_maps_outcome() -> None:
    """latest_outcome → data-tile: completed reads fresh-kill; every attention-demanding closure
    (crashed / needs_help / aborted — and, fail-visible, an UNKNOWN outcome, keeping render total)
    reads cracked-earth (the documented ADR proxy until park-and-persist lands
    awaiting_intervention); None with a real last_started_at = the latest run is still open →
    hunting. 'dormant' (the None/None pairing — 4b) is covered by test_render_dormant_tile."""
    cases = [
        ("completed", "fresh-kill"),
        ("crashed", "cracked-earth"),
        ("needs_help", "cracked-earth"),
        ("aborted", "cracked-earth"),
        ("some-future-outcome", "cracked-earth"),  # unknown: fail-visible, never invisible
        (None, "hunting"),
    ]
    for outcome, tile_state in cases:
        html = render.render(_board(territories=(_tile(latest_outcome=outcome),)))
        assert f'data-tile="{tile_state}"' in html, f"{outcome!r} should map to {tile_state!r}"


def test_render_open_run_and_its_tile_coexist() -> None:
    """A pure-render contract (plan Flag B): a live card and its territory's tile NEVER suppress
    each other — no tile-vs-card deduping in the renderer. (Live-assembler note: an in-flight hunt
    IS the territory's latest run, so the live tile reads 'hunting' during a hunt; the closed
    outcome returns on the 2a finish pulse. This hand-built state pins only the renderer.)"""
    html = render.render(_board(runs=(_OPEN_RUN,), territories=(_tile("mock-gym", "completed"),)))
    assert html.count('<article class="run"') == 1
    assert "run-open" in html
    assert 'data-tile="fresh-kill"' in html


def test_render_territory_name_is_escaped() -> None:
    """Invariant 3 extends to the new markup: a hostile territory name renders inert in the tile —
    the escape wall grows with the board, it doesn't just survive it."""
    html = render.render(_board(territories=(_tile("<script>alert(1)</script>", "completed"),)))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


# ── Step 5: HP/stamina bars, the sprite-state hook, the dormant tile ─────────────────────────────


def _armed_run(
    *,
    spent: float = 0.0,
    ceiling: float | None = None,
    turns: int = 0,
    max_iters: int | None = None,
    tool: str | None = "sniff",
) -> RunView:
    """An open run with the 4a-recorded caps set (or unset — the pre-4a shape)."""
    return RunView(
        run_id="run-armed",
        current_tool=tool,
        thinking="",
        spent_usd=spent,
        prey_count=0,
        error_count=0,
        turns=turns,
        territory="mock-gym",
        outcome=None,
        cost_ceiling_usd=ceiling,
        max_iterations=max_iters,
    )


def test_render_run_hp_bar_depletes_with_spend() -> None:
    """The ADR's 'budget guards visible as HP mechanics' (Pillar 5 §6): a quarter of the ceiling
    spent leaves a 75%-full depleting HP bar on the run card."""
    html = render.render(_board(runs=(_armed_run(spent=0.25, ceiling=1.0),)))
    assert 'class="bar hp"' in html
    assert "width:75%" in html


def test_render_run_stamina_bar_depletes_with_turns() -> None:
    """Stamina = remaining turns: 4 of 10 spent → 60% full."""
    html = render.render(_board(runs=(_armed_run(turns=4, max_iters=10),)))
    assert 'class="bar stamina"' in html
    assert "width:60%" in html


def test_render_bars_clamp_at_zero() -> None:
    """The last brain call can overshoot the ceiling (the breaker checks BEFORE a call) — both
    bars floor at 0%, never a negative width."""
    html = render.render(_board(runs=(_armed_run(spent=1.5, ceiling=1.0, turns=12, max_iters=10),)))
    assert html.count("width:0%") == 2
    assert "width:-" not in html


def test_render_run_without_ceilings_draws_no_bars() -> None:
    """A pre-4a run (None ceilings) draws no bar at all — the card renders intact (total),
    exactly as before Step 5."""
    html = render.render(_board(runs=(_armed_run(),)))
    assert 'class="bar' not in html
    assert "run-armed" in html


def test_render_hud_daemon_hp_bar() -> None:
    """The daemon-level HP bar in the HUD: global spend against the injected daemon ceiling
    (4b); no ceiling → no bar, the dollar figure alone as before."""
    armed = ViewState(
        high_water=1,
        runs=(),
        pen=(),
        spent_usd=0.5,
        phase=Phase.DAY,
        daemon_spend_ceiling_usd=2.0,
    )
    html = render.render(armed)
    assert 'class="bar hp"' in html
    assert "width:75%" in html
    assert 'class="bar' not in render.render(_board())  # uninjected: no bar anywhere


def test_render_sprite_state_attribute() -> None:
    """The Step-7 sprite hook: an open tool → data-rex="hunting"; no open tool → "idle". A state
    attribute the CSS keyframes will bind to — no art here."""
    assert 'data-rex="hunting"' in render.render(_board(runs=(_armed_run(tool="sniff"),)))
    assert 'data-rex="idle"' in render.render(_board(runs=(_armed_run(tool=None),)))


def test_render_thinking_shows_only_the_tail() -> None:
    """The consciousness bubble is a live ticker, not an archive: render slices the RAW thinking
    string to its tail and THEN escapes (slicing after escaping could split an `&lt;`-style
    entity), keeping the reducer lossless for ghost scrubbing — presentation truncation only."""
    prefix = "PREFIX-" + "x" * 400
    tail = "the live tail of Rex's reasoning <b>escaped</b>"
    run = RunView(
        run_id="r",
        current_tool=None,
        thinking=prefix + tail,
        spent_usd=0.0,
        prey_count=0,
        error_count=0,
    )
    html = render.render(_board(runs=(run,)))
    assert "escaped" in html
    assert "&lt;b&gt;" in html  # the tail still crosses _esc (inv 3)
    assert "PREFIX-" not in html  # the archive stays in the log, not the bubble


def test_render_dormant_tile() -> None:
    """The 4b None/None pairing drawn: latest_outcome None with NO last_started_at = never
    hunted → dormant; None WITH a timestamp stays hunting (an open run). The 2b deferral closes."""
    dormant = TerritoryView(territory="fresh-lands", latest_outcome=None, last_started_at=None)
    html = render.render(_board(territories=(dormant,)))
    assert 'data-tile="dormant"' in html
    hunting = render.render(_board(territories=(_tile(latest_outcome=None),)))
    assert 'data-tile="hunting"' in hunting


@pytest.mark.anyio
async def test_viewstate_endpoint_renders_the_board(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The server-rendered board: /viewstate folds the seeded log into a ViewState and renders it.
    Called as the callable directly (the house idiom — no TestClient), pointed at a temp DB."""
    db_path = tmp_path / "rex.db"
    monkeypatch.setattr(server, "DB_PATH", str(db_path))
    conn = await db.connect(db_path)
    try:
        run_id = await db.start_run(conn, territory="mock-gym")
        await db.append_event(conn, run_id, events.ToolCallEvent(tool="sniff", raw_request=b"{}"))
        await verdicts.capture_prey(conn, run_id, territory="mock-gym", posting="posting:mock-gym")
    finally:
        await conn.close()
    resp = await server.get_viewstate()
    body = bytes(resp.body).decode()
    assert 'class="board"' in body
    assert "posting:mock-gym" in body
    assert "sniff" in body
