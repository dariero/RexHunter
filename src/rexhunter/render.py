"""The dumb renderer (ADR invariant 2) — draws a ViewState, nothing more.

A PURE, TOTAL function ``ViewState -> HTML`` fragment: no logic/I/O/clock, deterministic. The
frontend is a pure projection of the log (invariant 2); this is the last, presentation-only hop of
that projection. LLM output and scraped postings are untrusted input (invariant 3), so every
ViewState string crosses ``html.escape`` on the way into the markup — a posting containing
``<script>`` renders inert.

Rendering is NOT a learning objective (unlike the loop/log/hub), so the frugality rule inverts here:
no framework, no template engine, no build step — f-strings and stdlib ``html.escape``. The browser
is a dumb painter: it re-fetches this server-rendered board on each SSE tick (server.py), never
re-running the reducer. The board is a fragment (a ``<div class="board">``); the page shell (CSS +
the fetch/tick JS) lives in server.py and injects it.
"""

import html

from rexhunter.view import Phase, PreyCard, RunView, TerritoryView, ViewState


def _esc(value: str) -> str:
    """The one boundary untrusted ViewState strings cross into markup (invariant 3)."""
    return html.escape(value)


# The consciousness bubble shows a live ticker, not an archive: the RAW thinking is sliced to
# this tail and THEN escaped (slicing an escaped string could split an entity). The reducer stays
# lossless — the full stream is in the log for the ghost scrubber; this is presentation only.
_THINKING_TAIL_CHARS = 280


def _remaining(spent: float, ceiling: float | None) -> float | None:
    """The depleting-bar fraction (ADR Pillar 5 §6 — budget guards as HP/stamina mechanics).
    None = no denominator recorded (a pre-4a row, or a nonpositive ceiling): draw no bar.
    Otherwise clamp01(1 - spent/ceiling): the last brain call can overshoot the ceiling (the
    breaker checks BEFORE a call), so the floor is 0, never a negative width."""
    if ceiling is None or ceiling <= 0:
        return None
    return max(0.0, min(1.0, 1.0 - spent / ceiling))


def _bar(kind: str, remaining: float) -> str:
    # kind is one of our own literals ("hp"/"stamina") and the width a computed number — nothing
    # untrusted crosses into this markup.
    return f'<div class="bar {kind}"><div class="fill" style="width:{remaining:.0%}"></div></div>'


def _run_card(run: RunView) -> str:
    tool = _esc(run.current_tool) if run.current_tool is not None else "idle"
    rex = "hunting" if run.current_tool is not None else "idle"  # the Step-7 sprite hook
    hp = _remaining(run.spent_usd, run.cost_ceiling_usd)
    stamina = _remaining(run.turns, run.max_iterations)
    bars = (_bar("hp", hp) if hp is not None else "") + (
        _bar("stamina", stamina) if stamina is not None else ""
    )
    return (
        f'<article class="run" data-rex="{rex}"><h3>{_esc(run.run_id)}</h3>'
        f'<div class="tool">🔧 {tool}</div>'
        f'<div class="thinking">{_esc(run.thinking[-_THINKING_TAIL_CHARS:])}</div>'
        f"{bars}"
        f'<div class="stats">🎯 {run.prey_count} · ⚠️ {run.error_count} '
        f"· ${run.spent_usd:.4f}</div></article>"
    )


def _tile_state(tile: TerritoryView) -> str:
    """TerritoryView → the tile's scene state. latest_outcome None splits on the 4b None/None
    pairing: a real last_started_at means the latest run is OPEN (Rex hunts there NOW —
    "hunting"); None means never hunted — "dormant" (the 2b deferral, closed). "completed" is a
    fresh kill. Everything else — crashed / needs_help / aborted, and any unknown future
    outcome — is cracked earth: the attention-demanding tile (a PROXY from runs.outcome until
    park-and-persist lands awaiting_intervention, per the ADR), doubling as the fail-visible
    fallback that keeps this total (an unrecognized closure demands attention, never hides)."""
    if tile.latest_outcome is None:
        return "hunting" if tile.last_started_at is not None else "dormant"
    if tile.latest_outcome == "completed":
        return "fresh-kill"
    return "cracked-earth"


def _territory_tile(tile: TerritoryView) -> str:
    # The state rides a data-tile attribute the CSS paints (sprite art is a later step); the
    # territory name is config/scrape-adjacent input and crosses _esc like everything else (inv 3).
    return (
        f'<article class="tile" data-tile="{_tile_state(tile)}">'
        f'<span class="territory">{_esc(tile.territory)}</span></article>'
    )


# The verdict actions each status offers, matching the transition map (verdicts.py:44-48): awaiting
# accepts feast/release/amber; ambered re-enters; feasted/released are terminal (no button). The
# verdict strings are the events.Verdict wire values the shell POSTs to /verdict.
def _button(prey_id: str, verdict: str, label: str) -> str:
    return f'<button data-prey-id="{_esc(prey_id)}" data-verdict="{verdict}">{label}</button>'


def _actions(card: PreyCard) -> str:
    if card.status == "awaiting_verdict":
        return (
            '<span class="actions"><input class="reason" placeholder="reason…">'
            f"{_button(card.prey_id, 'feast', 'Feast')}"
            f"{_button(card.prey_id, 'release', 'Release')}"
            f"{_button(card.prey_id, 'amber', 'Amber')}</span>"
        )
    if card.status == "ambered":
        return f'<span class="actions">{_button(card.prey_id, "reenter", "Reenter")}</span>'
    return ""  # feasted / released: terminal, not re-votable


def _prey_row(card: PreyCard) -> str:
    badge = _esc(card.status.replace("_", " ").upper())
    return (
        f'<article class="prey status-{_esc(card.status)}">'
        f'<span class="territory">{_esc(card.territory)}</span>'
        f'<span class="posting">{_esc(card.posting)}</span>'
        f'<span class="badge">{badge}</span>{_actions(card)}</article>'
    )


def render(state: ViewState) -> str:
    """Draw the ViewState as an HTML board fragment. Pure, total, deterministic — the same input
    always yields the same markup (the render analogue of the reducer's determinism law). Cards
    are LIVE runs only (outcome is None, the runs ⊕ overlay); a closed run's story is its
    territory's tile — the 2b card→tile handoff, ticked live by the run-finished pulse (2a)."""
    icon = "☀️" if state.phase is Phase.DAY else "🌙"
    runs = "".join(_run_card(run) for run in state.runs if run.outcome is None) or (
        '<p class="empty">no active hunts</p>'
    )
    tiles = "".join(_territory_tile(tile) for tile in state.territories)
    pen = "".join(_prey_row(card) for card in state.pen) or '<p class="empty">pen empty</p>'
    # The daemon-level HP bar (4b's injected ceiling vs the global spend fold) — the terrarium's
    # whole-wallet health; no ceiling injected → the dollar figure alone, as ever.
    daemon_hp = _remaining(state.spent_usd, state.daemon_spend_ceiling_usd)
    hud_bar = _bar("hp", daemon_hp) if daemon_hp is not None else ""
    return (
        f'<div class="board" data-phase="{state.phase.value}">'
        f'<header class="hud"><span class="phase">{icon} {state.phase.value.upper()}</span>'
        f'{hud_bar}<span class="spend">${state.spent_usd:.4f}</span></header>'
        f'<section class="territories">{tiles}</section>'
        f'<section class="runs">{runs}</section>'
        f'<section class="pen"><h2>Prey Pen</h2>{pen}</section>'
        f"</div>"
    )
