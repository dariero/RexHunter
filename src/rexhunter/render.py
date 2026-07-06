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

from rexhunter.view import Phase, PreyCard, RunView, ViewState


def _esc(value: str) -> str:
    """The one boundary untrusted ViewState strings cross into markup (invariant 3)."""
    return html.escape(value)


def _run_card(run: RunView) -> str:
    tool = _esc(run.current_tool) if run.current_tool is not None else "idle"
    return (
        f'<article class="run"><h3>{_esc(run.run_id)}</h3>'
        f'<div class="tool">🔧 {tool}</div>'
        f'<div class="thinking">{_esc(run.thinking)}</div>'
        f'<div class="stats">🎯 {run.prey_count} · ⚠️ {run.error_count} '
        f"· ${run.spent_usd:.4f}</div></article>"
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
    always yields the same markup (the render analogue of the reducer's determinism law)."""
    icon = "☀️" if state.phase is Phase.DAY else "🌙"
    runs = "".join(_run_card(run) for run in state.runs) or '<p class="empty">no active hunts</p>'
    pen = "".join(_prey_row(card) for card in state.pen) or '<p class="empty">pen empty</p>'
    return (
        f'<div class="board" data-phase="{state.phase.value}">'
        f'<header class="hud"><span class="phase">{icon} {state.phase.value.upper()}</span>'
        f'<span class="spend">${state.spent_usd:.4f}</span></header>'
        f'<section class="runs">{runs}</section>'
        f'<section class="pen"><h2>Prey Pen</h2>{pen}</section>'
        f"</div>"
    )
