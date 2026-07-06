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
from rexhunter.view import Phase, PreyCard, RunView, ViewState

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
