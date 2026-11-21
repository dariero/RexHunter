"""The scripted keyword baseline (`P2.4`, D9, fork `baseline`): the stub daemon's brain on the gym.

A title matches when it holds "AI" as a word, in capitals, or "machine learning" in any case. A
hunt lists page 1, reads each match and catches what it read. The names are looked up inside each
test, so on a tree without the baseline each test fails on its own line. Every request goes to the
in-process gym; there is no model and no spend.
"""

from pathlib import Path

import httpx2
import pytest

from rexhunter import db, events
from rexhunter.board.catalogue import GYM_BOARDS, GYM_CATALOGUE
from rexhunter.board.client import BoardClient
from rexhunter.board.gym import build_gym
from rexhunter.board.tools import build_registry
from rexhunter.loop import run_hunt

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("AI Engineer, Agents", True),
        ("Staff Engineer, AI Infrastructure", True),
        ("Senior Machine Learning Engineer", True),
        ("Head of MACHINE LEARNING", True),
        ("Air Quality Analyst", False),
        ("Email Marketing Manager", False),
        ("Domain Name Specialist", False),
        ("Maintenance Planner", False),
        ("Aircraft Painter", False),
        ("Machine Operator", False),
        ("Learning and Development Partner", False),
        ("Applied ai Engineer", False),
        ("LLM Evaluation Engineer", False),
    ],
)
def test_a_title_matches_on_ai_as_a_word_or_machine_learning(title: str, expected: bool) -> None:
    from rexhunter.board.baseline import matches

    assert matches(title) is expected


def test_the_baseline_never_matches_a_planted_decoy() -> None:
    from rexhunter.board.baseline import matches

    decoys = [p.title for b in GYM_CATALOGUE.values() for p in b.postings if p.planted == "decoy"]
    assert decoys and not any(matches(title) for title in decoys)


async def test_a_baseline_hunt_reads_the_matches_and_catches_what_it_read(tmp_path: Path) -> None:
    from rexhunter.board.baseline import baseline_brain_for

    conn = await db.connect(tmp_path / "rex.db")
    try:
        transport = httpx2.ASGITransport(app=build_gym(GYM_CATALOGUE))
        async with BoardClient(transport) as client:
            registry = build_registry(client, GYM_BOARDS)
            run_id = await run_hunt(
                conn,
                territory="gym-lever",
                brain=baseline_brain_for(GYM_BOARDS)("gym-lever"),
                registry=registry,
            )
        evs = await db.read_events(conn, run_id)
        ((outcome,),) = await conn.execute_fetchall(
            "SELECT outcome FROM runs WHERE id = ?", (run_id,)
        )
    finally:
        await conn.close()

    calls = [e for e in evs if isinstance(e, events.ToolCallEvent)]
    reads = [events.PostingRef.model_validate_json(c.raw_request) for c in calls[1:]]
    captured = [e for e in evs if isinstance(e, events.PreyCapturedEvent)]
    postings = GYM_CATALOGUE["wombatworks"].postings
    expected = [p.posting_id for p in postings[:3]]  # the three Lever titles that match
    assert [c.tool for c in calls] == ["list_postings"] + ["read_posting"] * 3
    assert [r.posting_id for r in reads] == expected
    assert [(c.board, c.posting_id) for c in captured] == [("gym-lever", p) for p in expected]
    assert outcome == "completed"
