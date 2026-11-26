"""Eval tier 2's score (`P2.5`): a run's catch against the gym's planted marks, folded from its log.

The score's names are looked up inside each test, so on a tree without `board.score` each test
fails on its own line. The recorded runs are the six first-wave hunts of 26 Nov
(tests/test_gym_sonnet_5_5_hunt_replay.py), each replayed from its fixture streams into a fresh log
and scored there; the other logs come from the keyword baseline or a scripted brain on the
in-process gym. No model, no spend, no network.
"""

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import aiosqlite
import anthropic
import httpx2
import pytest

from rexhunter import brain, db
from rexhunter.board import Board
from rexhunter.board.adapters import ADAPTERS, body_text
from rexhunter.board.baseline import baseline_brain_for
from rexhunter.board.catalogue import EVAL_BOARDS, EVAL_CATALOGUE, GYM_BOARDS, GYM_CATALOGUE
from rexhunter.board.client import BoardClient
from rexhunter.board.gym import build_gym
from rexhunter.board.tools import GYM_SYSTEM_PROMPT, build_registry, gym_seed
from rexhunter.events import PostingRef, ToolResultEvent
from rexhunter.loop import Brain, Decision, HuntComplete, ToolCallDecision, run_hunt

pytestmark = pytest.mark.anyio

type BrainFactory = Callable[[Sequence[Decision]], Brain]
MockAnthropic = Callable[..., anthropic.AsyncAnthropic]

BOARDS: dict[str, Board] = GYM_BOARDS | EVAL_BOARDS
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gym_sonnet_5_5_2026-11-26"
# The recorded first-wave runs, one per territory.
RECORDED = {
    "gym-greenhouse": "f0a5507e-d644-4d7d-938e-4a920c2a07ae",
    "gym-lever": "b929510c-5b9a-4b7d-b27a-4f4e0b2aadbd",
    "gym-storm": "1303d493-99d6-42a0-8af4-41d2c973396b",
    "gym-mojibake": "19618d54-198e-45dd-b356-29b730da7972",
    "gym-drift": "f726301f-e5ba-485a-918a-29e527009eae",
    "gym-pages": "e1e6b3d5-f943-4a20-b925-faca41f4ce24",
}


class Pinned(NamedTuple):
    outcome: tuple[str, str | None]
    outcome_met: bool
    prey_on_board: int
    prey_read: int
    decoys_read: int
    prey_caught: int
    decoys_caught: int
    capture_errors: int
    calls: int
    cost_usd: float
    precision: float | None
    recall: float | None


PINNED = {
    "gym-greenhouse": Pinned(("completed", None), True, 4, 4, 0, 4, 0, 0, 6, 0.0277, 1.0, 1.0),
    "gym-lever": Pinned(("completed", None), True, 4, 4, 0, 4, 0, 0, 6, 0.030868, 1.0, 1.0),
    "gym-storm": Pinned(("completed", None), True, 4, 4, 0, 4, 0, 0, 6, 0.030848, 1.0, 1.0),
    "gym-mojibake": Pinned(("completed", None), True, 4, 4, 0, 4, 0, 0, 6, 0.029138, 1.0, 1.0),
    # Its planned stop: nothing read, so no precision, and the outcome it should have.
    "gym-drift": Pinned(("aborted", "tool failure"), True, 4, 0, 0, 0, 0, 0, 1, 0.00275, None, 0.0),
    # Fifteen prey read and no decoy, then the cost ceiling before any catch.
    "gym-pages": Pinned(
        ("aborted", "cost ceiling"), False, 20, 15, 0, 0, 0, 0, 18, 0.20813, None, 0.0
    ),
}


def _gym() -> httpx2.ASGITransport:
    return httpx2.ASGITransport(app=build_gym(GYM_CATALOGUE | EVAL_CATALOGUE))


async def _replay(conn: aiosqlite.Connection, territory: str, mock: MockAnthropic) -> str:
    # The daemon's live brain on the gym, served the run's stored streams in order, over a fresh
    # gym (a storm counts down per gym instance) and the daemon's registry.
    streams = iter(sorted((FIXTURE_DIR / f"hunt_{RECORDED[territory]}").iterdir()))

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=next(streams).read_bytes())

    board_client = BoardClient(httpx2.ASGITransport(app=build_gym(GYM_CATALOGUE)))
    async with board_client, mock(handler, api_key="replay") as client:
        registry = build_registry(board_client, GYM_BOARDS)
        brain_for = brain.adapter_brain_for(
            client=client,
            model="claude-sonnet-5-5",
            registry=registry,
            max_tokens=brain.HUNT_MAX_TOKENS,
            thinking=brain.HUNT_THINKING,
            stream=True,
            system=GYM_SYSTEM_PROMPT,
            seed=gym_seed,
        )
        return await run_hunt(
            conn,
            territory=territory,
            brain=brain_for(territory),
            registry=registry,
            cost_ceiling_usd=0.20,
            daemon_spend_ceiling_usd=0.50,
        )


async def test_the_baseline_floor_on_gym_mimic(tmp_path: Path) -> None:
    from rexhunter.board.score import planted_marks, score_log

    conn = await db.connect(tmp_path / "rex.db")
    try:
        async with BoardClient(_gym()) as client:
            await run_hunt(
                conn,
                territory="gym-mimic",
                brain=baseline_brain_for(BOARDS)("gym-mimic"),
                registry=build_registry(client, BOARDS),
            )
        (score,) = await score_log(conn, planted_marks(BOARDS, GYM_CATALOGUE | EVAL_CATALOGUE))
        penned = await conn.execute_fetchall("SELECT posting_id FROM prey ORDER BY posting_id")
    finally:
        await conn.close()

    # The baseline reads and catches every title that names AI or machine learning: two prey and
    # all four decoys. It misses the two prey whose titles name neither.
    assert [str(p) for (p,) in penned] == [
        "4004001",
        "4004002",
        "4004005",
        "4004006",
        "4004007",
        "4004008",
    ]
    tally = score.tally
    assert (score.territory, score.outcome, score.outcome_met) == ("gym-mimic", "completed", True)
    assert (tally.prey_read, tally.decoys_read) == (2, 4)
    assert (tally.prey_caught, tally.decoys_caught) == (2, 4)
    assert tally.prey_on_board - tally.prey_caught == 2
    assert tally.precision == pytest.approx(2 / 6)
    assert tally.recall == 0.5
    assert (tally.capture_errors, tally.calls, tally.cost_usd, score.off_board) == (0, 0, 0.0, 0)


@pytest.mark.parametrize("territory", list(RECORDED))
async def test_each_recorded_run_scores_as_pinned(
    territory: str, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    from rexhunter.board.score import planted_marks, score_log

    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await _replay(conn, territory, mock_anthropic)
        (score,) = await score_log(conn, planted_marks(BOARDS, GYM_CATALOGUE | EVAL_CATALOGUE))
    finally:
        await conn.close()

    pinned, tally = PINNED[territory], score.tally
    assert (score.run_id, score.territory, score.off_board) == (run_id, territory, 0)
    assert (score.outcome, score.abort_reason) == pinned.outcome
    assert score.outcome_met is pinned.outcome_met
    assert (
        tally.prey_on_board,
        tally.prey_read,
        tally.decoys_read,
        tally.prey_caught,
        tally.decoys_caught,
        tally.capture_errors,
        tally.calls,
    ) == pinned[2:9]
    assert tally.cost_usd == pytest.approx(pinned.cost_usd, abs=1e-9)
    assert (tally.precision, tally.recall) == (pinned.precision, pinned.recall)


def test_precision_is_undefined_on_an_empty_catch() -> None:
    from rexhunter.board.score import Tally, planted_marks, score_run, total

    def tally(prey_caught: int, decoys_caught: int, prey_on_board: int = 4) -> Tally:
        return Tally(
            prey_on_board=prey_on_board,
            prey_read=3,
            decoys_read=1,
            prey_caught=prey_caught,
            decoys_caught=decoys_caught,
            capture_errors=0,
            calls=0,
            cost_usd=0.0,
        )

    assert tally(0, 0).precision is None  # not 0, which reads as all decoys, nor 1
    assert tally(0, 0).recall == 0.0
    assert total([tally(0, 0), tally(0, 0)]).precision is None
    assert [tally(1, 0).precision, tally(0, 1).precision, tally(1, 1).precision] == [1.0, 0.0, 0.5]
    assert tally(0, 0, prey_on_board=0).recall is None  # a board with no prey has no recall

    marks = planted_marks(BOARDS, GYM_CATALOGUE | EVAL_CATALOGUE)
    nothing = score_run(
        [], marks, run_id="r", territory="gym-mimic", outcome="completed", abort_reason=None
    )
    assert (nothing.tally.precision, nothing.tally.recall) == (None, 0.0)


async def test_a_listed_only_posting_is_not_read(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    from rexhunter.board.score import planted_marks, score_log

    mimic = "gym-mimic"
    decisions: list[Decision] = [
        ToolCallDecision(tool="list_postings", args={"board": mimic, "page": 1}),
        ToolCallDecision(tool="read_posting", args={"board": mimic, "posting_id": "4004001"}),
        ToolCallDecision(tool="read_posting", args={"board": mimic, "posting_id": "4004005"}),
        # A prey of another board, read: it counts apart, never toward this board's tally.
        ToolCallDecision(
            tool="read_posting", args={"board": "gym-greenhouse", "posting_id": "4001001"}
        ),
        # 4004003 was listed and never read: the catch names it, capture refuses it.
        HuntComplete(
            catch=[
                PostingRef(board=mimic, posting_id="4004001"),
                PostingRef(board=mimic, posting_id="4004003"),
            ]
        ),
    ]
    conn = await db.connect(tmp_path / "rex.db")
    try:
        async with BoardClient(_gym()) as client:
            run_id = await run_hunt(
                conn,
                territory=mimic,
                brain=scripted_brain(decisions),
                registry=build_registry(client, BOARDS),
            )
        log = await db.read_events(conn, run_id)
        (score,) = await score_log(conn, planted_marks(BOARDS, GYM_CATALOGUE | EVAL_CATALOGUE))
    finally:
        await conn.close()

    (listed,) = [e for e in log if isinstance(e, ToolResultEvent) and e.tool == "list_postings"]
    page = ADAPTERS["greenhouse"].page(body_text(listed.raw_response), mimic, 1)
    assert len(page.postings) == 8  # all eight were listed
    tally = score.tally
    assert (tally.prey_read, tally.decoys_read) == (1, 1)  # only the two read on this board
    assert score.off_board == 1
    assert (tally.prey_caught, tally.decoys_caught, tally.capture_errors) == (1, 0, 1)
    assert (tally.precision, tally.recall) == (1.0, 0.25)


async def test_the_score_is_a_projection_of_the_log(tmp_path: Path) -> None:
    from rexhunter.board.score import planted_marks, score_log, score_run

    marks = planted_marks(BOARDS, GYM_CATALOGUE | EVAL_CATALOGUE)
    path = tmp_path / "rex.db"
    conn = await db.connect(path)
    try:
        async with BoardClient(_gym()) as client:
            registry = build_registry(client, BOARDS)
            for territory in ("gym-greenhouse", "gym-mimic"):
                await run_hunt(
                    conn,
                    territory=territory,
                    brain=baseline_brain_for(BOARDS)(territory),
                    registry=registry,
                )
    finally:
        await conn.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    ro = await aiosqlite.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = await ro.execute_fetchall("SELECT type, name FROM sqlite_master ORDER BY name")
        first = await score_log(ro, marks)
        second = await score_log(ro, marks)
        refolded = [
            score_run(
                await db.read_events(ro, s.run_id),
                marks,
                run_id=s.run_id,
                territory=s.territory,
                outcome=s.outcome,
                abort_reason=s.abort_reason,
            )
            for s in first
        ]
        tables_after = await ro.execute_fetchall(
            "SELECT type, name FROM sqlite_master ORDER BY name"
        )
        changes = ro.total_changes
    finally:
        await ro.close()

    assert [s.territory for s in first] == ["gym-greenhouse", "gym-mimic"]
    assert first == second == refolded  # the same log, the same score
    # Nothing was written: no change, no new table, the file as it was.
    assert changes == 0
    assert list(tables_after) == list(tables)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_the_wave_total_counts_the_runs_expected_to_complete() -> None:
    from rexhunter.board.score import RunScore, Tally, wave_total

    def run(territory: str, prey: int, decoys: int, outcome: tuple[str, str | None]) -> RunScore:
        tally = Tally(
            prey_on_board=4,
            prey_read=prey,
            decoys_read=decoys,
            prey_caught=prey,
            decoys_caught=decoys,
            capture_errors=0,
            calls=2,
            cost_usd=0.01,
        )
        expected = ("aborted", "tool failure") if territory == "gym-drift" else ("completed", None)
        return RunScore(
            run_id=territory,
            territory=territory,
            outcome=outcome[0],
            abort_reason=outcome[1],
            expected=expected,
            off_board=0,
            tally=tally,
        )

    wave = wave_total(
        [
            run("gym-greenhouse", 1, 0, ("completed", None)),
            run("gym-mimic", 1, 3, ("completed", None)),
            run("gym-drift", 4, 4, ("aborted", "tool failure")),  # its planned stop: left out
        ]
    )
    assert (wave.prey_on_board, wave.prey_caught, wave.decoys_caught, wave.calls) == (8, 2, 3, 4)
    assert wave.cost_usd == pytest.approx(0.02)
    # Micro, not macro: 2 of 5 caught are prey (the runs' own precisions, 1.0 and 0.25, average
    # 0.625).
    assert wave.precision == pytest.approx(0.4)
    assert wave.recall == pytest.approx(0.25)
