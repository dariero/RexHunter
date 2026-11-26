"""gym-mimic (`P2.5`): the eval's own board, a clean Greenhouse board in a catalogue of its own,
served beside the six and outside the default schedule, whose decoys look like prey by title.

The names are looked up inside each test, so on a tree without the eval catalogue each test fails
on its own line. Every request goes to the in-process gym; nothing reaches a network.
"""

import httpx2
import pytest

from rexhunter import stub
from rexhunter.board import Board
from rexhunter.board.baseline import matches
from rexhunter.board.catalogue import GYM_BOARDS, GYM_CATALOGUE
from rexhunter.board.gym import HazardPlan, build_gym

pytestmark = pytest.mark.anyio

# The design's table, in order: title, location, team, planted.
ROWS = [
    ("AI Engineer, Retrieval", "Sydney", "Applied AI", "prey"),
    ("Machine Learning Engineer, Ranking", "Melbourne", "Models", "prey"),
    ("Software Engineer, Search Quality", "Brisbane", "Search", "prey"),
    ("Backend Engineer, Assistant Platform", "Remote", "Platform", "prey"),
    ("AI Sales Executive", "Sydney", "Sales", "decoy"),
    ("Technical Recruiter, Machine Learning", "Melbourne", "People", "decoy"),
    ("Applied AI Lead", "Perth", "Marketing", "decoy"),
    ("Machine Learning Specialist", "Remote", "Training", "decoy"),
]
PREY_TEXT = (
    "You will design, build and evaluate production systems on large language models, from "
    "retrieval and tool use to the tests that catch a regression before users do."
)
DECOY_TEXT = "The role sits outside engineering and works with teams across the business."
CLOSING = "Research & Development at a glance: small teams, big problems."


def test_gym_mimic_holds_four_prey_and_four_decoys() -> None:
    from rexhunter.board.catalogue import EVAL_BOARDS, EVAL_CATALOGUE

    assert EVAL_BOARDS == {
        "gym-mimic": Board(
            name="gym-mimic", ats="greenhouse", token="numbatlabs", base_url="http://gym/greenhouse"
        )
    }
    assert list(EVAL_CATALOGUE) == ["numbatlabs"]
    board = EVAL_CATALOGUE["numbatlabs"]
    assert (board.ats, board.plan) == ("greenhouse", HazardPlan())
    assert [(p.title, p.location, p.team, p.planted) for p in board.postings] == ROWS
    assert [p.posting_id for p in board.postings] == [
        "4004001",
        "4004002",
        "4004003",
        "4004004",
        "4004005",
        "4004006",
        "4004007",
        "4004008",
    ]
    # Each description is written as every board's is; only its middle paragraph says prey or decoy.
    for p in board.postings:
        assert p.description == (
            f"Numbat Labs is hiring a {p.title} in {p.location}, in the {p.team} team.",
            PREY_TEXT if p.planted == "prey" else DECOY_TEXT,
            CLOSING,
        )

    # Its decoys look like prey by title: the keyword baseline takes every decoy and two of the
    # four prey.
    assert [matches(p.title) for p in board.postings] == [True, True, False, False] + [True] * 4

    # Served beside the six, never one of them, and never scheduled.
    assert not set(EVAL_BOARDS) & set(GYM_BOARDS)
    assert not set(EVAL_CATALOGUE) & set(GYM_CATALOGUE)
    assert "gym-mimic" not in stub.DEFAULT_SCHEDULE
    build_gym(GYM_CATALOGUE | EVAL_CATALOGUE)


async def test_the_eval_catalogue_never_serves_its_mark() -> None:
    from rexhunter.board.catalogue import EVAL_CATALOGUE

    bodies: list[bytes] = []
    transport = httpx2.ASGITransport(app=build_gym(GYM_CATALOGUE | EVAL_CATALOGUE))
    async with httpx2.AsyncClient(transport=transport, base_url="http://gym") as gym:
        for token, board in EVAL_CATALOGUE.items():
            assert board.ats == "greenhouse"
            for params in ({}, {"content": "true"}):
                listed = await gym.get(f"/greenhouse/v1/boards/{token}/jobs", params=params)
                assert listed.status_code == 200
                bodies.append(listed.content)
            for posting in board.postings:
                job = await gym.get(f"/greenhouse/v1/boards/{token}/jobs/{posting.posting_id}")
                assert job.status_code == 200
                bodies.append(job.content)

    assert {p.planted for b in EVAL_CATALOGUE.values() for p in b.postings} == {"prey", "decoy"}
    assert len(bodies) == 2 + 8
    for body in bodies:
        assert b"planted" not in body
        assert b'"prey"' not in body
        assert b'"decoy"' not in body
