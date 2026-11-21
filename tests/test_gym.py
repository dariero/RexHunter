"""The gym (`P2.4`): Greenhouse- and Lever-shaped routes over an invented catalogue, with
injected hazards.

Each body is checked against the shape the board docs give (read 21 Nov 2026): Greenhouse's
`jobs` envelope with `meta.total`, each job's `id` (an integer), `title`, `updated_at`,
`location.name` and `absolute_url`, and `content` (entity-escaped HTML) on a job and on a list
asked for with `content=true`; Lever's bare array of postings with `id`, `text`, `categories`,
`hostedUrl`, `applyUrl`, `description` and `descriptionPlain`, paged by `skip` and `limit`. The
planted prey or decoy mark never leaves the gym, and a gym's answers depend only on its catalogue,
its plans and the order of the requests it got. Every request goes in-process through httpx2's
`ASGITransport`; nothing reaches a network.
"""

from collections.abc import Mapping
from typing import Any

import httpx2
import pytest

from rexhunter.board import PAGE_SIZE
from rexhunter.board.catalogue import GYM_CATALOGUE
from rexhunter.board.gym import GymBoard, GymPosting, HazardPlan, build_gym

pytestmark = pytest.mark.anyio

GREENHOUSE_JOB = {"id", "title", "updated_at", "location", "absolute_url"}
LEVER_POSTING = {
    "id",
    "text",
    "categories",
    "hostedUrl",
    "applyUrl",
    "description",
    "descriptionPlain",
}


def _client(boards: Mapping[str, GymBoard]) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=build_gym(boards)), base_url="http://gym"
    )


async def test_the_gym_greenhouse_list_matches_the_documented_shape() -> None:
    async with _client(GYM_CATALOGUE) as gym:
        listed = await gym.get("/greenhouse/v1/boards/quollridgelabs/jobs")
        with_content = await gym.get(
            "/greenhouse/v1/boards/quollridgelabs/jobs", params={"content": "true"}
        )
        first = listed.json()["jobs"][0]["id"]
        job = await gym.get(f"/greenhouse/v1/boards/quollridgelabs/jobs/{first}")
        missing = await gym.get("/greenhouse/v1/boards/quollridgelabs/jobs/1")
        unknown = await gym.get("/greenhouse/v1/boards/nobody/jobs")

    body = listed.json()
    assert listed.status_code == 200
    assert listed.headers["content-type"] == "application/json"
    assert set(body) == {"jobs", "meta"}
    assert body["meta"] == {"total": len(body["jobs"])} == {"total": 8}
    for each in body["jobs"]:
        assert set(each) == GREENHOUSE_JOB
        assert type(each["id"]) is int
        assert set(each["location"]) == {"name"}
    for each in with_content.json()["jobs"]:
        assert set(each) == GREENHOUSE_JOB | {"content"}
        assert each["content"].startswith("&lt;p&gt;")
    assert job.status_code == 200
    assert set(job.json()) == GREENHOUSE_JOB | {"content"}
    assert job.json()["id"] == first
    assert (missing.status_code, unknown.status_code) == (404, 404)


async def test_the_gym_lever_list_matches_the_documented_shape() -> None:
    async with _client(GYM_CATALOGUE) as gym:
        listed = await gym.get(
            "/lever/v0/postings/wombatworks", params={"mode": "json", "skip": "0", "limit": "20"}
        )
        window = await gym.get(
            "/lever/v0/postings/wombatworks", params={"mode": "json", "skip": "2", "limit": "3"}
        )
        first = listed.json()[0]["id"]
        posting = await gym.get(f"/lever/v0/postings/wombatworks/{first}")
        missing = await gym.get("/lever/v0/postings/wombatworks/nope")
        wrong_shape = await gym.get("/lever/v0/postings/quollridgelabs")  # a Greenhouse token

    body: list[dict[str, Any]] = listed.json()
    assert listed.status_code == 200
    assert len(body) == 8
    for each in body:
        assert set(each) == LEVER_POSTING
        assert type(each["id"]) is str
        assert set(each["categories"]) == {"location", "commitment", "team", "department"}
    assert window.json() == body[2:5]
    assert posting.status_code == 200
    assert posting.json() == body[0]
    assert (missing.status_code, wrong_shape.status_code) == (404, 404)


async def test_planted_labels_never_appear_in_any_gym_response() -> None:
    bodies: list[bytes] = []
    async with _client(GYM_CATALOGUE) as gym:
        for token, board in GYM_CATALOGUE.items():
            if board.ats == "greenhouse":
                listing, params = f"/greenhouse/v1/boards/{token}/jobs", {"content": "true"}
            else:
                listing, params = f"/lever/v0/postings/{token}", {"mode": "json"}
            # past any storm, to the 200s
            answers = [await gym.get(listing, params=params) for _ in range(board.plan.storm + 1)]
            assert answers[-1].status_code == 200
            bodies += [answer.content for answer in answers]
            for posting in board.postings:
                path = (
                    f"/greenhouse/v1/boards/{token}/jobs/{posting.posting_id}"
                    if board.ats == "greenhouse"
                    else f"/lever/v0/postings/{token}/{posting.posting_id}"
                )
                response = await gym.get(path)
                assert response.status_code == 200
                bodies.append(response.content)

    assert {p.planted for b in GYM_CATALOGUE.values() for p in b.postings} == {"prey", "decoy"}
    for body in bodies:
        assert b"planted" not in body
        assert b'"prey"' not in body
        assert b'"decoy"' not in body


async def test_a_gym_hunt_is_deterministic() -> None:
    # The requests one hunt on each hazard board would send, in order, twice, on two fresh gyms.
    requests = [
        "/lever/v0/postings/echidnaanalytics?mode=json&skip=0&limit=20",
        "/lever/v0/postings/echidnaanalytics?mode=json&skip=0&limit=20",
        "/lever/v0/postings/echidnaanalytics?mode=json&skip=0&limit=20",
        "/greenhouse/v1/boards/platypusdynamics/jobs",
        "/greenhouse/v1/boards/cafekookaburra/jobs/4003001",
        "/lever/v0/postings/dingorobotics?mode=json&skip=40&limit=20",
        "/greenhouse/v1/boards/quollridgelabs/jobs/1",
    ]

    async def answers() -> list[tuple[int, list[tuple[str, str]], bytes]]:
        async with _client(GYM_CATALOGUE) as gym:
            return [
                (r.status_code, list(r.headers.items()), r.content)
                for r in [await gym.get(path) for path in requests]
            ]

    first, second = await answers(), await answers()
    assert first == second
    assert [status for status, _, _ in first] == [429, 429, 200, 200, 200, 200, 404]
    assert dict(first[0][1])["retry-after"] == "2"
    assert first[5][2] == b"[]"


def _gym_posting(posting_id: str, title: str = "AI Engineer") -> GymPosting:
    return GymPosting(
        posting_id=posting_id,
        title=title,
        location="Sydney",
        team="Applied AI",
        description=("Build agents.",),
        planted="prey",
    )


@pytest.mark.parametrize(
    ("ats", "posting_ids", "plan", "title"),
    [
        ("greenhouse", ("job-1",), HazardPlan(), "AI Engineer"),
        ("greenhouse", ("1",), HazardPlan(paging="ignore_skip"), "AI Engineer"),
        ("lever", ("lv-1",), HazardPlan(drift="id_as_string"), "AI Engineer"),
        ("lever", ("lv-1",), HazardPlan(drift="envelope_renamed"), "AI Engineer"),
        ("lever", ("lv-1",), HazardPlan(meta_total_lie=True), "AI Engineer"),
        ("lever", ("lv-1",), HazardPlan(paging="exact_multiple"), "AI Engineer"),
        ("greenhouse", ("1",), HazardPlan(encoding="invalid_utf8"), "AI Engineer"),
        ("greenhouse", ("1",), HazardPlan(encoding="double_encoded"), "AI Engineer"),
        ("greenhouse", ("1",), HazardPlan(encoding="invalid_utf8"), "AI Engineer \N{EN DASH} 2"),
    ],
    ids=[
        "greenhouse-id-not-digits",
        "greenhouse-paging",
        "lever-id-as-string",
        "lever-envelope",
        "lever-meta-total",
        "lever-exact-multiple-of-one",
        "mojibake-all-ascii",
        "double-encoded-all-ascii",
        "invalid-utf8-beyond-latin-1",
    ],
)
def test_a_plan_its_board_cannot_carry_is_refused_at_build(
    ats: str, posting_ids: tuple[str, ...], plan: HazardPlan, title: str
) -> None:
    postings = tuple(_gym_posting(posting_id, title) for posting_id in posting_ids)
    board = GymBoard(
        ats="greenhouse" if ats == "greenhouse" else "lever",
        postings=postings,
        plan=plan,
    )
    with pytest.raises(ValueError, match="gym board 'testco'"):
        build_gym({"testco": board})


def test_the_catalogue_builds_and_its_pages_board_is_two_whole_pages() -> None:
    build_gym(GYM_CATALOGUE)
    assert len(GYM_CATALOGUE["dingorobotics"].postings) == 2 * PAGE_SIZE
