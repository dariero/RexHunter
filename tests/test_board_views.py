"""The posting views on the live shapes (`P5.1`): what Rex is shown of a Lever posting's lists and
closing, and of Greenhouse content escaped once or twice.

After `descriptionPlain`, a Lever posting carries `lists`, each a `text` and its `content` as
unstyled HTML, and `additionalPlain`, which may be an empty string (the Lever Postings API, re-read
27 Nov 2026). Greenhouse's `content` is the job's HTML converted to entities, and its docs' example
reads as escaped twice. Every gym posting view reads as it did before both changes, pinned by
digest. Each body is a literal or the gym's own; nothing reaches a network.
"""

import hashlib
import html
import json
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError

from rexhunter.board import DESCRIPTION_CAP, Ats
from rexhunter.board.adapters import ADAPTERS
from rexhunter.board.catalogue import EVAL_BOARDS, EVAL_CATALOGUE, GYM_BOARDS, GYM_CATALOGUE
from rexhunter.board.client import BoardClient
from rexhunter.board.gym import build_gym
from rexhunter.board.tools import build_registry
from rexhunter.events import PostingRef

pytestmark = pytest.mark.anyio

BOARDS = GYM_BOARDS | EVAL_BOARDS
CATALOGUE = GYM_CATALOGUE | EVAL_CATALOGUE
LV_POSTING: dict[str, Any] = {
    "id": "lv-1",
    "text": "AI Engineer",
    "categories": {"location": "Sydney"},
    "hostedUrl": "https://jobs.lever.co/acme/lv-1",
    "descriptionPlain": "Acme builds agents.\n",
}
GH_JOB: dict[str, Any] = {
    "id": 7,
    "title": "AI Engineer",
    "location": {"name": "Sydney"},
    "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/7",
}
LISTS = [
    {"text": "What you will do", "content": "<li>Build evals</li><li>Ship agents &amp; tools</li>"},
    {"text": "What you bring", "content": "<li>Python</li>"},
]
# The views of every gym posting of each shape, joined by newlines, as 251c4a5 rendered them.
GYM_VIEWS_SHA256: dict[Ats, str] = {
    "greenhouse": "6a02667b5410d08742317aee45119e384d3a634d7762f60330ccc9c6ca61e83d",
    "lever": "4f7df31f44841d2edae93f6fc10bfbdaa81347607c95e7c11aebe09ca6f6e3cd",
}


def _lever(**fields: Any) -> str:
    return ADAPTERS["lever"].posting(json.dumps(LV_POSTING | fields), "b").description


def _greenhouse(content: str) -> str:
    body = json.dumps(GH_JOB | {"content": content})
    return ADAPTERS["greenhouse"].posting(body, "b").description


def _escaped(markup: str, times: int) -> str:
    for _ in range(times):
        markup = html.escape(markup)
    return markup


async def _gym_views(ats: Ats) -> list[str]:
    # read_posting's view of each gym posting of one shape, from the bytes a fresh gym sends,
    # asked again past a storm.
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError(f"a view sent a request: {request.url}")

    gym = httpx2.ASGITransport(app=build_gym(CATALOGUE))
    views: list[str] = []
    async with (
        BoardClient(httpx2.MockTransport(refuse)) as offline,
        httpx2.AsyncClient(transport=gym) as http,
    ):
        view = build_registry(offline, BOARDS).get("read_posting").view
        for name, board in BOARDS.items():
            if board.ats != ats:
                continue
            gym_board = CATALOGUE[board.token]
            for posting in gym_board.postings:
                url = ADAPTERS[ats].read_url(board, posting.posting_id)
                answer = await http.get(url)
                for _ in range(gym_board.plan.storm):
                    if answer.status_code == 200:
                        break
                    answer = await http.get(url)
                assert answer.status_code == 200
                ref = PostingRef(board=name, posting_id=posting.posting_id)
                views.append(view(ref.model_dump_json().encode(), answer.content))
    return views


def _digest(views: list[str]) -> str:
    return hashlib.sha256("\n".join(views).encode()).hexdigest()


def test_a_lever_posting_shows_its_lists_and_closing() -> None:
    closing = "Visa sponsorship available."
    assert _lever(lists=LISTS, additionalPlain=closing) == (
        "Acme builds agents. What you will do Build evals Ship agents & tools What you bring"
        " Python Visa sponsorship available."
    )
    assert _lever(lists=LISTS[1:]) == "Acme builds agents. What you bring Python"
    assert _lever(additionalPlain=closing) == "Acme builds agents. Visa sponsorship available."


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"lists": []},
        {"additionalPlain": ""},
        {"lists": None, "additionalPlain": None},
        {"lists": [], "additionalPlain": ""},
    ],
    ids=["absent", "no-lists", "empty-closing", "null", "both-empty"],
)
def test_absent_or_empty_lists_and_closing_add_nothing(fields: dict[str, Any]) -> None:
    assert _lever(**fields) == "Acme builds agents."


def test_the_lists_and_closing_count_inside_the_cap() -> None:
    long_list = [{"text": "Details", "content": "<p>" + "word " * (DESCRIPTION_CAP // 5) + "</p>"}]
    shown = _lever(lists=long_list, additionalPlain="Closing.")
    cut = f" [description cut at {DESCRIPTION_CAP} characters]"
    assert shown.startswith("Acme builds agents. Details word word")
    assert shown.endswith(cut)
    assert len(shown) == DESCRIPTION_CAP + len(cut)
    assert "Closing." not in shown


@pytest.mark.parametrize(
    "fields",
    [
        {"lists": "anything"},
        {"lists": [{"text": "Requirements"}]},
        {"lists": [{"content": "<li>Python</li>"}]},
        {"lists": [{"text": "Requirements", "content": 3}]},
        {"lists": [{"text": None, "content": "<li>Python</li>"}]},
        {"additionalPlain": 3},
    ],
    ids=[
        "lists-string",
        "entry-no-content",
        "entry-no-text",
        "content-int",
        "text-null",
        "closing-int",
    ],
)
def test_a_retyped_list_or_closing_fails_the_boundary(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _lever(**fields)


async def test_a_gym_lever_posting_reads_as_before() -> None:
    views = await _gym_views("lever")
    assert len(views) == 8 + 8 + 40  # gym-lever, gym-storm and gym-pages
    assert _digest(views) == GYM_VIEWS_SHA256["lever"]


def test_double_escaped_greenhouse_content_reads_as_text() -> None:
    markup = "<p>Build evals &amp; agents.</p><ul><li>Python</li><li>SQL</li></ul>"
    assert _greenhouse(_escaped(markup, 2)) == "Build evals & agents. Python SQL"
    assert _greenhouse(_escaped(markup, 1)) == "Build evals & agents. Python SQL"


def test_greenhouse_content_is_unescaped_at_most_twice() -> None:
    # A third level of escaping stays as text.
    assert _greenhouse(_escaped("<p>Build evals.</p>", 3)) == "<p>Build evals.</p>"


async def test_single_escaped_greenhouse_content_reads_as_before() -> None:
    views = await _gym_views("greenhouse")
    assert len(views) == 8 + 8 + 8 + 8  # gym-greenhouse, gym-drift, gym-mojibake and gym-mimic
    assert _digest(views) == GYM_VIEWS_SHA256["greenhouse"]
    assert json.loads(views[0])["description"] == (
        "Quoll Ridge Labs is hiring a AI Engineer, Agents in Sydney, in the Applied AI team. You"
        " will design, build and evaluate production systems on large language models, from"
        " retrieval and tool use to the tests that catch a regression before users do. Research &"
        " Development at a glance: small teams, big problems."
    )
