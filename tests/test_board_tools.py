"""The board tools on the gym (`P2.4`, DoD #2's gym clause): `list_postings` and
`read_posting` against injected hazards, each driven through `run_hunt` by a brain that reads no
model, against a fresh gym.

Schema drift, a rate-limit storm, mojibake and a pagination trap each produce typed events that
carry the board's raw bytes, and each run ends in a typed outcome; only a posting the run itself
read reaches the pen, with the board's bytes for it. The model is shown each result through the
tool's view, rendered from the stored bytes alone. Every request goes in-process through httpx2's
`ASGITransport`, recorded on the way; nothing reaches a network, and no body comes from
tests/fixtures/.
"""

import base64
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2
import pytest

from rexhunter import brain, db, events
from rexhunter.board import DESCRIPTION_CAP, MAX_PAGES, PAGE_SIZE, Ats, Board
from rexhunter.board.catalogue import GYM_BOARDS, GYM_CATALOGUE
from rexhunter.board.client import BoardClient
from rexhunter.board.gym import Drift, GymBoard, GymPosting, HazardPlan, build_gym
from rexhunter.board.tools import UNREADABLE, build_registry
from rexhunter.events import ErrorEvent, PreyCapturedEvent, ToolResultEvent
from rexhunter.loop import (
    BACKOFF_BASE_S,
    RETRY_AFTER_CAP_S,
    Brain,
    Context,
    Decision,
    HuntComplete,
    NeedsHelp,
    ThinkingSink,
    ToolCallDecision,
    run_hunt,
)
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

type BrainFactory = Callable[[Sequence[Decision]], Brain]
type BrainFor = Callable[[ToolRegistry], Brain]
BOARD = "test-board"
TOKEN = "testco"
GH_LIST = f"/greenhouse/v1/boards/{TOKEN}/jobs"


class Recorder(httpx2.AsyncBaseTransport):
    """The gym's transport, keeping every request it carries."""

    def __init__(self, inner: httpx2.AsyncBaseTransport) -> None:
        self.inner = inner
        self.requests: list[httpx2.Request] = []

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return await self.inner.handle_async_request(request)


def _posting(n: int, ats: Ats, **fields: Any) -> GymPosting:
    values: dict[str, Any] = {
        "posting_id": str(5000 + n) if ats == "greenhouse" else f"lv-{n}",
        "title": f"AI Engineer {n}",
        "location": "Sydney",
        "team": "Applied AI",
        "description": ("Build and evaluate agents.",),
        "planted": "prey",
    }
    return GymPosting(**(values | fields))


def _gym(
    ats: Ats, plan: HazardPlan | None = None, postings: Sequence[GymPosting] | None = None
) -> tuple[dict[str, Board], dict[str, GymBoard]]:
    if postings is None:
        postings = [_posting(n, ats) for n in range(1, 4)]
    board = GymBoard(ats=ats, postings=tuple(postings), plan=plan or HazardPlan())
    return {BOARD: Board(name=BOARD, ats=ats, token=TOKEN, base_url=f"http://gym/{ats}")}, {
        TOKEN: board
    }


@dataclass
class Hunt:
    events: list[events.TrajectoryEvent]
    outcome: tuple[str, str | None]
    requests: list[httpx2.Request]
    registry: ToolRegistry
    context: list[events.TrajectoryEvent]
    stored: list[str]  # the payload column, as stored
    pen: list[tuple[Any, ...]]


async def _hunt(
    tmp_path: Path,
    boards: Mapping[str, Board],
    gym: Mapping[str, GymBoard],
    brain_for: BrainFor,
    *,
    territory: str = BOARD,
    board: httpx2.AsyncBaseTransport | None = None,
) -> Hunt:
    recorder = Recorder(board or httpx2.ASGITransport(app=build_gym(gym)))
    conn = await db.connect(tmp_path / "rex.db")
    try:
        async with BoardClient(recorder) as client:
            registry = build_registry(client, boards)
            run_id = await run_hunt(
                conn, territory=territory, brain=brain_for(registry), registry=registry
            )
        ((outcome, reason),) = await conn.execute_fetchall(
            "SELECT outcome, abort_reason FROM runs WHERE id = ?", (run_id,)
        )
        stored = await conn.execute_fetchall(
            "SELECT payload FROM trajectory_events WHERE run_id = ? ORDER BY seq", (run_id,)
        )
        pen = await conn.execute_fetchall(
            "SELECT board, posting_id, posting, status FROM prey ORDER BY captured_at"
        )
        return Hunt(
            events=await db.read_events(conn, run_id),
            outcome=(str(outcome), reason),
            requests=recorder.requests,
            registry=registry,
            context=await db.read_context(conn, run_id),
            stored=[str(row[0]) for row in stored],
            pen=[tuple(row) for row in pen],
        )
    finally:
        await conn.close()


def _scripted(scripted_brain: BrainFactory, decisions: Sequence[Decision]) -> BrainFor:
    def brain_for(_registry: ToolRegistry) -> Brain:
        return scripted_brain(decisions)

    return brain_for


def _list(page: int = 1, board: str = BOARD) -> ToolCallDecision:
    return ToolCallDecision(tool="list_postings", args={"board": board, "page": page})


def _read(posting_id: str, board: str = BOARD) -> ToolCallDecision:
    return ToolCallDecision(tool="read_posting", args={"board": board, "posting_id": posting_id})


def _ref(posting_id: str, board: str = BOARD) -> events.PostingRef:
    return events.PostingRef(board=board, posting_id=posting_id)


def _of[T](kind: type[T], evs: Sequence[events.TrajectoryEvent]) -> list[T]:
    return [e for e in evs if isinstance(e, kind)]


async def _gym_body(gym: Mapping[str, GymBoard], path: str) -> bytes:
    # The same answer from a fresh gym: what the board sent, fetched apart from the hunt.
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=build_gym(gym)), base_url="http://gym"
    ) as client:
        return (await client.get(path)).content


def _shown(hunt: Hunt, territory: str = BOARD) -> list[Any]:
    # What the model is shown for each result: the tool_result contents of the projection.
    messages = brain.project_messages(territory, hunt.context, hunt.registry)
    return [
        json.loads(block["content"])
        for message in messages[1:]
        if message["role"] == "user"
        for block in message["content"]
        if not block.get("is_error")
    ]


def _follow_pages(
    board: str = BOARD, then: Callable[[list[str], int], list[Decision]] | None = None
) -> BrainFor:
    """A brain that follows `next_page` from what it is shown, then does `then(ids, pages)`."""

    def brain_for(registry: ToolRegistry) -> Brain:
        tail: list[Decision] = []

        async def brain(context: Context, _sink: ThinkingSink) -> Decision:
            if tail:
                return tail.pop(0)
            results = [e for e in context if isinstance(e, ToolResultEvent)]
            if not results:
                return _list(1, board)
            view = registry.get("list_postings").view
            page = json.loads(view(results[-1].raw_request, results[-1].raw_response))
            if page["next_page"] is not None:
                return _list(page["next_page"], board)
            pages = [json.loads(view(r.raw_request, r.raw_response)) for r in results]
            ids = [p["posting_id"] for shown in pages for p in shown["postings"]]
            tail.extend(then(ids, len(pages)) if then else [NeedsHelp()])
            return tail.pop(0)

        return brain

    return brain_for


async def _drifted_list_is_fatal(
    tmp_path: Path, scripted_brain: BrainFactory, drift: Drift, marker: bytes
) -> None:
    boards, gym = _gym("greenhouse", HazardPlan(drift=drift))
    hunt = await _hunt(tmp_path, boards, gym, _scripted(scripted_brain, [_list()]))

    (error,) = _of(ErrorEvent, hunt.events)
    assert len(hunt.requests) == 1
    assert (error.tool, error.retryable) == ("list_postings", False)
    assert "fails the boundary" in error.error
    gym_bytes = await _gym_body(gym, GH_LIST)
    assert error.raw_response == gym_bytes
    assert marker in gym_bytes
    assert hunt.outcome == ("aborted", "tool failure")


# ── Schema drift ────────────────────────────────────────────────────────────


async def test_an_added_field_is_tolerated(tmp_path: Path, scripted_brain: BrainFactory) -> None:
    boards, gym = _gym("greenhouse", HazardPlan(drift="added_field"))
    decisions = [_list(), _read("5001"), NeedsHelp()]
    hunt = await _hunt(tmp_path, boards, gym, _scripted(scripted_brain, decisions))

    results = _of(ToolResultEvent, hunt.events)
    assert [r.tool for r in results] == ["list_postings", "read_posting"]
    assert all(b'"remote": true' in r.raw_response for r in results)
    assert _of(ErrorEvent, hunt.events) == []
    assert hunt.outcome == ("needs_help", None)


async def test_a_renamed_field_is_fatal_and_keeps_its_bytes(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    await _drifted_list_is_fatal(
        tmp_path, scripted_brain, "renamed_field", b'"name": "AI Engineer 1"'
    )


async def test_a_retyped_field_is_fatal(tmp_path: Path, scripted_brain: BrainFactory) -> None:
    await _drifted_list_is_fatal(tmp_path, scripted_brain, "id_as_string", b'"id": "5001"')


async def test_a_renamed_envelope_is_fatal(tmp_path: Path, scripted_brain: BrainFactory) -> None:
    await _drifted_list_is_fatal(tmp_path, scripted_brain, "envelope_renamed", b'{"postings": [')


# ── A rate-limit storm ──────────────────────────────────────────────────────


async def test_a_storm_within_budget_waits_the_board_retry_after(
    tmp_path: Path, scripted_brain: BrainFactory, retry_waits: list[float]
) -> None:
    boards, gym = _gym("lever", HazardPlan(storm=2, retry_after="2"))
    hunt = await _hunt(tmp_path, boards, gym, _scripted(scripted_brain, [_list(), NeedsHelp()]))

    errors = _of(ErrorEvent, hunt.events)
    assert [(e.retryable, e.raw_response) for e in errors] == [
        (True, b'{"error": "Too Many Requests"}')
    ] * 2
    assert len(retry_waits) == 2 and all(wait >= 2.0 for wait in retry_waits)
    assert [r.tool for r in _of(ToolResultEvent, hunt.events)] == ["list_postings"]
    assert len(hunt.requests) == 3
    assert hunt.outcome == ("needs_help", None)


async def test_a_storm_past_the_budget_ends_typed(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    boards, gym = _gym("lever", HazardPlan(storm=3))
    hunt = await _hunt(tmp_path, boards, gym, _scripted(scripted_brain, [_list()]))

    errors = _of(ErrorEvent, hunt.events)
    assert [e.retryable for e in errors] == [True, True, True]
    assert all(e.raw_response == b'{"error": "Too Many Requests"}' for e in errors)
    assert _of(ToolResultEvent, hunt.events) == []
    assert len(hunt.requests) == 3
    assert hunt.outcome == ("aborted", "tool failure")


async def test_a_board_retry_after_is_cut_to_the_cap(
    tmp_path: Path, scripted_brain: BrainFactory, retry_waits: list[float]
) -> None:
    boards, gym = _gym("lever", HazardPlan(storm=1, retry_after="600"))
    hunt = await _hunt(tmp_path, boards, gym, _scripted(scripted_brain, [_list(), NeedsHelp()]))

    assert retry_waits == [RETRY_AFTER_CAP_S]
    assert hunt.outcome == ("needs_help", None)


async def test_an_unreadable_retry_after_falls_back_to_jitter(
    tmp_path: Path, scripted_brain: BrainFactory, retry_waits: list[float]
) -> None:
    plan = HazardPlan(storm=1, retry_after="Sun, 22 Nov 2026 08:00:00 GMT")
    boards, gym = _gym("lever", plan)
    hunt = await _hunt(tmp_path, boards, gym, _scripted(scripted_brain, [_list(), NeedsHelp()]))

    assert len(hunt.requests) == 2
    assert len(retry_waits) == 1 and 0.0 <= retry_waits[0] <= BACKOFF_BASE_S
    assert hunt.outcome == ("needs_help", None)


async def test_a_503_is_retryable(
    tmp_path: Path, scripted_brain: BrainFactory, retry_waits: list[float]
) -> None:
    boards, gym = _gym("greenhouse", HazardPlan(storm=1, storm_status=503))
    hunt = await _hunt(tmp_path, boards, gym, _scripted(scripted_brain, [_list(), NeedsHelp()]))

    (error,) = _of(ErrorEvent, hunt.events)
    assert (error.retryable, error.raw_response) == (True, b'{"error": "Service Unavailable"}')
    assert len(retry_waits) == 1 and retry_waits[0] <= BACKOFF_BASE_S
    assert [r.tool for r in _of(ToolResultEvent, hunt.events)] == ["list_postings"]
    assert hunt.outcome == ("needs_help", None)


# ── Mojibake ────────────────────────────────────────────────────────────────


def _cafe(ats: Ats) -> list[GymPosting]:
    return [_posting(1, ats, title="AI Engineer, Café Search", location="Montréal")]


async def test_invalid_utf8_is_fatal_with_its_bytes(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    boards, gym = _gym("greenhouse", HazardPlan(encoding="invalid_utf8"), _cafe("greenhouse"))
    hunt = await _hunt(tmp_path, boards, gym, _scripted(scripted_brain, [_list()]))

    (error,) = _of(ErrorEvent, hunt.events)
    assert (error.tool, error.retryable, error.error) == (
        "list_postings",
        False,
        "FatalToolError('board body is not valid UTF-8')",
    )
    gym_bytes = await _gym_body(gym, GH_LIST)
    assert error.raw_response == gym_bytes
    assert b"Caf\xe9 Search" in gym_bytes
    (stored,) = [json.loads(p) for p in hunt.stored if json.loads(p)["type"] == "error"]
    assert base64.b64decode(stored["raw_response"]) == error.raw_response
    assert hunt.outcome == ("aborted", "tool failure")


async def test_double_encoded_text_is_kept_byte_for_byte(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    boards, gym = _gym("greenhouse", HazardPlan(encoding="double_encoded"), _cafe("greenhouse"))
    decisions = [_read("5001"), HuntComplete(catch=[_ref("5001")])]
    hunt = await _hunt(tmp_path, boards, gym, _scripted(scripted_brain, decisions))

    (capture,) = _of(PreyCapturedEvent, hunt.events)
    assert capture.raw_posting == await _gym_body(gym, f"{GH_LIST}/5001")
    assert "Café".encode().decode("latin-1").encode() in capture.raw_posting
    assert hunt.pen == [
        (
            BOARD,
            "5001",
            "AI Engineer, CafÃ© Search \N{EN DASH} MontrÃ©al (test-board)",
            "awaiting_verdict",
        )
    ]
    assert hunt.outcome == ("completed", None)


async def test_a_utf8_bom_is_tolerated(tmp_path: Path, scripted_brain: BrainFactory) -> None:
    boards, gym = _gym("lever", HazardPlan(encoding="bom"))
    decisions = [_list(), _read("lv-1"), HuntComplete(catch=[_ref("lv-1")])]
    hunt = await _hunt(tmp_path, boards, gym, _scripted(scripted_brain, decisions))

    results = _of(ToolResultEvent, hunt.events)
    assert [r.raw_response[:3] for r in results] == [b"\xef\xbb\xbf"] * 2
    assert _shown(hunt)[0]["postings"][0]["posting_id"] == "lv-1"
    assert hunt.pen == [
        (BOARD, "lv-1", "AI Engineer 1 \N{EN DASH} Sydney (test-board)", "awaiting_verdict")
    ]
    assert hunt.outcome == ("completed", None)


async def test_the_posting_view_is_plain_text(tmp_path: Path, scripted_brain: BrainFactory) -> None:
    long_text = "word " * (DESCRIPTION_CAP // 4)
    postings = [
        _posting(1, "greenhouse", description=("Research & Development: build agents.", "Ship.")),
        _posting(2, "greenhouse", description=(long_text,)),
    ]
    boards, gym = _gym("greenhouse", postings=postings)
    decisions = [_read("5001"), _read("5002"), NeedsHelp()]
    hunt = await _hunt(tmp_path, boards, gym, _scripted(scripted_brain, decisions))

    stored = _of(ToolResultEvent, hunt.events)[0].raw_response
    assert b"&lt;p&gt;Research &amp;amp; Development" in stored
    short, cut = _shown(hunt)
    assert short["description"] == "Research & Development: build agents. Ship."
    assert cut["description"] == (
        long_text.strip()[:DESCRIPTION_CAP] + f" [description cut at {DESCRIPTION_CAP} characters]"
    )
    assert set(short) == {"board", "posting_id", "title", "location", "url", "description"}


# ── Pagination traps ────────────────────────────────────────────────────────


async def test_a_board_that_ignores_skip_is_bounded(tmp_path: Path) -> None:
    postings = [_posting(n, "lever") for n in range(1, PAGE_SIZE + 6)]
    boards, gym = _gym("lever", HazardPlan(paging="ignore_skip"), postings)

    def catch(ids: list[str], pages: int) -> list[Decision]:
        # lv-1 was on every page the board sent: the catch names it once per page.
        return [_read("lv-1"), HuntComplete(catch=[_ref("lv-1")] * ids.count("lv-1"))]

    hunt = await _hunt(tmp_path, boards, gym, _follow_pages(then=catch))

    lists = [r for r in hunt.requests if r.url.path.endswith(f"/{TOKEN}")]
    assert len(lists) == MAX_PAGES
    assert [r.url.params["skip"] for r in lists] == [str(n * PAGE_SIZE) for n in range(MAX_PAGES)]
    assert [page["next_page"] for page in _shown(hunt)[:MAX_PAGES]] == [
        *range(2, MAX_PAGES + 1),
        None,
    ]
    assert len(_of(PreyCapturedEvent, hunt.events)) == 1
    assert [row[:2] for row in hunt.pen] == [(BOARD, "lv-1")]
    assert hunt.outcome == ("completed", None)


async def test_an_exact_multiple_ends_on_the_empty_page(tmp_path: Path) -> None:
    postings = [_posting(n, "lever") for n in range(1, 2 * PAGE_SIZE + 1)]
    boards, gym = _gym("lever", HazardPlan(paging="exact_multiple"), postings)
    hunt = await _hunt(tmp_path, boards, gym, _follow_pages())

    assert len(hunt.requests) == 3
    assert [(len(p["postings"]), p["next_page"]) for p in _shown(hunt)] == [
        (PAGE_SIZE, 2),
        (PAGE_SIZE, 3),
        (0, None),
    ]
    assert hunt.outcome == ("needs_help", None)


async def test_a_lying_meta_total_is_ignored(tmp_path: Path, scripted_brain: BrainFactory) -> None:
    boards, gym = _gym("greenhouse", HazardPlan(meta_total_lie=True))
    hunt = await _hunt(tmp_path, boards, gym, _scripted(scripted_brain, [_list(), NeedsHelp()]))

    (result,) = _of(ToolResultEvent, hunt.events)
    assert json.loads(result.raw_response)["meta"] == {"total": 1003}
    (page,) = _shown(hunt)
    assert [p["posting_id"] for p in page["postings"]] == ["5001", "5002", "5003"]
    assert page["next_page"] is None


@pytest.mark.parametrize("page", [0, MAX_PAGES + 1])
async def test_an_out_of_range_page_never_executes(
    tmp_path: Path, scripted_brain: BrainFactory, page: int
) -> None:
    boards, gym = _gym("lever")
    hunt = await _hunt(tmp_path, boards, gym, _scripted(scripted_brain, [_list(page)]))

    (error,) = _of(ErrorEvent, hunt.events)
    assert (error.tool, error.error) == ("list_postings", "invalid tool args")
    assert hunt.requests == []
    assert hunt.outcome == ("aborted", "tool failure")
    schema = hunt.registry.get("list_postings").json_schema["properties"]["page"]
    assert (schema["minimum"], schema["maximum"], schema["default"]) == (1, MAX_PAGES, 1)


# ── Tiny Arms, the network, the log ─────────────────────────────────────────


async def test_the_board_client_sends_get_only(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    decisions: list[Decision] = []
    for name in ("gym-greenhouse", "gym-lever"):
        first = GYM_CATALOGUE[GYM_BOARDS[name].token].postings[0].posting_id
        decisions += [_list(board=name), _read(first, board=name)]
    hunt = await _hunt(
        tmp_path,
        GYM_BOARDS,
        GYM_CATALOGUE,
        _scripted(scripted_brain, [*decisions, NeedsHelp()]),
        territory="gym-greenhouse",
    )

    assert {r.method for r in hunt.requests} == {"GET"} and len(hunt.requests) == 4
    assert not any(
        hasattr(BoardClient, verb)
        for verb in ("post", "put", "patch", "delete", "request", "send", "stream")
    )
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=build_gym(GYM_CATALOGUE)), base_url="http://gym"
    ) as gym:
        applied = await gym.post("/greenhouse/v1/boards/quollridgelabs/jobs/4001001", json={})
    assert applied.status_code == 405


async def test_list_postings_returns_the_board_body_byte_for_byte(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    hunt = await _hunt(
        tmp_path,
        GYM_BOARDS,
        GYM_CATALOGUE,
        _scripted(scripted_brain, [_list(board="gym-greenhouse"), NeedsHelp()]),
        territory="gym-greenhouse",
    )

    (result,) = _of(ToolResultEvent, hunt.events)
    assert result.raw_response == await _gym_body(
        GYM_CATALOGUE, "/greenhouse/v1/boards/quollridgelabs/jobs"
    )
    assert result.raw_request == b'{"board":"gym-greenhouse","page":1}'


async def test_a_read_that_answers_another_posting_is_fatal(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    other = {"id": 7, "title": "AI Engineer", "location": {"name": "Sydney"}}
    body = json.dumps(other | {"absolute_url": "http://gym/x", "content": ""}).encode()
    boards, gym = _gym("greenhouse")
    board = httpx2.MockTransport(lambda _request: httpx2.Response(200, content=body))
    hunt = await _hunt(
        tmp_path, boards, gym, _scripted(scripted_brain, [_read("5001")]), board=board
    )

    (error,) = _of(ErrorEvent, hunt.events)
    assert (error.error, error.raw_response) == (
        "FatalToolError(\"board answered posting '7'\")",
        body,
    )
    assert hunt.outcome == ("aborted", "tool failure")


async def test_a_view_of_bytes_it_cannot_parse_is_a_fixed_text() -> None:
    # Never reached by a result this code stored (its bytes passed the same parse first): an
    # older or edited log renders the fixed text, never the bytes, and never raises.
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError(f"a view sent a request: {request.url}")

    async with BoardClient(httpx2.MockTransport(refuse)) as client:
        registry = build_registry(client, GYM_BOARDS)
    lists, reads = registry.get("list_postings").view, registry.get("read_posting").view
    page = b'{"board":"gym-lever","page":1}'
    ref = b'{"board":"gym-lever","posting_id":"x"}'
    assert lists(b'{"board":"gym-nowhere","page":1}', b"[]") == UNREADABLE
    assert lists(page, b"\xe9") == UNREADABLE
    assert lists(page, b'{"jobs": []}') == UNREADABLE
    assert reads(b'{"board":"gym-lever"}', b"{}") == UNREADABLE
    assert reads(ref, b"[]") == UNREADABLE
    stored = ToolResultEvent(tool="read_posting", raw_request=ref, raw_response=b"<html>")
    (_, result) = brain.project_messages("gym-lever", [stored], registry)
    assert result["content"][0]["content"] == UNREADABLE
    assert lists(page, b"[]") == '{"board":"gym-lever","page":1,"postings":[],"next_page":null}'


@pytest.mark.parametrize("decision", [_list(board="gym-nowhere"), _read("1", "gym-nowhere")])
async def test_an_unknown_board_never_reaches_the_network(
    tmp_path: Path, scripted_brain: BrainFactory, decision: ToolCallDecision
) -> None:
    hunt = await _hunt(tmp_path, GYM_BOARDS, GYM_CATALOGUE, _scripted(scripted_brain, [decision]))

    (error,) = _of(ErrorEvent, hunt.events)
    assert error.error == "FatalToolError(\"unknown board: 'gym-nowhere'\")"
    assert error.raw_response is None
    assert hunt.requests == []
    assert hunt.outcome == ("aborted", "tool failure")


async def test_a_gym_run_projects_from_its_log_with_no_network(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    decisions = [
        _list(board="gym-lever"),
        _read("00005e1f-0000-4000-8000-000000000001", "gym-lever"),
        NeedsHelp(),
    ]
    hunt = await _hunt(
        tmp_path,
        GYM_BOARDS,
        GYM_CATALOGUE,
        _scripted(scripted_brain, decisions),
        territory="gym-lever",
    )

    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError(f"the projection sent a request: {request.url}")

    async with BoardClient(httpx2.MockTransport(refuse)) as offline:
        replayed = build_registry(offline, GYM_BOARDS)
        again = brain.project_messages("gym-lever", hunt.context, replayed)
    assert again == brain.project_messages("gym-lever", hunt.context, hunt.registry)
    page, posting = _shown(hunt, "gym-lever")
    assert [p["title"] for p in page["postings"]][:2] == [
        "AI Platform Engineer",
        "Machine Learning Researcher",
    ]
    assert (posting["title"], posting["location"]) == ("AI Platform Engineer", "Sydney")


async def test_a_gym_posting_read_and_caught_is_penned_with_its_bytes(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    ref = _ref("4001001", "gym-greenhouse")
    decisions = [
        _list(board=ref.board),
        _read(ref.posting_id, ref.board),
        HuntComplete(catch=[ref]),
    ]
    hunt = await _hunt(
        tmp_path,
        GYM_BOARDS,
        GYM_CATALOGUE,
        _scripted(scripted_brain, decisions),
        territory=ref.board,
    )

    (capture,) = _of(PreyCapturedEvent, hunt.events)
    read = _of(ToolResultEvent, hunt.events)[-1]
    gym_bytes = await _gym_body(GYM_CATALOGUE, "/greenhouse/v1/boards/quollridgelabs/jobs/4001001")
    assert capture.raw_posting == read.raw_response == gym_bytes
    assert (capture.board, capture.posting_id) == (ref.board, ref.posting_id)
    assert hunt.pen == [
        (
            "gym-greenhouse",
            "4001001",
            "AI Engineer, Agents \N{EN DASH} Sydney (gym-greenhouse)",
            "awaiting_verdict",
        )
    ]
    assert hunt.outcome == ("completed", None)


@pytest.mark.parametrize("name", ["gym-greenhouse", "gym-lever"])
async def test_the_pen_label_agrees_with_the_posting_view(
    tmp_path: Path, scripted_brain: BrainFactory, name: str
) -> None:
    ids = [p.posting_id for p in GYM_CATALOGUE[GYM_BOARDS[name].token].postings]
    decisions: list[Decision] = [_read(posting_id, name) for posting_id in ids]
    decisions.append(HuntComplete(catch=[_ref(posting_id, name) for posting_id in ids]))
    hunt = await _hunt(
        tmp_path, GYM_BOARDS, GYM_CATALOGUE, _scripted(scripted_brain, decisions), territory=name
    )

    shown = _shown(hunt, name)
    labels = [c.label for c in _of(PreyCapturedEvent, hunt.events)]
    assert len(labels) == len(ids) == 8
    assert labels == [f"{v['title']} \N{EN DASH} {v['location']} ({v['board']})" for v in shown]
