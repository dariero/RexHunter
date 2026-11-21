"""The board client and the two adapters (`P2.4`): the boundary a board's bytes cross.

`BoardClient.get` returns a 200's body as it arrived and turns every other answer into the loop's
tool-error taxonomy with the body kept (invariant 6): a 429, a 5xx or a transport failure is
retryable, with the board's `retry-after`; any other status is fatal, and a redirect is never
followed. The adapters ask the paths the Greenhouse and Lever docs give (read 21 Nov 2026) and
validate a body on the fields Rex uses only: an unknown field passes, a missing, renamed or
retyped one fails, and so does a body that is not UTF-8. Every response here is a literal behind
an httpx2 `MockTransport`; nothing reaches a network.
"""

import json
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError

from rexhunter.board import PAGE_SIZE, Board
from rexhunter.board.adapters import ADAPTERS, body_text
from rexhunter.board.client import BoardClient
from rexhunter.loop import FatalToolError, RetryableToolError

pytestmark = pytest.mark.anyio

GREENHOUSE = Board(
    name="b", ats="greenhouse", token="quoll ridge labs", base_url="http://gym/greenhouse"
)
LEVER = Board(name="b", ats="lever", token="wombatworks", base_url="http://gym/lever")
GH_JOB: dict[str, Any] = {
    "id": 7,
    "title": "AI Engineer",
    "updated_at": "2026-11-01T09:00:00-04:00",
    "location": {"name": "Sydney"},
    "absolute_url": "http://gym/greenhouse/quollridgelabs/jobs/7",
    "content": "&lt;p&gt;Build agents.&lt;/p&gt;",
}
LV_POSTING: dict[str, Any] = {
    "id": "lv-1",
    "text": "AI Engineer",
    "categories": {"location": "Sydney", "team": "Applied AI"},
    "hostedUrl": "http://gym/lever/wombatworks/lv-1",
    "descriptionPlain": "Build agents.",
}


def _answering(*responses: httpx2.Response) -> tuple[BoardClient, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []
    queue = list(responses)

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return queue.pop(0)

    return BoardClient(httpx2.MockTransport(handler)), seen


async def test_a_200_body_is_returned_as_received() -> None:
    body = b'{ "jobs" : [ ],\n "meta": {"total": 0} }'
    client, seen = _answering(httpx2.Response(200, content=body))
    async with client:
        assert await client.get("http://gym/greenhouse/v1/boards/x/jobs") == body
    assert [r.method for r in seen] == ["GET"]


@pytest.mark.parametrize(
    ("status", "header", "retry_after"),
    [(429, "3", 3.0), (500, None, None), (503, "-1", None), (529, "soon", None)],
)
async def test_a_429_or_a_5xx_is_retryable_with_its_body_and_retry_after(
    status: int, header: str | None, retry_after: float | None
) -> None:
    headers = {} if header is None else {"retry-after": header}
    client, _ = _answering(httpx2.Response(status, content=b"busy", headers=headers))
    async with client:
        with pytest.raises(RetryableToolError) as caught:
            await client.get("http://gym/lever/v0/postings/x")
    assert (caught.value.raw_response, caught.value.retry_after) == (b"busy", retry_after)
    assert str(caught.value) == f"board answered {status}"


@pytest.mark.parametrize("status", [404, 400, 401, 307])
async def test_any_other_status_is_fatal_with_its_body(status: int) -> None:
    redirect = {"location": "http://elsewhere.invalid/jobs"}
    client, seen = _answering(httpx2.Response(status, content=b"no", headers=redirect))
    async with client:
        with pytest.raises(FatalToolError) as caught:
            await client.get("http://gym/greenhouse/v1/boards/x/jobs")
    assert caught.value.raw_response == b"no"
    assert len(seen) == 1  # a redirect is never followed


@pytest.mark.parametrize("failure", [httpx2.ConnectError, httpx2.ReadTimeout])
async def test_a_transport_failure_is_retryable_with_no_body(
    failure: type[httpx2.TransportError],
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise failure("down", request=request)

    async with BoardClient(httpx2.MockTransport(handler)) as client:
        with pytest.raises(RetryableToolError) as caught:
            await client.get("http://gym/lever/v0/postings/x")
    assert (caught.value.raw_response, caught.value.retry_after) == (None, None)


async def test_the_adapters_ask_the_documented_paths() -> None:
    gh, lv = ADAPTERS["greenhouse"], ADAPTERS["lever"]
    assert gh.list_request(GREENHOUSE, 3) == (
        "http://gym/greenhouse/v1/boards/quoll%20ridge%20labs/jobs",
        {},
    )
    assert (
        gh.read_url(GREENHOUSE, "7")
        == "http://gym/greenhouse/v1/boards/quoll%20ridge%20labs/jobs/7"
    )
    assert lv.list_request(LEVER, 3) == (
        "http://gym/lever/v0/postings/wombatworks",
        {"mode": "json", "skip": str(2 * PAGE_SIZE), "limit": str(PAGE_SIZE)},
    )
    # A posting id is one path segment, whatever it holds.
    assert lv.read_url(LEVER, "../x/y") == "http://gym/lever/v0/postings/wombatworks/..%2Fx%2Fy"


def test_the_boundary_reads_only_the_fields_rex_uses() -> None:
    gh_list = json.dumps({"jobs": [GH_JOB | {"departments": [], "extra": {"a": 1}}], "meta": None})
    page = ADAPTERS["greenhouse"].page(gh_list, "b", 1)
    assert (page.postings[0].posting_id, page.postings[0].title) == ("7", "AI Engineer")
    posting = ADAPTERS["lever"].posting(json.dumps(LV_POSTING | {"lists": "anything"}), "b")
    assert (posting.title, posting.location, posting.description) == (
        "AI Engineer",
        "Sydney",
        "Build agents.",
    )
    no_location: dict[str, Any] = LV_POSTING | {"categories": {}}
    assert ADAPTERS["lever"].posting(json.dumps(no_location), "b").location == ""


@pytest.mark.parametrize(
    ("ats", "body"),
    [
        ("greenhouse", {k: v for k, v in GH_JOB.items() if k != "absolute_url"}),
        ("greenhouse", {("name" if k == "title" else k): v for k, v in GH_JOB.items()}),
        ("greenhouse", GH_JOB | {"id": "7"}),
        ("greenhouse", GH_JOB | {"id": True}),
        ("greenhouse", GH_JOB | {"location": {"name": None}}),
        ("lever", {k: v for k, v in LV_POSTING.items() if k != "descriptionPlain"}),
        ("lever", LV_POSTING | {"id": 1}),
        ("lever", LV_POSTING | {"categories": {"location": 3}}),
    ],
    ids=[
        "gh-missing",
        "gh-renamed",
        "gh-id-string",
        "gh-id-bool",
        "gh-null-location",
        "lv-missing",
        "lv-id-int",
        "lv-location-int",
    ],
)
def test_a_missing_renamed_or_retyped_field_fails_the_boundary(
    ats: str, body: dict[str, Any]
) -> None:
    adapter = ADAPTERS["greenhouse" if ats == "greenhouse" else "lever"]
    with pytest.raises(ValidationError):
        adapter.posting(json.dumps(body), "b")


def test_a_body_that_is_not_utf8_fails_and_a_bom_does_not() -> None:
    clean = json.dumps(LV_POSTING | {"text": "Café"}, ensure_ascii=False).encode()
    assert ADAPTERS["lever"].posting(body_text(b"\xef\xbb\xbf" + clean), "b").title == "Café"
    with pytest.raises(UnicodeDecodeError):
        body_text(clean.replace("é".encode(), b"\xe9"))
