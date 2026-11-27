"""The live boards (`P5.1`): the board list and the host allowlist.

`live_client` refuses a URL whose scheme is not https, whose host is not in `LIVE_HOSTS`, or that
carries userinfo or a port other than 443, with a fatal tool error raised before any request, and
like every board client it sends GET only and follows no redirect (invariant 4). Every board in
`LIVE_BOARDS` answers over https on an allowlisted host of its own shape. Every response here is a
literal behind an httpx2 `MockTransport`; nothing reaches a network.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import httpx2
import pytest

from rexhunter import db
from rexhunter.board import Board
from rexhunter.board.live import LIVE_BOARDS, LIVE_HOSTS, live_client
from rexhunter.board.tools import build_registry
from rexhunter.events import ErrorEvent
from rexhunter.loop import Brain, Decision, FatalToolError, ToolCallDecision, run_hunt

pytestmark = pytest.mark.anyio

type BrainFactory = Callable[[Sequence[Decision]], Brain]

# The hosts each documented shape answers on: the Greenhouse Job Board API, and the Lever Postings
# API's global and EU hosts (re-read 27 Nov 2026).
ATS_HOSTS = {
    "greenhouse": frozenset({"boards-api.greenhouse.io"}),
    "lever": frozenset({"api.lever.co", "api.eu.lever.co"}),
}


def _recording() -> tuple[httpx2.MockTransport, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, content=b"[]")

    return httpx2.MockTransport(handler), seen


def test_the_allowlist_is_the_three_public_ats_hosts() -> None:
    assert LIVE_HOSTS == frozenset({"boards-api.greenhouse.io", "api.lever.co", "api.eu.lever.co"})
    assert ATS_HOSTS["greenhouse"] | ATS_HOSTS["lever"] == LIVE_HOSTS


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/v1/boards/acme/jobs",
        "https://boards-api.greenhouse.io.evil.example/v1/boards/acme/jobs",
        "https://boards-api.greenhouse.io@evil.example/v1/boards/acme/jobs",
        "https://boards-api.greenhouse.io./v1/boards/acme/jobs",
        "https://api.lever.co.evil.example/v0/postings/acme",
        "https://evilapi.lever.co/v0/postings/acme",
        "https://jobs.lever.co/acme",
        "https://boards.greenhouse.io/acme",
        "https://api.anthropic.com/v1/messages",
        "/v0/postings/acme",
        "https://alice@api.lever.co/v0/postings/acme",
        "https://alice:s3cret@api.lever.co/v0/postings/acme",
        "https://api.lever.co:8443/v0/postings/acme",
    ],
)
async def test_the_live_client_refuses_a_host_outside_the_allowlist(url: str) -> None:
    transport, seen = _recording()
    async with live_client(transport) as client:
        with pytest.raises(FatalToolError, match="board host refused") as caught:
            await client.get(url, {"mode": "json"})
    assert caught.value.raw_response is None
    assert seen == []
    text = f"{caught.value} {caught.value!r}"
    assert "alice" not in text and "s3cret" not in text


@pytest.mark.parametrize("host", sorted(LIVE_HOSTS))
async def test_the_live_client_refuses_plain_http(host: str) -> None:
    transport, seen = _recording()
    async with live_client(transport) as client:
        with pytest.raises(FatalToolError, match="board host refused"):
            await client.get(f"http://{host}/v0/postings/acme")
    assert seen == []


@pytest.mark.parametrize(
    "url",
    [
        "https://boards-api.greenhouse.io/v1/boards/acme/jobs",
        "https://api.lever.co/v0/postings/acme",
        "https://api.eu.lever.co/v0/postings/acme",
        "https://API.Lever.co/v0/postings/acme",
        "https://api.lever.co:443/v0/postings/acme",
    ],
)
async def test_the_live_client_reaches_an_allowlisted_host_by_get(url: str) -> None:
    transport, seen = _recording()
    async with live_client(transport) as client:
        assert await client.get(url, {"mode": "json"}) == b"[]"
    assert [(r.method, r.url) for r in seen] == [("GET", httpx2.URL(url, params={"mode": "json"}))]


async def test_the_live_client_follows_no_redirect() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        moved = {"location": "https://api.lever.co/v0/postings/other"}
        return httpx2.Response(302, content=b"moved", headers=moved)

    async with live_client(httpx2.MockTransport(handler)) as client:
        with pytest.raises(FatalToolError, match="board answered 302"):
            await client.get("https://api.lever.co/v0/postings/acme")
    assert len(seen) == 1


async def test_a_refused_host_ends_its_run_with_no_request(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    # Fatal, so the loop never retries it: one ErrorEvent with no body, and no request sent.
    transport, seen = _recording()
    acme = Board(name="acme", ats="lever", token="acme", base_url="https://acme.example")
    listing = ToolCallDecision(tool="list_postings", args={"board": "acme", "page": 1})
    conn = await db.connect(tmp_path / "rex.db")
    try:
        async with live_client(transport) as client:
            registry = build_registry(client, {"acme": acme})
            run_id = await run_hunt(
                conn, territory="acme", brain=scripted_brain([listing]), registry=registry
            )
        events = await db.read_events(conn, run_id)
        ((outcome, reason),) = await conn.execute_fetchall(
            "SELECT outcome, abort_reason FROM runs WHERE id = ?", (run_id,)
        )
    finally:
        await conn.close()
    (error,) = [e for e in events if isinstance(e, ErrorEvent)]
    assert (error.tool, error.retryable, error.raw_response) == ("list_postings", False, None)
    assert error.error == "FatalToolError('board host refused: https://acme.example')"
    assert seen == []
    assert (outcome, reason) == ("aborted", "tool failure")


def _faults(name: str, board: Board) -> list[str]:
    url = httpx2.URL(board.base_url)
    checks = {
        "name": board.name == name,
        "https": url.scheme == "https",
        "allowlisted": url.host in LIVE_HOSTS,
        "shape": url.host in ATS_HOSTS[board.ats],
        "bare": board.base_url == f"https://{url.host}",  # no port, path, query or userinfo
    }
    return [check for check, held in checks.items() if not held]


def test_every_live_board_is_https_on_an_allowlisted_host() -> None:
    # The check finds each fault on its own, so it holds over an empty list too.
    gh, lv, eu = (
        "https://boards-api.greenhouse.io",
        "https://api.lever.co",
        "https://api.eu.lever.co",
    )
    assert _faults("a", Board("a", "greenhouse", "acme", gh)) == []
    assert _faults("a", Board("a", "lever", "acme", lv)) == []
    assert _faults("a", Board("a", "lever", "acme", eu)) == []
    assert _faults("a", Board("b", "lever", "acme", lv)) == ["name"]
    assert _faults("a", Board("a", "lever", "acme", "http://api.lever.co")) == ["https", "bare"]
    assert _faults("a", Board("a", "lever", "acme", "https://jobs.lever.co")) == [
        "allowlisted",
        "shape",
    ]
    assert _faults("a", Board("a", "greenhouse", "acme", lv)) == ["shape"]
    assert _faults("a", Board("a", "lever", "acme", f"{lv}:8443")) == ["bare"]
    assert _faults("a", Board("a", "lever", "acme", f"{lv}/v0")) == ["bare"]
    faults = {name: _faults(name, board) for name, board in LIVE_BOARDS.items()}
    assert {name: found for name, found in faults.items() if found} == {}
