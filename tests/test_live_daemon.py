"""The daemon's board set (`P5.1`): `REXHUNTER_BOARDS` picks the gym, by default, or the live
boards, never both. Live boards are hunted once a day, and a start with an unknown value or no live
board is refused before the log is opened.

Through the real lifespan, offline: the live boards here are a test's own, patched in, and
`brain.live_transport` is replaced by an httpx2 `MockTransport` in every test that starts the
daemon, so the live client runs behind its allowlist over literals; nothing reaches a network.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest

from rexhunter import brain, db, server, stub
from rexhunter.board import Board, live
from rexhunter.board.catalogue import EVAL_BOARDS, GYM_BOARDS
from rexhunter.board.client import BoardClient
from rexhunter.loop import FatalToolError
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

ACME = Board(name="acme", ats="lever", token="acme", base_url="https://api.lever.co")
EUCO = Board(name="euco", ats="lever", token="euco", base_url="https://api.eu.lever.co")
GYM_URL = "http://gym/lever/v0/postings/wombatworks"


@pytest.fixture
def live_boards(monkeypatch: pytest.MonkeyPatch) -> dict[str, Board]:
    boards = {"acme": ACME, "euco": EUCO}
    monkeypatch.setattr(live, "LIVE_BOARDS", boards)
    return boards


def _network(monkeypatch: pytest.MonkeyPatch, answer: bytes | None = None) -> list[httpx2.Request]:
    # brain.live_transport's stand-in: with no answer, any request it carries fails the test.
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if answer is None:
            raise AssertionError(f"a request reached the network transport: {request.url}")
        return httpx2.Response(200, content=answer)

    monkeypatch.setattr(brain, "live_transport", lambda: httpx2.MockTransport(handler))
    return seen


async def _poll(db_path: Path, sql: str, done: int) -> None:
    # Bounded, so the suite never hangs on a hunt that never closes.
    conn = await db.connect(db_path)
    try:
        for _ in range(500):
            ((count,),) = await conn.execute_fetchall(sql)
            if int(count) >= done:
                return
            await asyncio.sleep(0.01)
    finally:
        await conn.close()
    raise AssertionError(f"never reached {done}: {sql}")


async def _runs(db_path: Path) -> list[tuple[str, str | None, str | None]]:
    conn = await db.connect(db_path)
    try:
        rows = await conn.execute_fetchall(
            "SELECT territory, outcome, abort_reason FROM runs ORDER BY territory"
        )
    finally:
        await conn.close()
    return [(str(row[0]), row[1], row[2]) for row in rows]


def test_the_daemon_hunts_the_gym_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REXHUNTER_BOARDS", raising=False)
    assert stub.boards_mode() == "gym"
    schedule, _brain_for, registry_for, _cap, _drafter = stub.daemon_config()
    assert schedule == stub.DEFAULT_SCHEDULE == dict.fromkeys(GYM_BOARDS, 30.0)
    assert registry_for is stub.gym_registry


def test_the_live_schedule_is_daily(
    live_boards: dict[str, Board], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REXHUNTER_BOARDS", "live")
    assert stub.LIVE_INTERVAL == 24 * 3600.0
    schedule, _brain_for, registry_for, _cap, _drafter = stub.daemon_config()
    assert schedule == dict.fromkeys(live_boards, 86400.0)
    assert registry_for is stub.live_registry


@pytest.mark.parametrize("value", ["staging", "GYM", "live,gym", ""])
async def test_an_unknown_boards_value_refuses_to_start(
    value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "rex.db"
    monkeypatch.setenv("REXHUNTER_BOARDS", value)
    monkeypatch.setattr(server, "DB_PATH", str(db_path))
    _network(monkeypatch)
    with pytest.raises(ValueError, match=r"unknown REXHUNTER_BOARDS"):
        async with server.app.router.lifespan_context(server.app):
            pass
    assert not db_path.exists()  # refused before the log is opened


async def test_live_mode_refuses_an_empty_board_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "rex.db"
    monkeypatch.setenv("REXHUNTER_BOARDS", "live")
    monkeypatch.setattr(live, "LIVE_BOARDS", {})
    monkeypatch.setattr(server, "DB_PATH", str(db_path))
    _network(monkeypatch)
    with pytest.raises(RuntimeError, match=r"REXHUNTER_BOARDS=live needs a board"):
        async with server.app.router.lifespan_context(server.app):
            pass
    assert not db_path.exists()


async def test_live_mode_builds_its_registry_over_the_live_client(
    live_boards: dict[str, Board], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The stub brain is the keyword baseline over the live boards: it lists page 1 of each, finds
    # no title to read in an empty list, and completes with an empty catch, at no spend.
    db_path = tmp_path / "rex.db"
    monkeypatch.setenv("REXHUNTER_BOARDS", "live")
    monkeypatch.delenv("REXHUNTER_BRAIN", raising=False)
    monkeypatch.setattr(server, "DB_PATH", str(db_path))
    seen = _network(monkeypatch, answer=b"[]")
    async with server.app.router.lifespan_context(server.app):
        await _poll(db_path, "SELECT COUNT(*) FROM runs WHERE outcome IS NOT NULL", 2)
    assert sorted((r.method, str(r.url)) for r in seen) == [
        ("GET", "https://api.eu.lever.co/v0/postings/euco?mode=json&skip=0&limit=20"),
        ("GET", "https://api.lever.co/v0/postings/acme?mode=json&skip=0&limit=20"),
    ]
    assert await _runs(db_path) == [("acme", "completed", None), ("euco", "completed", None)]


async def test_the_live_daemon_refuses_a_host_off_the_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A board whose host is not allowlisted is never reached: its hunt ends at the refusal.
    db_path = tmp_path / "rex.db"
    rogue = Board(name="rogue", ats="lever", token="rogue", base_url="https://evil.example")
    monkeypatch.setattr(live, "LIVE_BOARDS", {"rogue": rogue})
    monkeypatch.setenv("REXHUNTER_BOARDS", "live")
    monkeypatch.delenv("REXHUNTER_BRAIN", raising=False)
    monkeypatch.setattr(server, "DB_PATH", str(db_path))
    seen = _network(monkeypatch)
    async with server.app.router.lifespan_context(server.app):
        await _poll(db_path, "SELECT COUNT(*) FROM runs WHERE outcome IS NOT NULL", 1)
    assert seen == []
    assert await _runs(db_path) == [("rogue", "aborted", "tool failure")]


@pytest.mark.parametrize("boards", ["gym", "live"])
async def test_one_daemon_never_hunts_both_sets(
    boards: str,
    live_boards: dict[str, Board],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The lifespan hands the scheduler one set's schedule, a registry over that set's boards, and
    # that set's board client: the gym's answers no live host, and the live one refuses the gym.
    monkeypatch.setenv("REXHUNTER_BOARDS", boards)
    monkeypatch.setattr(server, "DB_PATH", str(tmp_path / "rex.db"))
    _network(monkeypatch)
    clients: list[BoardClient] = []

    def spying(registry_for: Callable[[BoardClient], ToolRegistry]) -> Any:
        def spy(client: BoardClient) -> ToolRegistry:
            clients.append(client)
            return registry_for(client)

        return spy

    monkeypatch.setattr(stub, "gym_registry", spying(stub.gym_registry))
    monkeypatch.setattr(stub, "live_registry", spying(stub.live_registry))
    handed: list[tuple[Any, ...]] = []
    started = asyncio.Event()

    async def scheduler_spy(*args: Any, **_kwargs: Any) -> None:
        handed.append(args)
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(server, "run_scheduler", scheduler_spy)
    async with server.app.router.lifespan_context(server.app):
        await asyncio.wait_for(started.wait(), timeout=5)
        ((_path, schedule),) = handed
        (client,) = clients
        if boards == "gym":
            assert set(schedule) == set(GYM_BOARDS)
            assert await client.get(GYM_URL)  # the in-process gym answers
            with pytest.raises(FatalToolError, match="board answered 404"):
                await client.get("https://api.lever.co/v0/postings/acme")  # the gym's own 404
        else:
            assert set(schedule) == set(live_boards)
            with pytest.raises(FatalToolError, match="board host refused: http://gym"):
                await client.get(GYM_URL)
    assert not set(schedule) & set(GYM_BOARDS if boards == "live" else live_boards)


def test_no_live_board_shares_a_gym_name() -> None:
    # A restart derives each territory's first hunt from its runs by name, so a shared name would
    # let one set's runs hold the other's board.
    assert not set(live.LIVE_BOARDS) & set(GYM_BOARDS | EVAL_BOARDS)
