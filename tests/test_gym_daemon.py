"""The daemon hunts the gym (`P2.4`, D9, fork `client-home`): the six gym territories are the
default schedule, the stub daemon hunts them with the keyword baseline, and the live daemon brain
sends the gym prompt and the board tools.

Through the real lifespan, offline: the stub daemon spends nothing, and the live one runs over an
httpx2 `MockTransport` with a dummy key. The lifespan builds the board client once, over the
in-process gym; `/viewstate`, which reads the schedule on every request, builds none.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from rexhunter import brain, db, server, stub
from rexhunter.board.catalogue import GYM_BOARDS, GYM_CATALOGUE
from rexhunter.board.client import BoardClient

pytestmark = pytest.mark.anyio


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


def test_the_default_schedule_hunts_the_gym_boards() -> None:
    assert stub.DEFAULT_SCHEDULE == dict.fromkeys(GYM_BOARDS, 30.0)
    assert list(stub.DEFAULT_SCHEDULE) == [
        "gym-greenhouse",
        "gym-lever",
        "gym-drift",
        "gym-storm",
        "gym-mojibake",
        "gym-pages",
    ]
    assert {GYM_BOARDS[name].token for name in stub.DEFAULT_SCHEDULE} == set(GYM_CATALOGUE)


async def test_the_stub_daemon_pens_gym_prey_with_no_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "rex.db"
    monkeypatch.delenv("REXHUNTER_BRAIN", raising=False)
    monkeypatch.setattr(server, "DB_PATH", str(db_path))
    closed: list[BoardClient] = []
    real_aclose = BoardClient.aclose

    async def counting_aclose(self: BoardClient) -> None:
        closed.append(self)
        await real_aclose(self)

    monkeypatch.setattr(BoardClient, "aclose", counting_aclose)

    async with server.app.router.lifespan_context(server.app):
        assert closed == []  # the lifespan's one board client is open while it hunts
        await _poll(
            db_path,
            "SELECT COUNT(*) FROM runs WHERE outcome = 'completed'"
            " AND territory IN ('gym-greenhouse', 'gym-lever')",
            2,
        )

    conn = await db.connect(db_path)
    try:
        rows = await conn.execute_fetchall("SELECT board, posting_id FROM prey ORDER BY board")
        captures = [
            e
            for (run_id,) in await conn.execute_fetchall("SELECT id FROM runs")
            for e in await db.read_events(conn, str(run_id))
            if e.type == "prey_captured"
        ]
        ((usage,),) = await conn.execute_fetchall(
            "SELECT COUNT(*) FROM trajectory_events WHERE type = 'usage'"
        )
    finally:
        await conn.close()

    planted = {
        (name, p.posting_id): p.planted
        for name, board in GYM_BOARDS.items()
        for p in GYM_CATALOGUE[board.token].postings
    }
    penned = {(str(b), str(p)) for b, p in rows}
    assert {key for key in penned if key[0] == "gym-greenhouse"} == {
        ("gym-greenhouse", "4001001"),
        ("gym-greenhouse", "4001002"),
        ("gym-greenhouse", "4001003"),
    }
    assert len({key for key in penned if key[0] == "gym-lever"}) == 3
    assert all(planted[key] == "prey" for key in penned)  # the baseline pens no decoy
    assert len(captures) == len(penned)
    assert int(usage) == 0  # no spend: the stub daemon makes no brain call
    assert len(closed) == 1  # and it is closed at shutdown


def _sse(data: dict[str, object]) -> bytes:
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()


NEEDS_HELP_STREAM = b"".join(
    [
        _sse(
            {
                "type": "message_start",
                "message": {"model": brain.SMOKE_MODEL, "content": [], "usage": {}},
            }
        ),
        _sse(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "needs_help",
                    "input": {},
                },
            }
        ),
        _sse({"type": "content_block_stop", "index": 0}),
        _sse({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {}}),
        _sse({"type": "message_stop"}),
    ]
)


async def test_the_live_daemon_brain_sends_the_gym_prompt_and_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rexhunter.board.tools import GYM_SYSTEM_PROMPT, gym_seed

    db_path = tmp_path / "rex.db"
    monkeypatch.setenv("REXHUNTER_BRAIN", "live")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setattr(server, "DB_PATH", str(db_path))
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, content=NEEDS_HELP_STREAM)

    factory = brain.anthropic_client

    def offline_client(api_key: str, *, timeout: float) -> anthropic.AsyncAnthropic:
        return factory(api_key, transport=httpx2.MockTransport(handler), timeout=timeout)

    monkeypatch.setattr(brain, "anthropic_client", offline_client)

    async with server.app.router.lifespan_context(server.app):
        await _poll(db_path, "SELECT COUNT(*) FROM runs WHERE outcome = 'needs_help'", 1)

    first = bodies[0]
    assert first["system"] == GYM_SYSTEM_PROMPT
    (territory,) = [t for t in GYM_BOARDS if first["messages"][0]["content"] == gym_seed(t)]
    assert first["messages"] == [{"role": "user", "content": gym_seed(territory)}]
    assert [tool["name"] for tool in first["tools"]] == [
        "list_postings",
        "read_posting",
        "hunt_complete",
        "needs_help",
    ]
    assert (first["thinking"], first["stream"], first["max_tokens"]) == (
        brain.HUNT_THINKING,
        True,
        brain.HUNT_MAX_TOKENS,
    )
    assert {body["system"] for body in bodies} == {GYM_SYSTEM_PROMPT}


async def test_viewstate_builds_no_board_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[object] = []
    real_init = BoardClient.__init__

    def counting_init(self: BoardClient, transport: httpx2.AsyncBaseTransport) -> None:
        built.append(transport)
        real_init(self, transport)

    monkeypatch.setattr(BoardClient, "__init__", counting_init)
    monkeypatch.setattr(server, "DB_PATH", str(tmp_path / "rex.db"))

    first = await server.get_viewstate()
    second = await server.get_viewstate()

    assert built == []
    body = bytes(first.body).decode()
    assert all(name in body for name in GYM_BOARDS)
    assert "mock-gym" not in body
    assert bytes(second.body) == bytes(first.body)
