"""Slice C · live verdict streaming.

An APPLIED verdict POST fans an id-less notification frame to every open board (hub.notify), so all
viewers refresh to the new pen status — not just the tab that clicked. It fires post-commit
(write-ahead, invariant 1) and is id-less, so the trajectory-id resume cursor is untouched
(``pen_events`` has its own id sequence and must not collide with the ``/events`` stream). A no-op
verdict (already resolved) fans out nothing.

Drives the real ``/verdict`` route via ``httpx.ASGITransport`` (no lifespan → no daemon), with the
hub set on ``app.state`` by hand.
"""

from pathlib import Path

import httpx
import pytest

from rexhunter import db, server, verdicts
from rexhunter.hub import Hub

pytestmark = pytest.mark.anyio


async def _seed_prey(db_path: Path) -> str:
    conn = await db.connect(db_path)
    try:
        run_id = await db.start_run(conn, territory="mock-gym")
        return await verdicts.capture_prey(conn, run_id, territory="mock-gym", posting="posting:x")
    finally:
        await conn.close()


async def test_applied_verdict_notifies_every_live_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "DB_PATH", str(tmp_path / "rex.db"))
    hub = Hub()
    server.app.state.hub = hub
    prey_id = await _seed_prey(tmp_path / "rex.db")
    _, queue = hub.register()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://t"
    ) as client:
        resp = await client.post("/verdict", json={"prey_id": prey_id, "verdict": "feast"})

    assert resp.json() == {"applied": True}
    env = queue.get_nowait()
    assert (
        env.id is None
    )  # id-less: never advances Last-Event-ID nor collides with the trajectory stream
    assert '"verdict":"feast"' in env.data  # carries the verdict payload (also shown in the #feed)
    assert env.sse().startswith("data:")  # a message frame → the browser onmessage → board refresh


async def test_noop_verdict_notifies_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "DB_PATH", str(tmp_path / "rex.db"))
    hub = Hub()
    server.app.state.hub = hub
    prey_id = await _seed_prey(tmp_path / "rex.db")
    _, queue = hub.register()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://t"
    ) as client:
        first = await client.post("/verdict", json={"prey_id": prey_id, "verdict": "feast"})
        assert first.json() == {"applied": True}
        queue.get_nowait()  # drain the applied notification
        # a SECOND feast on the now-feasted prey is a status-guarded no-op (verdicts.py:127)
        second = await client.post("/verdict", json={"prey_id": prey_id, "verdict": "feast"})

    assert second.json() == {"applied": False}
    assert queue.empty()  # a no-op verdict fans out nothing
