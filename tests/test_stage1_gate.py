"""Stage 1 gate (ADR definition-of-done #1).

A real hunt subprocess writes through rexhunter.db and is SIGKILLed mid-append.
After the "restart" (reopening the log): every confirmed-committed event must be
queryable, and the dangling `outcome IS NULL` run must be marked 'crashed' at boot.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from rexhunter import db

pytestmark = pytest.mark.anyio

SRC = str(Path(__file__).resolve().parents[1] / "src")
CONFIRMED_APPENDS = 25

KilledHunt = tuple[Path, str, list[int]]

WRITER = """
import asyncio
import sys

from rexhunter import db


async def hunt(db_path: str) -> None:
    conn = await db.connect(db_path)
    run_id = await db.start_run(conn, territory="gate")
    print(run_id, flush=True)
    seq = 0
    while True:
        event_id = await db.append_event(conn, run_id, type="sniff", payload=f"prey-{seq}")
        print(event_id, flush=True)
        seq += 1


asyncio.run(hunt(sys.argv[1]))
"""


@pytest.fixture
def killed_hunt(tmp_path: Path) -> KilledHunt:
    """Run a hunt in a real subprocess, SIGKILL it mid-hunt, return its confirmed commits."""
    db_path = tmp_path / "rex.db"
    proc = subprocess.Popen(
        [sys.executable, "-c", WRITER, str(db_path)],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": SRC},
    )
    out = proc.stdout
    assert out is not None
    run_id = out.readline().strip()
    confirmed = [int(out.readline()) for _ in range(CONFIRMED_APPENDS)]
    proc.kill()  # SIGKILL: no atexit, no graceful close - a mid-hunt power cut
    proc.wait()
    return db_path, run_id, confirmed


async def test_pre_kill_events_survive_restart(killed_hunt: KilledHunt) -> None:
    db_path, run_id, confirmed = killed_hunt

    conn = await db.connect(db_path)  # the restart
    try:
        async with conn.execute("PRAGMA journal_mode") as cur:
            mode = await cur.fetchone()
        assert mode is not None and mode[0] == "wal"

        async with conn.execute(
            "SELECT id, seq, payload FROM trajectory_events WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ) as cur:
            rows = list(await cur.fetchall())

        ids = [int(row[0]) for row in rows]
        seqs = [int(row[1]) for row in rows]
        payloads = [str(row[2]) for row in rows]

        assert len(rows) >= CONFIRMED_APPENDS
        assert set(confirmed) <= set(ids)  # every confirmed append survived the kill
        assert seqs == list(range(len(rows)))  # per-run replay cursor is gapless and ordered
        assert payloads[:CONFIRMED_APPENDS] == [f"prey-{i}" for i in range(CONFIRMED_APPENDS)]
    finally:
        await conn.close()


async def test_dangling_run_marked_crashed_at_boot(killed_hunt: KilledHunt) -> None:
    db_path, run_id, _ = killed_hunt

    conn = await db.connect(db_path)
    try:
        async with conn.execute("SELECT outcome FROM runs WHERE id = ?", (run_id,)) as cur:
            row = await cur.fetchone()
        assert row is not None and row[0] is None  # the kill left the run dangling

        assert await db.mark_crashed_runs(conn) == 1  # boot detects and marks it

        async with conn.execute(
            "SELECT outcome, ended_at FROM runs WHERE id = ?", (run_id,)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None and row[0] == "crashed"
        assert row[1] is not None  # ended_at backfilled from the last committed event

        assert await db.mark_crashed_runs(conn) == 0  # second boot is a no-op
    finally:
        await conn.close()
