"""SQLite write-ahead trajectory log (ADR pillar 1).

Commit before publish (invariant 1); single writer per run (invariant 7);
events append-only and immutable. `payload` stays a raw string until Stage 2.
"""

import uuid
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    territory    TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    ended_at     TEXT,
    outcome      TEXT,
    abort_reason TEXT
);

CREATE TABLE IF NOT EXISTS trajectory_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL REFERENCES runs(id),
    seq        INTEGER NOT NULL,
    type       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_events_run ON trajectory_events (run_id, seq);
"""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


async def connect(path: str | Path) -> aiosqlite.Connection:
    conn = await aiosqlite.connect(path)
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA busy_timeout=5000")
    await conn.execute("PRAGMA foreign_keys=ON")
    await conn.executescript(_SCHEMA)
    await conn.commit()
    return conn


async def start_run(conn: aiosqlite.Connection, *, territory: str) -> str:
    run_id = str(uuid.uuid4())
    await conn.execute(
        "INSERT INTO runs (id, territory, started_at) VALUES (?, ?, ?)",
        (run_id, territory, _utcnow()),
    )
    await conn.commit()
    return run_id


async def append_event(conn: aiosqlite.Connection, run_id: str, *, type: str, payload: str) -> int:
    # seq is assigned inside the INSERT itself: the per-run cursor cannot race or gap.
    cursor = await conn.execute(
        """
        INSERT INTO trajectory_events (run_id, seq, type, payload, created_at)
        SELECT ?, COALESCE(MAX(seq) + 1, 0), ?, ?, ?
        FROM trajectory_events WHERE run_id = ?
        """,
        (run_id, type, payload, _utcnow(), run_id),
    )
    await conn.commit()
    event_id = cursor.lastrowid
    if event_id is None:  # pragma: no cover - an INSERT always sets a rowid
        raise RuntimeError("append_event: INSERT returned no rowid")
    return event_id


async def finish_run(
    conn: aiosqlite.Connection, run_id: str, *, outcome: str, abort_reason: str | None = None
) -> None:
    await conn.execute(
        "UPDATE runs SET ended_at = ?, outcome = ?, abort_reason = ?"
        " WHERE id = ? AND outcome IS NULL",
        (_utcnow(), outcome, abort_reason, run_id),
    )
    await conn.commit()


async def mark_crashed_runs(conn: aiosqlite.Connection) -> int:
    """Boot-time sweep: a run still open at boot did not survive its process (ADR DoD #1).

    ended_at is backfilled from the run's last committed event - the crash time is unknown,
    the last evidence is not.
    """
    cursor = await conn.execute(
        """
        UPDATE runs
        SET outcome = 'crashed',
            ended_at = (SELECT MAX(created_at) FROM trajectory_events WHERE run_id = runs.id)
        WHERE outcome IS NULL
        """
    )
    await conn.commit()
    return cursor.rowcount
