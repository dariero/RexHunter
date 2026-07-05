"""SQLite write-ahead trajectory log (ADR pillar 1).

Commit before publish (invariant 1); single writer per run (invariant 7);
events append-only and immutable. `payload` is a typed trajectory event
(see events.py) serialised to JSON; reads cross the validation boundary
(invariant 3) via read_events -> events.decode_event.
"""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from rexhunter import events

# The write-ahead publish hook (ADR pillar 3): a committed event, offered to the broadcast hub.
# `(global_id, stored_payload_string)` — the exact JSON in the `payload` column, so a live-spliced
# viewer and a catch-up viewer render byte-identical feeds. Typed here (the lowest layer) and reused
# by the trajectory writers; the hub supplies the concrete callback. db.py never imports the hub —
# the callback is injected (inversion of control), keeping pillar 1 decoupled from pillar 3.
type PublishFn = Callable[[int, str], None]

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

-- The prey pen (ADR pillar 4). `prey` is a PROJECTION (invariant 2): base fields land at
-- capture (with a PreyCapturedEvent in trajectory_events), status/decided_at/reason/provenance
-- fold from pen_events. It is not a second source of truth — it is rebuildable from the logs.
CREATE TABLE IF NOT EXISTS prey (
    id          TEXT PRIMARY KEY,                  -- uuid4; mirrored on the capture event
    run_id      TEXT NOT NULL REFERENCES runs(id), -- the hunt that caught it (provenance)
    territory   TEXT NOT NULL,
    posting     TEXT NOT NULL,
    status      TEXT NOT NULL,                     -- awaiting_verdict|feasted|released|ambered
    captured_at TEXT NOT NULL,
    decided_at  TEXT,                              -- set on a verdict, cleared on re-entry
    reason      TEXT,                              -- RELEASE: labelled rejection data
    provenance  TEXT                               -- AMBER: why shelved
);

-- The verdict log (ADR pillar 4): NOT run-scoped — a verdict arrives after the hunt has exited,
-- from a different writer (the POST handler), so it cannot be a trajectory event of the run
-- (invariant 7). Append-only; `prey.status` is its projection. Global `id` is the pen cursor.
CREATE TABLE IF NOT EXISTS pen_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    prey_id    TEXT NOT NULL REFERENCES prey(id),
    type       TEXT NOT NULL,                      -- discriminator, mirrors the payload
    payload    TEXT NOT NULL,                      -- JSON-serialised VerdictEvent
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pen_events_prey ON pen_events (prey_id, id);

-- Follow-up work queue (ADR pillar 4): a FEAST enqueues a draft_pitch job here, atomically with
-- the verdict. Picked up by the same background machinery as hunts; the draft (a stub this slice)
-- lands in `result` for human editing — Rex drafts, never sends (invariant 4).
CREATE TABLE IF NOT EXISTS jobs (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,                      -- 'draft_pitch'
    prey_id    TEXT NOT NULL REFERENCES prey(id),
    status     TEXT NOT NULL,                      -- queued|running|done
    result     TEXT,                               -- the stub draft, once the worker runs
    created_at TEXT NOT NULL
);
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


# seq is assigned INSIDE the INSERT (no read-then-write gap): COALESCE(MAX(seq) + 1, 0) scoped
# to this run by `WHERE run_id = ?`. That scope is load-bearing under concurrency (invariant 7) —
# SQLite serialises the statement under the WAL write lock, so a concurrent writer on a DIFFERENT
# run never bumps this run's MAX. tests/test_invariant7.py derives the unscoped mutant by removing
# exactly that clause from this constant, proving scope is the sole variable.
APPEND_EVENT_SQL = (
    "INSERT INTO trajectory_events (run_id, seq, type, payload, created_at) "
    "SELECT ?, COALESCE(MAX(seq) + 1, 0), ?, ?, ? "
    "FROM trajectory_events WHERE run_id = ?"
)


async def append_event(
    conn: aiosqlite.Connection,
    run_id: str,
    event: events.TrajectoryEvent,
    *,
    publish: PublishFn | None = None,
) -> int:
    # The event is already typed (validated at construction) - serialising it OUT is not a
    # boundary crossing. model_dump_json() fills the payload column; event.type mirrors the
    # discriminator into the type column for SQL-side filtering.
    payload = event.model_dump_json()
    cursor = await conn.execute(
        APPEND_EVENT_SQL,
        (run_id, event.type, payload, _utcnow(), run_id),
    )
    await conn.commit()
    event_id = cursor.lastrowid
    if event_id is None:  # pragma: no cover - an INSERT always sets a rowid
        raise RuntimeError("append_event: INSERT returned no rowid")
    # Write-ahead (invariant 1): publish ONLY after the commit above returns the global id. The log
    # is truth; this is the notification. A missed publish is recoverable from the log, so this
    # never blocks — the hub drops a slow viewer rather than the writer awaiting it.
    if publish is not None:
        publish(event_id, payload)
    return event_id


async def read_events(conn: aiosqlite.Connection, run_id: str) -> list[events.TrajectoryEvent]:
    """Per-run replay read, routed through the validation boundary (invariant 3).

    Every stored payload crosses events.decode_event here; a corrupt or stale row raises
    rather than leaking an untyped string into the system. Ordered by the per-run `seq`
    cursor - the ghost-replay ordering.
    """
    async with conn.execute(
        "SELECT payload FROM trajectory_events WHERE run_id = ? ORDER BY seq", (run_id,)
    ) as cursor:
        rows = await cursor.fetchall()
    return [events.decode_event(str(row[0])) for row in rows]


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
