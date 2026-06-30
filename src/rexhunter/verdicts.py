"""The prey pen & the human-verdict state machine (ADR pillar 4).

"Pause" here is not a parked coroutine — it is a row state on disk. Hunts run to completion and
exit; captured prey lands as `prey` rows with status='awaiting_verdict'. A human verdict
(Feast / Release / Amber) is the ONLY path that transitions such a row (invariant 4, Tiny Arms):
there is no registered tool that submits/sends/applies, so the verdict is the human's own DB
write, not a confirmation dialog over an action Rex could otherwise take.

Two logs, one projection (invariant 2). The capture is a run-scoped trajectory event (the owning
hunt writes it, invariant 7); the verdict is a `pen_events` row (NOT run-scoped — it arrives
after the run has exited, from the POST handler). `prey` is the projection of both, maintained
transactionally and rebuildable by folding the events — never a second source of truth.

Unit 1 (here): `capture_prey`. The verdict machine and the enqueued follow-up job follow.
"""

import uuid
from datetime import UTC, datetime

import aiosqlite

from rexhunter import db, events

AWAITING = "awaiting_verdict"


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


async def capture_prey(
    conn: aiosqlite.Connection, run_id: str, *, territory: str, posting: str
) -> str:
    """Capture one posting into the pen. Returns the new prey_id.

    The PreyCapturedEvent (run-scoped, invariant 7; raw posting bytes, invariant 6) and the prey
    projection row are written in ONE transaction — a single commit. A crash mid-capture leaves
    NEITHER durable (atomicity), so the log and the pen never disagree about a capture.
    """
    prey_id = str(uuid.uuid4())
    now = _utcnow()
    event = events.PreyCapturedEvent(
        prey_id=prey_id, territory=territory, raw_posting=posting.encode()
    )
    # Reuse the log's append SQL but DON'T go through db.append_event — its per-call commit would
    # split this into two transactions. Both writes ride one commit below.
    await conn.execute(
        db.APPEND_EVENT_SQL,
        (run_id, event.type, event.model_dump_json(), now, run_id),
    )
    await conn.execute(
        "INSERT INTO prey (id, run_id, territory, posting, status, captured_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (prey_id, run_id, territory, posting, AWAITING, now),
    )
    await conn.commit()
    return prey_id
