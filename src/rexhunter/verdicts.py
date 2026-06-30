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
from collections.abc import Sequence
from datetime import UTC, datetime

import aiosqlite

from rexhunter import db, events
from rexhunter.events import Verdict, VerdictEvent

AWAITING = "awaiting_verdict"

# verdict -> (required source status, resulting status). This table IS the guard: a verdict fires
# ONLY from its source status; from anywhere else it is a no-op. The SAME table drives the live
# UPDATE (below) and the fold (the projection rebuild), so the row and the log cannot disagree.
_TRANSITIONS: dict[Verdict, tuple[str, str]] = {
    Verdict.FEAST: (AWAITING, "feasted"),
    Verdict.RELEASE: (AWAITING, "released"),
    Verdict.AMBER: (AWAITING, "ambered"),
    Verdict.REENTER: ("ambered", AWAITING),
}


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


async def submit_verdict(
    conn: aiosqlite.Connection,
    prey_id: str,
    verdict: Verdict,
    *,
    reason: str | None = None,
    provenance: str | None = None,
) -> bool:
    """Apply a verdict as a guarded, idempotent transition. ``True`` = transitioned, ``False`` =
    no-op (the row was not in the verdict's required source state).

    One transaction: a status-guarded ``UPDATE ... WHERE id=? AND status=<source>``; the
    VerdictEvent is appended ONLY if the UPDATE matched the row (rowcount==1). A double-click, a
    replayed POST, or a verdict on an already-resolved row matches zero rows → no UPDATE, no
    event, nothing enqueued: a harmless no-op (idempotency). RELEASE requires a reason (the
    labelled-rejection payoff), enforced before any write. (FEAST's job enqueue lands in unit 3,
    inside this same rowcount==1 branch, keeping the flip + event + job one atomic step.)
    """
    if verdict is Verdict.RELEASE and reason is None:
        raise ValueError("RELEASE requires a reason (labelled rejection data)")

    source, target = _TRANSITIONS[verdict]
    now = _utcnow()
    decided_at = None if target == AWAITING else now  # re-entry returns to pending: clear it
    cursor = await conn.execute(
        "UPDATE prey SET status = ?, decided_at = ?,"
        " reason = COALESCE(?, reason), provenance = COALESCE(?, provenance)"
        " WHERE id = ? AND status = ?",
        (target, decided_at, reason, provenance, prey_id, source),
    )
    if cursor.rowcount != 1:
        await conn.rollback()  # guard didn't match: discard, append nothing — the no-op path
        return False

    event = VerdictEvent(prey_id=prey_id, verdict=verdict, reason=reason, provenance=provenance)
    await conn.execute(
        "INSERT INTO pen_events (prey_id, type, payload, created_at) VALUES (?, ?, ?, ?)",
        (prey_id, event.type, event.model_dump_json(), now),
    )
    await conn.commit()
    return True


async def read_pen_events(conn: aiosqlite.Connection, prey_id: str) -> list[VerdictEvent]:
    """One prey's verdict log, in order, routed through the validation boundary (invariant 3)."""
    async with conn.execute(
        "SELECT payload FROM pen_events WHERE prey_id = ? ORDER BY id", (prey_id,)
    ) as cursor:
        rows = await cursor.fetchall()
    return [events.decode_verdict_event(str(row[0])) for row in rows]


def fold(verdict_events: Sequence[VerdictEvent]) -> tuple[str, str | None, str | None]:
    """Reconstruct ``(status, reason, provenance)`` by replaying the verdict log — the projection's
    rebuild. Applies each event only from its required source status (the same guard the live
    UPDATE uses), so folding the log reproduces the prey row exactly: the column never drifts.
    """
    status, reason, provenance = AWAITING, None, None
    for event in verdict_events:
        source, target = _TRANSITIONS[event.verdict]
        if status != source:
            continue  # defensive: the log only holds applied transitions, so this never fires
        status = target
        if event.reason is not None:
            reason = event.reason
        if event.provenance is not None:
            provenance = event.provenance
    return status, reason, provenance
