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

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from rexhunter import db, events
from rexhunter.events import Verdict, VerdictEvent

logger = logging.getLogger("rexhunter")

AWAITING = "awaiting_verdict"

# The follow-up job queue (a FEAST enqueues one). A draft_pitch job moves queued → running → done.
DRAFT_PITCH = "draft_pitch"
QUEUED, RUNNING, DONE = "queued", "running", "done"

# A pitch drafter: read the prey it is given, return the draft text. The stub this slice does no
# LLM work; P5 swaps in the paid drafter. Injected (not imported) so this module stays free.
Drafter = Callable[[aiosqlite.Connection, str], Awaitable[str]]

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
    if verdict is Verdict.FEAST:
        # The follow-up draft_pitch job, enqueued ATOMICALLY with the flip + event (same txn, gated
        # by the rowcount==1 above). Exactly-once: one FEAST → one job, even under a concurrent
        # double-FEAST — the loser matched zero rows and never reached here. The handler does no
        # heavy work; the worker drafts (Tiny Arms: Rex drafts, never sends).
        await conn.execute(
            "INSERT INTO jobs (id, kind, prey_id, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), DRAFT_PITCH, prey_id, QUEUED, now),
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


# ── The follow-up worker (ADR pillar 4): the same background machinery as hunts ───────────────


async def claim_job(conn: aiosqlite.Connection) -> tuple[str, str] | None:
    """Atomically claim the oldest queued job: flip exactly one queued→running, return its
    (job_id, prey_id), or None if the queue is empty. The guarded flip keeps the claim safe even
    if a second worker ever races — each job is handed to at most one (jobs is multi-writer)."""
    async with conn.execute(
        "SELECT id, prey_id FROM jobs WHERE status = ? ORDER BY created_at, id LIMIT 1", (QUEUED,)
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return None
    job_id, prey_id = str(row[0]), str(row[1])
    cursor = await conn.execute(
        "UPDATE jobs SET status = ? WHERE id = ? AND status = ?", (RUNNING, job_id, QUEUED)
    )
    await conn.commit()
    if cursor.rowcount != 1:
        return None  # lost the race; the next poll picks up whatever is left
    return job_id, prey_id


async def complete_job(conn: aiosqlite.Connection, job_id: str, result: str) -> None:
    """Record the drafted pitch and mark the job done. The draft LANDS in `result` for human
    editing — there is no send/submit path (invariant 4, Tiny Arms)."""
    await conn.execute(
        "UPDATE jobs SET status = ?, result = ? WHERE id = ?", (DONE, result, job_id)
    )
    await conn.commit()


async def requeue_running_jobs(conn: aiosqlite.Connection) -> int:
    """Boot sweep (the jobs analogue of db.mark_crashed_runs): a job left 'running' means the
    worker died mid-draft. Reset it to 'queued' so it is picked up again. Safe because the stub
    drafter is pure; P5's paid drafter will need an idempotency guard before re-running."""
    cursor = await conn.execute("UPDATE jobs SET status = ? WHERE status = ?", (QUEUED, RUNNING))
    await conn.commit()
    return cursor.rowcount


async def run_job_worker(
    db_path: str | Path, *, drafter: Drafter, poll_interval: float = 0.5
) -> None:
    """Drain the jobs queue forever: claim a queued job, run the drafter, record the draft; sleep
    when the queue is empty. The same background shape as the hunt scheduler — its own connection
    (jobs is multi-writer), cancellable (daemon shutdown tears it down via its finally)."""
    conn = await db.connect(db_path)
    try:
        while True:
            claimed = await claim_job(conn)
            if claimed is None:
                await asyncio.sleep(poll_interval)  # idle: nothing queued
                continue
            job_id, prey_id = claimed
            draft = await drafter(conn, prey_id)
            await complete_job(conn, job_id, draft)
    finally:
        # The connection's worker thread is non-daemon: it MUST close or it blocks interpreter
        # shutdown (and leaks). The close runs inside the very cancellation that triggers daemon
        # shutdown, which would otherwise interrupt a bare `await conn.close()` — so, exactly like
        # run_hunt's shielded finish_run, shield the close and drive it to done through any
        # re-cancel that lands mid-flight.
        closing = asyncio.ensure_future(conn.close())
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                pass
