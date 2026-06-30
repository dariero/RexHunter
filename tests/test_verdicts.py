"""P4 · prey pen + verdict machine.

Unit 1 (here): capture on completion. A hunt that finishes with a catch writes prey rows with
status='awaiting_verdict'; the capture is a run-scoped trajectory event (invariant 7) carrying
the raw posting bytes (invariant 6), and the event + row land in ONE transaction (atomicity).
Later units grow this file with the verdict state machine and the enqueued follow-up job.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import aiosqlite
import pytest

from rexhunter import db, verdicts
from rexhunter.events import PreyCapturedEvent, Verdict
from rexhunter.loop import Brain, Decision, HuntComplete, ToolCallDecision, run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

BrainFactory = Callable[[Sequence[Decision]], Brain]

PreyRow = tuple[str, str, str, str, str, str | None]


async def pen_rows(conn: aiosqlite.Connection) -> list[PreyRow]:
    async with conn.execute(
        "SELECT id, run_id, territory, posting, status, decided_at FROM prey ORDER BY captured_at"
    ) as cur:
        return [
            (str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]), r[5])
            for r in await cur.fetchall()
        ]


async def test_completed_hunt_captures_prey_as_awaiting_verdict(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    reg = ToolRegistry()

    @reg.tool
    async def sniff(prey: str) -> str:
        return f"posting:{prey}"

    conn = await db.connect(tmp_path / "rex.db")
    try:
        brain = scripted_brain(
            [
                ToolCallDecision(tool=sniff.__name__, args={"prey": "acme"}),
                HuntComplete(catch=["posting:acme", "posting:globex"]),
            ]
        )
        run_id = await run_hunt(conn, territory="mock-gym", brain=brain, registry=reg)

        rows = await pen_rows(conn)
        assert len(rows) == 2
        assert all(r[1] == run_id for r in rows)  # run provenance: which hunt caught it
        assert all(r[2] == "mock-gym" for r in rows)  # territory
        assert {r[3] for r in rows} == {"posting:acme", "posting:globex"}
        assert all(r[4] == "awaiting_verdict" for r in rows)  # the pending status
        assert all(r[5] is None for r in rows)  # not yet decided

        # capture is a run-scoped trajectory event (invariant 7) carrying raw posting bytes (inv 6)
        events = await db.read_events(conn, run_id)
        captures = [e for e in events if isinstance(e, PreyCapturedEvent)]
        assert len(captures) == 2
        assert {c.raw_posting for c in captures} == {b"posting:acme", b"posting:globex"}
        assert {c.prey_id for c in captures} == {r[0] for r in rows}  # event ↔ row pairing
    finally:
        await conn.close()


async def test_completion_without_catch_pens_nothing(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    reg = ToolRegistry()
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await run_hunt(
            conn, territory="t", brain=scripted_brain([HuntComplete()]), registry=reg
        )
        assert await pen_rows(conn) == []
        events = await db.read_events(conn, run_id)
        assert [e for e in events if isinstance(e, PreyCapturedEvent)] == []
    finally:
        await conn.close()


async def test_capture_prey_is_atomic(tmp_path: Path) -> None:
    # The PreyCapturedEvent and the prey row are ONE transaction: if the row INSERT fails, the
    # event must not durably land either. Proven on a FRESH connection (an uncommitted write is
    # invisible to a second reader), so this is real durability, not same-txn read-your-write.
    conn = await db.connect(tmp_path / "rex.db")
    run_id = await db.start_run(conn, territory="t")

    real_execute = conn.execute

    def fail_on_prey_insert(sql: str, *args: object, **kwargs: object) -> object:
        if "INSERT INTO prey" in sql:
            raise RuntimeError("disk full mid-capture")
        return real_execute(sql, *args, **kwargs)  # type: ignore[arg-type]

    conn.execute = fail_on_prey_insert  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await verdicts.capture_prey(conn, run_id, territory="t", posting="posting:acme")
    conn.execute = real_execute  # type: ignore[method-assign]
    await conn.rollback()  # discard the broken transaction
    await conn.close()

    fresh = await db.connect(tmp_path / "rex.db")  # the "second reader" / a restart
    try:
        events = await db.read_events(fresh, run_id)
        assert [e for e in events if isinstance(e, PreyCapturedEvent)] == []  # event rolled back
        async with fresh.execute("SELECT COUNT(*) FROM prey") as cur:
            row = await cur.fetchone()
        assert row is not None and row[0] == 0  # row rolled back
    finally:
        await fresh.close()


# ── Unit 2: the verdict state machine (guarded, idempotent) + the projection ──────────────


async def _pen_one(
    conn: aiosqlite.Connection, *, territory: str = "t", posting: str = "posting:x"
) -> str:
    """Capture one posting and return its prey_id (the verdict tests' fixture)."""
    run_id = await db.start_run(conn, territory=territory)
    return await verdicts.capture_prey(conn, run_id, territory=territory, posting=posting)


async def prey_status(conn: aiosqlite.Connection, prey_id: str) -> tuple[str, str | None]:
    async with conn.execute("SELECT status, decided_at FROM prey WHERE id = ?", (prey_id,)) as cur:
        row = await cur.fetchone()
    assert row is not None
    return str(row[0]), row[1]


async def pen_event_count(conn: aiosqlite.Connection, prey_id: str) -> int:
    async with conn.execute("SELECT COUNT(*) FROM pen_events WHERE prey_id = ?", (prey_id,)) as cur:
        row = await cur.fetchone()
    assert row is not None
    return int(row[0])


async def test_feast_transitions_awaiting_to_feasted(tmp_path: Path) -> None:
    conn = await db.connect(tmp_path / "rex.db")
    try:
        prey_id = await _pen_one(conn)
        assert await verdicts.submit_verdict(conn, prey_id, Verdict.FEAST) is True
        status, decided_at = await prey_status(conn, prey_id)
        assert status == "feasted"
        assert decided_at is not None  # a verdict stamps decided_at
        assert await pen_event_count(conn, prey_id) == 1  # exactly one VerdictEvent appended
    finally:
        await conn.close()


async def test_release_records_reason_and_requires_one(tmp_path: Path) -> None:
    conn = await db.connect(tmp_path / "rex.db")
    try:
        prey_id = await _pen_one(conn)
        # reason is the labelled-rejection payoff — required, enforced before any DB write.
        with pytest.raises(ValueError):
            await verdicts.submit_verdict(conn, prey_id, Verdict.RELEASE)
        assert (await prey_status(conn, prey_id))[0] == "awaiting_verdict"  # unchanged
        assert await pen_event_count(conn, prey_id) == 0

        assert await verdicts.submit_verdict(conn, prey_id, Verdict.RELEASE, reason="not AI-eng")
        async with conn.execute("SELECT status, reason FROM prey WHERE id = ?", (prey_id,)) as cur:
            row = await cur.fetchone()
        assert row is not None and row[0] == "released" and row[1] == "not AI-eng"
    finally:
        await conn.close()


async def test_amber_records_provenance_and_can_reenter(tmp_path: Path) -> None:
    conn = await db.connect(tmp_path / "rex.db")
    try:
        prey_id = await _pen_one(conn)
        assert await verdicts.submit_verdict(conn, prey_id, Verdict.AMBER, provenance="maybe later")
        async with conn.execute(
            "SELECT status, provenance, decided_at FROM prey WHERE id = ?", (prey_id,)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None and row[0] == "ambered" and row[1] == "maybe later"
        assert row[2] is not None

        # ambered is the ONLY state that re-enters the pen; re-entry clears decided_at.
        assert await verdicts.submit_verdict(conn, prey_id, Verdict.REENTER) is True
        status, decided_at = await prey_status(conn, prey_id)
        assert status == "awaiting_verdict" and decided_at is None
        assert await pen_event_count(conn, prey_id) == 2  # amber + reenter both logged
    finally:
        await conn.close()


async def test_double_feast_is_a_noop(tmp_path: Path) -> None:
    # Idempotency: the status guard makes a replayed / double-clicked FEAST a harmless no-op —
    # the second call matches zero rows, transitions nothing, appends NO second event.
    conn = await db.connect(tmp_path / "rex.db")
    try:
        prey_id = await _pen_one(conn)
        assert await verdicts.submit_verdict(conn, prey_id, Verdict.FEAST) is True
        assert await verdicts.submit_verdict(conn, prey_id, Verdict.FEAST) is False  # no-op
        assert (await prey_status(conn, prey_id))[0] == "feasted"
        assert await pen_event_count(conn, prey_id) == 1  # one event, not two
    finally:
        await conn.close()


async def test_verdict_on_a_resolved_row_is_a_noop(tmp_path: Path) -> None:
    conn = await db.connect(tmp_path / "rex.db")
    try:
        prey_id = await _pen_one(conn)
        assert await verdicts.submit_verdict(conn, prey_id, Verdict.FEAST) is True
        # feasted is terminal: RELEASE requires status='awaiting_verdict', so it no-ops.
        assert await verdicts.submit_verdict(conn, prey_id, Verdict.RELEASE, reason="x") is False
        assert (await prey_status(conn, prey_id))[0] == "feasted"  # untouched
        assert await pen_event_count(conn, prey_id) == 1
    finally:
        await conn.close()


async def test_reenter_only_from_ambered(tmp_path: Path) -> None:
    conn = await db.connect(tmp_path / "rex.db")
    try:
        prey_id = await _pen_one(conn)  # status awaiting_verdict, never ambered
        assert await verdicts.submit_verdict(conn, prey_id, Verdict.REENTER) is False  # no-op
        assert (await prey_status(conn, prey_id))[0] == "awaiting_verdict"
        assert await pen_event_count(conn, prey_id) == 0
    finally:
        await conn.close()


async def test_projection_never_drifts_from_the_verdict_log(tmp_path: Path) -> None:
    # The payoff of choosing the event form: prey is a PROJECTION. Fold the verdict log and it
    # must reproduce the row exactly — the column can never silently drift from the log.
    conn = await db.connect(tmp_path / "rex.db")
    try:
        prey_id = await _pen_one(conn)
        await verdicts.submit_verdict(conn, prey_id, Verdict.AMBER, provenance="first look")
        await verdicts.submit_verdict(conn, prey_id, Verdict.REENTER)
        await verdicts.submit_verdict(conn, prey_id, Verdict.RELEASE, reason="stale")

        log = await verdicts.read_pen_events(conn, prey_id)
        folded = verdicts.fold(log)  # (status, reason, provenance)

        async with conn.execute(
            "SELECT status, reason, provenance FROM prey WHERE id = ?", (prey_id,)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert folded == (row[0], row[1], row[2])
        assert folded[0] == "released"  # awaiting →amber →awaiting →released
    finally:
        await conn.close()
