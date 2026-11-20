"""Every stored timestamp is ISO 8601 in UTC, taken when its row is written.

Runs, trajectory events, prey, pen events and jobs each stamp a TEXT column with `db.utcnow()`.
SQL compares and folds those columns as text (the boot sweep's `MAX(created_at)`, the job queue's
`ORDER BY created_at`), so one format everywhere is what keeps text order equal to time order.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rexhunter import db, verdicts
from rexhunter.events import ToolCallEvent, Verdict

pytestmark = pytest.mark.anyio

STAMPS = {
    "runs.started_at": "SELECT started_at FROM runs",
    "runs.ended_at": "SELECT ended_at FROM runs",
    "trajectory_events.created_at": "SELECT created_at FROM trajectory_events",
    "prey.captured_at": "SELECT captured_at FROM prey",
    "prey.decided_at": "SELECT decided_at FROM prey",
    "pen_events.created_at": "SELECT created_at FROM pen_events",
    "jobs.created_at": "SELECT created_at FROM jobs",
}


async def test_every_stored_timestamp_is_iso_8601_utc_from_its_write(tmp_path: Path) -> None:
    before = datetime.now(UTC)
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await db.start_run(conn, territory="mock-gym")
        event = ToolCallEvent(tool="sniff", raw_request=b"{}", tool_use_id="toolu_1")
        await db.append_event(conn, run_id, event)
        prey_id = await verdicts.capture_prey(conn, run_id, territory="mock-gym", raw_posting=b"p")
        assert await verdicts.submit_verdict(conn, prey_id, Verdict.FEAST)  # stamps the pen + job
        await db.finish_run(conn, run_id, outcome="completed")
        stored: dict[str, list[str | None]] = {}
        for column, sql in STAMPS.items():
            async with conn.execute(sql) as cur:
                stored[column] = [row[0] for row in await cur.fetchall()]
    finally:
        await conn.close()
    after = datetime.now(UTC)

    for column, values in stored.items():
        assert values, column
        for value in values:
            assert isinstance(value, str), (column, value)
            stamp = datetime.fromisoformat(value)
            assert stamp.utcoffset() == timedelta(0), (column, value)
            assert stamp.isoformat() == value, (column, value)  # stored in that one form
            assert before <= stamp <= after, (column, value)
