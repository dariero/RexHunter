"""Catch by reference: a hunt names its catch as (board, posting_id) references, and capture
resolves each against the run's own posting-source results.

A reference the run read is penned with the board's bytes (invariant 6) and with its board and
posting id; one the run never read, or only saw in a list, is a `<capture>` ErrorEvent and is not
penned; a posting already in the pen, in any status, is never penned again, and the unique index
decides a race between two hunts. The pen label is derived from the stored bytes, the same in the
log's projection, the snapshot and `prey.posting`. An older `prey` table gains the columns, and a
capture event from before this change still decodes.

The new names are looked up inside each test, so on a tree without them each test fails on its
own line. Every body is built here from literals; nothing under tests/fixtures/ is read or
written. No brain call reaches a network: the brains are scripted.
"""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
import pytest
from pydantic import ValidationError

from rexhunter import db, events, verdicts, view
from rexhunter.events import PreyCapturedEvent, Verdict

pytestmark = pytest.mark.anyio

GREENHOUSE = json.dumps(
    {
        "id": 101,
        "title": "Staff ML Engineer",
        "location": {"name": "Remote"},
        "absolute_url": "https://boards.example.invalid/quokkalabs/jobs/101",
        "content": "&lt;p&gt;Train and ship models.&lt;/p&gt;",
    }
).encode()


async def _pen(conn: aiosqlite.Connection) -> list[tuple[Any, ...]]:
    rows = await conn.execute_fetchall(
        "SELECT board, posting_id, posting, status FROM prey ORDER BY captured_at"
    )
    return [tuple(r) for r in rows]


def _captures(evs: Sequence[events.TrajectoryEvent]) -> list[PreyCapturedEvent]:
    return [e for e in evs if isinstance(e, PreyCapturedEvent)]


def test_a_posting_ref_is_two_strings_and_nothing_else() -> None:
    ref = events.PostingRef.model_validate({"board": "gym-greenhouse", "posting_id": "101"})
    same = events.PostingRef(board="gym-greenhouse", posting_id="101")
    assert ref == same
    assert len(dict.fromkeys([ref, same])) == 1  # frozen: equal references are one key
    for wrong in (
        {"board": "gym-greenhouse"},
        {"board": "gym-greenhouse", "posting_id": 101},
        {"board": "gym-greenhouse", "posting_id": "101", "url": "https://example.invalid"},
    ):
        with pytest.raises(ValidationError):
            events.PostingRef.model_validate(wrong)


@pytest.mark.parametrize(
    ("verdict", "status"),
    [
        (None, "awaiting_verdict"),
        (Verdict.FEAST, "feasted"),
        (Verdict.RELEASE, "released"),
        (Verdict.AMBER, "ambered"),
    ],
)
async def test_a_posting_already_in_the_pen_is_not_penned_again(
    verdict: Verdict | None, status: str, tmp_path: Path
) -> None:
    conn = await db.connect(tmp_path / "rex.db")
    try:
        first_run = await db.start_run(conn, territory="gym-greenhouse")
        first = await verdicts.capture_prey(
            conn,
            first_run,
            territory="gym-greenhouse",
            raw_posting=GREENHOUSE,
            board="gym-greenhouse",
            posting_id="101",
        )
        if verdict is not None:
            assert await verdicts.submit_verdict(conn, first, verdict, reason="r")
        second_run = await db.start_run(conn, territory="gym-greenhouse")
        again = await verdicts.capture_prey(
            conn,
            second_run,
            territory="gym-greenhouse",
            raw_posting=b"the same posting, read again",
            board="gym-greenhouse",
            posting_id="101",
        )
        assert again == first  # the row that already holds it
        assert [p[3] for p in await _pen(conn)] == [status]  # one row, its status untouched
        assert _captures(await db.read_events(conn, second_run)) == []  # no second event
    finally:
        await conn.close()


async def test_an_old_prey_table_gains_the_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "rex.db"
    async with aiosqlite.connect(db_path) as raw:  # a file from before catch by reference
        await raw.executescript(
            """
            CREATE TABLE runs (id TEXT PRIMARY KEY, territory TEXT NOT NULL,
                started_at TEXT NOT NULL, ended_at TEXT, outcome TEXT, abort_reason TEXT,
                cost_ceiling_usd REAL, max_iterations INTEGER);
            CREATE TABLE prey (id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                territory TEXT NOT NULL, posting TEXT NOT NULL, status TEXT NOT NULL,
                captured_at TEXT NOT NULL, decided_at TEXT, reason TEXT, provenance TEXT);
            INSERT INTO runs (id, territory, started_at, outcome)
                VALUES ('r0', 'mock-gym', '2026-11-01T00:00:00+00:00', 'completed');
            INSERT INTO prey (id, run_id, territory, posting, status, captured_at)
                VALUES ('p0', 'r0', 'mock-gym', 'posting:mock-gym', 'awaiting_verdict',
                        '2026-11-01T00:00:00+00:00');
            """
        )
    conn = await db.connect(db_path)
    try:
        columns = {str(r[1]) for r in await conn.execute_fetchall("PRAGMA table_info(prey)")}
        assert {"board", "posting_id"} <= columns
        assert await _pen(conn) == [(None, None, "posting:mock-gym", "awaiting_verdict")]
        indexes = {str(r[1]): r for r in await conn.execute_fetchall("PRAGMA index_list(prey)")}
        assert indexes["idx_prey_posting"][2] == 1  # unique
        assert indexes["idx_prey_posting"][4] == 1  # partial: rows from before have no board
    finally:
        await conn.close()


def test_a_capture_event_from_before_p2_4_still_decodes() -> None:
    # The guard: a stored capture written before catch by reference decodes, and its pen card
    # shows its bytes decoded, as it did.
    payload = json.dumps(
        {
            "type": "prey_captured",
            "prey_id": "p0",
            "territory": "mock-gym",
            "raw_posting": "cG9zdGluZzptb2NrLWd5bQ==",  # base64 of b"posting:mock-gym"
        }
    )
    event = events.decode_event(payload)
    assert isinstance(event, PreyCapturedEvent)
    assert event.raw_posting == b"posting:mock-gym"
    assert getattr(event, "board", "") == "" and getattr(event, "posting_id", "") == ""
    row = view.LogRow(id=1, seq=0, run_id="r0", created_at="2026-11-01T00:00:00+00:00", event=event)
    state = view.project([row], datetime(2026, 11, 21, 9, tzinfo=UTC))
    assert [card.posting for card in state.pen] == ["posting:mock-gym"]
