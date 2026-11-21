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

import asyncio
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import aiosqlite
import pytest
from pydantic import ValidationError

from rexhunter import brain, db, events, server, stub, verdicts, view, viewstate
from rexhunter.events import ErrorEvent, PreyCapturedEvent, ToolCallEvent, Verdict
from rexhunter.loop import Brain, Decision, HuntComplete, ToolCallDecision, run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

BrainFactory = Callable[[Sequence[Decision]], Brain]
type Row = tuple[str, str | None]

GREENHOUSE = json.dumps(
    {
        "id": 101,
        "title": "Staff ML Engineer",
        "location": {"name": "Remote"},
        "absolute_url": "https://boards.example.invalid/quokkalabs/jobs/101",
        "content": "&lt;p&gt;Train and ship models.&lt;/p&gt;",
    }
).encode()
LEVER = json.dumps(
    {
        "id": "lv-7",
        "text": "AI Platform Engineer",
        "categories": {"location": "Sydney", "team": "Platform"},
        "hostedUrl": "https://jobs.example.invalid/wombatworks/lv-7",
        "descriptionPlain": "Build the platform.",
    }
).encode()
BODIES: dict[tuple[str, str], bytes] = {
    ("gym-greenhouse", "101"): GREENHOUSE,
    ("gym-lever", "lv-7"): LEVER,
}


def _ref(board: str, posting_id: str) -> events.PostingRef:
    return events.PostingRef(board=board, posting_id=posting_id)


def _registry(bodies: dict[tuple[str, str], bytes] | None = None) -> ToolRegistry:
    table = BODIES if bodies is None else bodies

    async def read_posting(board: str, posting_id: str) -> bytes:
        """Read one posting."""
        return table[(board, posting_id)]

    async def list_postings(board: str) -> bytes:
        """List a board's postings."""
        ids = [posting_id for (b, posting_id) in table if b == board]
        return json.dumps({"board": board, "postings": ids}).encode()

    async def glance(board: str, posting_id: str) -> bytes:
        """Glance at a posting: the same arguments, but no posting source."""
        return table[(board, posting_id)]

    reg = ToolRegistry()
    reg.tool(read_posting, posting_source=True)
    reg.tool(list_postings)
    reg.tool(glance)
    return reg


def _read(ref: events.PostingRef) -> ToolCallDecision:
    return ToolCallDecision(tool="read_posting", args=ref.model_dump())


async def _row(conn: aiosqlite.Connection, run_id: str) -> Row:
    async with conn.execute(
        "SELECT outcome, abort_reason FROM runs WHERE id = ?", (run_id,)
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    return str(row[0]), row[1]


async def _hunt(
    conn: aiosqlite.Connection,
    scripted_brain: BrainFactory,
    decisions: Sequence[Decision],
    reg: ToolRegistry | None = None,
    *,
    territory: str = "gym-greenhouse",
) -> tuple[str, list[events.TrajectoryEvent], Row]:
    run_id = await run_hunt(
        conn,
        territory=territory,
        brain=scripted_brain(decisions),
        registry=_registry() if reg is None else reg,
    )
    return run_id, await db.read_events(conn, run_id), await _row(conn, run_id)


async def _pen(conn: aiosqlite.Connection) -> list[tuple[Any, ...]]:
    rows = await conn.execute_fetchall(
        "SELECT board, posting_id, posting, status FROM prey ORDER BY captured_at"
    )
    return [tuple(r) for r in rows]


def _captures(evs: Sequence[events.TrajectoryEvent]) -> list[PreyCapturedEvent]:
    return [e for e in evs if isinstance(e, PreyCapturedEvent)]


def _capture_errors(evs: Sequence[events.TrajectoryEvent]) -> list[ErrorEvent]:
    return [e for e in evs if isinstance(e, ErrorEvent) and e.tool == "<capture>"]


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


def test_hunt_complete_schema_carries_board_and_posting_id() -> None:
    tools = {
        tool["name"]: tool["input_schema"] for tool in brain.build_tools(stub.build_registry())
    }
    schema = tools["hunt_complete"]
    items = schema["properties"]["catch"]["items"]
    assert items == {"$ref": "#/$defs/PostingRef"}
    ref = schema["$defs"]["PostingRef"]
    assert ref == {
        "additionalProperties": False,
        "properties": {"board": {"type": "string"}, "posting_id": {"type": "string"}},
        "required": ["board", "posting_id"],
        "type": "object",
    }  # the class docstring is not in it: written for readers of the code, not for the model


def test_a_posting_source_must_take_exactly_a_reference() -> None:
    reg = ToolRegistry()

    async def read_posting(board: str, posting_id: str) -> bytes:
        return b""

    async def by_board(board: str) -> bytes:
        return b""

    async def by_number(board: str, posting_id: int) -> bytes:
        return b""

    reg.tool(read_posting, posting_source=True)
    assert reg.get("read_posting").posting_source is True
    for wrong in (by_board, by_number):
        with pytest.raises(TypeError, match="a posting source takes exactly"):
            reg.tool(wrong, posting_source=True)


async def test_a_read_posting_is_penned_with_the_board_bytes(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    ref = _ref("gym-greenhouse", "101")
    conn = await db.connect(tmp_path / "rex.db")
    try:
        _, evs, row = await _hunt(conn, scripted_brain, [_read(ref), HuntComplete(catch=[ref])])
        assert row == ("completed", None)
        (capture,) = _captures(evs)
        assert capture.raw_posting == GREENHOUSE  # the board's bytes, as the read returned them
        assert (capture.board, capture.posting_id) == ("gym-greenhouse", "101")
        assert [p[:2] for p in await _pen(conn)] == [("gym-greenhouse", "101")]
        assert _capture_errors(evs) == []
    finally:
        await conn.close()


async def test_the_latest_read_of_a_posting_gives_its_bytes(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    reads = iter([b"first read", GREENHOUSE])
    reg = ToolRegistry()

    async def read_posting(board: str, posting_id: str) -> bytes:
        return next(reads)

    reg.tool(read_posting, posting_source=True)
    ref = _ref("gym-greenhouse", "101")
    conn = await db.connect(tmp_path / "rex.db")
    try:
        _, evs, _ = await _hunt(
            conn, scripted_brain, [_read(ref), _read(ref), HuntComplete(catch=[ref])], reg
        )
        assert [c.raw_posting for c in _captures(evs)] == [GREENHOUSE]
    finally:
        await conn.close()


async def test_an_unread_reference_is_rejected_and_never_penned(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    read, unread = _ref("gym-greenhouse", "101"), _ref("gym-greenhouse", "999")
    conn = await db.connect(tmp_path / "rex.db")
    try:
        _, evs, row = await _hunt(
            conn, scripted_brain, [_read(read), HuntComplete(catch=[unread, read])]
        )
        assert row == ("completed", None)  # the run completes; the resolved rest is penned
        (error,) = _capture_errors(evs)
        assert (error.retryable, error.error) == (False, "unresolved catch reference")
        assert json.loads(error.raw_request) == {"board": "gym-greenhouse", "posting_id": "999"}
        assert [p[:2] for p in await _pen(conn)] == [("gym-greenhouse", "101")]
    finally:
        await conn.close()


async def test_a_listed_but_unread_reference_is_rejected(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    # The posting appears in a list, and in the result of a tool that takes the same arguments
    # but is no posting source: neither is the run's read of it.
    ref = _ref("gym-greenhouse", "101")
    decisions = [
        ToolCallDecision(tool="list_postings", args={"board": "gym-greenhouse"}),
        ToolCallDecision(tool="glance", args=ref.model_dump()),
        HuntComplete(catch=[ref]),
    ]
    conn = await db.connect(tmp_path / "rex.db")
    try:
        _, evs, row = await _hunt(conn, scripted_brain, decisions)
        assert row == ("completed", None)
        assert len(_capture_errors(evs)) == 1
        assert _captures(evs) == []
        assert await _pen(conn) == []
    finally:
        await conn.close()


async def test_a_duplicate_reference_in_one_catch_is_penned_once(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    read, unread = _ref("gym-greenhouse", "101"), _ref("gym-greenhouse", "999")
    catch = [read, read, unread, unread]
    conn = await db.connect(tmp_path / "rex.db")
    try:
        _, evs, _ = await _hunt(conn, scripted_brain, [_read(read), HuntComplete(catch=catch)])
        assert len(_captures(evs)) == 1
        assert len(_capture_errors(evs)) == 1  # each distinct reference is handled once
        assert len(await _pen(conn)) == 1
    finally:
        await conn.close()


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


async def test_two_concurrent_hunts_catching_one_posting_pen_it_once(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    ref = _ref("gym-greenhouse", "101")
    db_path = tmp_path / "rex.db"
    first, second = await db.connect(db_path), await db.connect(db_path)
    try:
        results = await asyncio.gather(
            *(
                _hunt(conn, scripted_brain, [_read(ref), HuntComplete(catch=[ref])])
                for conn in (first, second)
            )
        )
        assert [row for _, _, row in results] == [("completed", None)] * 2
        assert sum(len(_captures(evs)) for _, evs, _ in results) == 1
        assert [_capture_errors(evs) for _, evs, _ in results] == [
            [],
            [],
        ]  # a duplicate is no error
        assert len(await _pen(first)) == 1
    finally:
        await first.close()
        await second.close()


@pytest.mark.parametrize(
    ("bodies", "ref", "label"),
    [
        (
            BODIES,
            ("gym-greenhouse", "101"),
            "Staff ML Engineer \N{EN DASH} Remote (gym-greenhouse)",
        ),
        (BODIES, ("gym-lever", "lv-7"), "AI Platform Engineer \N{EN DASH} Sydney (gym-lever)"),
        ({("gym-x", "1"): b'{"title": "ML Engineer"}'}, ("gym-x", "1"), "ML Engineer (gym-x)"),
        ({("gym-x", "1"): b"no title here"}, ("gym-x", "1"), "no title here"),
    ],
    ids=["greenhouse", "lever", "no-location", "no-title"],
)
async def test_the_pen_label_is_derived_from_the_board_bytes(
    bodies: dict[tuple[str, str], bytes],
    ref: tuple[str, str],
    label: str,
    tmp_path: Path,
    scripted_brain: BrainFactory,
) -> None:
    posting = _ref(*ref)
    conn = await db.connect(tmp_path / "rex.db")
    try:
        await _hunt(
            conn,
            scripted_brain,
            [_read(posting), HuntComplete(catch=[posting])],
            _registry(bodies),
            territory=ref[0],
        )
        assert [p[2] for p in await _pen(conn)] == [label]  # prey.posting
        pen = cast("list[dict[str, str]]", (await server.snapshot_state(conn))["pen"])
        assert [card["posting"] for card in pen] == [label]  # the snapshot
        state = await viewstate.build_viewstate(conn, datetime(2026, 11, 21, 9, tzinfo=UTC))
        assert [card.posting for card in state.pen] == [label]  # the log's projection
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


def test_project_messages_skips_a_capture_error() -> None:
    call = ToolCallEvent(tool="sniff", raw_request=b"{}", tool_use_id="toolu_1")
    error = ErrorEvent(
        tool="<capture>", retryable=False, error="unresolved catch reference", raw_request=b"{}"
    )
    assert brain.project_messages("gym", [call, error]) == brain.project_messages("gym", [call])
