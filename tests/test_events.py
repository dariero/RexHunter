"""Stage 2 · slice 1 gate — typed event model + the validation boundary (invariant 3).

Two halves of one boundary:
  - round-trip / discriminator-mirror: a typed event survives append → read identical,
    and the DB `type` column stays in lockstep with the embedded discriminator.
  - boundary rejection (first-class negative): hostile bytes are rejected at the read
    crossing (`decode_event`), never surfaced as a silent untyped string.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rexhunter import db
from rexhunter.events import (
    ErrorEvent,
    SniffEvent,
    ToolCallEvent,
    ToolResultEvent,
    decode_event,
)


@pytest.mark.anyio
async def test_round_trip_fidelity(tmp_path: Path) -> None:
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await db.start_run(conn, territory="gate")
        original = SniffEvent(prey="AI Engineer")
        await db.append_event(conn, run_id, original)

        [restored] = await db.read_events(conn, run_id)
        assert restored == original  # equal by value: serialise → deserialise is lossless
    finally:
        await conn.close()


@pytest.mark.anyio
async def test_type_column_mirrors_discriminator(tmp_path: Path) -> None:
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await db.start_run(conn, territory="gate")
        await db.append_event(conn, run_id, SniffEvent(prey="Eval Engineer"))

        async with conn.execute(
            "SELECT type, payload FROM trajectory_events WHERE run_id = ?", (run_id,)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        column_type, payload = str(row[0]), str(row[1])

        # the SQL `type` column is a denormalised copy of the in-payload discriminator;
        # they must never drift (ADR: "the type column mirrors the discriminator").
        assert column_type == decode_event(payload).type == "sniff"
    finally:
        await conn.close()


# Each is bytes that must NOT round-trip into a typed event.
HOSTILE_PAYLOADS = [
    pytest.param('{"type": "sniff", "prey":', id="malformed-json"),
    pytest.param('{"type": "bark", "prey": "rabbit"}', id="unknown-type"),  # union tag invalid
    pytest.param('{"prey": "rabbit"}', id="type-absent"),  # no discriminator at all
    pytest.param('{"type": "sniff"}', id="missing-required-prey"),
    pytest.param('{"type": "sniff", "prey": "x", "claws": 3}', id="unknown-extra-field"),
]


@pytest.mark.parametrize("payload", HOSTILE_PAYLOADS)
def test_decode_event_rejects_hostile_input(payload: str) -> None:
    # the read crossing is the boundary: hostile bytes raise, never return a silent string.
    with pytest.raises(ValidationError):
        decode_event(payload)


@pytest.mark.anyio
async def test_read_path_routes_through_the_crossing(tmp_path: Path) -> None:
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await db.start_run(conn, territory="gate")
        # hand-insert a corrupt row, bypassing append_event's typed write boundary -
        # simulates a stale / hand-edited / older-version row in a durable log.
        await conn.execute(
            "INSERT INTO trajectory_events (run_id, seq, type, payload, created_at)"
            " VALUES (?, 0, 'sniff', ?, ?)",
            (run_id, '{"type": "sniff"}', "2026-01-01T00:00:00+00:00"),
        )
        await conn.commit()

        with pytest.raises(ValidationError):
            await db.read_events(conn, run_id)  # read path rejects, never a mystery string
    finally:
        await conn.close()


def test_discriminator_dispatches_to_the_right_member() -> None:
    # With >=2 members the union is discriminated by `type`: each tagged payload must decode
    # to its OWN class (not the first member), and round-trip equal.
    samples = [
        SniffEvent(prey="x"),
        ToolCallEvent(tool="fetch", raw_request=b"{}"),
        ToolResultEvent(tool="fetch", raw_request=b"{}", raw_response=b"{}"),
        ErrorEvent(tool="fetch", retryable=False, error="boom", raw_request=b"{}"),
    ]
    for event in samples:
        restored = decode_event(event.model_dump_json())
        assert type(restored) is type(event)
        assert restored == event


# Raw I/O fields are bytes (invariant 6). Pydantic's JSON default encodes bytes as UTF-8 and
# RAISES on non-UTF-8 input; the base64 config on _Event makes binary survive losslessly.
# model_dump_json() on these would raise without that config - so reaching the asserts at all
# already proves the encoding decision.
BINARY = b"\x00\xff\xfe binary \x01"


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(
            ToolResultEvent(tool="f", raw_request=b"{}", raw_response=BINARY), id="tool_result"
        ),
        pytest.param(
            ErrorEvent(tool="f", retryable=True, error="x", raw_request=b"{}", raw_response=BINARY),
            id="error",
        ),
    ],
)
def test_raw_bytes_survive_base64_round_trip(event: ToolResultEvent | ErrorEvent) -> None:
    restored = decode_event(event.model_dump_json())
    assert restored == event
    assert isinstance(restored, ToolResultEvent | ErrorEvent)
    assert restored.raw_response == BINARY  # binary preserved, not UTF-8 mangled


@pytest.mark.anyio
async def test_tool_result_binary_round_trips_through_db(tmp_path: Path) -> None:
    # The full path: append_event (model_dump_json, base64) -> SQLite -> read_events
    # (decode_event, base64). A dead run carrying binary tool I/O is a pytest fixture for free.
    conn = await db.connect(tmp_path / "rex.db")
    try:
        run_id = await db.start_run(conn, territory="gate")
        original = ToolResultEvent(
            tool="fetch_posting", raw_request=b'{"board":"x"}', raw_response=BINARY
        )
        await db.append_event(conn, run_id, original)

        [restored] = await db.read_events(conn, run_id)
        assert restored == original
    finally:
        await conn.close()
