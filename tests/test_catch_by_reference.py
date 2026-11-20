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
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rexhunter import events, view
from rexhunter.events import PreyCapturedEvent

pytestmark = pytest.mark.anyio


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
