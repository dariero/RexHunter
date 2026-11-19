"""db.connect under a cancel: the cause of the intermittent aiosqlite warning (W3.4b).

The warning ("Event loop is closed" from an aiosqlite worker thread, a
PytestUnhandledThreadExceptionWarning that surfaced in tests/test_stage5_gate.py) came from a
connection db.connect opened and never closed: a daemon shutdown that cancelled a task while it
was still inside db.connect left the connection's worker thread running, and aiosqlite's
`__del__` later stopped it on whatever event loop was current, by then a closed one. Bisected to
test_stage4_gate.py::test_boot_crash_sweep_marks_the_run_but_leaves_prey_untouched, whose short
body shuts the daemon down while its job worker may still be connecting.
"""

import asyncio
import threading
from pathlib import Path

import aiosqlite
import pytest

from rexhunter import db

pytestmark = pytest.mark.anyio


def _workers() -> set[threading.Thread]:
    return {t for t in threading.enumerate() if "_connection_worker_thread" in t.name}


async def test_a_cancel_inside_connect_closes_the_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = asyncio.Event()

    async def held(self: aiosqlite.Connection, sql_script: str) -> aiosqlite.Cursor:
        # The schema bootstrap never finishes: the cancel lands after the connection opened.
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError(f"unreachable: {self!r} {sql_script[:20]!r}")

    before = _workers()
    monkeypatch.setattr(aiosqlite.Connection, "executescript", held)
    task = asyncio.create_task(db.connect(tmp_path / "rex.db"))
    await entered.wait()
    opened = _workers() - before
    assert len(opened) == 1  # the connection db.connect opened, held inside its bootstrap
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for thread in opened:
        thread.join(timeout=1.0)
    assert [t.name for t in opened if t.is_alive()] == []  # and it was closed, its thread done
