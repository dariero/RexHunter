"""eval_score (`P2.5`): eval tier 2's score of an existing log, printed from a read-only
connection that leaves the file as it was.

The log is the six first-wave hunts of 26 Nov (tests/test_gym_sonnet_5_5_hunt_replay.py), replayed
from their fixture streams into one file, each over its own fresh gym. The CLI's names are looked
up inside each test, so on a tree without it each test fails on its own line. No model, no spend,
no network.
"""

import asyncio
import hashlib
import re
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import anthropic
import httpx2
import pytest

from rexhunter import brain, db
from rexhunter.board.catalogue import GYM_BOARDS, GYM_CATALOGUE
from rexhunter.board.client import BoardClient
from rexhunter.board.gym import build_gym
from rexhunter.board.tools import GYM_SYSTEM_PROMPT, build_registry, gym_seed
from rexhunter.loop import run_hunt

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gym_sonnet_5_5_2026-11-26"
# The recorded first-wave runs, in the order they are replayed.
RECORDED = {
    "gym-greenhouse": "f0a5507e-d644-4d7d-938e-4a920c2a07ae",
    "gym-lever": "b929510c-5b9a-4b7d-b27a-4f4e0b2aadbd",
    "gym-storm": "1303d493-99d6-42a0-8af4-41d2c973396b",
    "gym-mojibake": "19618d54-198e-45dd-b356-29b730da7972",
    "gym-drift": "f726301f-e5ba-485a-918a-29e527009eae",
    "gym-pages": "e1e6b3d5-f943-4a20-b925-faca41f4ce24",
}
# What eval_score prints for that log: the pinned scores of tests/test_board_score.py, the wave's
# micro totals over the five runs expected to complete, and the log's whole spend.
PINNED = (
    "\n".join(
        [
            "territory       outcome                      as expected  prey  read p/d  caught"
            " p/d  unread catch  off board  calls  cost     precision  recall",
            "gym-greenhouse  completed                    yes          4     4/0       4/0   "
            "      0             0          6      $0.0277  1.00       1.00",
            "gym-lever       completed                    yes          4     4/0       4/0   "
            "      0             0          6      $0.0309  1.00       1.00",
            "gym-storm       completed                    yes          4     4/0       4/0   "
            "      0             0          6      $0.0308  1.00       1.00",
            "gym-mojibake    completed                    yes          4     4/0       4/0   "
            "      0             0          6      $0.0291  1.00       1.00",
            "gym-drift       aborted: tool failure        yes          4     0/0       0/0   "
            "      0             0          1      $0.0027  \N{EN DASH}          0.00",
            "gym-pages       aborted: cost ceiling        no           20    15/0      0/0   "
            "      0             0          18     $0.2081  \N{EN DASH}          0.00",
            "",
            "wave            5 runs expected to complete               36    31/0      16/0  "
            "      0                        42     $0.3267  1.00       0.44",
            "",
            "prey: on the board; read and caught: prey/decoys; unread catch: catch references"
            " capture refused; as expected: gym-drift aborted: tool failure, every other boar"
            "d completed; precision is undefined (\N{EN DASH}) on an empty catch.",
            "spend: $0.3294, the whole log",
        ]
    )
    + "\n"
)


def _reached(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise AssertionError("eval_score opened the log for writing")


async def _replay_into(path: Path, mock: MockAnthropic) -> None:
    # The daemon's live brain on the gym, each run served its stored streams in order over a fresh
    # gym (a storm counts down per gym instance), all six into one log under the live caps.
    conn = await db.connect(path)
    try:
        for territory, run_id in RECORDED.items():
            streams = iter(sorted((FIXTURE_DIR / f"hunt_{run_id}").iterdir()))

            def handler(_request: httpx2.Request, streams: Any = streams) -> httpx2.Response:
                return httpx2.Response(200, content=next(streams).read_bytes())

            board = BoardClient(httpx2.ASGITransport(app=build_gym(GYM_CATALOGUE)))
            async with board, mock(handler, api_key="replay") as client:
                registry = build_registry(board, GYM_BOARDS)
                brain_for = brain.adapter_brain_for(
                    client=client,
                    model="claude-sonnet-5-5",
                    registry=registry,
                    max_tokens=brain.HUNT_MAX_TOKENS,
                    thinking=brain.HUNT_THINKING,
                    stream=True,
                    system=GYM_SYSTEM_PROMPT,
                    seed=gym_seed,
                )
                await run_hunt(
                    conn,
                    territory=territory,
                    brain=brain_for(territory),
                    registry=registry,
                    cost_ceiling_usd=0.20,
                    daemon_spend_ceiling_usd=0.50,
                )
    finally:
        await conn.close()


def _score(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    from rexhunter import eval_score

    monkeypatch.setattr(sys, "argv", ["rexhunter.eval_score", str(path)])
    eval_score.main()


def test_eval_score_prints_the_pinned_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_anthropic: MockAnthropic,
) -> None:
    path = tmp_path / "rex.db"
    asyncio.run(_replay_into(path, mock_anthropic))
    _score(monkeypatch, path)
    assert capsys.readouterr().out == PINNED


def test_eval_score_leaves_the_db_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_anthropic: MockAnthropic,
) -> None:
    path = tmp_path / "rex.db"
    asyncio.run(_replay_into(path, mock_anthropic))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(db, "connect", _reached)  # it never opens a writer

    _score(monkeypatch, path)

    assert capsys.readouterr().out.startswith("territory")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert [p.name for p in tmp_path.iterdir()] == ["rex.db"]  # no -wal or -shm left beside it

    missing = tmp_path / "missing.db"
    with pytest.raises(SystemExit) as refused:
        _score(monkeypatch, missing)
    assert str(refused.value.code).startswith(f"no log at {missing}: ")
    assert not missing.exists()  # a missing log is refused, never created


def test_eval_score_reads_a_live_log_through_its_wal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rexhunter import eval_score

    path = tmp_path / "rex.db"
    asyncio.run(_bootstrap(path))  # the schema, in WAL mode, its writer closed
    # A writer still open, its run committed to the WAL and not yet checkpointed into the file.
    writer = sqlite3.connect(path)
    try:
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute(
            "INSERT INTO runs (id, territory, started_at) VALUES ('live', 'gym-mimic', 'now')"
        )
        writer.commit()
        assert (tmp_path / "rex.db-wal").exists()
        assert eval_score.read_only_uri(path).endswith("?mode=ro")
        _score(monkeypatch, path)
    finally:
        writer.close()

    rows = [re.split(r"\s{2,}", line) for line in capsys.readouterr().out.splitlines()]
    assert [row[:3] for row in rows if row[0] == "gym-mimic"] == [["gym-mimic", "live", "no"]]


async def _bootstrap(path: Path) -> None:
    conn = await db.connect(path)
    await conn.close()
