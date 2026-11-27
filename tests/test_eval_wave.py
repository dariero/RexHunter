"""The one-wave harness (`P2.5`): each gym territory and gym-mimic hunted once into a fresh log,
then scored.

Offline throughout: the baseline mode spends nothing and builds no client, and the count-only and
live modes run over an httpx2 MockTransport with a dummy key, so nothing is sent anywhere. The
harness's names are looked up inside each test, so on a tree without it each test fails on its own
line.
"""

import asyncio
import hashlib
import json
import sqlite3
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NoReturn

import anthropic
import httpx2
import pytest

from rexhunter import brain, db, scheduler
from rexhunter.board.score import RunScore, score_log
from rexhunter.board.tools import GYM_SYSTEM_PROMPT, gym_seed

TERRITORIES = [
    "gym-greenhouse",
    "gym-lever",
    "gym-drift",
    "gym-storm",
    "gym-mojibake",
    "gym-pages",
    "gym-mimic",
]
# The wire check's body 10 is the daemon's live first request on gym-greenhouse, body 11 gym-mimic's
# and body 12 gym-greenhouse's count body: bodies 7, 9 and 8, each with prompt caching asked for
# (P5.1).
BODY_10 = "2ae0821a905cd9dd9d8ac35446765d99bd394b4a153d97ae55dc3c55f6c42492"
BODY_11 = "d0cd6ff6cadd35b8cdf10d2ed1a6e49d45f0cbd0f5a86811293936972a1ea0a2"
BODY_12 = "457ddb6676681959b28f0e324f4e10cddec9648cfe0070d96212676962cc8ab6"
DUMMY_KEY = "sk-test-not-real"

type Handler = Callable[[httpx2.Request], httpx2.Response]


def _sse(data: dict[str, object]) -> bytes:
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()


NEEDS_HELP_STREAM = b"".join(
    [
        _sse(
            {
                "type": "message_start",
                "message": {"model": brain.SMOKE_MODEL, "content": [], "usage": {}},
            }
        ),
        _sse(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "needs_help",
                    "input": {},
                },
            }
        ),
        _sse({"type": "content_block_stop", "index": 0}),
        _sse({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {}}),
        _sse({"type": "message_stop"}),
    ]
)


def _reached(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise AssertionError("the wave reached a path it must not")


@pytest.fixture(autouse=True)
def no_daemon_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    # The daemon's scheduler never returns: a wave that turned to it fails each test here at once,
    # never hangs the suite.
    monkeypatch.setattr(scheduler, "run_scheduler", _reached)


def _wave(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    from rexhunter import eval_wave

    monkeypatch.setattr(sys, "argv", ["rexhunter.eval_wave", *args])
    eval_wave.main()


def _offline(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> None:
    # The production client factory, over a MockTransport: what the wave would send, sent nowhere.
    factory = brain.anthropic_client

    def offline_client(api_key: str, *, timeout: float) -> anthropic.AsyncAnthropic:
        return factory(api_key, transport=httpx2.MockTransport(handler), timeout=timeout)

    monkeypatch.setattr(brain, "anthropic_client", offline_client)


def _runs(path: Path) -> list[tuple[str, str, str | None, float, int]]:
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        rows = conn.execute(
            "SELECT territory, outcome, abort_reason, cost_ceiling_usd, max_iterations FROM runs"
        ).fetchall()
    finally:
        conn.close()
    return sorted(
        (str(t), str(o), None if r is None else str(r), float(c), int(m)) for t, o, r, c, m in rows
    )


def _scores(path: Path) -> dict[str, RunScore]:
    from rexhunter.eval_score import MARKS

    async def read() -> Sequence[RunScore]:
        conn = await db.connect(path)
        try:
            return await score_log(conn, MARKS)
        finally:
            await conn.close()

    return {score.territory: score for score in asyncio.run(read())}


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm"])
@pytest.mark.parametrize("mode", ["--baseline", "--count-only", "live"])
def test_the_wave_refuses_an_existing_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, suffix: str
) -> None:
    taken = tmp_path / f"wave.db{suffix}"
    taken.write_bytes(b"taken")
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_KEY)
    monkeypatch.setenv("REXHUNTER_BRAIN", "live")
    monkeypatch.setattr(brain, "anthropic_client", _reached)
    monkeypatch.setattr(scheduler, "run_hunts", _reached)

    with pytest.raises(SystemExit) as refused:
        _wave(monkeypatch, "--db", str(tmp_path / "wave.db"), *([] if mode == "live" else [mode]))

    assert str(refused.value.code).startswith(f"refusing {taken}: ")
    assert taken.read_bytes() == b"taken"
    assert [p.name for p in tmp_path.iterdir()] == [taken.name]  # nothing else was written


@pytest.mark.parametrize("flag", ["--count", "--base"])
def test_the_wave_refuses_an_abbreviated_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    from rexhunter import eval_wave

    # The mode is read by its exact name (here and in `paid_entry`), so a prefix that argparse
    # would take for it must be refused, or `--count` would run the paid wave.
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_KEY)
    monkeypatch.setenv("REXHUNTER_BRAIN", "live")
    monkeypatch.setattr(eval_wave, "run_live_wave", _reached)
    monkeypatch.setattr(eval_wave, "count_first_bodies", _reached)
    monkeypatch.setattr(scheduler, "run_hunts", _reached)
    monkeypatch.setattr(brain, "anthropic_client", _reached)

    with pytest.raises(SystemExit) as refused:
        _wave(monkeypatch, "--db", str(tmp_path / "wave.db"), flag)

    assert refused.value.code == 2  # argparse's usage error
    assert list(tmp_path.iterdir()) == []


def test_the_wave_hunts_each_territory_once_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "wave.db"
    monkeypatch.delenv("REXHUNTER_BRAIN", raising=False)
    _wave(monkeypatch, "--db", str(path), "--baseline")  # it returns: one wave, then done

    # One run per territory, under the wave's per-run caps, each ended.
    assert _runs(path) == sorted(
        (t, "aborted", "tool failure", 0.40, 50)
        if t == "gym-drift"
        else (t, "completed", None, 0.40, 50)
        for t in TERRITORIES
    )
    # The baseline's floor per board: prey caught, decoys caught, prey on the board.
    scores = _scores(path)
    assert {
        t: (s.tally.prey_caught, s.tally.decoys_caught, s.tally.prey_on_board)
        for t, s in scores.items()
    } == {
        "gym-greenhouse": (3, 0, 4),
        "gym-lever": (3, 0, 4),
        "gym-drift": (0, 0, 4),
        "gym-storm": (3, 0, 4),
        "gym-mojibake": (4, 0, 4),
        "gym-pages": (9, 0, 20),
        "gym-mimic": (2, 4, 4),
    }
    out = capsys.readouterr().out.splitlines()
    assert out[0].split()[0] == "territory"
    assert sorted(line.split()[0] for line in out[1:8]) == sorted(TERRITORIES)
    assert out[-1] == "spend: $0.0000, the whole log"


def test_the_wave_passes_its_two_ceilings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rexhunter import eval_wave

    calls: list[dict[str, Any]] = []
    run_hunts = scheduler.run_hunts

    async def spy(db_path: Path, territories: Sequence[str], **kwargs: Any) -> list[str | None]:
        calls.append({"territories": list(territories), **kwargs})
        return await run_hunts(db_path, territories, **kwargs)

    monkeypatch.setattr(scheduler, "run_hunts", spy)
    monkeypatch.setattr(scheduler, "run_scheduler", _reached)
    monkeypatch.delenv("REXHUNTER_BRAIN", raising=False)
    _wave(monkeypatch, "--db", str(tmp_path / "wave.db"), "--baseline")

    (call,) = calls
    assert call["territories"] == TERRITORIES
    assert (call["cost_ceiling_usd"], call["daemon_spend_ceiling_usd"]) == (0.40, 0.75)
    assert (call["max_concurrent"], call["max_iterations"]) == (4, 50)
    assert (eval_wave.RUN_CEILING_USD, eval_wave.WAVE_CEILING_USD) == (0.40, 0.75)


def test_the_baseline_wave_builds_no_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "wave.db"
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("REXHUNTER_BRAIN", raising=False)
    monkeypatch.setattr(brain, "paid_entry", _reached)
    monkeypatch.setattr(brain, "anthropic_client", _reached)
    monkeypatch.setattr(brain, "select_brain_for", _reached)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", _reached)

    _wave(monkeypatch, "--db", str(path), "--baseline")

    assert len(_runs(path)) == len(TERRITORIES)


def test_the_wave_sends_the_daemons_first_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sent: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, content=NEEDS_HELP_STREAM)

    monkeypatch.setenv("REXHUNTER_BRAIN", "live")
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_KEY)
    _offline(monkeypatch, handler)
    path = tmp_path / "wave.db"
    _wave(monkeypatch, "--db", str(path))

    # One live call per territory (each asks for help), and no pre-flight.
    assert [str(r.url) for r in sent] == [brain.MESSAGES_URL] * len(TERRITORIES)
    first: dict[str, bytes] = {}
    for request in sent:
        body = json.loads(request.content)
        assert body["system"] == GYM_SYSTEM_PROMPT
        assert (body["thinking"], body["stream"]) == (brain.HUNT_THINKING, True)
        assert body["cache_control"] == {"type": "ephemeral"}
        first[body["messages"][0]["content"]] = request.content
    assert set(first) == {gym_seed(t) for t in TERRITORIES}
    assert hashlib.sha256(first[gym_seed("gym-greenhouse")]).hexdigest() == BODY_10
    assert hashlib.sha256(first[gym_seed("gym-mimic")]).hexdigest() == BODY_11
    assert {(t, o) for t, o, *_ in _runs(path)} == {(t, "needs_help") for t in TERRITORIES}
    assert "ceilings: $0.40 a run, $0.75 the wave" in capsys.readouterr().out


def test_the_count_only_wave_counts_each_first_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sent: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, json={"input_tokens": 1234})

    monkeypatch.delenv("REXHUNTER_BRAIN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_KEY)
    _offline(monkeypatch, handler)
    monkeypatch.setattr(scheduler, "run_hunts", _reached)
    path = tmp_path / "wave.db"
    _wave(monkeypatch, "--db", str(path), "--count-only")

    # Seven counts, one per territory's first body, and no messages request at all.
    assert [str(r.url) for r in sent] == [brain.COUNT_TOKENS_URL] * len(TERRITORIES)
    bodies = [json.loads(r.content) for r in sent]
    assert [b["messages"] for b in bodies] == [
        [{"role": "user", "content": gym_seed(t)}] for t in TERRITORIES
    ]
    assert all("max_tokens" not in b and b["system"] == GYM_SYSTEM_PROMPT for b in bodies)
    assert all(b["cache_control"] == {"type": "ephemeral"} for b in bodies)
    assert hashlib.sha256(sent[0].content).hexdigest() == BODY_12
    assert capsys.readouterr().out.splitlines() == [
        *(f"[count-only] {t}: input_tokens=1234" for t in TERRITORIES),
        "[count-only] validated, NO paid call made.",
    ]
    assert not path.exists()  # a count writes no log
