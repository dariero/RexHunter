"""Offline replay of the first live daemon hunts on Claude Sonnet 5.5 (DoD #5, no network).

On 19 Nov 2026 UTC the daemon ran six `mock-gym` hunts live (`REXHUNTER_BRAIN=live`,
`claude-sonnet-5-5`, adaptive thinking with `display: "summarized"`, streamed,
`COST_CEILING_USD=0.05`, `DAEMON_SPEND_CEILING_USD=0.05`, `MAX_ITERATIONS` unset, so 50), and each
brain attempt stored its exchange in the daemon's log. The fixture directory
`tests/fixtures/daemon_sonnet_5_5_2026-11-20/` keeps two of those runs, one per outcome, each
response's bytes as stored: `058dc53e…` (completed, an empty catch) and `5e555e12…` (needs_help).
The request bytes are not kept: the terminal tools' schemas have changed since the run, so a stored
request would pin an old wire, and the replay checks each request's shape instead.

Re-driving each run through the adapter, on its recorded model and under the daemon's settings,
reproduces its log: the outcome and the run's caps, every event but the exchanges payload for
payload (against a digest of the live log's payloads), and each exchange's status and response
bytes.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import anthropic
import httpx2
import pytest

from rexhunter import brain, cost, db, stub
from rexhunter.events import (
    BrainExchangeEvent,
    ThinkingDelta,
    ToolCallEvent,
    ToolResultEvent,
    TrajectoryEvent,
    UsageEvent,
)
from rexhunter.loop import run_hunt
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "daemon_sonnet_5_5_2026-11-20"
RECORDED_MODEL = "claude-sonnet-5-5"  # both runs were recorded on it (each stream's "model")
TERRITORY = "mock-gym"
# The daemon's settings for these hunts: the two ceilings the live environment set, server.py's
# default MAX_ITERATIONS, and the scheduler's default tool timeout and retry budget.
COST_CEILING_USD = 0.05
DAEMON_SPEND_CEILING_USD = 0.05
MAX_ITERATIONS = 50
TOOL_TIMEOUT_S = 30.0
RETRY_BUDGET = 2

type ToolUse = tuple[str, str, dict[str, Any]]


class Run(NamedTuple):
    run_id: str
    outcome: tuple[str, str | None]
    sha256: dict[str, str]  # each call's stream, pinned: a rewritten fixture fails here first
    tool_uses: list[ToolUse]  # each call's one tool_use, read off its stream
    usage: list[tuple[int, int]]  # each call's (input, output) tokens
    deltas: list[int]  # the ThinkingDeltas relayed during each call
    spend_usd: float  # the run's fold under today's table
    log_sha256: str  # the live log's payloads but the exchanges', in seq order, "\n"-joined


RUNS = [
    Run(
        run_id="058dc53e-6c44-4e76-bf09-10e4b9f44ce1",
        outcome=("completed", None),
        sha256={
            "call_01.json": "45e5f4cb5005270c8cebb4f456f535201be5b7fd036090fcabb9b93d67c3f851",
            "call_02.json": "649c4fa3b85005691d5dbf735c29488bf893a1633e32c612ff7cbffb8b90c4d4",
            "call_03.json": "09b3a415840dec283a2dc2a099850af6a5952fe0f9ad1bcf8c0fd0bed57445be",
        },
        tool_uses=[
            (
                "toolu_01GJaeCcsvmkmoXDDE4KZJuG",
                "sniff",
                {"prey": "mock-gym AI-engineering job postings"},
            ),
            ("toolu_01LtbZjZ7VHm7pidT3sZhLfL", "sniff", {"prey": "mock-gym"}),
            ("toolu_012Ri54rLywiwLNDZt6W33MY", "hunt_complete", {"catch": []}),
        ],
        usage=[(1073, 59), (1158, 149), (1324, 165)],
        deltas=[0, 21, 20],
        spend_usd=0.01084,  # (1073 + 1158 + 1324) x $2/M + (59 + 149 + 165) x $10/M
        log_sha256="b93ae5ba7f66bd5d2dbb32f01bb4429c9da6605563cf2ea384e25ed2b1e0ffe8",
    ),
    Run(
        run_id="5e555e12-f9a7-47d9-b190-10ef0bb766d4",
        outcome=("needs_help", None),
        sha256={
            "call_01.json": "9e25f43912aa3cdc4e4081a9472b3bd622a995dcf74730521d20ba98297fb4c3",
            "call_02.json": "cdaac1b10aa049f3cc3ee46962bdbdda0f2205998f8114e4ff5564a8326e0184",
            "call_03.json": "8bf74da2b1bfabe9a4e05e98256773e03e93a8c6f4b5c62d660ebcdfc0cf0b99",
        },
        tool_uses=[
            (
                "toolu_01GrmcqVvwHo2uuwjZs88gpB",
                "sniff",
                {"prey": "mock-gym AI-engineering job postings"},
            ),
            ("toolu_01DRseHAtY97r9cttKFiqpwT", "sniff", {"prey": "mock-gym"}),
            ("toolu_014UYkDqEXHc7fXmGm7xZLHd", "needs_help", {}),
        ],
        usage=[(1073, 59), (1158, 197), (1372, 46)],
        deltas=[0, 21, 12],
        spend_usd=0.010226,  # (1073 + 1158 + 1372) x $2/M + (59 + 197 + 46) x $10/M
        log_sha256="eff4c1168a26080e627363cdeb1d5b28e273acd6083742a3d7a17e2c4b72d33b",
    ),
]
IDS = [run.run_id[:8] for run in RUNS]

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]


class Replay(NamedTuple):
    events: list[TrajectoryEvent]
    payloads: list[tuple[str, str]]  # each row's (type, payload) as stored, in seq order
    runs_row: tuple[Any, ...]  # (outcome, abort_reason, cost_ceiling_usd, max_iterations)
    requests: list[httpx2.Request]


def _streams(run: Run) -> list[bytes]:
    return [(FIXTURE_DIR / f"hunt_{run.run_id}" / name).read_bytes() for name in run.sha256]


def _assemble(raw: bytes) -> dict[str, Any]:
    assembler = brain.StreamAssembler()
    for event in brain.iter_sse_events(raw):
        assembler.feed(event)
    return assembler.assembled()


async def sniff(prey: str) -> str:
    # The daemon's board tool without stub.sniff's 5 s beat: it echoes the scent, so every
    # tool_result replays byte for byte.
    return f"posting:{prey}"


# The tool description the daemon sent, which the model reads: stub.sniff's docstring.
sniff.__doc__ = stub.sniff.__doc__


async def _replay(run: Run, db_path: Path, mock_anthropic: MockAnthropic) -> Replay:
    streams = _streams(run)
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, content=streams[len(requests) - 1])

    reg = ToolRegistry()
    reg.tool(sniff)
    async with mock_anthropic(handler, api_key="replay") as client:
        # The daemon's live brain (`brain.select_brain_for`), on the recorded model.
        brain_for = brain.adapter_brain_for(
            client=client,
            model=RECORDED_MODEL,
            registry=reg,
            max_tokens=brain.HUNT_MAX_TOKENS,
            thinking=brain.HUNT_THINKING,
            stream=True,
        )
        conn = await db.connect(db_path)
        try:
            run_id = await run_hunt(
                conn,
                territory=TERRITORY,
                brain=brain_for(TERRITORY),
                registry=reg,
                tool_timeout_s=TOOL_TIMEOUT_S,
                retry_budget=RETRY_BUDGET,
                max_iterations=MAX_ITERATIONS,
                cost_ceiling_usd=COST_CEILING_USD,
                daemon_spend_ceiling_usd=DAEMON_SPEND_CEILING_USD,
            )
            events = await db.read_events(conn, run_id)
            rows = await conn.execute_fetchall(
                "SELECT type, payload FROM trajectory_events WHERE run_id = ? ORDER BY seq",
                (run_id,),
            )
            caps = await conn.execute_fetchall(
                "SELECT outcome, abort_reason, cost_ceiling_usd, max_iterations FROM runs"
                " WHERE id = ?",
                (run_id,),
            )
        finally:
            await conn.close()
    payloads = [(str(row[0]), str(row[1])) for row in rows]
    return Replay(events, payloads, tuple(next(iter(caps))), requests)


def _expected_types(run: Run) -> list[str]:
    # A call's deltas are written during the call, before its exchange; a sniff then writes its
    # call and result, and the terminal call writes nothing more.
    types: list[str] = []
    for (_id, name, _input), deltas in zip(run.tool_uses, run.deltas, strict=True):
        types += ["thinking_delta"] * deltas + ["brain_exchange", "usage"]
        if name == "sniff":
            types += ["tool_call", "tool_result"]
    return types


def test_the_fixture_holds_the_two_recorded_runs_unchanged() -> None:
    assert sorted(p.name for p in FIXTURE_DIR.iterdir()) == sorted(
        f"hunt_{run.run_id}" for run in RUNS
    )
    for run in RUNS:
        run_dir = FIXTURE_DIR / f"hunt_{run.run_id}"
        assert sorted(p.name for p in run_dir.iterdir()) == list(run.sha256)
        for name, digest in run.sha256.items():
            assert hashlib.sha256((run_dir / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("run", RUNS, ids=IDS)
def test_each_stream_assembles_to_the_recorded_turn(run: Run) -> None:
    streams = _streams(run)
    assert len(streams) == len(run.tool_uses) == len(run.usage) == len(run.deltas) == 3
    for raw, (tool_use_id, name, tool_input), (n_in, n_out), deltas in zip(
        streams, run.tool_uses, run.usage, run.deltas, strict=True
    ):
        turn = _assemble(raw)
        assert turn["model"] == RECORDED_MODEL
        assert turn["stop_reason"] == "tool_use"
        # Call one did not think; each later call carries one signed thinking block first.
        types = ["thinking", "tool_use"] if deltas else ["tool_use"]
        assert [block["type"] for block in turn["content"]] == types
        tool_use = turn["content"][-1]
        assert (tool_use["id"], tool_use["name"], tool_use["input"]) == (
            tool_use_id,
            name,
            tool_input,
        )
        assert (turn["usage"]["input_tokens"], turn["usage"]["output_tokens"]) == (n_in, n_out)


@pytest.mark.parametrize("run", RUNS, ids=IDS)
async def test_the_replay_reproduces_the_live_log(
    run: Run, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    replay = await _replay(run, tmp_path / "rex.db", mock_anthropic)
    streams = _streams(run)
    assert len(replay.requests) == len(streams)  # one request per stored attempt, as live
    assert replay.runs_row == (*run.outcome, COST_CEILING_USD, MAX_ITERATIONS)
    assert [kind for kind, _payload in replay.payloads] == _expected_types(run)

    # Every event but the exchanges is the live log's, payload for payload.
    log = "\n".join(payload for kind, payload in replay.payloads if kind != "brain_exchange")
    assert hashlib.sha256(log.encode()).hexdigest() == run.log_sha256

    # Each exchange stores its response as served, the fixture's bytes, with its status.
    exchanges = [e for e in replay.events if isinstance(e, BrainExchangeEvent)]
    assert [(x.status, x.raw_response) for x in exchanges] == [(200, raw) for raw in streams]

    # The deltas relayed during each call join to that call's assembled thinking.
    relayed: list[str] = []
    pending: list[str] = []
    for event in replay.events:
        if isinstance(event, ThinkingDelta):
            pending.append(event.text)
        elif isinstance(event, BrainExchangeEvent):
            relayed.append("".join(pending))
            pending = []
    turns = [_assemble(raw)["content"] for raw in streams]
    assert relayed == [
        "".join(block["thinking"] for block in turn if block["type"] == "thinking")
        for turn in turns
    ]
    assert sum(isinstance(e, ThinkingDelta) for e in replay.events) == sum(run.deltas)

    usage = [e for e in replay.events if isinstance(e, UsageEvent)]
    assert [(u.model, u.input_tokens, u.output_tokens) for u in usage] == [
        (RECORDED_MODEL, n_in, n_out) for n_in, n_out in run.usage
    ]
    assert cost.fold_cost(replay.events) == pytest.approx(run.spend_usd, abs=1e-9)

    # Each sniff: one ToolCallEvent and one ToolResultEvent, paired by the provider's tool_use_id.
    sniffs = [tool_use for tool_use in run.tool_uses if tool_use[1] == "sniff"]
    calls = [e for e in replay.events if isinstance(e, ToolCallEvent)]
    results = [e for e in replay.events if isinstance(e, ToolResultEvent)]
    assert [(c.tool_use_id, c.tool, json.loads(c.raw_request)) for c in calls] == sniffs
    assert [(r.tool_use_id, json.loads(r.raw_response)) for r in results] == [
        (tool_use_id, f"posting:{tool_input['prey']}") for tool_use_id, _name, tool_input in sniffs
    ]


@pytest.mark.parametrize("run", RUNS, ids=IDS)
async def test_each_request_sends_back_the_stored_turns(
    run: Run, tmp_path: Path, mock_anthropic: MockAnthropic
) -> None:
    replay = await _replay(run, tmp_path / "rex.db", mock_anthropic)
    turns = [_assemble(raw)["content"] for raw in _streams(run)]
    bodies = [json.loads(request.content) for request in replay.requests]
    assert len(bodies) == len(turns) == 3
    for n, (request, body) in enumerate(zip(replay.requests, bodies, strict=True)):
        # The shape and the wire the daemon used: the recorded model, adaptive thinking with
        # summarized display, auto tool choice with one tool call per turn, streamed.
        assert body["model"] == RECORDED_MODEL
        assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
        assert body["max_tokens"] == 4096
        assert body["stream"] is True
        assert request.headers.get_list("authorization") == ["Bearer replay"]
        assert request.headers.get_list("anthropic-version") == ["2026-10-21"]
        # Call n sends back the n turns before it, each exactly as stored: dicts and key order.
        messages = body["messages"]
        assert [m["role"] for m in messages] == ["user", *["assistant", "user"] * n]
        for k in range(n):
            sent = messages[1 + 2 * k]["content"]
            assert sent == turns[k]
            assert [list(block) for block in sent] == [list(block) for block in turns[k]]
        if n:
            assert messages[: 2 * n - 1] == bodies[n - 1]["messages"]  # append-only


def test_the_replay_sends_the_daemon_tool_description() -> None:
    reg = ToolRegistry()
    reg.tool(sniff)
    assert brain.build_tools(reg) == brain.build_tools(stub.build_registry())
