"""P2.3 gate — invariant 7 ("single writer per run") under real concurrency.

Until P2.3, one hunt = one writer, so invariant 7 held by accident. These tests run genuine
concurrent writers (one aiosqlite connection per run — the model the ADR's "SQLite write-lock
serialises writers" presupposes) and prove the per-run `seq` cursor stays collision-free,
gapless, and uncontaminated across runs.

The headline proof is NOT atomicity. Under per-hunt connections + invariant 7, appends within a
run are strictly sequential, so `seq` is gapless by *sequentiality*. The property production
actually leans on is the per-run SCOPE `WHERE run_id = ?` — so the teeth-proof is a
ONE-VARIABLE mutation: the same N-run harness, the only difference being that clause (derived
by removing it from `db.APPEND_EVENT_SQL` itself, so scope is provably the sole variable).

The same-run two-writer tests are explicit BELT-AND-SUSPENDERS: invariant 7 forbids two writers
per run, so they probe robustness *beyond* the guarantee, not the production property.
"""

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

import aiosqlite
import pytest

from rexhunter import db
from rexhunter.events import SniffEvent, TrajectoryEvent
from rexhunter.loop import Brain, Decision, HuntComplete, ToolCallDecision
from rexhunter.scheduler import run_hunts
from rexhunter.tools import ToolRegistry

pytestmark = pytest.mark.anyio

BrainFactory = Callable[[Sequence[Decision]], Brain]
Append = Callable[[aiosqlite.Connection, str, TrajectoryEvent], Awaitable[None]]
_FIXED_TS = "2026-06-23T00:00:00+00:00"  # created_at is irrelevant to seq; pin it in mutants


async def seqs_of(conn: aiosqlite.Connection, run_id: str) -> list[int]:
    async with conn.execute(
        "SELECT seq FROM trajectory_events WHERE run_id = ? ORDER BY seq", (run_id,)
    ) as cur:
        return [int(row[0]) for row in await cur.fetchall()]


# ── Unit 0 · connection-model spike (de-risk before anything stacks on it) ────


async def test_spike_per_hunt_connections_write_concurrently_without_lock(tmp_path: Path) -> None:
    # N runs, each its own connection, all appending K events at once. Modest N*K and no shared
    # connection: writes genuinely contend for SQLite's write lock, resolved by busy_timeout.
    db_path = tmp_path / "rex.db"
    boot = await db.connect(db_path)  # bootstrap schema + WAL once, before the write storm
    await boot.close()

    n, k = 6, 12

    async def writer(i: int) -> str:
        conn = await db.connect(db_path)
        try:
            run_id = await db.start_run(conn, territory=f"t{i}")
            for j in range(k):
                await db.append_event(conn, run_id, SniffEvent(prey=f"r{i}-e{j}"))
            return run_id
        finally:
            await conn.close()

    run_ids = await asyncio.gather(*(writer(i) for i in range(n)))  # no "database is locked"

    reader = await db.connect(db_path)
    try:
        for run_id in run_ids:
            assert await seqs_of(reader, run_id) == list(range(k))  # gapless, no collision
    finally:
        await reader.close()


# ── Unit 2 · the gate — N concurrent REAL hunts ──────────────────────────────


async def test_concurrent_hunts_seq_gapless_and_uncontaminated(
    tmp_path: Path, scripted_brain: BrainFactory
) -> None:
    n, m = 8, 4  # 8 hunts, all concurrent; m tool calls -> 2m events/run (call + result)
    db_path = tmp_path / "rex.db"
    reg = ToolRegistry()

    @reg.tool
    async def noop() -> str:
        return "ok"

    def brain_for(_territory: str) -> Brain:
        calls: list[Decision] = [ToolCallDecision(tool=noop.__name__, args={}) for _ in range(m)]
        return scripted_brain([*calls, HuntComplete()])

    run_ids = await run_hunts(
        db_path,
        [f"t{i}" for i in range(n)],
        brain_for=brain_for,
        registry=reg,
        max_concurrent=n,  # all at once — maximal write-lock contention
    )
    assert all(rid is not None for rid in run_ids)

    reader = await db.connect(db_path)
    try:
        # per run: gapless 0..2m-1 (no collision, no gap, no contamination from siblings)
        for rid in run_ids:
            assert rid is not None
            assert await seqs_of(reader, rid) == list(range(2 * m))

        # cross-run: every run has exactly its own 2m events, bounded 0..2m-1
        async with reader.execute(
            "SELECT run_id, COUNT(*), MIN(seq), MAX(seq) FROM trajectory_events GROUP BY run_id"
        ) as cur:
            groups = list(await cur.fetchall())
        assert len(groups) == n
        assert all(
            int(c) == 2 * m and int(lo) == 0 and int(hi) == 2 * m - 1 for _, c, lo, hi in groups
        )

        # global id cursor survives concurrency: distinct + strictly increasing
        async with reader.execute("SELECT id FROM trajectory_events ORDER BY id") as cur:
            ids = [int(r[0]) for r in await cur.fetchall()]
        assert len(ids) == n * 2 * m
        assert ids == sorted(set(ids))  # no duplicate global ids, monotonic

        # dual-cursor agreement: within a run, global id order == per-run seq order
        for rid in run_ids:
            assert rid is not None
            async with reader.execute(
                "SELECT id FROM trajectory_events WHERE run_id = ? ORDER BY seq", (rid,)
            ) as cur:
                ids_by_seq = [int(r[0]) for r in await cur.fetchall()]
            assert ids_by_seq == sorted(ids_by_seq)
    finally:
        await reader.close()


# ── Unit 2 · the teeth — one-variable scope mutation ─────────────────────────


async def _barrier_harness(append: Append, db_path: Path, n: int, k: int) -> list[list[int]]:
    # Synchronised rounds: an asyncio.Barrier makes all N writers append their j-th event in
    # lockstep, so global write order is round-robin. Under an UNSCOPED MAX that forces each
    # run's seqs into disjoint windows ([rN, (r+1)N)) — a deterministic failure, not luck.
    boot = await db.connect(db_path)
    await boot.close()
    barrier = asyncio.Barrier(n)

    async def writer(i: int) -> str:
        conn = await db.connect(db_path)
        try:
            run_id = await db.start_run(conn, territory=f"t{i}")
            for j in range(k):
                await barrier.wait()  # round rendezvous: all N writers append together
                await append(conn, run_id, SniffEvent(prey=f"r{i}-e{j}"))
            return run_id
        finally:
            await conn.close()

    run_ids = await asyncio.gather(*(writer(i) for i in range(n)))
    reader = await db.connect(db_path)
    try:
        return [await seqs_of(reader, rid) for rid in run_ids]
    finally:
        await reader.close()


@pytest.mark.parametrize("scoped", [True, False], ids=["real_scoped", "mutant_unscoped"])
async def test_per_run_scope_is_load_bearing(tmp_path: Path, scoped: bool) -> None:
    n, k = 8, 10
    mutant_sql = db.APPEND_EVENT_SQL.replace(" WHERE run_id = ?", "")
    assert mutant_sql != db.APPEND_EVENT_SQL  # the one-clause mutation actually fired

    if scoped:

        async def append(conn: aiosqlite.Connection, run_id: str, event: TrajectoryEvent) -> None:
            await db.append_event(conn, run_id, event)  # the production path, scoped
    else:

        async def append(conn: aiosqlite.Connection, run_id: str, event: TrajectoryEvent) -> None:
            # IDENTICAL to db.append_event except the unscoped SQL and its dropped run_id param.
            await conn.execute(mutant_sql, (run_id, event.type, event.model_dump_json(), _FIXED_TS))
            await conn.commit()

    per_run = await _barrier_harness(append, tmp_path / "rex.db", n, k)

    if scoped:
        assert all(s == list(range(k)) for s in per_run)  # scope present -> isolated, gapless
    else:
        # scope dropped -> each run's seqs are globally-interleaved windows, never range(k)
        assert all(s != list(range(k)) for s in per_run)


# ── Unit 2 · belt-and-suspenders — robust even if invariant 7 were breached ──


async def _start_one_run(db_path: Path) -> str:
    conn = await db.connect(db_path)
    try:
        return await db.start_run(conn, territory="shared")
    finally:
        await conn.close()


async def test_same_run_naive_read_then_write_collides(tmp_path: Path) -> None:
    # BELT-AND-SUSPENDERS (not the headline — invariant 7 forbids this): two writers on ONE run.
    # This proves the harness has teeth — a naive read-then-write loses the race deterministically.
    db_path = tmp_path / "rex.db"
    run_id = await _start_one_run(db_path)
    barrier = asyncio.Barrier(2)

    async def naive_writer() -> None:
        conn = await db.connect(db_path)
        try:
            async with conn.execute(
                "SELECT COALESCE(MAX(seq) + 1, 0) FROM trajectory_events WHERE run_id = ?",
                (run_id,),
            ) as cur:
                row = await cur.fetchone()
            seq = int(row[0]) if row is not None else 0
            await barrier.wait()  # both read the SAME max before either inserts
            await conn.execute(
                "INSERT INTO trajectory_events (run_id, seq, type, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (run_id, seq, "sniff", "{}", _FIXED_TS),
            )
            await conn.commit()
        finally:
            await conn.close()

    with pytest.raises(sqlite3.IntegrityError):  # UNIQUE(run_id, seq) catches the duplicate
        await asyncio.gather(naive_writer(), naive_writer())


async def test_same_run_atomic_insert_serialises_concurrent_writers(tmp_path: Path) -> None:
    # The real atomic INSERT...SELECT survives the SAME same-run race the naive version loses:
    # the WAL write lock serialises the two statements -> seq 0 then 1, no collision.
    db_path = tmp_path / "rex.db"
    run_id = await _start_one_run(db_path)
    barrier = asyncio.Barrier(2)

    async def atomic_writer(tag: str) -> None:
        conn = await db.connect(db_path)
        try:
            await barrier.wait()  # rendezvous, then both call the real atomic append at once
            await db.append_event(conn, run_id, SniffEvent(prey=tag))
        finally:
            await conn.close()

    await asyncio.gather(atomic_writer("a"), atomic_writer("b"))  # no IntegrityError

    reader = await db.connect(db_path)
    try:
        assert await seqs_of(reader, run_id) == [0, 1]  # gapless, no collision
    finally:
        await reader.close()
