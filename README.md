# RexHunter 🦖

**A tiny-armed T-Rex that hunts AI-engineering jobs while you sleep – and brings them back for you to judge.**

Local-first, single-user autonomous job-hunting agent on a "terrarium" model: a background
daemon hunts roles on a schedule, survives the hazards of a mock job-board gym, and drops
promising postings into a pen for a human verdict. State is event-sourced – a monotonic SQLite
log is the single source of truth – and the frontend is a retro-game projection of that log:
the agent's actual trajectory replayed as a creature moving through its world.

> **Status: build-in-public, early.** The daemon persists every event to the SQLite WAL log
> (Stage 1 done, gate-tested). Stage 2 – the agent loop & tool harness – is next.

## Core constraints

Four of seven invariants; the rest are in the [ADR](rexhunter-adr.md).

- **Write-ahead log.** Every event commits to SQLite *before* anyone sees it. The log is truth;
  the live stream is only a notification – a missed message is replayed from the log. Durability
  lands before the LLM: a crash at step 9 of a 10-step hunt must never re-pay for steps 1–8.
- **Projection, never authority.** Everything the UI shows is derived from the log. The live
  view and the ghost replay of a finished hunt are the same renderer reading the same events at
  different cursors.
- **Validate at the boundary.** Raw bytes (scraped HTML, HTTP responses, LLM output) cross
  exactly one Pydantic validation line on entry; only typed objects exist inside. LLM output is
  untrusted input.
- **Tiny Arms.** Rex structurally cannot apply, send, or submit – no such tool is registered.
  A human "Feast" verdict is a database state transition, not a confirmation dialog.

## Architecture at a glance

One data path, end to end. A hunt writes events; everything else is a reader.

```
   hunt task                                         browser
  (agent loop)                                    (retro-game UI)
      |                                                 ^
      | 1. append event                                 | 4. SSE stream
      v                                                 |    (Last-Event-ID resume)
  +-----------+   2. after commit   +---------------+   |
  | SQLite    | ------------------> | broadcast hub | --+
  | WAL log   |   3. notify         | (per-viewer   |
  | (truth)   |                     |  async queues)|
  +-----------+                     +---------------+
      ^
      | many readers, one writer (WAL); a ghost replay
      | reads the same log at a different cursor
```

Downstream is a firehose (SSE, `Last-Event-ID` resume); upstream is the occasional verdict
(plain HTTP POST). The hub may drop messages – correctness lives in the log, not the stream.

## Tech stack

- **Python 3.14+ / asyncio** – one process, one event loop, no external services.
- **FastAPI + uvicorn** – lifespan launches the background daemon; SSE + POST endpoints.
- **SQLite + aiosqlite, WAL mode** – the trajectory store: one file, one writer, many readers.
- **Pydantic v2** – events are a discriminated union, validated at the network edge.
- **Hand-rolled agent loop – no LangChain, no LangGraph.** The ~150-line loop, the tool
  contract, and the trajectory store *are* the point of this project; a framework would hide
  exactly the control flow and durable-state machinery it exists to build.
- **Tooling:** `ruff`, `pyright` (strict), `pytest` – pinned in `pyproject.toml`.

## Build status

| Stage | What | Status |
|---|---|---|
| 0 | Prototype daemon – FastAPI lifespan + background loop + SSE feed + in-memory list | **Done** |
| 1 | **Persistence** – in-memory list to SQLite WAL event log (`runs`, `trajectory_events`) | **Done** |
| 2 | Loop & tool harness – `@rex_tool` registry, validate then execute then append | Planned |
| 3 | Durable pause & human-in-the-loop – `awaiting_verdict` rows, Feast / Release / Amber | Planned |
| 4 | Brain socket – provider-agnostic LLM, native tool calling, thinking-token relay | Planned |

Stage gates are test-first. Stage 1's gate (`tests/test_stage1_gate.py`): a real hunt
subprocess is `kill -9`ed mid-append; every confirmed event must survive the restart and the
dangling run must be marked `'crashed'` at boot.

## Getting started

Requires [uv](https://docs.astral.sh/uv/) – it manages the Python 3.14 toolchain and the virtualenv.

```bash
git clone https://github.com/dariero/RexHunter.git
cd RexHunter
uv sync                            # creates .venv, installs runtime + dev tools from uv.lock

uv run uvicorn --app-dir src rexhunter.server:app --reload   # daemon → http://127.0.0.1:8000

# quality gates
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest                      # incl. the Stage 1 kill -9 crash-durability gate
```

## Full architecture

[`rexhunter-adr.md`](rexhunter-adr.md) – the complete design record: five pillars, seven
invariants, and the rejection record for every alternative considered (Postgres, Kafka,
LangGraph, Temporal, and more).
