# RexHunter 🦖

**A tiny-armed T-Rex that hunts AI-engineering jobs while you sleep – and brings them back for you to judge.**

RexHunter is a local-first, single-user autonomous job-hunting agent built on a "terrarium"
model: a background daemon that hunts roles on a schedule, survives the hazards of a mock
job-board gym, and drops promising postings into a pen for a human verdict. State is
event-sourced – a monotonic SQLite log is the single source of truth – and the front end is a
retro-game projection of that log: you watch the agent's actual trajectory replay as a creature
moving through its world, not a dashboard of metrics. The arms are tiny on purpose (see below).

> **Status: build-in-public, very early.** Today the daemon is a ~50-line Hello World – a
> background loop streaming text over SSE. The architecture below is the design it's being built
> toward, stage by stage. See [Build status](#build-status) for what's real vs planned.

## Why it's built this way

The interesting part isn't the job-hunting – it's the constraints the system holds itself to.
Four of the load-bearing ones:

- **Write-ahead log.** Every event is committed to SQLite *before* anyone sees it. The log is
  the truth; the live stream is only a notification – miss a message, replay it from the log.
  (This is why durability lands before the LLM: a crash at step 9 of a 10-step hunt must never
  re-pay for steps 1-8.)
- **Projection, never authority.** Nothing the UI shows is real state – it's all derived from
  the log. The live view and the "ghost replay" of a finished hunt are the *same* renderer
  reading the *same* events at different cursors.
- **Validate at the boundary.** Raw bytes from the internet – scraped HTML, HTTP responses, and
  LLM output – cross exactly one Pydantic validation line on the way in. Inside it, everything
  is typed. LLM output is treated as untrusted input, same as a scraped web page.
- **Tiny Arms.** Rex physically cannot apply, send, or submit anything – not by policy, but
  because no such tool is registered. A human "Feast" verdict is a database state transition,
  not an "are you sure?" dialog. The agent hunts; the human decides.

These are four of seven invariants. The rest live in the [ADR](rexhunter-adr.md).

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

Downstream is a firehose (SSE); upstream is the occasional verdict (plain HTTP POST). The hub
is allowed to drop messages – correctness lives in the log, not the stream.

*Today only the skeleton exists: a loop, an in-memory list standing in for the log, and an SSE
feed to a `<pre>` tag. Stage 1 replaces the list with the real SQLite WAL log.*

## Tech stack

- **Python 3.14+ / asyncio** – one process, one event loop, no external services.
- **FastAPI + uvicorn** – lifespan launches the background daemon; SSE + POST endpoints.
- **SQLite + aiosqlite, WAL mode** – the trajectory store: one file, `cp` is your backup, one
  writer and many readers. (Lands in Stage 1.)
- **Pydantic v2** – events are a discriminated union, validated at the network edge.
- **Server-Sent Events** – downstream streaming with a protocol-level resume cursor
  (`Last-Event-ID`); plain POST upstream for the rare commands.
- **A hand-rolled agent loop – no LangChain, no LangGraph.** Deliberate: the ~150-line loop, the
  tool contract, and the trajectory store *are* the point of this project. A framework would
  hide exactly the control flow and durable-state machinery it exists to build. (LangGraph is
  genuinely good – it's rejected because adopting it would delete the learning objective and
  add a second source of truth competing with the log.)

Tooling: `ruff`, `pyright` (strict), `pytest`, pinned in `pyproject.toml`.

## Build status

Build-in-public. Honest state, not aspiration:

| Stage | What | Status |
|---|---|---|
| 0 | Hello World daemon – FastAPI lifespan + background loop + SSE feed + in-memory list | **Done** |
| 1 | **Persistence** – in-memory list to SQLite WAL event log (`runs`, `trajectory_events`) | **In progress** |
| 2 | Loop & tool harness – `@rex_tool` registry, validate then execute then append | Planned |
| 3 | Durable pause & human-in-the-loop – `awaiting_verdict` rows, Feast / Release / Amber | Planned |
| 4 | Brain socket – provider-agnostic LLM, native tool calling, thinking-token relay | Planned |

Stage gates are test-first – Stage 1's is: `kill -9` mid-hunt, restart, every pre-kill event
still queryable. Already wired around the daemon: strict tooling, a green-gated pre-push hook,
and CI. The SQLite log itself is the current work and not yet landed.

## Getting started

Requires [uv](https://docs.astral.sh/uv/) – it manages the Python 3.14 toolchain and the virtualenv for you.

```bash
git clone https://github.com/dariero/RexHunter.git
cd RexHunter
uv sync                        # creates .venv, installs runtime + dev tools from uv.lock

# run the daemon, then open http://127.0.0.1:8000
uv run uvicorn main:app --reload

# the real quality gates
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest        # no tests yet – Stage 1 writes the first
```

You'll see Rex sniff the air for fresh prey every five seconds. That's the whole of Stage 0 – a
heartbeat with a story. The interesting code is still ahead.

## Full architecture

[`rexhunter-adr.md`](rexhunter-adr.md) is the complete design record – five pillars, seven
invariants, and an honest rejection of every alternative considered (Postgres, Kafka, LangGraph,
Temporal, and more). Start there for the why behind every decision above.
