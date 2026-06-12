# RexHunter🦖 – Working Contract

Local-first, single-user autonomous job-hunting agent on a **'terrarium' model**: a
background daemon hunts AI-engineering roles on a schedule, fights mock-gym hazards, and
captures postings into a prey pen for human verdicts (Feast / Release / Amber). State is
**event-sourced** – a monotonic SQLite log is the single source of truth – and the
retro-game frontend is a **pure projection** of that log, streamed over SSE.

The build deliberately **hand-rolls** the agent loop, tool contract, durable-pause
machinery, and trajectory store, because building those primitives *is* the learning
objective. Boring technology everywhere else. A dependency must replace >100 lines we'd
otherwise understand, or it stays out.

**North Star:** `rexhunter-adr.md` – the full architecture. **Read it before any
architecture decision.** This file is the day-to-day contract; the ADR is the law. If they
disagree, the ADR wins and this file gets corrected in the same commit.

## The seven invariants

Violating one is an **architecture change, not a refactor** – stop, name the number, update
the ADR. These are the laws; everything else is detail.

1. **Write-ahead** – commit the event to SQLite *before* publishing to the broadcast hub.
  Log is truth; stream is notification.
2. **Projection** – all displayed state derives from the log (or tables maintained
  transactionally with it). Frontend is never authoritative. Live renderer and ghost
   replayer are one renderer, two cursors.
3. **Validate at the boundary** – raw bytes (HTTP, scraped HTML, LLM output) cross one
  Pydantic boundary on entry; typed objects only inside. **LLM output is untrusted input.**
4. **Tiny Arms** – Rex cannot submit/send/apply, *structurally*. No such tool is registered.
  A human verdict is a DB state transition, not a confirmation dialog.
5. **Derive, don't store** – HP, run stats, day/night phase are folded from events or the
  clock, never a second mutable copy.
6. **Raw I/O snapshots** – every tool event carries the raw request + response bytes. Dead
  runs become pytest fixtures for free.
7. **Single writer per run** – only the owning hunt task appends to a run. Readers unlimited
  (WAL makes this safe).

## Stack

- **Runtime:** Python 3.14+, `asyncio`.
- **Web/host:** FastAPI (lifespan-launched background daemon + SSE endpoints). *Inherited
from the Hello World; not named in the ADR.*
- **Persistence:** SQLite single file, `aiosqlite`, WAL + `busy_timeout=5000`. Core tables
`runs` + `trajectory_events`, dual cursor: global `id` (SSE stream) + per-run `seq` (ghost
replay). Append-only, immutable.
- **Typing:** Pydantic v2. Events are a discriminated union serialized to JSON; the `type`
column mirrors the discriminator.
- **Agent loop:** hand-rolled async loop; `@rex_tool` decorator derives JSON schema from the
typed signature (one definition = schema + validator + handler). **No agent framework.**
- **Transport:** SSE downstream (`EventSource`, `Last-Event-ID` resume); plain HTTP POST
upstream for verdicts/config.
- **LLM (Stage 4):** provider-agnostic `brain()` – loop imports no vendor SDK. Native tool
calling, constrained decoding where available, `ThinkingDelta` event relay, per-event
cost accounting.
- **Tooling:** `ruff` (lint + format), `pyright` (typecheck, strict), `pytest` (async tests via
anyio's built-in pytest plugin – no separate `pytest-anyio` dep). Pins + config in `pyproject.toml`.
Pyright over mypy: a tie on correctness for our Pydantic v2 + aiosqlite stack (both clean, no
plugin), but Pyright is the engine Cursor already runs – one type-truth in editor and gate.

## Project structure

Emerges over the stages; target shape:

```
src/rexhunter/
├── db.py            # connect(), schema bootstrap, start_run, append_event   ← Stage 1
├── events.py        # Pydantic discriminated-union event types               ← Stage 2
├── hub.py           # in-process broadcast hub (per-viewer queues)
├── loop.py          # hand-rolled plan→tool→observe agent loop               ← Stage 2
├── tools/           # @rex_tool registry + tool handlers                     ← Stage 2
├── verdicts.py      # awaiting_verdict state machine, park-and-persist       ← Stage 3
├── brain.py         # provider-agnostic LLM socket                           ← Stage 4
├── scheduler.py     # per-territory deadline scheduler                       ← polish
├── server.py        # FastAPI app: lifespan, SSE feed, POST endpoints
└── board/           # mock job board (gym) + live Greenhouse/Lever adapters
tests/               # mirrors src/, one failing test gate per stage
```

## Commands

```bash
# Commands assume the project venv (./.venv) – activate it, or prefix each with .venv/bin/.
ruff check . && ruff format .   # lint + format
pyright                         # strict typecheck, whole project (reads [tool.pyright])
pytest -q                       # all tests   ·   pytest tests/test_x.py -q   for one file
uvicorn main:app --reload       # run the daemon (→ rexhunter.server:app after the src/ move)

# verify the write-ahead log once Stage 1 creates it — expect: wal, then ok
sqlite3 rexhunter.db 'PRAGMA journal_mode; PRAGMA integrity_check;'
```

## Build sequence – one stage at a time

Dependency-ordered from the Hello World (FastAPI + loop + SSE + in-memory list). **Do not
start Stage N+1 until Stage N's gate is green.** Gates are in the ADR's Definition-of-done.

- **▶ Stage 1 – Persistence (CURRENT).** In-memory list → SQLite WAL log.
*Gate:* `kill -9` mid-hunt → restart → all pre-kill events queryable; dangling
`outcome IS NULL` runs marked `'crashed'` at boot.
- **Stage 2 – Loop & tool harness.** `@rex_tool` registry, validate→execute→append,
retryable-vs-fatal taxonomy.
- **Stage 3 – Durable pause & HITL.** `awaiting_verdict` rows, Feast/Release/Amber machine.
- **Stage 4 – Brain socket.** `brain()`, native tool calling, thinking-delta relay.

`payload` stays a raw string until Stage 2. Don't introduce event types or the LLM loop
early.

## How to work with me

- **Test-first.** Write the stage's failing gate, watch it fail, build to green.
- **Act on obvious moves** within the current stage – write code + tests, fix bugs, follow
established patterns. Don't ask permission for the mechanical next step.
- **Ask before:** any paid LLM call (quote rough cost first), irreversible actions, or
architecture forks (anything hitting an ADR swap trigger – Postgres, external pub/sub,
multi-process, a new framework). Forks are ADR edits, not code edits.
- **Push back with the invariant number** if I reach for a framework or shortcut that breaks
a law (e.g. *"publishes before the commit – breaks invariant 1"*). Don't quietly comply.

## Solo workflow – direct-to-main

Solo engineer, no feature-branch / PR overhead – commit and push straight to `main`.
Machine-enforced quality replaces team review, in two honest layers:

- **Local pre-push gate** (`.githooks/pre-push`) – the fast inner loop. Every push runs
`ruff check` + `ruff format --check` + `pyright` + `pytest`; any red aborts the push. Arm it
once per clone: `git config core.hooksPath .githooks`. *It is bypassable with `--no-verify` –
a convenience gate, not a wall.*
- **Server-side protection** – GitHub branch protection on `main`: blocks force-pushes and
deletion (admins included); CI (`.github/workflows/ci.yml`) runs on every push. Honest limit:
required status checks gate PR *merges*, not direct pushes — a direct `git push` lands first and
CI reports red *after*. Direct-to-main trades the hard pre-merge gate for speed: the pre-push hook
makes red rare, CI makes it loud and fast; only a branch → required-check → merge flow makes red
*structurally unable* to reach `main`.

Never `--no-verify` into `main`. If the gate is red, the fix is green code, not a bypass.

## Accepted limits – do not pre-solve

Deliberate, documented in the ADR: single-process in-process hub; SQLite single-writer;
lossy broadcast hub (never revisit – log is truth); no auth on localhost; live adapters hit
**public ATS APIs only (Greenhouse / Lever)** – never scrape ToS-prohibited boards.