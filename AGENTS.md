# RexHunter🦖 – Working Contract

Local-first, single-user autonomous job-hunting agent on a **'terrarium' model**: a
background daemon hunts AI-engineering roles on a schedule, fights mock-gym hazards, and
captures postings into a prey pen for human verdicts (Feast / Release / Amber). State is
**event-sourced** – a monotonic SQLite log is the single source of truth – and the
retro-game frontend is a **pure projection** of that log, streamed over SSE.

The build deliberately **hand-rolls** the agent loop, tool contract, durable-pause
machinery, and trajectory store – building those primitives *is* the learning objective.
Boring technology everywhere else. A dependency must replace >100 lines we'd otherwise
understand, or it stays out – one logged exception: the Anthropic SDK (W2.0 decision D1,
ADR header note).

**North Star:** `rexhunter-adr.md` – the full architecture. **Read it before any
architecture decision.** This file is the day-to-day contract; the ADR is the law. If they
disagree, the ADR wins and this file gets corrected in the same commit.

## The seven invariants

Violating one is an **architecture change, not a refactor** – stop, name the number, update
the ADR.

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

- **Runtime:** Python 3.15, `asyncio`.
- **Web/host:** FastAPI (lifespan-launched background daemon + SSE endpoints). Inherited
  from the prototype baseline; the ADR names it only in that baseline's build-sequence row.
- **Persistence:** SQLite single file, `aiosqlite`, WAL + `busy_timeout=5000`. Core tables
  `runs` + `trajectory_events`, dual cursor: global `id` (SSE stream) + per-run `seq` (ghost
  replay). Append-only, immutable.
- **Typing:** Pydantic v2. Events are a discriminated union serialized to JSON; the `type`
  column mirrors the discriminator.
- **Agent loop:** hand-rolled async loop; `@rex_tool` decorator derives JSON schema from the
  typed signature (one definition = schema + validator + handler). **No agent framework.**
  A retried brain call or tool waits a jittered backoff first (`loop.backoff_delay`, W3.2-B2).
- **Transport:** SSE downstream (`EventSource`, `Last-Event-ID` resume); plain HTTP POST
  upstream for verdicts (`POST /verdict`; no config endpoint is built).
- **LLM (`P5`):** provider-agnostic `brain()` – loop imports no vendor SDK. The adapter
  (`brain.py`) runs on the Anthropic SDK in raw mode (`with_raw_response` /
  `with_streaming_response`, `max_retries=0` – the loop owns retries); our SSE assembler and
  `parse_decision` still see the provider's raw bytes. The wire is pinned explicitly (W2.1b):
  `Authorization: Bearer <key>`, `anthropic-version: 2026-10-21` and an explicit `base_url`, with
  `x-api-key` and `anthropic-beta` omitted; today the pins equal the SDK 2.6.0 defaults and hold
  if a later SDK changes them. Fixtures recorded before W2.1b used `x-api-key` and `2023-06-01`;
  they hold response bytes only and replay unchanged. `brain.anthropic_client` is the one client
  factory: no redirects (the SDK default follows them cross-origin, resending the paid body on a
  307/308; httpx2 strips the Bearer key there, not an `x-api-key`), TLS verified against
  certifi – the pre-W2.0 trust store, not the system Keychain. `ANTHROPIC_CUSTOM_HEADERS` (W2.1):
  every name it lists gets the pinned value if
  it is a pinned name, in any spelling, and `omit` otherwise, so no value from that variable
  reaches the wire. The residual: a listed name the SDK sets itself (the client-level defaults
  `Accept`, `Content-Type`, `User-Agent`, `X-Stainless-Async` and the `X-Stainless-*` platform
  headers, and the per-request `X-Stainless` timeout, read-timeout and retry-count headers) loses
  its value on the wire, dropped or downgraded to the httpx2 default, so the wire differs from the
  unset case; whether the API accepts a POST without `Content-Type` is unverified. `Host`,
  `Content-Length`, `Connection` and `Accept-Encoding` (httpx2 sets them per request),
  `X-Stainless-Raw-Response` and the pinned names keep their values (a listed spelling of a
  pinned name can become the header's name; the value stays pinned). A listed name that is not
  valid UTF-8 makes every request fail before it is sent, as it did before W2.1. A client
  `copy()` or `with_options()` keeps these entries but re-reads the variable, so a name or
  spelling added to it after the factory ran is not covered on the copy and can reach the wire, a
  second spelling of a pinned name included; src/ never copies the client. Native tool calling
  (constrained decoding is not built: no request carries `strict` or `output_config`),
  `ThinkingDelta` event relay, per-event cost accounting.
  Each `ToolCallEvent` the live brain writes stores the whole assistant turn (`assistant_content`,
  W2.3), and the next call sends it back as stored, every block in order. The live model is
  `claude-sonnet-5-5` (`SMOKE_MODEL`, W2.5); a thinking-off hunt sends that model's row of
  `brain.THINKING_FLOOR` (`between_tools`; the `claude-sonnet-5` row was removed in W5), and a model with no
  row fails closed with `ValueError` before any request (W2.2). The loop acts only on a whole turn
  that stopped for its tool call (W3.1): `_drive_stream` ends a call on an SSE error frame, on a
  stream that is not whole (cut before `message_stop`, a block left open, no stop reason) and on a
  tool input that does not parse, and `parse_decision` on any `stop_reason` but `tool_use`, each
  before any dispatch. A 200 it refuses is still counted, its UsageEvent before its ErrorEvent
  (W3.2-U2a), and both spend ceilings are checked before every brain attempt (W3.2-C1). Each brain
  attempt reads only the run's `db.CONTEXT_TYPES` (`db.read_context`, W3.3-W) and, once it has sent
  a request, stores a `BrainExchangeEvent` before its other events (W3.3): the body httpx2 sent, the
  body as received (content-decoded) and the status, never published and left out of the SSE
  backfill and the projection. An attempt that ends at the `<loop>` backstop stores none (W3.3-E2).
- **Tooling:** `ruff` (lint + format), `pyright` (typecheck, strict – one type-truth in
  editor and gate), `pytest` (async tests via anyio's built-in pytest plugin – no separate
  `pytest-anyio` dep). Config and version floors in `pyproject.toml`; exact pins in `uv.lock`.

## Project structure

Emerges over the stages; target shape:

```
src/rexhunter/
├── db.py            # connect(), schema bootstrap, start_run, append_event   ← P1
├── events.py        # Pydantic discriminated-union event types               ← P2.1
├── hub.py           # in-process broadcast hub (per-viewer queues)           ← P3
├── loop.py          # hand-rolled plan→tool→observe agent loop               ← P2.2
├── tools/           # @rex_tool registry + tool handlers                     ← P2.2
├── verdicts.py      # awaiting_verdict state machine, park-and-persist       ← P4
├── brain.py         # provider-agnostic LLM socket                           ← P5
├── scheduler.py     # per-territory deadline scheduler                       ← P2.3
├── server.py        # FastAPI app: lifespan, SSE feed, POST endpoints
└── board/           # mock job board (gym) + live Greenhouse/Lever adapters
tests/               # mirrors src/, one gate per slice
```

## Commands

```bash
# Env is uv-managed (package=false app): `uv sync` creates the in-repo ./.venv from uv.lock.
uv run ruff check . && uv run ruff format .   # lint + format
uv run pyright                  # strict typecheck, whole project (reads [tool.pyright])
uv run pytest -q                # all tests   ·   uv run pytest tests/test_x.py -q  for one file
uv run uvicorn --app-dir src rexhunter.server:app --reload   # run the daemon

# verify the write-ahead log — expect: wal, then ok
sqlite3 rexhunter.db 'PRAGMA journal_mode; PRAGMA integrity_check;'
```

## Build sequence

Canonical order + gates live in **`rexhunter-adr.md` § Build sequence** (the one place).
"What's next" resolves across two files — the ADR gives the order, this section gives the
current position; this is the entry point, not a one-file answer. Slices are named by
**slice-ID** (`P<pillar>`) only — *never by a build-order number* (a bare "3" collides with
`P3`); say the slice-ID, or "the slice after `P2.2`". Don't start a slice until the prior
slice's gate (ADR Definition-of-done) is green.

**Current position (22 Nov 2026, after PR #23):** every slice in the ADR's table has landed, and
`P2.4`, the gym board, was the last (PRs #20 to #23). The daemon hunts the six gym territories, with
the scripted keyword baseline as its no-spend brain. The next slice is Darie's choice. Open: the live
Greenhouse and Lever adapters (`P5`'s live ATS unit), park-and-persist on drift (the unmet clause of
ADR DoD #4), extraction with strict mode (`P5`), and eval tiers 2 and 3.

**Build log:** `docs/build-log.md` has one entry per slice and upgrade wave, with the commits, the
forks and who chose them, the tests, the mutants and the wire baselines. Read it when a task needs
the history behind a decision. A finished slice or wave adds its entry at the end of that file and
updates the paragraph above in the same commit. This file keeps no history of its own.

`payload` is a typed event union (`events.py`); the agent loop & `@rex_tool` harness landed in
`P2.2` (`loop.py`, `tools/`). The no-spend stub brain (`stub.py`) drives by default; the live
`brain()` (`P5`, paid) runs only when `REXHUNTER_BRAIN=live` arms it (`brain.select_brain_for`).

## How to work with me

- **Test-first.** Write the stage's failing gate, watch it fail, build to green.
- **Act on obvious moves** within the current stage – write code + tests, fix bugs, follow
  established patterns. Don't ask permission for the mechanical next step.
- **Ask before:** any paid LLM call (quote rough cost first), irreversible actions, or
  architecture forks (anything hitting an ADR swap trigger – Postgres, external pub/sub,
  multi-process, a new framework). Forks are ADR edits, not code edits.
- **Push back with the invariant number** if I reach for a framework or shortcut that breaks
  a law (e.g. *"publishes before the commit – breaks invariant 1"*). Don't quietly comply.

## Solo workflow – branch + PR

Solo engineer, branch + PR: every change goes on a named branch and lands on `main` through a
pull request, merged with a merge commit (never squash or rebase), so the commit hashes this file
cites stay valid. GitHub deletes the head branch at the merge. Machine-enforced quality replaces
team review, in two layers:

- **Local pre-push gate** (`.githooks/pre-push`) – every push runs `ruff check` +
  `ruff format --check` + `pyright` + `pytest`; any red aborts the push. Arm once per clone:
  `git config core.hooksPath .githooks`. Bypassable with `--no-verify` – a convenience gate,
  not a wall.
- **Server-side protection on `main`** – CI (`.github/workflows/ci.yml`, job `quality`) runs on
  every pull request and on every push to `main`. `quality` is a required status check: a PR
  merges only with it green and with its branch up to date with `main` (strict), admins included.
  Branch protection and a repository ruleset block force-pushes to `main` and its deletion. The
  pre-push hook makes red rare; the required check keeps a red PR from merging.

Never `--no-verify` into `main`. If the gate is red, the fix is green code, not a bypass.

## Accepted limits – do not pre-solve

Deliberate, documented in the ADR: single-process in-process hub; SQLite single-writer;
lossy broadcast hub (never revisit – log is truth); no auth on localhost; live adapters hit
**public ATS APIs only (Greenhouse / Lever)** – never scrape ToS-prohibited boards.
