# RexHunter🦖 – Working Contract

Local-first, single-user autonomous job-hunting agent on a **'terrarium' model**: a
background daemon hunts AI-engineering roles on a schedule, fights mock-gym hazards, and
captures postings into a prey pen for human verdicts (Feast / Release / Amber). State is
**event-sourced** – a monotonic SQLite log is the single source of truth – and the
retro-game frontend is a **pure projection** of that log, streamed over SSE.

The build deliberately **hand-rolls** the agent loop, tool contract, durable-pause
machinery, and trajectory store – building those primitives *is* the learning objective.
Boring technology everywhere else. A dependency must replace >100 lines we'd otherwise
understand, or it stays out.

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

- **Runtime:** Python 3.14+, `asyncio`.
- **Web/host:** FastAPI (lifespan-launched background daemon + SSE endpoints). Inherited
  from the prototype baseline; not named in the ADR.
- **Persistence:** SQLite single file, `aiosqlite`, WAL + `busy_timeout=5000`. Core tables
  `runs` + `trajectory_events`, dual cursor: global `id` (SSE stream) + per-run `seq` (ghost
  replay). Append-only, immutable.
- **Typing:** Pydantic v2. Events are a discriminated union serialized to JSON; the `type`
  column mirrors the discriminator.
- **Agent loop:** hand-rolled async loop; `@rex_tool` decorator derives JSON schema from the
  typed signature (one definition = schema + validator + handler). **No agent framework.**
- **Transport:** SSE downstream (`EventSource`, `Last-Event-ID` resume); plain HTTP POST
  upstream for verdicts/config.
- **LLM (`P5`):** provider-agnostic `brain()` – loop imports no vendor SDK. Native tool
  calling, constrained decoding where available, `ThinkingDelta` event relay, per-event
  cost accounting.
- **Tooling:** `ruff` (lint + format), `pyright` (typecheck, strict – one type-truth in
  editor and gate), `pytest` (async tests via anyio's built-in pytest plugin – no separate
  `pytest-anyio` dep). Pins + config in `pyproject.toml`.

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
# Env is uv-managed (package=false app). The real venv lives at ~/.venvs/rexhunter —
# OUTSIDE iCloud, which silently corrupts an in-repo .venv; ./.venv is a symlink to it.
# First-time setup, or to heal a lost symlink:  bash scripts/setup-venv.sh   (then `uv run`).
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

**Current position:** `P4` ✅ **done** · `P5` Unit 1 (edge boundary) ✅ **done** (`020283e`) ·
Unit 2a (provider adapter, offline) ✅ **done** (`edd4ec8`) · Unit 2b (the first paid call) ✅
**done** — one real Sonnet 5 call landed and parsed (`tool=sniff`), captured as a golden fixture
(`tests/fixtures/smoke_sonnet5.json`, invariant 6; it carries a live adaptive-thinking sibling
block, so it validates the 2a lenient scan on genuine output). `SMOKE_MODEL = "claude-sonnet-5"`;
`build_tools` strips `title` from the derived schemas (Anthropic accepted them — confirmed by the
free `count_tokens` pre-flight); a shared `build_request_body` holds the Sonnet 5 guards (no
`temperature`/`top_p`/`top_k`, no `thinking`, no prefill); `smoke.py` is the paid entrypoint
(`--count-only` free pre-flight → one `messages` call → persist raw bytes → `parse_decision`),
hard one-call guard. **`P5` Unit 2c — brain-in-loop — ✅ done. ADR upgrade-map stage 3 CLOSED:** the
LLM drives a full hunt end to end (plan → tool runs → observation threads back into `messages` → next
iteration → typed outcome), replay-gated. *2c.1 (offline, no-spend) — ✅ done (`71ab898`):* the loop
projects the trajectory log into an Anthropic `messages` array (inv 2, `project_messages`), threads a
`HUNT_SYSTEM_PROMPT`, folds per-call usage into a running cost from `UsageEvent`s (inv 5,
`cost.fold_cost`) with a `COST_CEILING_USD` breaker, `classify` learns httpx transport/status failures
(429/5xx/timeouts → retryable), and `select_brain_for`/`REXHUNTER_BRAIN` gate the spender surface
(stub default, opt-in `live`). The live loop path sends `thinking: {type: "disabled"}` +
`tool_choice: {disable_parallel_tool_use: true}` — bare one-`tool_use` replay turns until thinking
capture lands in Unit 3. *2c.2 (gated live) — ✅ done:* one budget-capped hunt (`rexhunter.hunt_smoke`,
`COST_CEILING_USD=0.05`, `MAX_ITERATIONS=3`) on `SMOKE_MODEL` against the mock-gym board — 2 brain
calls, `needs_help`, $0.0149. **Call two fired: the reconstructed assistant `tool_use` + user
`tool_result` (paired by `tool_use_id`) was accepted by the real API** — the never-run path is proven.
Each call's raw bytes captured as golden fixtures (`tests/fixtures/hunt_smoke_call_0{1,2}.json`, inv 6);
`tests/test_hunt_replay.py` re-drives them offline to identical events (DoD #5). The daemon still drives
the **no-spend stub brain** (`stub.py`); daemon-lifespan live wiring is a later unit.
**▶ Next: `P5` Unit 3** (streaming `ThinkingDelta` relay) — where the vendor-SDK frugality call and the
deferred P3 broadcast hub re-open.

- **`P1` · Persistence — ✅ done.** In-memory list → SQLite WAL log.
  *Gate (green, `tests/test_stage1_gate.py`):* `kill -9` mid-hunt → restart → all pre-kill
  events queryable; dangling `outcome IS NULL` runs marked `'crashed'` at boot.
- **`P2.1` · Typed events — ✅ done.** `events.py` (`SniffEvent`, the `TrajectoryEvent`
  union, `decode_event`) + the read/write validation boundary (`db.py` typed
  `append_event` / `read_events`). *Gate (green):* `tests/test_events.py`.
- **`P2.2` · Loop & tool harness — ✅ done.** Hand-rolled `run_hunt` loop, `@rex_tool`
  registry (one signature → schema + validator + handler), retryable-vs-fatal taxonomy with
  per-tool timeout + retry budget; stub brain (no LLM, no spend). *Gate (green,
  `tests/test_stage2_gate.py`):* a tool that raises / hangs past timeout / an unknown tool
  name → typed events + clean outcome, never an unhandled escape (ADR **DoD #2**).
- **`P2.3` · Hunt scheduler — ✅ done (wired).** `scheduler.py`: `run_hunts` (bounded
  `TaskGroup` + `Semaphore`, one connection per run) and `run_scheduler` (per-territory
  monotonic-clock deadlines, no stored schedule). Each hunt is a single writer per run
  (invariant 7), failures are isolated (one hunt's abort never cancels or corrupts a sibling),
  and graceful shutdown closes a cancelled run via `run_hunt`'s shielded cancel path. *Gate
  (green, `tests/test_invariant7.py` + `tests/test_scheduler.py`):* N concurrent hunts → seq
  collision-free + gapless per run (proven by a one-clause unscoped-`MAX` mutant), each
  raising/hanging/unknown-tool failure isolated under the cap, no escape from the group (ADR
  **DoD #2 under concurrency**). *Wired (`P2.3-wiring`):* `run_scheduler` runs in `server.py`'s
  lifespan driving a no-spend stub brain (`stub.py`); `rex_loop` deleted. *Lifespan gate
  (green, `tests/test_server_lifespan.py`):* booted through the real `lifespan_context`, ≥2
  concurrent hunts spawn on startup and shutdown leaves no run `outcome IS NULL`, no orphaned
  task, no leaked connection.
- **`P4` · Durable pause & HITL — ✅ done.** `verdicts.py`: the prey pen + Feast/Release/Amber
  machine. Capture rides `HuntComplete` (run-scoped `PreyCapturedEvent` + prey row, one txn);
  each verdict is a status-guarded, idempotent transition appending a `VerdictEvent` to the
  non-run-scoped `pen_events` log, with `prey.status` its projection (`fold` proves no drift); a
  FEAST atomically enqueues a `draft_pitch` job drained by `run_job_worker` (stub drafter, no
  LLM); `POST /verdict` is the only path a captured row can move (invariant 4). Park-and-persist
  deferred. *Gate (green, `tests/test_stage4_gate.py` + `tests/test_verdicts.py`):* restart
  through the real `lifespan_context` → prey survives, verdicts still apply, double-FEAST and
  replay-across-restart are no-ops (one event, one job ever), and the boot crash-sweep marks a
  dangling run `'crashed'` while leaving the committed prey untouched (ADR **DoD #4**, prey-pen
  clause).
- **▶ `P5` · Brain socket — in progress.** *Unit 1 (the edge boundary) — ✅ done (`020283e`):*
  `brain.py` `parse_decision` (raw provider bytes → typed `Decision`, invariant 3), the
  malformed→`ErrorEvent` path (invariant 6), `tool_use_id` correlation on
  `ToolCallEvent`/`ToolResultEvent`; offline, no-spend — the whole of DoD #5, gate
  `tests/test_stage5_gate.py`. *Unit 2a (provider adapter, offline half) — active:*
  `adapter_brain_for` over **httpx** wraps `parse_decision` around the Anthropic Messages API,
  presents `sniff`/`hunt_complete`/`needs_help` as derived tool schemas (D1), rejects multiple
  `tool_use` blocks; injected transport, **no live call** (contract gate
  `tests/test_brain_adapter.py`). *Unit 2b (the first paid call) — ✅ done:* one live smoke on
  `claude-sonnet-5` through the 2a adapter (free `count_tokens` pre-flight → one `messages` call →
  persist raw bytes → `parse_decision`) landed and parsed to `ToolCallDecision(tool="sniff")`,
  captured as `tests/fixtures/smoke_sonnet5.json` (invariant 6, a real adaptive-thinking sibling
  block); Sonnet 5 guards (no sampling params / `thinking` / prefill) + `title`-stripped schemas;
  hard one-call guard; `smoke.py` entrypoint. Offline gates `tests/test_smoke_offline.py` +
  `tests/test_smoke_replay.py`. *Later units:* streaming thinking-delta relay (Unit 3),
  extraction/strict mode, cost ceiling, live-adapter wiring.
- **`P3` · Streaming hub.** Per-viewer broadcast queues + `Last-Event-ID` resume; prototype
  SSE is live, the real hub + DoD #3 gate are deferred (built late — see ADR order).

`payload` is a typed event union (`events.py`); the agent loop & `@rex_tool` harness landed in
`P2.2` (`loop.py`, `tools/`). The brain is still a **stub** — don't introduce the LLM
`brain()` early; it's `P5` (paid), and the loop stays free until then.

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
Machine-enforced quality replaces team review, in two layers:

- **Local pre-push gate** (`.githooks/pre-push`) – every push runs `ruff check` +
  `ruff format --check` + `pyright` + `pytest`; any red aborts the push. Arm once per clone:
  `git config core.hooksPath .githooks`. Bypassable with `--no-verify` – a convenience gate,
  not a wall.
- **Server-side protection** – GitHub branch protection on `main` blocks force-pushes and
  deletion (admins included); CI (`.github/workflows/ci.yml`) runs on every push. Limit:
  required status checks gate PR *merges*, not direct pushes – a direct push lands first and
  CI reports red *after*. The pre-push hook makes red rare; CI makes it loud and fast.

Never `--no-verify` into `main`. If the gate is red, the fix is green code, not a bypass.

## Accepted limits – do not pre-solve

Deliberate, documented in the ADR: single-process in-process hub; SQLite single-writer;
lossy broadcast hub (never revisit – log is truth); no auth on localhost; live adapters hit
**public ATS APIs only (Greenhouse / Lever)** – never scrape ToS-prohibited boards.
