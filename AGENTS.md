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
  `brain.THINKING_FLOOR` (`between_tools`; `disabled` on `claude-sonnet-5`), and a model with no
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

**Current position:** `P4` ✅ **done** · `P5` Unit 1 (edge boundary) ✅ **done** (`020283e`) ·
Unit 2a (provider adapter, offline) ✅ **done** (`edd4ec8`) · Unit 2b (the first paid call) ✅
**done** — one real Sonnet 5 call landed and parsed (`tool=sniff`), captured as a golden fixture
(`tests/fixtures/smoke_sonnet5.json`, invariant 6; it carries a live adaptive-thinking sibling
block, so it validates the 2a lenient scan on genuine output). `SMOKE_MODEL = "claude-sonnet-5"` (`"claude-sonnet-5-5"` since W2.5);
`build_tools` strips `title` from the derived schemas (Anthropic accepted them — confirmed by the
free `count_tokens` pre-flight); a shared `build_request_body` holds the Sonnet 5 guards (no
`temperature`/`top_p`/`top_k`, no `thinking`, no prefill; since W2.5 on Sonnet 5.5 too, as its docstring lists them); `smoke.py` is the paid entrypoint
(`--count-only` free pre-flight → one `messages` call → persist raw bytes → `parse_decision`),
hard one-call guard. **`P5` Unit 2c — brain-in-loop — ✅ done. ADR upgrade-map stage 3 CLOSED:** the
LLM drives a full hunt end to end (plan → tool runs → observation threads back into `messages` → next
iteration → typed outcome), replay-gated. *2c.1 (offline, no-spend) — ✅ done (`71ab898`):* the loop
projects the trajectory log into an Anthropic `messages` array (inv 2, `project_messages`), threads a
`HUNT_SYSTEM_PROMPT`, folds per-call usage into a running cost from `UsageEvent`s (inv 5,
`cost.fold_cost`) with a `COST_CEILING_USD` breaker, `classify` learns httpx transport/status failures
(429/5xx/timeouts → retryable; since W2.0 the adapter maps SDK/transport errors to a provider-neutral
`BrainCallError` that `classify` honours, and loop.py imports no httpx), and
`select_brain_for`/`REXHUNTER_BRAIN` gate the spender surface (stub default, opt-in `live`). The live
loop path sends `thinking: {type: "disabled"}` (`{type: "between_tools"}` since W2.5) + `tool_choice: {disable_parallel_tool_use: true}` –
bare one-`tool_use` replay turns until thinking
capture lands in Unit 3. *2c.2 (gated live) — ✅ done:* one budget-capped hunt (`rexhunter.hunt_smoke`,
`COST_CEILING_USD=0.05`, `MAX_ITERATIONS=3`) on `SMOKE_MODEL` against the mock-gym board — 2 brain
calls, `needs_help`, $0.0149. **Call two fired: the reconstructed assistant `tool_use` + user
`tool_result` (paired by `tool_use_id`) was accepted by the real API** — the never-run path is proven.
Each call's raw bytes captured as golden fixtures (`tests/fixtures/hunt_1ab26132…/call_0{1,2}.json`, inv 6);
`tests/test_hunt_replay.py` re-drives them offline to identical events (DoD #5). The daemon still drives
the **no-spend stub brain** (`stub.py`); daemon-lifespan live wiring is a later unit.
**▶ `P3` · Streaming hub — ✅ done. DoD #3 CLOSED.** The prototype poll is retired; the real in-process
broadcast hub now serves the live feed (pipeline: hunt task → SQLite append (write-ahead) → hub → SSE →
projection). *`P3.1` (hub + fan-out, offline) — ✅ done (`380e5af`):* `hub.py` — one bounded
`asyncio.Queue` per viewer; `publish()` offers a committed event to every queue via `put_nowait` and
DROPS a full queue (drop-and-resync), never awaiting (a slow viewer can't slow the hunter);
`Envelope(id, data)` carries the stored payload STRING so live-splice == catch-up bytes by construction;
a heartbeat is `id=None` → renders `": keep-alive\n\n"`, never advancing `Last-Event-ID`. Write-ahead
(inv 1) made structural: the publish choke SET is `{db.append_event, verdicts.capture_prey}` — the second
bypasses `append_event` for one-txn atomicity and the stub captures on every hunt, so BOTH publish,
strictly post-commit; `type PublishFn` threaded (default `None` → prototype/tests untouched) through loop +
scheduler; server lifespan feeds the hub + runs the heartbeat task. *`P3.2` (resume + gate) — ✅ done:*
`server.py` `snapshot_state` (open runs + prey pen + per-territory latest + `latest_id`, from the log,
inv 2), `catch_up` (`WHERE id > :last_seen`) and `stream_events` (register FIRST → catch-up → live-splice
with monotonic-id dedup; `finally` deregisters); `/events` repointed at the hub (`?since=` seed +
`Last-Event-ID` resume), `/snapshot` added, `/` does snapshot-then-stream. DoD #3 gate
(`tests/test_stage3_gate.py`): two viewers render byte-identical feeds; a viewer closed for a full hunt
reconnects via `Last-Event-ID` with zero gaps / zero dups (catch-up + splice + dedup, incl. the
≤-high-water drop); a full-queue viewer is dropped while the hunt completes unslowed. Verified
end-to-end through the real uvicorn server (curl `/events` streamed a full hunt's
`tool_call`/`tool_result`/`prey_captured` frames + heartbeats; `/snapshot` `latest_id` 0→3). Thinking stays
`{type:"disabled"}` (`{type:"between_tools"}` since W2.5) — P3 built its render target, does not consume it.
**▶ `P5` · Unit 3 — the `ThinkingDelta` relay — ✅ done.** Thinking is back on and Rex's reasoning
streams through the P3 hub. *`3a` (streaming transport, offline) — ✅ done (`a3e05b4`):* `brain.py`
`iter_sse_events` + `StreamAssembler` (90 lines, under the frugality threshold — no vendor SDK; invs
3/6 need the raw signed-block bytes the SDK's typed-delta layer hides; W2.0 kept them hand-rolled and
put only the transport on the SDK, raw mode – the D9 probe confirmed the typed layer re-serialises the
signed block, key order `signature,thinking,type`) fold a Sonnet 5 SSE response
into the SAME `_ProviderResponse` shape `parse_decision` consumes; display streams, execution waits
(tool_use `input` finalized only at `content_block_stop`). *`3b` (write-ahead relay, offline) — ✅ done
(`2c37708`):* `ThinkingDelta` joins the union (relay-only, skipped by `project_messages`); `Brain` gains
a call-time `ThinkingSink`, a loop-built closure over `(conn, run_id, publish)` that appends each delta
write-ahead (inv 1) then the hub broadcasts it — reusing `publish()`, no new fan-out. *`3c`
(signed-block replay + gate) — ✅ done, replay-gated (`c70f2f7`):* `ToolCallEvent.thinking` stores the
turn's VERBATIM signed block (inv 6); `project_messages` leads each reconstructed assistant turn with it,
echoed never rebuilt (the signature is bound to the exact bytes — a rebuild would 400). **One gated
thinking-on live hunt PROVEN** (`rexhunter.hunt_smoke_thinking`, `claude-sonnet-5`, adaptive+summarized,
`HUNT_MAX_TOKENS=4096`, `COST_CEILING_USD=0.20`, `MAX_ITERATIONS=3`): 2 brain calls, `completed`, $0.0187.
**Call two was accepted — the reconstructed assistant turn led with call one's verbatim signed thinking
block and the real API took it** (not a 400). Raw SSE streams captured as golden fixtures
(`tests/fixtures/hunt_32975e98.../call_0{1,2}.json`, inv 6); `tests/test_thinking_hunt_replay.py`
re-drives them offline to identical events (DoD #5). The daemon still drives the no-spend stub brain;
daemon-lifespan live wiring is a later unit.
**▶ `P5` · Daemon live-wiring — ✅ done.** The harness-proven streaming/thinking brain now runs in the
daemon: the lifespan streams Rex's reasoning to a real `/events` tab, bounded by a daemon-level (id-scoped)
spend budget. *`W.1` (brain selection + daemon budget, offline) — ✅ done:* the lifespan calls
`select_brain_for` honoring `REXHUNTER_BRAIN` (was hardcoded stub); `live` now selects the STREAMING/THINKING
adapter (`stream=True` + adaptive thinking, `HUNT_THINKING`/`HUNT_MAX_TOKENS` promoted to `brain.py`), not the
non-streaming one; `select_brain_for(default=)` routes `daemon_config`'s brain through stub mode (the
injectable lifespan seam) and the lifespan owns the paid client's close. The id-scoped ceiling
`DAEMON_SPEND_CEILING_USD` is the seq-scoped per-run ceiling's analogue: `daemon_spend_usd = cost.fold_cost`
over the whole log's `UsageEvent`s (`db.read_usage`, ALL runs along the global `id` cursor) — a fold over the
log, NOT a counter (inv 5: a counter drifts on a crash between the usage commit and the increment, and races
across concurrent hunts; the fold reconstructs spend exactly, one price table – an input to the fold, never
stored state; `prices=` defaults to `cost.PRICE_PER_MTOK` (W2.1) – two windows). `_bounded_hunt`
consults it BEFORE `run_hunt` and refuses to launch when over budget (Tiny Arms for money, inv 4 — the daemon
structurally cannot overspend; fail-closed whole-log window, rolling day-window deferred). `cost_ceiling_usd`
threaded scheduler→run_hunt; env knobs `DAEMON_SPEND_CEILING_USD`/`COST_CEILING_USD`/`MAX_ITERATIONS`. *`W.2`
(ThinkingSink under the real lifespan, offline) — ✅ done:* the loop-built sink closure (`conn`/`run_id`/hub
`publish`) proven write-ahead under `server.app.router.lifespan_context` — a delta-emitting brain injected via
`daemon_config`, viewer gated on a test `Event` to beat the delta race. Test-only (the plumbing was already
correct). *`W.3` (one gated live daemon hunt + browser smoke) — ✅ done, replay-gated:* one live hunt through
the real lifespan (`claude-sonnet-5`, thinking-on + streaming, `DAEMON_SPEND_CEILING_USD=0.005`, fresh
`REXHUNTER_DB`) — **$0.0224, ONE run, budget gate refused further hunts** (`daemon_spend_usd` == the run's
`fold_cost` == $0.0224 at the $3/$15 table the run was priced at; D6 reprices the same tokens to $0.014928).
The browser tab rendered Rex's reasoning streaming live: `thinking_delta` frames
committed DURING each brain call, interleaved with the verbatim-signed-block `tool_call`, `tool_result`,
`usage`, and an id-less `: keep-alive`. This hunt ended `needs_help` (the model judged the mock board
unworkable), so no `prey_captured` in THIS stream — that event's live streaming is covered by the free stub
smoke. Canonical `/events` stream captured as a fixture (`tests/fixtures/daemon_hunt_4688398b….sse`, inv 6);
`tests/test_daemon_stream_replay.py` re-drives it offline (monotonic ids, delta-before-tool_call, signed
block, paired tool_use_ids, usage folds to the spend, id-less heartbeat). Scope guard held: mock-gym only, no
ATS adapter, no frontend, no new tools. **Since `W.3`, the projection spine has shipped**
(`view`/`viewstate`/`render` – the pure `project(log)→ViewState` reducer, the ⊕ assembler, and the
retro-game renderer + verdict buttons + live verdict streaming; property-locked, commits
`0ceb1b3`→`0b40e20`) **and, on top of it, the game-state layer (`506ec2d`→`2d40d0d`, 2026-07-07,
all test-first + browser-verified):** the runs ⊕ overlay tier (run outcome/territory served from
the `runs` table — NO terminal event, the settled reconciliation stands; ghost cursor stamps the
CURRENT outcome, a pinned deferral); per-run budget ceilings recorded write-once at `start_run`
(4a — recorded input facts, not injected config; `db.connect` ALTERs the columns into pre-4a
files, so an upgrade never strands an existing log); the daemon ceiling + schedule INJECTED into
`finalize` the clock's way (4b — dormant territories representable via the
`last_started_at=None` pairing); the id-less `run_finished` pulse (`hub.notify` after the closing
commit — live closures only, structurally never shutdown-abort or boot-sweep); closed-run cards
pruned into per-territory tiles (fresh-kill / cracked-earth / hunting / dormant, cracked-earth
the documented proxy until park-and-persist); `turns` folded per ToolCallEvent; and depleting
HP/stamina bars + `data-rex` sprite hooks — **ADR Pillar 5 §6's "budget guards visible as
HP/stamina mechanics" is CLOSED.** *Then next: Step 7 — the aesthetic skin (pixel/CRT CSS, sprite
keyframes on `data-rex`, day/night scene, damage/feast cues, StaticFiles split — still no build
step); then `P5` extraction/strict mode + the live ATS (Greenhouse/Lever) adapter; then eval
tiers 2–3.*
**▶ W2.0 · Brain adapter on the Anthropic SDK – ✅ done (`2755e7f`→`7c4d883`, 2026-11-09).** The
adapter's transport moved from raw httpx to `anthropic` 2.6.0 in raw mode (D9a): `with_raw_response`
/ `with_streaming_response` hand the provider's own bytes to the hand-rolled `StreamAssembler` and
`parse_decision`, so every recorded fixture replays byte for byte (invariants 3 and 6 unchanged). No
model change: `claude-sonnet-5`, thinking defaults as before. `brain.anthropic_client` is the one
client factory (`max_retries=0` – the loop owns retries; wire pinned to `x-api-key` +
`anthropic-version: 2023-06-01` with an explicit `base_url`; no redirects; certifi trust store). SDK
and transport errors reach the loop as the provider-neutral `BrainCallError` (D1a); `loop.py`
imports no SDK and no httpx. Replay-gated offline, no paid call; 261 tests pass at `7c4d883`.
*Next: W2.1 (below). The API-version and Bearer-auth bump behind a free `count_tokens` check is
deferred to its own run.*
**▶ W2.1 · Prices and the custom-headers residual – ✅ done on branch `w2.1-prices-headers`
(`120457b`→`66e704d`, 2026-11-10), PR #1.** Offline: no paid call, no model change, no
wire bump. D6 prices `claude-sonnet-5` at the $2/$10 sticker ("is now the standard price"; was
$3/$15), and the table gains `claude-sonnet-5-5` ($2/$10), `claude-opus-5-5` ($4/$20) and
`claude-fable-5-1` ($10/$50), with a guard that the unknown-model fallback ($15/$75) is at least
the dearest row on each side. The table is an input to the fold (P1b, inv 5): `cost.cost_of` /
`fold_cost` take `prices=`, defaulting to `PRICE_PER_MTOK`; each replay test checks its live
report under the $3/$15 table its run was priced at, plus the exact fold under today's table.
Existing logs reprice: `daemon_spend_usd` over a pre-D6 log of `claude-sonnet-5` usage reads a
third lower.
`ANTHROPIC_CUSTOM_HEADERS` (H1a): `anthropic_client` gives every listed name the pinned value or
`omit`, so no value from the variable reaches the wire (residual in Stack → LLM). The review
found and closed R1, live at `cdc7c14`: a second spelling of a pinned name (the exact spelling
first, then a case variant) let the variable replace the key and the version, add
`Authorization`, and switch a beta on. 276 tests pass at `66e704d`. *Not in W2.1, each deferred to its own run:
the API-version and Bearer-auth bump (behind a free `count_tokens` check) and W2.2.*
**▶ W2.1b · Wire bump and F8 – ✅ done on branch `w2.1b-wire-bump` (`db94095`→`2a2a322`,
2026-11-14), PR #4.** One paid check, no model change (`SMOKE_MODEL` stays `claude-sonnet-5`).
F8: `smoke.py` no longer overwrites `tests/fixtures/smoke_sonnet5.json`. Each paid smoke writes a
new `smoke_<model>_<UTC time>Z.json`, reserved with `"xb"` before any call (an existing name
refuses at $0), dropped only while no paid reply has reached it, and written, flushed and closed
before parsing (inv 6); `fixture_dir` and `now` have no defaults. The wire (W1a): `anthropic_client`
pins `Authorization: Bearer <key>` and `anthropic-version: 2026-10-21`, with `x-api-key` and
`anthropic-beta` omitted – today's SDK 2.6.0 defaults, pinned so a later SDK default cannot move
them (a test fakes one); the SDK defaults alone (W1b) would fail open under H1a. Paid check
(Darie): `count_tokens` 200 over the new wire, then run `4c47b41b-5268-4da3-b787-9483b21d2c3f` on
`claude-sonnet-5`: 2 brain calls, `needs_help`, $0.0096. Its two responses are the new fixtures
(`tests/test_wire_bump_hunt_replay.py`, DoD #5); every earlier fixture replays unchanged. Also in
this run: `claude-fable-5-2` priced at $8/$40 (PR #3) and the branch + PR workflow (PR #2).
Keychain trust: not adopted – Darie, 13 Nov 2026 (the trust store stays certifi). 295 tests pass
at `2a2a322`. *Next: W2.2 (`between_tools`).*
**▶ W2.3 · The whole assistant turn, stored and echoed – ✅ done on branch `w2.3-verbatim-turn`
(`22c95b6`→`3c6269f`, 2026-11-14), PR #6.** One paid run, no model change (`SMOKE_MODEL` stays
`claude-sonnet-5`). D7 with forks V1, E1 and W1 (chosen 14 Nov under Darie's delegation): the new
`ToolCallEvent.assistant_content` holds the JSON bytes of the response's `content` array, every
block in order (V1). `parse_decision` fills it from the validated envelope on both the plain and the
streaming path; before, the plain path kept no block and the streaming path only its first
`thinking` block (F6). `project_messages` echoes it exactly as stored, the `tool_use` included with
its `caller` and the model's own `input` (E1: the docs say to send each assistant turn back exactly
as the API returned it, and on models with preserved thinking the prefix check compares every
earlier message). W1: new events fill only `assistant_content`; an event whose field is empty (every
event written before W2.3) keeps the old reconstruction – its `thinking` block, if any, then a
`tool_use` rebuilt from `id`, `name` and `raw_request`. `ToolCallDecision.thinking` and
`_extract_thinking_block` are gone. Free probe P (`count_tokens` on call two of the W2.1b hunt as
W2.3 builds it): P1 200 and P1b (P1 without `caller`) 200, both 1817 input tokens; the control P2
(an unknown key on the text block) 400, so `count_tokens` checks content blocks and the thinking-off
echo `[text, tool_use]` is accepted. Paid check (Darie): run `c1b41835-9206-4878-896f-fa13601a1f91`
on `claude-sonnet-5`, thinking on, streamed: 3 brain calls, `aborted` / `max iterations` (three
`sniff` calls, no `hunt_complete`), $0.0269. Call one is the first streamed response under
`2026-10-21`; call two, the first live request built from a stored whole turn (the signed thinking
block, then the `tool_use` with `caller`), was accepted; call three sent back two stored turns and
was accepted. Its three streams are the new fixtures (`tests/test_verbatim_turn_hunt_replay.py`,
DoD #5); every earlier fixture file is unchanged and still replays; in the three earlier hunt
replays, call two now carries the W2.3 echo, pinned offline. 325 tests pass at `3c6269f`. *Next: the
switch to Sonnet 5.5 (W2.2 + W2.5, plan B, Darie 14 Nov).*
**▶ W2.2 + W2.5 · The switch to Claude Sonnet 5.5 – ✅ done on branch `w2.2-thinking-floor`
(`c93e577`→`b6cd940`, 2026-11-14), PR #7, and branch `w2.5-sonnet-5-5` (`f8bc50a`→`8bed83c`,
2026-11-15), PR #8.** Plan B, forks chosen 14 Nov under Darie's delegation. W2.2 (F5, fork F1):
`brain.THINKING_FLOOR` holds the lowest thinking setting per model, `claude-sonnet-5`
`{type: "disabled"}` and `claude-sonnet-5-5` `{type: "between_tools"}` (the thinking docs and the
Sonnet 5.5 migration guide, read 14 Nov 2026: Sonnet 5.5 answers `disabled` with a 400, and
`between_tools` takes no other field). `thinking_floor(model)` returns a copy of the row; a model
with no row raises `ValueError` in `adapter_brain_for` and in the pre-flight body, before any
request (fail closed). Fork R1: each replay of a recorded run drives the adapter on the model it
was recorded on (`RECORDED_MODEL`) and asserts that model on the wire, and `disabled` on the
thinking-off runs. Part A left the Sonnet 5 wire unchanged (the pre-flight and adapter bodies are
byte-identical); 336 tests pass at `3ae8268`. W2.5 (D2): `SMOKE_MODEL = "claude-sonnet-5-5"`, so
every gated entrypoint and the daemon's `select_brain_for` (`REXHUNTER_BRAIN=live`) run on Sonnet
5.5; no live daemon run was made. Free probe K (`count_tokens` on Sonnet 5.5): K1, the thinking-off
body (`between_tools`), 200, 1743 input tokens; K2, the thinking-on body (adaptive,
`display: "summarized"`), 200, 1744; the controls 400, so `count_tokens` checks the thinking value:
K3 (`disabled`: "To turn thinking off on this model, send `"thinking": {"type": "between_tools"}`
instead of `{"type": "disabled"}`. The model does not think before responding. The short updates it
writes between tool calls come back as thinkin", cut there by the probe) and K4 (`between_tools`
with `display`: "Extra inputs are not permitted"). Paid checks (Darie, $0.028018 of a $0.25 cap):
run `5b62bc48-7577-4a52-b003-bdabf592061d`, thinking off, plain: 3 brain calls, `needs_help`,
$0.013476. Each turn came back `[text, tool_use]` with 0 thinking tokens and no progress-update
block; calls two and three sent back one and two stored thinking-off turns and were accepted (the
first live `messages` send-back of a stored thinking-off turn; the prefix check passed). Run
`54590aec-de50-4727-a0a2-9e533fb02efa`, thinking on, streamed: 3 brain calls, `needs_help`,
$0.014542. Call one did not think (`[tool_use]` only); calls two and three each carried one
summarized, signed thinking block; the SSE event and delta types are those of the Sonnet 5
recordings; call three sent back call two's signed block and was accepted (the first live send-back
of a Sonnet 5.5 thinking block). Their responses and streams are the new fixtures
(`tests/test_sonnet_5_5_hunt_replay.py`, `tests/test_sonnet_5_5_thinking_hunt_replay.py`, DoD #5);
every earlier fixture file is unchanged and still replays, each recorded hunt on `claude-sonnet-5`
(R1). Still offline only or unverified: a turn with two thinking blocks; whether a thinking block
is up-front reasoning or a progress update; the "interrupted before it finished" truncation shape;
the tool descriptions, as the replays register their own `sniff` while the live runs sent
`stub.sniff`'s docstring (call three of the thinking-on run quotes it); and a `display` dropped
from `HUNT_THINKING`, caught only by the wire literals of two replay tests. Earlier lines of this
section that name `claude-sonnet-5` as `SMOKE_MODEL` or `disabled` as the live thinking-off value
describe the state before this entry. 350 tests pass at `8bed83c`. *Next: W3. New for it:
`tests/test_hunt_smoke_thinking_offline.py` leaves two `rexhunter-think-*` temp directories per
suite run, and `tests/test_stage5_gate.py` intermittently raises an aiosqlite
`PytestUnhandledThreadExceptionWarning`.*
**▶ W3.1 · The cut-stream guard – ✅ done on branch `w3.1-cut-stream-guard` (`6717f4b`→`2df2e20`,
2026-11-15), PR #9.** Offline: no API call of any kind, no model change, no wire change (the four
wire-check bodies, a plain turn, a streamed turn, the pre-flight and an echoed second turn, are
byte-identical to `e1a22b4`'s; cp1 §8, cp5 §3.4). Labels are W3.1's (cp files: this run's
checkpoint reports). Forks chosen 15 Nov by rex-orchestrator under Darie's delegation: W3.1-S1,
W3.1-T3, W3.1-P1, W3.1-Z1 with Z-stop, W3.1-A1; W3.1-U1 by Darie herself. The hazards at `e1a22b4`
(cp1 §4): W3.1-H1, a cut `hunt_complete` (its start `input` `{}`) closed the run `completed` with
an empty catch; W3.1-H2, a cut `sniff` aborted as "invalid tool args" with no response bytes;
W3.1-H3, a closed `tool_use` whose input was not JSON raised `JSONDecodeError` in `_finalize`,
reaching the `<loop>` catch-all with no bytes, and hunt_smoke_thinking's reporter raised before
`write_fixtures`; W3.1-H4, an SSE `error` frame was ignored and the closed `tool_use` dispatched;
W3.1-H5, `stop_reason` was never read on the decision path (`parse_decision` acted on a `tool_use`
whatever the stop reason). Now a turn that did not finish never reaches `_run_tool` or a terminal
decision (`tests/test_cut_stream_guard.py`, 46 tests). `_drive_stream` checks, after the read:
W3.1-Z, the first `error` frame ends the read (Z-stop) and fails the call with the status its type
stands for (`brain.STREAM_ERROR_STATUS`, the 11 rows of the errors page, read 15 Nov 2026; a type
with no row is fatal, status None), retryable iff 429 or 5xx (`_retryable_status`, shared with
`brain_call_error`), its `raw_response` the frame's own `data` bytes; W3.1-T, a body that ends on a
frame boundary before `message_stop` is a retryable `BrainCallError` with the assembled bytes,
retried within `retry_budget` ("brain call failed"); W3.1-T3, a body that reached `message_stop`
with a block left open or no stop reason is a `BrainParseError`; W3.1-P, a closed `tool_use` input
that does not parse (`ValueError`, so an integer over 4300 digits too) keeps the block's start
input and is a `BrainParseError`, and the fold never raises on it, so the reporter reaches
`write_fixtures`. Then `parse_decision` (W3.1-S): any `stop_reason` but `tool_use`, `None`
included, is a `BrainParseError`, checked before the usage is read. Each `BrainParseError` ends the
run `aborted`, "malformed provider response" (W3.1-A1), with one `<brain>` ErrorEvent carrying the
bytes and the cause in `detail` (`stop_reason='max_tokens'`, the open blocks, the unparsed text).
Each check has its mutant, red then reverted (W3.1-M1 to W3.1-M22, cp2 to cp5b). The docs (stop
reasons, streaming, thinking, errors and the tool-use pages, read 15 Nov 2026, saved with sha256 in
cp1 §2): no page tells a client to run a tool when `stop_reason` is not `tool_use` (the thinking
page, read 19 Nov 2026, continues a turn that stopped on `max_tokens` soon after a tool call with
"a `tool_result` for each `tool_use` block in that turn": fork S2, not taken; W3.4b), and none states
where an error frame can arrive, what follows it, or whether a cut `tool_use` gets its
`content_block_stop`. No cut shape has been seen live: there was no paid run, and every cut and
error stream is built in the tests. The Unit 3a line above ("display streams, execution waits")
holds for every cut shape only from W3.1 on. 396 tests pass at `2df2e20`. Open (cp1 §6-7, cp2 §2,
cp5 §3.5, cp6): **W3.1-U** (Darie, W3.1-U1, 15 Nov): a call the guard refuses appends no
UsageEvent, so `cost.fold_cost`, the per-run ceiling and `daemon_spend_usd` do not count it;
measured with `retry_budget=2` and `max_iterations=3`, a plain or streamed `max_tokens` turn and a
stream cut before `message_delta` each gave 3 UsageEvents at `e1a22b4` (one per dispatched
iteration) and give 0 now, an error frame that ends the body after the `tool_use` gave 3 and gives
0 from W3.1-S1 on, and one inside an otherwise finished stream gave 3 and gives 0 from W3.1-Z1 on;
a 5xx or a reset mid-stream appended none before W3.1 either. **W3.1-O1 to W3.1-O5**, provider
bytes that still reach `<loop>` without them: a `data:` line that is not JSON (a body cut inside a
`data:` line too), a frame that is not an object, a frame missing `index`, `content_block` or
`delta`, a value of the wrong type in those or in `message` or `usage` (a non-integer `index`, a
non-object `content_block`, `delta`, `message` or `usage`), and a token count that is not a number
on a `tool_use` turn (`int()` in `_extract_usage`); W3.1-O1 to W3.1-O4 also make the reporter raise
before `write_fixtures`. **W3.1-K1**: Z-stop cannot act behind hunt_smoke's `CapturingTransport`,
which reads the whole body first (`CapturingTransport.handle_async_request`). The new retries run
with no backoff, like every brain retry (`loop._call_brain`). *Next: W3 run 2 (retries and backoff;
W3.1-U and W3.1-K1 sit next to it). `tests/test_hunt_smoke_thinking_offline.py` still leaves two
`rexhunter-think-*` temp directories per suite run (the two new harness tests leave none), and
`tests/test_stage5_gate.py` still intermittently raises the aiosqlite
`PytestUnhandledThreadExceptionWarning`.*
**▶ W3.2 · Retries that wait, both ceilings before every brain attempt, refused calls counted – ✅
done on branch `w3.2-retries-and-ceilings` (`4d72e82`→`8569e62`, 2026-11-15/16), PR #10.** Offline:
no API call of any kind, no model change, no wire change (the four wire-check bodies are
byte-identical to `8fdd92a`'s, and a retried body equals the first). Labels are W3.2's (cp files:
this run's checkpoint reports). Forks chosen 15 Nov by rex-orchestrator under Darie's delegation:
W3.2-U2a with X1-move, W3.2-C1, W3.2-B2 as written (Jfull + Cap8), W3.2-Y1; U2 and the R2-RV1 fix
are Darie's own rulings. "Fragility #2" and "#4" are items of the W3 plan (D8, Darie 9 Nov: fix all
four in W3): #2, a retry runs again at once; #4, the spend ceilings are checked too rarely (#1, the
context window, and #3, raw SSE bytes, are W3 run 3). The hazards at `8fdd92a`: W3.2-H1 (#2),
`_call_brain` re-sent a retryable BrainCallError at once; W3.2-H2 (#2), `_run_tool` re-ran a
TimeoutError or RetryableToolError at once; W3.2-H3 (#4), the daemon ceiling was checked once per
hunt, before `run_hunt`; W3.2-H4 (= W3.1-U), a 200 the boundary refused appended no UsageEvent;
W3.2-H5 (#4), the per-run ceiling was checked once per iteration, so a retried call made
`retry_budget + 1` requests with no check between them. **U2** (Darie, 15 Nov, replacing her W3.1-U1
ruling: "a refused call's usage gets a UsageEvent before its ErrorEvent") closes the W3.1 entry's
open W3.1-U. As built (W3.2-U2a, `4d72e82`): `BrainParseError` and `BrainCallError` carry `usage`;
`brain._refused_usage` reads a refused 200's count and never raises, and a count that does not read,
or that the fold cannot price (the W3.2-RV1 pricing check: `cost.cost_of` once, so an unpriceable
count cannot make every later fold raise), is not counted and its ErrorEvent says "usage not
counted"; `_call_brain` and `run_hunt`'s BrainParseError clause append it before the ErrorEvent
(write-ahead). X1-move: the accepted turn's `_extract_usage` now runs after the three later
refusals, so a refused turn with an unreadable count ends "malformed provider response" with its
bytes, not `<loop>`. The U2a table (cp1 §7; `retry_budget=2`, 100 input and 20 output tokens per
literal response): a plain or streamed `max_tokens` turn, W3.1-T3, W3.1-P, two or no `tool_use`
blocks, a `tool_use` without `id` and a bad `hunt_complete` catch went 0 → 1 UsageEvent ($0 →
$0.0004 each); a stream cut before `message_stop` ($0.0012) and a retryable error frame ($0.00063)
0 → 3, one per attempt; a fatal error frame 0 → 1; still not counted: a reset mid-stream
(W3.2-U2b), an error frame before `message_start`, a status error, an envelope-invalid body and a
count that does not read; every other outcome is unchanged. **C1** (`3f421ff`): `_call_brain`
re-reads the log and checks both ceilings before every attempt, a retry included: the run's fold
against `cost_ceiling_usd`, then, when set, `cost.fold_cost(db.read_usage(conn))` against
`daemon_spend_ceiling_usd`, which `run_hunt(daemon_spend_ceiling_usd=None)` gets from
`_bounded_hunt` (injected, never recorded on the runs row, 4b). New abort reason "daemon spend
ceiling" ("cost ceiling" when both trip: it is checked first). The bound: a run, or the daemon, goes
past its ceiling by at most the one call each hunt has in flight, of the calls the log counts
(W3.2-X16). Measured: a $0.01 daemon ceiling stops a hunt at 2 calls, $0.016 (3 calls, $0.024, at
`8fdd92a`); two concurrent hunts stay within the ceiling plus one call each (≤ $0.026; $0.048 at
`8fdd92a`). **B2 + Y1** (`a1ff3f5`): `loop.backoff_delay(retry, retry_after=None)` is full jitter, a
uniform draw in [0, min(8, 0.5 × 2^(retry-1))] s, then at least `min(retry_after, 8)` (60 s since W3.4);
`BACKOFF_BASE_S = 0.5` and `BACKOFF_CAP_S = 8.0` are anthropic 2.6.0's `INITIAL_RETRY_DELAY` and
`MAX_RETRY_DELAY` (`_constants.py:14-15`). `brain._retry_after` reads a status response's
`retry-after` in seconds only (finite, ≥ 0; an HTTP-date, `retry-after-ms` and anything else are
ignored). `_call_brain` and `_run_tool` await `loop.retry_wait` (default `asyncio.sleep`) after the
attempt's events commit, only before a retry; the next attempt re-checks both ceilings. Y1:
tests/conftest.py's autouse `retry_waits` records every wait and never sleeps. With the default
`retry_budget=2` a failing brain call now waits at most 0.5 s + 1 s, 3 s + 3 s with `retry-after:
3`, and at most 8 s + 8 s (60 s + 60 s since W3.4). Risk notes: a full-jitter wait can be near 0 s, so a retry can still
follow almost at once; a `retry-after` above 8 s is cut to 8 s (60 s since W3.4; W3.2-X6; the rate-limits docs say to
wait that many seconds), so such a retry likely fails within the budget, and making it fatal
instead is Darie's call (ADR:126). Parked, not taken: Jsdk (the SDK's own jitter) and Cap60 (Cap60 taken in W3.4).
**R2-RV1** (`8569e62`): found by review 2 (on C1), pre-existing at `8fdd92a`, widened by C1 (an
in-run `read_usage` before every attempt), fixed in W3.2 on Darie's ruling of 15 Nov (W3.2-F2a +
W3.2-F3). A daemon-shutdown cancel that landed inside an in-run SELECT (`async with
conn.execute(…)`'s `__aenter__`) while another connection committed left the stepped statement in
aiosqlite's worker, so the shielded `finish_run` failed at once with `database is locked`
(SQLITE_BUSY_SNAPSHOT): the run stayed `outcome IS NULL`, the cancel was lost, and `_bounded_hunt`
swallowed the error, so the territory loop relaunched hunts. Now `db.read_events` and
`db.read_usage` use `execute_fetchall` (one worker call, no statement left open); `run_hunt`'s
cancel clause always re-raises the cancel and logs a failed write, and the boot sweep then closes
that run 'crashed' (DoD #1); and `_bounded_hunt` re-raises a CancelledError for any exception
raised while it is being cancelled. The red tests hold the read with a progress handler (gates, not
timing); the F1 and F3 branches have their own tests. **Process:** reviews 1 to 3 ran as
three-reviewer workflows with a refuter per finding; from 15 Nov 23:30 AEDT, Darie's ruling: no
subagents, workflows or adversarial checks to the end of W3.2 (the rest self-reviewed in
session). 466 tests pass at `8569e62` (396 at `8fdd92a`): the new
tests/test_retries_and_ceilings.py and the fixture in tests/conftest.py; no existing assert
changed. No live run was made. Open: W3.2-U2b (a reset
mid-stream: billed, uncounted); W3.1-X9, W3.1-K1, W3.1-O1 to W3.1-O5 (the accepted-turn W3.1-O5
included); W3.2-X17 (hunt_smoke's `CapturingTransport` drops `retry-after`); Jsdk and Cap60
(parked); W3.2-N1 (a null `input_tokens` in `message_delta`); the count types: a negative count
lowers the fold (W3.2-RV12), a bool, numeric-string or float count is read as a number, and a
priceable huge count (313-314 digits) prices to inf and latches both ceilings (typed); W3.2-RV1b (an
accepted turn's unpriceable count, 315-4300 digits, is written and every later fold raises; since C1
a hunt under a daemon ceiling then ends `<loop>` at its next attempt); W3.2-X16 (the bound covers
only the calls the log counts); W3.2-R2-RV14 (budget aborts append no event: an ADR:126 note,
Darie's); W3.2-R2-M (the launch gate's `>=` unpinned); the `capture_prey` atomicity edge (a cancel
between its two INSERTs; INFERRED, not probed); and the intermittent aiosqlite
`PytestUnhandledThreadExceptionWarning` (tests/test_stage5_gate.py, as before W3.2). *Next: W3 run 3
(fragility #1 and #3). `tests/test_hunt_smoke_thinking_offline.py` still leaves two
`rexhunter-think-*` temp directories per suite run.*
**▶ W3.3 · The brain reads what it uses, and every attempt stores its bytes – ✅ done on branch
`w3.3-context-and-raw-bytes` (`c3b7372`→`1df8bdc`, 2026-11-17), PR #11.** Offline: no API call of
any kind, no model change, no wire change (the five wire-check bodies, a plain turn, a streamed
turn, the pre-flight, the second request of a two-turn hunt and a retried attempt, are
byte-identical to `070ed9a`'s). Labels are W3.3's (cp files: this run's checkpoint reports). Darie's
decisions of 17 Nov 2026, in her words: "Весь разговор, узкое чтение" (the model keeps the whole
conversation; the read from the log is narrowed) and "Новое событие на попытку" (one new event per
brain attempt, with its request and response bytes). Forks chosen 17 Nov by rex-orchestrator under
her delegation: W3.3-W, W3.3-E2, W3.3-R1, W3.3-S1, W3.3-X2. "Fragility #1" and "#3" are items of the
W3 plan (D8, Darie 9 Nov): #1, the brain's context read; #3, the raw SSE bytes, with the
request-bytes gap. The hazards at `070ed9a`: W3.3-H1 (#1), `_call_brain` read the whole run before
every attempt (`db.read_events`) and decoded every row, ThinkingDeltas included, though a call uses
four types, so a stored row of another type that did not validate ended the run at `<loop>`; W3.3-H2
(#3), a call's response bytes were not stored as received (the plain path kept only the parsed turn,
the stream path read text lines and kept a re-encoded assembly); W3.3-H3, the request body sent was
stored nowhere; W3.3-H4, a terminal decision appends no event (ADR:128, :220), so the last call's
bytes left no trace and a dead run could not become a fixture from its log alone (ADR:30). **W**
(`c3b7372`): `db.read_context` reads the run's `db.CONTEXT_TYPES` (`tool_call`, `tool_result`,
`error`, `usage`: the types `brain.project_messages` turns into `messages` and the per-run spend
fold reads), a type filter in SQL, one `execute_fetchall` (the R2-RV1 property); on both recorded
Sonnet 5.5 runs it projects the same messages as the whole read at every point of the run. **E2, R1,
S1, X2** (`1df8bdc`): the new `BrainExchangeEvent` (`raw_request: bytes`, `raw_response: bytes |
None`, `status: int | None`) rides on the Decision (an `exchange` field), on BrainParseError and on
BrainCallError, and the loop appends it before the attempt's UsageEvent, ErrorEvent or ToolCallEvent
(its ThinkingDeltas, written during the call, come first). `raw_request` is the body httpx2 put on
the wire (`http_request.content`, or the SDK error's `request`); `raw_response` is the body as
received, content-decoded: `read()` on the plain path, every chunk `iter_bytes` gave on the stream
path, framed into lines by `_received_lines` (httpx2's own `aiter_lines` over a tee, equal to
`iter_lines` at every chunk size tested), so a reset mid-stream keeps the bytes before the cut;
`status` lets a 429 or 529 attempt replay as one; both are None when no response arrived (a
connection failure). The event is committed and never published; `server.catch_up` and
`viewstate._LOG_ROWS_SELECT` leave it out and `view._apply` folds it to nothing. A brain that
reports no bytes (the stub and scripted brains) writes none. Every existing event keeps its bytes
(`raw_request=b""` on the `<brain>` and `<loop>` ErrorEvents; `BrainParseError.raw` and
`BrainCallError.raw_response` the assembled bytes on the stream path). Before commit, the wire check
found the carrier field in the terminal tools' `input_schema` (HuntComplete and NeedsHelp derive it;
every body about doubled): `SkipJsonSchema` keeps it out, pinned by
`test_the_exchange_never_reaches_the_tools_the_model_sees`. Bytes measured on the two recorded
Sonnet 5.5 runs, per call: requests 3689 to 4548 B (thinking off) and 3721 to 5851 B (on), responses
738 to 823 B and 2361 to 6474 B, stored payloads 5977 to 15549 B; a 50-call run is INFERRED at about
1.0 MB (off) to 3.5 MB (on) stored, quadratic in the calls. A run rebuilds from its own log: its
exchanges, served back in order, give the same requests byte for byte and identical events (DoD #5,
ADR:30; both recorded runs and a 529-then-needs_help run). Reds: the 8 hazard tests (W3.3-H1 ×2 at
`070ed9a`, W3.3-H2 to W3.3-H4 ×6 at `c3b7372`) and 31 existing exact event sequences (29 in
tests/test_retries_and_ceilings.py through `_hunt`, one in each Sonnet 5.5 replay), green by the
pinned edits; 0 `assert` lines changed or removed. The five pinned edits (W3.3-P1 to W3.3-P5): the
R2-RV1 hold's parametrize id `read_events` → `read_context`
(tests/test_retries_and_ceilings.py:1250, wording at :1208 and :1272); a filter that drops only
`BrainExchangeEvent` in `_hunt` (:257) and in each Sonnet 5.5 replay's `_replay`
(tests/test_sonnet_5_5_hunt_replay.py:117, tests/test_sonnet_5_5_thinking_hunt_replay.py:163); the
tests/test_hunt_smoke_thinking_offline.py:8 docstring. Mutants W3.3-M1 to W3.3-M13 (M6 measured as
W3.3-M6a and W3.3-M6b), each red on a scratch copy of its commit's tree. Docs (read 17 Nov 2026,
quotes in cp1 §2): the docs allow cutting older turns ("If your own code writes the summary, or cuts
older turns without one, remove the kept turns' thinking or send `"drop_block"`",
preserved-thinking/change-tools-or-instructions), at the cost of the kept turns' thinking on a model
with preserved thinking, Claude Sonnet 5.5 among them; a `tool_result` must immediately follow its
`tool_use`, and a tool-use turn's thinking blocks go back complete and unmodified. The ADR's revisit
trigger, 100,000 input tokens: about a tenth of Sonnet 5.5's 1M window, and 9 to 15 times the
projected input of a run's 50th call (about 6.8K tokens thinking off, 10.6K on, at the measured
growth per turn). No live run was made. 516 tests pass at `1df8bdc` (466 at `070ed9a`): the new
tests/test_context_read.py (5) and tests/test_brain_exchange.py (45). Open: W3.2-U2b (a reset
mid-stream is billed and uncounted; its partial bytes are now stored); W3.1-X9; W3.1-K1; W3.1-O1 to
W3.1-O5; the W3.2 count items (W3.2-N1, W3.2-RV12 and the count types, W3.2-RV1b, W3.2-X16) and the
rest of the W3.2 entry's open list; the attempts that end at `<loop>` (W3.1-O1 to W3.1-O5, a
non-network `APIConnectionError` cause, a failing sink write) store no exchange (W3.3-E2 as chosen;
the O-items' typed fix brings them in); a transport failure replays from the log only approximately
(its ErrorEvent names the exception); SSE line endings are UNVERIFIED (no page states them; every
recorded stream is LF only); W3.3-F1 (Darie's: at `070ed9a` the model already sees an optional
`usage` input on `hunt_complete` and `needs_help`; it is overwritten after validation, but a
malformed one refuses the turn, and the fix changes the wire) and W3.3-F2 (Darie's: the decisions
have no model_config, so before W3.3 an unknown `exchange` key in `hunt_complete`'s input was
ignored; now a malformed one refuses the turn and a well-formed one is overwritten); the wire
check's body 4 baseline is `f551d65e…0c05` (W3.2-X11: the ON FILE `25d12acc…3abe` is not
reproducible). *Next: W4. `tests/test_hunt_smoke_thinking_offline.py` still leaves two
`rexhunter-think-*` temp directories per suite run, and `tests/test_stage5_gate.py` still
intermittently raises the aiosqlite `PytestUnhandledThreadExceptionWarning`.*
**▶ W3.4a · Strict token counts, typed malformed frames, a counted reset, retry-after up to 60 s,
no usage in the terminal tools – ✅ done on branch `w3.4-spend-and-boundary` (`a19010f`→`7d18078`,
2026-11-17), PR #12.** Offline: no API call of any kind, no model change; one approved wire change
(F1). Labels are W3.4's (cp files: this run's checkpoint reports). Darie's decisions of 17 Nov
2026, in her words (each the planner's recommended option): "U2b: считать обрыв", "Строгие
счётчики", "Ждать до 60 с", "F1: убрать usage из схемы", "RV14: заметка в ADR" and "Оркестратор +
merge, живая охота". Forks chosen 17 Nov by rex-orchestrator under her delegation: counts-1 (one
reader; the bound `brain.MAX_TOKEN_COUNT = 10_000_000`; the pricing check dropped), frames-1 with
1a, reset-1 with 1a; N1 and `loop.RETRY_AFTER_CAP_S = 60.0` as planned. **Counts** (`a19010f`;
W3.2-RV12, the count types, W3.2-RV1b, W3.1-O5): `brain._read_usage`, the one reader for the
accepted and the refused path, reads `input_tokens` and `output_tokens` only, each a Python int that
is not a bool, 0 to 10,000,000 (a validation limit, ten times Claude Sonnet 5.5's 1M window; a call
then prices at most $900); an absent count reads 0, as before. A refused turn whose count fails is
not counted and its ErrorEvent says so; an accepted one is refused "malformed provider response"
with its bytes and exchange, and writes no UsageEvent. Reds at `9106c23`: a refused -5,000,000
count folded to -$9.9998; a bool, a numeric string, a float, a negative count and one past the
bound were counted, or dispatched on an accepted turn; a 401-digit accepted count was written and
the next fold raised (`<loop>`); a word, null or a list on an accepted turn reached `<loop>`. An int
over 4300 digits fails pydantic's envelope ("number out of range"), typed (4 guards). **N1**
(`e3f4fee`, W3.2-N1): a null count in `message_delta` keeps the earlier count, every key, as
anthropic 2.6.0's accumulator does (lib/streaming/_messages.py:547-556); red: message_start's 100
was overwritten. **Frames** (`193e901`, W3.1-O1 to O4): `_drive_stream`'s guard catches ValueError
(the int-limit one included) and `StreamAssembler.feed`'s new MalformedFrame (not an event type),
never the sink. A frame that does not parse, unterminated and last on a body that ended before
`message_stop`, is the cut (retried, W3.1-T); every other malformed frame is a BrainParseError with
the bytes and the counts so far and its exchange. Under 1a an `index` is an int and a
`content_block`, `delta`, `message` or `usage` an object; an absent optional field, an unknown event
type or field and a delta or stop for an index never started are read as before. Reds: 28 cases
ended at `<loop>` without bytes, and 4 values `int()` or `dict()` coerced were accepted. **Reset**
(`d3a72a2`, W3.2-U2b): a transport failure during the line read is the retryable BrainCallError,
its text unchanged (plus the note when the counts do not read), with the counts the frames
reported, so each attempt appends its UsageEvent before its ErrorEvent; `raw_response` stays None, a
frame whose closing blank line never came is dropped, and a reset after `message_stop` is counted,
then retried. The adapter's outer clause stays, for a failure outside the line read. Red: `[E, ERR]`
three times where `[E, U, ERR]`. **Cap60** (`2ccb2ad`): a brain retry waits at least
min(`retry-after`, 60 s); the jitter (0.5 s base, 8 s cap) and the tool retries are unchanged; red:
`retry-after` 30, 60 and 90 waited 8 s. **F1** (`7d18078`, W3.3-F1): `usage` leaves the
`hunt_complete` and `needs_help` schemas through `SkipJsonSchema`, like `exchange`; a model-sent
`usage` is still validated (W3.3-F2). **RV14** (W3.2-R2-RV14): a reconciliation note after ADR:130,
a budget abort closes its run through `runs.outcome` and appends no event, and a refused launch
starts no run. Wire (five bodies, a MockTransport capture): commits 1 to 5 byte-identical to
`9106c23` (1 `ac3af52e…7b66` 3739 B, 2 `21290c02…9c30` 3771, 3 `03abc056…08a4` 3721, 4
`f551d65e…0c05` 4150, 5 = 2); from F1 on, 1 `6122764f…4036` 2204, 2 `3c70bbbe…f358` 2236, 3
`10933c31…c7e4` 2186, 4 `b22273fd…4067` 2615, 5 = 2, each differing only by `usage` and its `$defs`
in the two terminal tools. Pinned edits W3.4-PA1 to PA13, all in tests/test_retries_and_ceilings.py
and tests/test_brain_exchange.py: :9-11 and :20-22 docstrings (PA11); :476-478 comment (PA13);
:567-568, the rename to `test_a_reset_mid_stream_is_counted` (PA9); :573 (PA1); :580-581 comment
(PA10); :585-586 (PA2, PA3); :948-949 (PA4); :1191 (PA5); test_brain_exchange.py :379 (PA6),
:414-417 comment (PA12), :422 (PA7), :425 (PA8). Mutants W3.4-MA1 to MA15, each red on a scratch
copy of its commit's tree. Docs read 17 Nov 2026: rate-limits, streaming, errors. Process: no
subagent or workflow; one advisor call at A1, none after (Darie: every check inside one session). No
live run was made. 605 tests pass at `7d18078` (516 at `9106c23`): the new
tests/test_spend_and_boundary.py (89). Open: W3.1-X9 (a retried stream's ThinkingDeltas repeat in
the feed); the network environment hardening (the SDK client reads proxies, `SSL_CERT_FILE`,
`SSLKEYLOGFILE` and OTel headers from the environment); W3.3-F2; W3.2-X16, narrowed (the bound
still covers only the calls the log counts: a count that does not read, a status error, an
envelope-invalid body and a stream that fails before `message_start` stay uncounted, and output
after the last frame received is not counted, INFERRED); the attempts that end at `<loop>` with no
exchange (a non-network `APIConnectionError` cause, a failing sink write); SSE line endings,
UNVERIFIED; the aiosqlite `PytestUnhandledThreadExceptionWarning` (tests/test_stage5_gate.py); Jsdk,
parked; **W3.4-D1** (an environment item, Darie's: with `int_max_str_digits` lowered below 4300 by
`PYTHONINTMAXSTRDIGITS` or `-X`, a plain-path int from that limit to 4300 digits ends `<loop>`, at
both count messages and `assistant_content`'s `json.dumps`, because pydantic's parser ignores the
lowered limit while `repr` enforces it; the default limit is typed on every path; a whole fix is an
environment policy, for example refusing to start below 4300); **W3.4-D2** (a value nested about
74,800 levels or more, in a field the API sends as a string, ends `<loop>` on the stream path at its
`str()` and `repr()` sites: the delta text fields, `stop_reason` in the refusal messages, the error
frame's type and the count messages; eight of the nine predate W3.4; the plain path refuses depth
past about 200). For Darie: keeping a whole turn despite a reset after `message_stop` (today
counted, then retried; a W4 question), and a `max iterations` line in the RV14 note (that abort
closes the same way, `loop._drive`). *Next: W3.4b (the smoke harnesses, the `capture_prey` cancel
edge, the hygiene and docs backlog) and the live daemon hunt.*
**▶ W3.4b · Smoke harnesses, capture atomicity, hygiene and the docs backlog – ✅ done on branch
`w3.4-harness-hygiene-docs` (`3f2fbcb`→`10c7f2e`, 2026-11-20), PR #13.** Offline: no API call of
any kind, no model change, no wire change (the five wire-check bodies are byte-identical to
`619e6ea`'s, the W3.4a F1 baselines, after every commit). Labels are W3.4's (cp files: this run's
checkpoint reports). The cp1 stop to rex-orchestrator was held by that session, undelivered, so
Darie answered in the worker's terminal on 20 Nov 2026: the recommended options of cp1, then
"decide on your own, autonomously, and do not ask me for confirmation". Forks: capture-1,
fixtures-1, telemetry-1; with them F8 d1 (a missing fixture dir refuses, never created),
`db_dir` as the temp-dir seam, and commit 8 with its mutant label MB11. **Capture** (`3f2fbcb`,
W3.1-K1, W3.2-X17): `CapturingTransport` passes the decoded body through as it arrives, captured
on the way, every header kept but `content-encoding` and `content-length`; the pre-flight passes
untouched; `captured` is a property. Reds: the tee read past the first error frame (`[1, 2] ==
[1]`) and dropped `retry-after: 3` (`[0.22…] == [3.0]`); MB1 (the body buffered again) and MB2
(`retry-after` dropped) each red. Live chunked or gzip streams through the tee are UNVERIFIED: only
a paid smoke exercises them. **Fixtures** (`4126b89`): both harnesses write what they captured in a
`finally`, before the reporter, a cancelled hunt naming its directory after its fresh DB's one
run. Reds: a cancel at call two, a 502 in HTML (hunt_smoke's reporter) and a frame that is not
JSON (hunt_smoke_thinking's) each lost every fixture (`[] == [...]`); MB3 (the write only on
success) red twice. The reporter can still raise after the write. **smoke.py** (`470d78b`, F8 c to
f): the paid path needs `REXHUNTER_BRAIN=live` (`--count-only` the key only); a transport error at
the pre-flight, the paid call or its body read is a typed exit; a failing unlink no longer masks
the error (c); a missing fixture dir refuses at $0 (d); any `OSError` reserving is the `[fixture]`
exit (e); the UTC name is pinned under a UTC+14 process zone, killing the `astimezone()` survivor
that passed under `TZ=UTC` (f). Reds: 7; MB4 (the guard removed) 1, MB5 (`APIConnectionError`
escaping) 4. F8 (a) and (b) stay accepted limits, untested. **capture_prey** (`9806ae0`): VERIFIED
(it was INFERRED): a cancel on its first INSERT, then `run_hunt`'s `finish_run` commit, left a
`PreyCapturedEvent` and no prey row, and so did a failed prey INSERT followed by the `<loop>`
ErrorEvent's commit (found here). It now rolls back on any BaseException; reds `(1, 0, …) == (0, 0,
…)` on both paths, held by a progress handler; MB6 red twice. **Temp dirs** (`33c6eb2`):
`run_live_hunt(db_dir=)`, set to `tmp_path` by the offline tests, with an autouse `mkdtemp` guard
(red twice before the seam); `rexhunter-think-*` 398 → 398 across a suite run (two more per run
before); MB7 red twice; the paid path keeps the system temp dir. **Pins** (`319f097`): a launch at
exactly the daemon ceiling is refused (W3.2-R2-M; the `>` mutant MB8 passed all 605 tests before);
the live daemon brain's request carries `display: "summarized"` (MB9 red 3, two of them the replay
literals; on Sonnet 5.5 the default is "omitted", the thinking docs, read 19 Nov 2026); the six
replays of recorded hunts register `stub.sniff`'s docstring, the description every live run sent
(6 reds before, MB10 red); the stage-1 gate's pipe closes in a `with` block (F12/R9; 0
ResourceWarnings under `-W always::ResourceWarning`, 2 before). **Chore** (`be93f4e`, F11):
`.hypothesis/` in `.gitignore`; `actions/checkout@d23441a…` (`# v6.1.0`, the commit `v6` points at,
so no bump; v7.0.1 exists, Darie's); fastapi telemetry pinned off and tested (inert today: only
`opentelemetry-api` is installed). **The aiosqlite warning** (`0c5d443`): CLOSED. A cancel inside
`db.connect` after `aiosqlite.connect()` leaked the connection, and aiosqlite's `__del__` later
stopped its worker on a closed event loop. Bisected to test_stage4_gate.py's crash-sweep test (5
warnings in 20 runs with stage 5; 0 in 20 after; 0 in 6 full suite runs); `db.connect` closes the
connection on any BaseException in its bootstrap; on-demand red, the worker thread alive after the
cancel; MB11 red. **Docs** (`10c7f2e`): the Stack lines on constrained decoding and the config
POST mark what is not built; a pointer on the W3.1 entry's stop-reason sentence; five
p1-walkthrough citations; the four `loop.py:NN` citations name `loop._run_tool`. Test edits:
PB1 (test_smoke_offline.py:315, the transport row now expects `SystemExit`, with its comment),
PB2 (:376, `REXHUNTER_BRAIN=live`), PB3 (test_hunt_smoke_thinking_offline.py:7-9 docstring), PB4
(its three calls gain `db_dir=tmp_path`), PB5 (the six replays' `sniff`), PB6
(test_stage1_gate.py's fixture), PB7 (test_view.py:48, :221, test_viewstate.py:29 citations),
PB8 (comments on the `assert` lines at test_smoke_offline.py:85 and test_stage_2c.py:699, their
expressions unchanged). 633 tests pass at `10c7f2e` (605 at `619e6ea`): the new
tests/test_smoke_harnesses.py (7) and tests/test_db_connect.py (1), and 20 more in six files. No
subagent, workflow or advisor call. Open: capture-2 (fixtures from the log's exchanges, W4); C10
(`.claude/hooks/guard-bash.sh:33-35` still says direct-to-main, Darie's); and from the W3.4a list
W3.1-X9, the network environment hardening, W3.3-F2, W3.2-X16 as narrowed, the `<loop>` attempts
with no exchange, SSE line endings, Jsdk, W3.4-D1, W3.4-D2 and its two questions for Darie. *Next:
the first live daemon hunt on Sonnet 5.5 (Darie's command), then W4.*
**▶ W3.4 Part C · The first live daemon hunt on Sonnet 5.5 – ✅ done (2026-11-20), PR #14.** On
`main` `dc4c8cf`, in Darie's terminal with the key set: the free pre-flight `PYTHONPATH=src uv run
python -m rexhunter.hunt_smoke_thinking --count-only`, then `REXHUNTER_BRAIN=live
REXHUNTER_DB=<live dir>/rex.db DAEMON_SPEND_CEILING_USD=0.05 COST_CEILING_USD=0.05 uv run uvicorn
--app-dir src rexhunter.server:app` (`MAX_ITERATIONS` unset: 50). A first attempt without the key
refused at $0 in both places, the pre-flight and the lifespan's `select_brain_for` (fail closed);
it left only an empty schema in the DB. The daemon was not stopped after its first run, so it ran
six `mock-gym` hunts, each launched 5 s after the last closed (21:37:29Z to 21:39:29Z), until the
launch gate refused the seventh: the whole-log fold $0.051306 against the $0.05 ceiling, $0.001306
over, inside the one-call-in-flight bound (W3.2-C1). Outcomes: five `needs_help`, one `completed`
with an empty catch (no prey). Per run: $0.010226, $0.006392, $0.006872, $0.010840, $0.006842,
$0.010134. 15 brain calls, each one attempt, status 200 and `stop_reason` `tool_use`: no retry, no
ErrorEvent, no error frame, no cut; 9 `sniff` calls; 191 ThinkingDeltas; 239 events. Call one of
every run: 1073 input and 59 output tokens, `[tool_use]` only (no thinking, as in W2.5), and its
request byte-identical to the wire check's body 2 (`3c70bbbe…f358`, 2236 B), so the offline
baseline is the live wire. Calls two and three each carried one summarized, signed thinking block
and sent the stored turns back; all were accepted. Every request's `hunt_complete` and
`needs_help` schemas carry no `usage` and no `$defs` (W3.3-F1, live). The stored stream bodies,
2357 to 9831 B, hold no CRLF: LF only, as received (`_received_lines` keeps the chunks), which
settles SSE line endings for what this run saw. DoD #5 on the live log: the six runs, rebuilt in
order on one fresh DB from their own BrainExchangeEvents (the same caps and the same $0.05 daemon
ceiling), give byte-identical requests (15 of 15), identical event payloads (239 of 239) and the
same outcomes; the rebuilt fold is $0.051306 and refuses a seventh launch. The DB copy read was
taken while the daemon idled (`PRAGMA integrity_check` ok). The DB stays outside the repo
(`~/dev/agent-reports/RexHunter/live-2026-11-20/rex.db`); making fixtures from it is Darie's.
Still unseen live: hunt_smoke's capture-1 tee, a retry, a cut stream and an error frame. *Next:
W4.*
**▶ W4a · No class docstrings in the terminal schemas, a live-hunt replay, small docs – ✅ done on
branch `w4-wire-fixture-docs` (`7c553a7`→`cb04b01`, 2026-11-20), PR #15.** Offline: no API call of
any kind, no model change; one approved wire change (`doc`). Labels are W4's (cp files: this run's
checkpoint reports). Darie's answers of 20 Nov 2026 to the planner's four questions, each the
recommended option: the docstrings out of the schema; a fixture from the live hunt; R10, `httpx`
to the dev group (W4b); the orchestrator answers the stops and merges, after a ping; A8 and A9
left as they are; "max iterations" in the RV14 note; pointers in the history lines; C10, a comment
in the hook. Forks chosen 20 Nov by rex-orchestrator under her delegation: doc-1, live-1. The
channel: a user-level `crossSessionInbound: hold` setting made the orchestrator's session hold every
inbound message (likely the W3.4b cause too), so the ping from `rex-w4` waited for Darie's approval
there; she approved it and removed the setting, the pong came back, and cp1 to cp3 arrived with no
hold. **doc**
(`7c553a7`): the class docstrings of `HuntComplete` and `NeedsHelp` (loop.py:180, :197), notes for
readers ("The outcome lives in runs (like P1) - no terminal event", "P2.2 ends the run cleanly"),
reached the model in every request as the top-level `description` of the `hunt_complete` and
`needs_help` input schemas. `build_tools` now drops that key after `_strip_titles`; the tool-level
descriptions, the properties and any field description stay. Wire (five bodies, a MockTransport
capture), the new baselines: 1 `bb2ef203…c24d` 1487 B, 2 `6b9e9a17…98a8` 1519, 3 `3f01481b…c293`
1469, 4 `08e8682c…fd31` 1898, 5 = 2; each 717 B shorter than at b7cf381 and equal to it, byte for
byte, with the two `"description"` spans cut. Reds at b7cf381: tests/test_terminal_schemas.py:26
and :65 (`'description'` still in the schema). Mutants W4-MW1 to MW3, 2, 1 and 1 red; MW3 (field
descriptions dropped too) goes red only through a stand-in model, as no terminal field has a
description. **live** (`12cb4b5`): `tests/fixtures/daemon_sonnet_5_5_2026-11-20/` keeps two of
Part C's six runs, one per outcome, `058dc53e…` (completed, an empty catch) and `5e555e12…`
(needs_help): each response's bytes as the daemon's log stored them (`call_0{1,2,3}.json`, 2372 to
6532 B), read from a copy of the live DB, byte-equal to the stored `raw_response`, with no secret
pattern and LF only. No request bytes are kept, so the fixture does not pin the wire.
tests/test_daemon_sonnet_5_5_hunt_replay.py (8) re-drives each run through the adapter on
`claude-sonnet-5-5` under the daemon's caps and reproduces its log (DoD #5 on live daemon data):
the outcome and the run's caps, the event types, the payloads other than the exchanges (a sha256
against the live log's), each exchange's status and response bytes, the relayed thinking (41 and
33 deltas), the usage and its fold ($0.010840, $0.010226), the tool calls, and each request's shape
and its echo of the stored turns. In a scratch check outside the suite, the replay on b7cf381
rebuilt all six live requests byte for byte; from `7c553a7` each is 717 B shorter and equal to the
live request less the two keys.
Mutants W4-MW4 (one response byte changed) 2 red, MW5 (the wrong model) 4 red. **Docs:** the
ADR:132 note gains a "max iterations" sentence (`0ee5252`, W3.2-R2-RV14); five history lines get a
pointer each, with no reflow (`cb04b01`: :145, :148, :159, :188, :784); and
`.claude/hooks/guard-bash.sh:33-35` states the PR rule, its code unchanged (C10). Test edits:
none. 644 tests pass at `cb04b01` (633 at b7cf381): tests/test_terminal_schemas.py (3) and the
replay (8). No subagent, workflow or advisor call. Open: the W3.4b entry's list less C10;
capture-2 was not chosen. *Next: W4b (the behaviour-preserving refactor).*
**▶ W4b · A behaviour-preserving refactor – ✅ done on branch `w4-refactor` (`fbf4fc3`→`89ed4a0`,
2026-11-20), PR #16.** No API call of any kind, no model change, no wire change (the five wire-check
bodies are byte-identical to the W4a baselines after every commit). The network was used once, in
B1's scratch probe: `uv run` fetched the locked wheels from PyPI into a scratch venv, and `uv lock`
fetched package metadata; nothing global was installed, no version moved, and the branch's lock was
resolved offline. Labels are W4's.
Forks chosen 20 Nov by rex-orchestrator under Darie's delegation: r1-a with characterisation tests
first, r2-a, r8-b; W4-R6 dropped. Each commit was applied byte for byte from a scratch chain of the
same steps on `git archive 71d31ad`, and each commit's tree equals its step's. **W4-R5**
(`fbf4fc3`): one `db.utcnow`; verdicts.py drops its copy. **W4-R4** (`8515467`):
`db.drive_to_done`, the shielded wait both shutdown cleanups ran. `run_hunt`'s `finish_run` waits
inside `suppress(Exception)`, the old `break`, then logs and re-raises the cancel; the job worker's
`conn.close` lets a failed close propagate, as before. **Tests** (`695f380`):
tests/test_paid_entrypoint_gates.py (6) pins the exact `SystemExit` message of smoke, hunt_smoke
and hunt_smoke_thinking with no key, and on the paid path without `REXHUNTER_BRAIN=live`, each
before any client or path is reached. Before, no test pinned the key exits or the hunts' refusals,
and smoke's refusal only by a substring. **W4-R1** (`0cff322`): `brain.paid_entry(before=, paid=)`
is the three mains' gate: the key, `--count-only`, then the `live` opt-in, in that order, every
message unchanged. In the two hunts the opt-in now runs before `asyncio.run`, not inside it.
**W4-R7** (`acf6df9`): `lazy from rexhunter import stub` (PEP 810) at the top of brain.py, in place
of the only function-level import in src. **W4-R2** (`a44534e`): `loop._attempts` (22 lines) is
the one retry loop of `_run_tool` and `_call_brain`. Every event order is unchanged, and every
pinned sequence passes unedited. One stored text changes: a failing attempt's `ErrorEvent.detail`
traceback now names `_attempts` (and `attempt` for the brain) where it named `_run_tool` or
`_call_brain`. The error, the exception line and the raw bytes are the same, and no test or
invariant covers frame names. **W4-R10** (`89ed4a0`): `httpx` moves to the dev group at 0.28.1 (no
module in src imports it; two tests do). uv.lock changes only in `rexhunter`'s own entries (2
insertions, 2 deletions, 39 packages, no version moved), resolved with `uv lock --offline`.
Dropped: **W4-R6**, as hub.py:31-32 raises on an empty env value at import while server.py's
`_env_int` and `_env_float` map it to the default, so one shared parser would change one of them;
**W4-R8**, as pyright 1.1.414's typeshed gives `frozendict` an `__init__(self)` with no other
parameter, so every non-empty `frozendict(...)` is an error under strict (7 errors on 5 forms). No
constant can change type without a `# type: ignore` or a pyright bump, which is Darie's. Mutants,
each on a `git archive` of its commit (the repo untouched):
- W4-MW6b (`utcnow` returns None): red through NOT NULL;
- W4-MW7b (the cleanup unshielded): 1 red, tests/test_scheduler.py:272;
- W4-MW8 (the opt-in dropped): 4 red, tests/test_smoke_offline.py:419 and the three refusal tests;
- W4-MW10 (the key check dropped): 3 red, tests/test_paid_entrypoint_gates.py:62;
- W4-MW9 (one retry fewer): 61 red.

Two mutants found gaps. W4-MW6 (`utcnow` returns "") turned nothing red: no test pins a
timestamp's value or format. W4-MW7 (a re-cancel re-raised) turned nothing red: the shield keeps the
write running, and the second-cancel test reads the outcome after it has landed. src grows from
4893 to 4905 lines: loop.py 632 → 647, brain.py 1277 → 1293, db.py 261 → 276, verdicts.py
252 → 242, smoke.py 229 → 221, hunt_smoke.py 252 → 244, hunt_smoke_thinking.py 201 → 193. Test
edits: none. 650 tests pass at `89ed4a0` (644 at 71d31ad). Wire: 1 `bb2ef203…c24d` 1487, 2
`6b9e9a17…98a8` 1519, 3 `3f01481b…c293` 1469, 4 `08e8682c…fd31` 1898, 5 = 2. No subagent, workflow
or advisor call. Open: the two gaps above, W4-R6 and W4-R8 as dropped, and the W4a entry's list.
*Next: Darie's.*

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
  `tests/test_stage5_gate.py`. *Unit 2a (provider adapter, offline half) – ✅ done (`edd4ec8`):*
  `adapter_brain_for` over **httpx** (since W2.0: over the Anthropic SDK in raw mode) wraps
  `parse_decision` around the Anthropic Messages API, presents
  `sniff`/`hunt_complete`/`needs_help` as derived tool schemas (D1), rejects multiple
  `tool_use` blocks; injected transport, **no live call** (contract gate
  `tests/test_brain_adapter.py`). *Unit 2b (the first paid call) — ✅ done:* one live smoke on
  `claude-sonnet-5` through the 2a adapter (free `count_tokens` pre-flight → one `messages` call →
  persist raw bytes → `parse_decision`) landed and parsed to `ToolCallDecision(tool="sniff")`,
  captured as `tests/fixtures/smoke_sonnet5.json` (invariant 6, a real adaptive-thinking sibling
  block); Sonnet 5 guards (no sampling params / `thinking` / prefill; since W2.5 on Sonnet 5.5 too, as `build_request_body`'s docstring lists them) + `title`-stripped schemas;
  hard one-call guard; `smoke.py` entrypoint. Offline gates `tests/test_smoke_offline.py` +
  `tests/test_smoke_replay.py`. *Unit 3 (streaming `ThinkingDelta` relay + signed-block replay) — ✅
  done, replay-gated (`c70f2f7`):* hand-rolled SSE `StreamAssembler` (no vendor SDK; since W2.0 it
  reads the SDK's raw stream, still hand-rolled), the write-ahead `ThinkingDelta` relay through the
  P3 hub, and `project_messages` echoing the verbatim signed thinking
  block; one gated live hunt (`claude-sonnet-5`, thinking on, $0.0187) proved call two accepted with
  the reconstructed signed-block turn, captured + offline-replayed (`tests/test_thinking_hunt_replay.py`).
  *Later units:* extraction/strict mode, the live ATS (Greenhouse/Lever) adapter.
- **`P3` · Streaming hub — ✅ done.** Per-viewer broadcast queues + `Last-Event-ID` resume; the
  prototype poll is retired and `/events` streams from the real in-process hub. *Gate (green,
  `tests/test_stage3_gate.py`):* two viewers render byte-identical feeds; a viewer closed for a full
  hunt reconnects via `Last-Event-ID` with zero gaps + zero duplicates (snapshot + catch-up +
  live-splice, monotonic-id dedup); a full-queue viewer is dropped without slowing the hunt (ADR
  **DoD #3**). Offline hub unit (`tests/test_hub.py`) proves the write-ahead order structurally: an
  envelope reaches a viewer only after a separate reader sees the committed row (inv 1).

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
