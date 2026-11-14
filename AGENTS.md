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
- **Transport:** SSE downstream (`EventSource`, `Last-Event-ID` resume); plain HTTP POST
  upstream for verdicts/config.
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
  second spelling of a pinned name included; src/ never copies the client. Native tool calling,
  constrained decoding where available, `ThinkingDelta` event relay, per-event cost accounting.
  Each `ToolCallEvent` the live brain writes stores the whole assistant turn (`assistant_content`,
  W2.3), and the next call sends it back as stored, every block in order.
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
(429/5xx/timeouts → retryable; since W2.0 the adapter maps SDK/transport errors to a provider-neutral
`BrainCallError` that `classify` honours, and loop.py imports no httpx), and
`select_brain_for`/`REXHUNTER_BRAIN` gate the spender surface (stub default, opt-in `live`). The live
loop path sends `thinking: {type: "disabled"}` + `tool_choice: {disable_parallel_tool_use: true}` –
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
`{type:"disabled"}` — P3 built its render target, does not consume it.
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
  block); Sonnet 5 guards (no sampling params / `thinking` / prefill) + `title`-stripped schemas;
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
