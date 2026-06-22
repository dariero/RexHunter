# RexHunter🦖 – System Design Map & Architectural Decision Record

**Status:** Accepted · **Date:** 2026-06-11 · **Owner:** Darie
**Scope:** Local-first autonomous job-hunting agent ('terrarium' model) – background daemon, event-sourced state, retro-game frontend projection, human-in-the-loop verdicts.
**Build philosophy:** Hand-rolled primitives over frameworks where the primitive *is* the learning objective. Boring technology everywhere else. Every dependency must replace more than ~100 lines we would otherwise understand.

---

## Decision summary

| # | Decision | Chosen | Rejected |
|---|----------|--------|----------|
| 1 | Persistence & trajectory store | SQLite (WAL) monotonic event log | PostgreSQL, Kafka/Redis Streams, vector DBs, JSONL files |
| 2 | Agent loop & tool harness | Hand-rolled async loop + decorator-derived tool registry | LangChain, LangGraph, CrewAI, PydanticAI |
| 3 | Real-time UI transport | SSE over write-ahead log + in-process broadcast hub | WebSockets, long-polling, short polling |
| 4 | Human-in-the-loop pause | Durable table state (`awaiting_verdict`) + park-and-persist hybrid | RAM-held `asyncio.Future`, Temporal, Celery |
| 5 | LLM integration | Native tool calling + constrained decoding, Pydantic validation at the network edge | Prompt-begging for JSON, regex extraction, framework output parsers |

## System invariants (cross-cutting laws)

These bind every pillar. A change that violates one of these is an architecture change, not a refactor, and requires updating this ADR.

1. **Write-ahead rule.** Every event is committed to the SQLite log *before* it is published to the broadcast hub. The log is truth; the stream is notification. Any consumer that misses a publish recovers from the log.
2. **Projection law.** All state shown anywhere (UI sprites, HP bars, territory status, prey pen) is derived from the event log or from tables transactionally maintained alongside it. Nothing in the frontend is authoritative. The live renderer and the ghost-run replayer are the same renderer fed by different cursors.
3. **Validate at the boundary, typed everywhere inside.** Raw bytes (HTTP responses, LLM output, scraped HTML) cross exactly one Pydantic validation boundary on entry. From that line inward, only typed objects exist. LLM output is untrusted input and is treated with the same suspicion as scraped HTML.
4. **Tiny Arms consent law.** RexHunter cannot perform any externally visible side effect (submitting, sending, applying) – structurally, not by convention. There is no tool registered that can do it. Human approval is a database state transition, not a UI confirmation dialog.
5. **Derive, don't store.** Aggregate state (HP, run statistics, day/night phase) is computed by folding over events or from the clock, never stored as a second mutable copy that can drift from the log.
6. **Raw I/O snapshots.** Every tool event payload carries the raw request and raw response bytes it operated on. Consequences: ghost replays need no network, and every dead run is automatically a pytest regression fixture.
7. **Single writer per run.** Only the hunt task that owns a run appends to that run's events. Readers are unlimited (WAL mode guarantees this is safe).

---

## Pillar 1 – The Monotonic Event Log & Persistence

### The Architecture (The What)

Two tables form the entire persistence core:

```sql
CREATE TABLE runs (
    id           TEXT PRIMARY KEY,            -- uuid4
    territory    TEXT NOT NULL,
    started_at   TEXT NOT NULL,               -- ISO 8601 UTC
    ended_at     TEXT,
    outcome      TEXT,                        -- 'completed' | 'aborted' | 'crashed' | NULL while live
    abort_reason TEXT
);

CREATE TABLE trajectory_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,  -- global monotonic stream cursor
    run_id     TEXT NOT NULL REFERENCES runs(id),
    seq        INTEGER NOT NULL,                   -- per-run replay cursor
    type       TEXT NOT NULL,                      -- discriminator of the Pydantic event union
    payload    TEXT NOT NULL,                      -- JSON-serialised typed event
    created_at TEXT NOT NULL,
    UNIQUE (run_id, seq)
);
CREATE INDEX idx_events_run ON trajectory_events (run_id, seq);
```

The **dual-cursor design** is deliberate. The global `id` orders events across all runs and is the SSE stream cursor (`Last-Event-ID`, `WHERE id > :last_seen`). The per-run `seq` orders events within one hunt and is the ghost-replay cursor (`WHERE run_id = :run ORDER BY seq`). Two consumers, two orderings, one table.

Events are **append-only and immutable**. Payloads are a Pydantic discriminated union (`PlanEvent | ToolCallEvent | ToolResultEvent | ThinkingDelta | DamageEvent | ErrorEvent | ...`) serialised to JSON; the `type` column mirrors the discriminator for SQL-side filtering. Exceptions are not special – they are events whose payload contains a traceback.

Operational settings, applied once at startup: `PRAGMA journal_mode=WAL` (readers never block the single writer – the SSE feed queries while a hunt writes) and `PRAGMA busy_timeout=5000` (brief contention waits instead of throwing). Driver: `aiosqlite`.

Derived stores (the found-jobs table, dedupe hashes, prey pen status) live in adjacent tables maintained transactionally with their originating events. They are queryable conveniences; the log remains the source of truth, and any derived table can be rebuilt by replaying the log.

### The Justification (The Why)

One data structure serves four consumers identified during design: the live UI feed, reconnect resynchronisation, the agent's own working-memory context, and the ghost-run audit trail. Event sourcing is not adopted as fashion – it is the minimal structure that serves all four without duplication.

Durability before intelligence is a wallet decision: once `brain()` is paid API calls, a crash at step 9 of a 10-step hunt must not force re-paying for steps 1–8. Durability buys the receipt; payload completeness (invariant 6) buys *resume* and *replay*.

SQLite specifically: zero operational surface (no container, no port, no credentials), a single file whose backup strategy is `cp`, transactional appends measured in microseconds, and concurrency semantics (one writer, many readers under WAL) that exactly match the system's shape. For a single-process, single-user, local-first daemon, SQLite is not the compromise option – it is the correct option.

### The Counter-Options (The Alternatives)

**PostgreSQL.** The default 'serious' choice: true multi-writer concurrency, network access, richer types, mature migration tooling.

**Kafka / Redpanda / Redis Streams.** Purpose-built append-only logs with native consumer-offset semantics, replay, and fan-out.

**Vector databases (Qdrant, Chroma, pgvector) as trajectory store.** Store events as embedded documents; retrieve by similarity.

**JSONL flat files.** One file per run, one JSON object per line. Maximum simplicity.

### The Rejection (Why Not Chosen)

**PostgreSQL** adds a network hop, a running service, connection pooling, and migration ceremony to buy multi-process write concurrency the system does not have – there is exactly one writer process by design (invariant 7). The honest swap trigger: if RexHunter ever becomes multi-process (separate worker pool) or needs remote access, Postgres wins, and the repository layer exists so that swap touches one module. Until that trigger fires, Postgres is operational weight with no payoff.

**Kafka-class systems** solve fan-out and offsets at a scale of thousands of consumers across machines. The prototype baseline demonstrated that a consumer offset is an integer (`sent`) and is now a `WHERE id > ?` clause. Importing a distributed log to replace one indexed column is the definition of résumé-driven architecture.

**Vector DBs as the trajectory store** misunderstand the queries. Trajectory access is ordered and exact ('events of run X in sequence', 'everything after id 412') – relational queries. Embeddings answer similarity questions, which trajectories never ask. A vector index may *later* join the system as a derived index over job postings (semantic dedupe, match scoring) – derived, rebuildable, never authoritative.

**JSONL files** lose transactional pairing with the `runs` table, indexed cursor queries, and safe concurrent reads during writes. They are fine for export, wrong for the live store.

---

## Pillar 2 – The Hand-Rolled Loop & Tool Harness

### The Architecture (The What)

The agent loop is owned code, roughly:

```
while not done:
    context  = assemble_context(run_id)        # window over recent trajectory events
    decision = await brain(context)            # returns ToolCallDecision | HuntComplete | NeedsHelp
    match decision:
        case ToolCallDecision():
            args   = validate(decision)        # Pydantic boundary - invalid args never execute
            result = await execute(decision.tool, args)   # timeout + retry policy per tool
            append_event(ToolResultEvent(...))
        case HuntComplete():
            done = True
        case NeedsHelp():
            park_and_persist(...)              # see Pillar 4
```

Tools are plain async Python functions registered through a `@rex_tool` decorator that reads the Pydantic-typed signature and derives the JSON schema via `model_json_schema()`. One definition yields three artifacts: the schema sent to the LLM, the validator the arguments pass through, and the handler that executes. Schema drift between declaration and implementation is structurally impossible.

The harness owns an explicit error taxonomy: retryable (network blips, 429s, transient 5xx) versus fatal (validation failures, unknown tools, budget exhaustion), with per-tool timeout and retry budgets. Every transition – plan, dispatch, result, retry, damage, abort – is appended as a typed event before anything else happens (invariant 1). Concurrent hunts run inside a bounded task group; a hung tool is cancelled at its deadline and recorded, never silently awaited forever.

### The Justification (The Why)

Two reasons, one practical and one strategic.

Practical: **the loop is the observability surface.** Because every state transition is code we wrote, emitting the trajectory event is one line at the site of the transition – not an instrumentation callback bolted onto someone else's control flow. The terrarium's hunt log, the ghost replays, and the agent's context window all consume the same events the loop naturally produces. Debugging means reading ~150 lines of our own Python.

Strategic: this project exists to convert evaluation expertise into demonstrated core engineering. 'I hand-rolled the agent loop, the tool contract enforcement, and the trajectory store' is the portfolio claim; delegating those three things to a framework deletes the claim.

### The Counter-Options (The Alternatives)

**LangChain (AgentExecutor / AgentRunnable).** The incumbent; enormous integration catalogue, chains, callbacks.

**LangGraph.** Graph-structured agent runtime with checkpointing, interrupts, and durable state – genuinely well-designed.

**CrewAI.** Role-based multi-agent orchestration ('researcher agent', 'writer agent') with built-in delegation patterns.

**PydanticAI.** Typed-first agent framework from the Pydantic team; philosophically the closest neighbour.

### The Rejection (Why Not Chosen)

**LangChain** inverts the abstraction this system needs. Its control flow is opaque by default; recovering 'what exactly was sent to the model and what came back' requires the callback/tracing system – instrumenting from outside what an owned loop emits from inside. Prompt assembly is hidden behind chain composition, which is precisely the surface a hand-rolled system must own. Validation failures surface as framework exceptions distant from the Pydantic boundary that should have caught them. The historical API churn is a secondary cost; the primary cost is observability by archaeology.

**LangGraph deserves an honest rejection, because it is good.** Its checkpointer and `interrupt()` solve real problems – specifically the problems Pillars 1 and 4 solve. That is exactly why it is rejected: adopting it would outsource the durable-state and pause machinery this project exists to teach, and its checkpoint store would become a second source of truth competing with the event log, violating the projection law. The unvarnished version: at a company, with deadlines, LangGraph would be a defensible choice. For this build, it amputates the learning objective. Knowing LangGraph's design well enough to explain why we rebuilt its core by hand is the interview asset.

**CrewAI** operates at the wrong abstraction level entirely: role-play prompt scaffolding over a loop, with weak typing at the boundaries and 'collaboration' implemented as prompt concatenation. RexHunter is one agent with strict contracts, not a troupe.

**PydanticAI** is rejected with respect rather than criticism – it embodies the same validate-at-boundary philosophy. But its agent loop and tool registry are the ~150 lines whose construction is the point. Post-build, reimplementing one hunt in PydanticAI as a comparison benchmark is a worthwhile exercise, not a threat.

---

## Pillar 3 – Real-Time Consciousness Streaming

### The Architecture (The What)

The pipeline: **hunt task → SQLite append (write-ahead) → in-process broadcast hub → SSE endpoint → browser projection.**

The broadcast hub is a small object holding one bounded `asyncio.Queue` per connected viewer. After the database commit returns the new global event id, the event envelope is offered to every viewer queue. The hub is *allowed to be lossy* precisely because of the write-ahead rule – correctness lives in the log.

Tab connect/reconnect is **snapshot + catch-up + live splice**: the client fetches a snapshot endpoint (territory states, prey pen, open runs, latest event id), renders instantly, then opens the SSE stream sending that id as `Last-Event-ID`. The server replays `WHERE id > :last_seen` from the log, then splices into the live queue. Monotonic ids make deduplication trivial – the renderer ignores anything ≤ its high-water mark. The browser never sees a gap and never double-applies.

Backpressure policy: **drop-and-resync.** If a viewer's bounded queue overflows (backgrounded tab, sleeping laptop), the hub drops that viewer entirely; the browser's built-in `EventSource` reconnect triggers the snapshot dance. A slow viewer can never block or slow the hunter. Heartbeat comments are emitted every ~15s to keep intermediaries from killing idle connections.

Upstream traffic (verdicts, territory config) is plain HTTP POST – request/response semantics with status codes and idempotency, exactly what actions want.

### The Justification (The Why)

The data flow is fundamentally asymmetric: a continuous firehose downstream, occasional discrete commands upstream. SSE matches this shape natively and brings three free gifts: it is plain HTTP (debuggable with `curl`, friendly to every proxy), the browser auto-reconnects without any client code, and `Last-Event-ID` is a *protocol-level* resume cursor – the resync mechanism is a standard, not an invention. The snapshot/catch-up/live-splice pattern is the same one used by collaborative editors and market-data feeds; implementing it over SSE + SQLite is the smallest honest version of a pattern with real systems pedigree.

### The Counter-Options (The Alternatives)

**WebSockets.** Full-duplex persistent connection; the reflex choice for 'real-time'.

**Long-polling.** Client holds a request open until data arrives, then immediately re-requests.

**Short polling.** Client asks 'anything new?' on an interval. (The prototype's internal generator effectively did this against a list.)

### The Rejection (Why Not Chosen)

**WebSockets** purchase bidirectional streaming the system does not use – verdicts are rare, discrete, and better served by POST (status codes, retries, idempotency keys come free; a WS message gets none of these without inventing a protocol). The hidden cost is the reconnect-and-resume story: WS has no `Last-Event-ID`, so gap recovery after a dropped connection means hand-rolling sequence acknowledgement – reimplementing SSE's standard feature, badly, on a transport that is also harder to proxy and impossible to `curl`. Swap trigger: if the terrarium ever needs high-frequency client→server streaming (e.g. live joystick control of Rex), revisit. It will not.

**Long-polling** is SSE with extra connection churn and none of the cursor semantics – strictly dominated here.

**Short polling** is acknowledged as the legitimate prototype fallback (it is how the prototype's feed generator watched the list), but at terrarium timescales it forces a bad trade between latency and wasted wakeups, and the UI's 'consciousness stream' feel dies at any polite polling interval.

**Accepted limit:** the in-process hub binds streaming to a single backend process. Horizontal scaling would require external pub/sub (e.g. Redis) between writer and SSE servers. The terrarium is single-user and single-process by design; this limit is accepted and documented rather than pre-solved.

---

## Pillar 4 – The Durable Pause & Human-in-the-Loop

### The Architecture (The What)

Hunts **run to completion** – there is no suspended coroutine waiting days for a click. Captured prey is written to the pen as rows with `status = 'awaiting_verdict'`; the hunt logs `HuntCompleted` and exits cleanly. Rex sitting at the gate is not a paused task – it is *rows in a table with a pending status*.

The verdict is a state machine on the prey row:

```
awaiting_verdict --FEAST--> feasted    (enqueues draft_pitch job; pitch lands as draft for human editing)
awaiting_verdict --RELEASE--> released (rejection reason recorded - labelled data for future scorer tuning)
awaiting_verdict --AMBER--> ambered    (shelved with provenance; can return to awaiting_verdict)
```

Each verdict POST is an idempotent transition: a status guard (`UPDATE ... WHERE status = 'awaiting_verdict'`) makes double-clicks and replayed requests harmless. Follow-up work (pitch drafting) is enqueued as a job row, picked up by the same background machinery as hunts – the verdict handler itself does no LLM work.

For genuinely mid-run interventions with short horizons (Cracked Earth schema-drift emergencies), the **park-and-persist hybrid**: the run first appends a durable `awaiting_intervention` event, *then* awaits an in-memory `asyncio.Future` with a timeout. A click within the window resumes hot from RAM; a timeout or process death closes the run at the durable checkpoint, and the territory shows cracked earth until attended. Fast path in memory, truth on disk.

The Tiny Arms law (invariant 4) completes the design: no registered tool can submit, send, or apply. The human verdict is not a confirmation dialog over an action the agent could take – it is the only path by which certain state transitions can occur at all.

### The Justification (The Why)

A verdict can take three days; the process will restart dozens of times in that window (deploys, reboots, Ctrl-C during development). Any design holding workflow state in RAM dies at the first restart. The durable-state design has **crash-equivalence**: a restart is indistinguishable from a moment of inactivity, because all state that matters was already in the database. This is the central insight of durable-execution systems (Temporal, Restate) – 'pause' is not a property of a coroutine; it is a workflow state at a persistence boundary – implemented here at hand scale. Asking 'how do I hold an async task in mid-air' was the wrong question; 'where is the durable state boundary' is the right one, and answering it by hand is the senior-engineer move this pillar exists to practise.

The reason-coded RELEASE verdict is a quiet second payoff: every rejection is a labelled preference example, accumulating training data for match-scorer tuning as a side effect of normal use.

### The Counter-Options (The Alternatives)

**RAM-held `asyncio.Future` / `Event`.** The textbook pattern: park the coroutine, resolve the future from the POST handler.

**Temporal (or Restate / durable-execution engines).** Workflow-as-code with transparent persistence of execution state; `await human_signal()` survives restarts natively.

**Celery + result backend.** Classic task queue with state in Redis/DB.

**LangGraph `interrupt()` + checkpointer.** Framework-native HITL pause.

### The Rejection (Why Not Chosen)

**Pure Futures** fail on horizon: they evaporate on restart, leak on abandoned verdicts, and cannot survive a deploy. The pattern is retained *only* for the short-horizon hot path inside park-and-persist, where the durable checkpoint underneath makes the in-memory loss harmless. Knowing the pattern and knowing its blast radius are both required.

**Temporal** is the correct answer at company scale and the rejection is purely contextual: it requires an external cluster (or cloud dependency) to run a single-user terrarium, and – the recurring theme – it abstracts exactly the durable-boundary thinking this pillar exists to internalise. Building the hand version is what makes Temporal's value articulable later.

**Celery** models 'run this task', not 'this workflow is waiting on a human'; pause semantics would be bolted onto a tool designed for fire-and-forget execution, with workflow state smeared between broker, backend, and application tables.

**LangGraph interrupts** are rejected by inheritance from Pillar 2: adopting the checkpointer for pauses creates the second source of truth the projection law forbids.

---

## Pillar 5 – The Brain Socket & Modern LLM Integration

### The Architecture (The What)

`brain()` sits behind a small provider-agnostic protocol; the loop never imports a vendor SDK directly. Per call:

1. **Tools** are presented as JSON schemas *generated* from `@rex_tool` signatures (`model_json_schema()`) – never hand-written. The schema the model sees, the validator the arguments pass, and the executing function are one definition.
2. **Native tool calling**: the provider returns a typed `tool_use` block (name, structured arguments, `tool_use_id`). No text parsing exists anywhere in the path. The `tool_use_id` is stored on both `ToolCallEvent` and `ToolResultEvent` – the provider's own correlation key doubles as the log's pairing key.
3. **The edge boundary**: the raw provider response is validated into the decision union (`ToolCallDecision | HuntComplete | NeedsHelp`) at one line. Validation failure is a typed `ErrorEvent` with the raw payload attached (invariant 6) – diagnosable from the ghost replay, never a mystery string.
4. **Structured outputs / constrained decoding** for extraction tasks (`extract_requirements → JobRequirements`): where supported, the provider's constrained decoding makes schema-violating output unrepresentable, converting 'parse and pray' into 'cannot be wrong by construction'. Where unsupported, the fallback is validate-and-retry with a hard retry budget – never regex.
5. **Streaming**: provider deltas arrive over SSE (the same protocol the terrarium speaks to the browser – the system is an SSE relay end to end). Thinking deltas are appended as `ThinkingDelta` events – write-ahead, then broadcast – rendering Rex's actual chain of thought as the live hunt feed. **Display streams; execution waits**: partial tool-call JSON is rendered for suspense but a tool executes only when its block is complete and validated.
6. **Budget guards**: token and cost accounting are recorded per event; runs carry a max-iteration circuit breaker and a cost ceiling, both enforced by the loop, both visible as the HP/stamina mechanics in the terrarium.

### The Justification (The Why)

The governing principle: **LLM output is untrusted input.** It is probabilistic text from a network service and deserves exactly the suspicion given to scraped HTML – which means validation at the network edge, with everything inside the boundary statically typed. Native tool calling and constrained decoding move correctness from 'prompt persuasion' (probabilistic) to 'protocol guarantee' (structural). Failures then surface at one known line with the raw payload attached, instead of as regex mismatches three functions deep. The thinking-token relay is the same justification wearing the game's costume: the hunt log stops being flavour text *about* the brain and becomes a window *into* it, at zero additional architecture because the event pipeline already existed.

### The Counter-Options (The Alternatives)

**Prompt-begging + regex.** 'Respond ONLY in JSON' in the system prompt; extract with `re.search`; retry on failure.

**Instructor-style validate-and-retry loops.** Pydantic validation with automatic re-prompting on failure – philosophically aligned with this design.

**Framework output parsers (LangChain `OutputParser` family).** Parsing and fixing logic encapsulated in chain components.

**Raw JSON mode without schema.** Provider flags guaranteeing syntactic JSON, with shape unenforced.

### The Rejection (Why Not Chosen)

**Prompt-begging + regex** is probabilistic correctness: it degrades silently under model updates, partial matches produce plausible-but-wrong extractions (the worst failure class – wrong data that looks right), every retry burns paid tokens, and free-text parsing is an injection surface (a job posting containing instruction-shaped text gets interpolated into prompts whose output is then trusted). In a system whose inputs include arbitrary text from the public internet, the parsing layer must be a wall, not a sieve.

**Instructor** is rejected gently – it *is* validate-at-boundary, and was the right pattern before native support matured. Native constrained decoding supersedes its retry loop wherever available (guaranteed beats retried); where it isn't, our fallback reimplements instructor's core in a few owned lines rather than adopting a dependency for them.

**Framework output parsers** inherit Pillar 2's rejection: they relocate the validation boundary into opaque framework internals, where failures surface as framework exceptions far from the edge.

**Raw JSON mode** guarantees validity, not conformity – syntactically perfect JSON in the wrong shape still crashes the typed core. Schema-bound modes exist; use them.

---

## Known accepted limits & swap triggers

| Limit | Accepted because | Revisit when |
|---|---|---|
| Single process, in-process hub | Single-user terrarium | Multiple backend processes or remote viewers |
| SQLite single-writer | One hunt scheduler owns all writes | Worker-pool architecture → PostgreSQL |
| Lossy broadcast hub | Log is truth (invariant 1) | Never – this is the design, not a debt |
| No auth on the local UI | localhost-only daemon | Any network exposure beyond localhost |
| Live adapters limited to public ATS APIs (Greenhouse/Lever) | ToS-compliant, stable contracts | Never scrape ToS-prohibited boards |

## Build sequence (canonical)

The one place the build order lives. Slices are **ordered top-to-bottom, earliest first**; the
next slice to build is the earliest one not yet marked done in the projections. Each slice is
identified **only by its slice-ID** — `P<pillar>`, pillar-keyed so the integer is stable
architectural identity that is never renumbered — **and its name.**

**Naming rule (binds the whole scheme):** *never refer to a slice by a build-order number.* A
bare position like "slice 3" collides with `P3` the instant the prefix is dropped — which is how
the original three-way numbering drift began. Refer to a slice by its slice-ID, or relationally
("the slice after `P2.2`"). This list therefore carries **no order integers** — order is row
position, not a number.

This list is **status-free**: it is the stable plan, and by the projection law (invariant 2,
applied to the docs) status is never stored here, only derived downstream. Live status lives in
its two projections — `CLAUDE.md` (current working position) and `README.md` (public
done/planned) — both of which reference these slice-IDs and never restate the order as integers.

| Slice | Pillar | Gate |
|-------|--------|------|
| Prototype baseline — FastAPI lifespan + loop + SSE + in-memory list | — | — |
| **`P1` · Persistence** — SQLite WAL event log (`runs`, `trajectory_events`) | 1 | DoD #1 |
| **`P2.1` · Typed events** — discriminated-union model + validation boundary | 2 | `tests/test_events.py` (sub-gate of DoD #2) |
| **`P2.2` · Loop & tool harness** — agent loop + `@rex_tool` registry + error taxonomy | 2 | DoD #2 |
| **`P2.3` · Hunt scheduler** — concurrent-hunt orchestration + per-territory deadlines | 2 | DoD #2 under concurrency (invariant 7) |
| **`P4` · Durable pause & HITL** — `awaiting_verdict`, Feast/Release/Amber machine | 4 | DoD #4 |
| **`P5` · Brain socket** — provider-agnostic LLM, native tool calling (paid) | 5 | DoD #5 |
| **`P3` · Streaming hub** — per-viewer broadcast queues, `Last-Event-ID` resume | 3 | DoD #3 |

Order is **dependency order, not pillar order.** `P3` (Streaming) sits last because the
prototype's naive SSE feed covers it for now and the real broadcast hub is deferred; its pillar
identity (3) is fixed regardless of build position. The gate column points at the per-pillar
Definition-of-done below — and `DoD #N` is itself pillar-keyed (`P3` → `DoD #3`), so a gate can
never drift from its slice.

**`P2.3` bundles two concerns** under one slice: (a) **concurrent-hunt orchestration** — the
bounded task group, squarely Pillar 2 and the concrete enforcement of invariant 7; and (b)
**per-territory deadline timing** — operational scheduling with no native pillar, parked in
Pillar 2. It is a **candidate to decompose into two slices when built**; kept as one until the
work proves the seam.

## Definition of done per pillar (test-first)

1. **Log:** `kill -9` mid-hunt → restart → every pre-kill event queryable; dangling `outcome IS NULL` runs detected and marked `'crashed'` at boot.
2. **Loop:** a tool that raises, a tool that hangs past timeout, and an LLM returning an unknown tool name each produce typed events and a clean run outcome – never an unhandled exception escaping the loop.
3. **Streaming:** two tabs render identical feeds; a tab closed for a full hunt reopens with zero gaps and zero duplicates; a viewer with a full queue is dropped without slowing the hunt.
4. **Pause:** restart the process with prey in the pen → verdicts still work; double-clicking FEAST is a no-op; park-and-persist resumes hot within the window and checkpoints cleanly past it.
5. **Brain:** replaying a recorded run's raw payloads through the parsers requires no network and reproduces identical events; a malformed provider response becomes an `ErrorEvent` carrying the raw payload, with the run ending in a typed outcome.

---

*This document is the North Star. Deviations are allowed – silently undocumented deviations are not. When reality teaches us something this spec got wrong, the spec gets the update and the commit message gets the story.* 🦖
