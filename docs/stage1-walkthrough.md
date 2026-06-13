# Stage 1 walkthrough – the trajectory log, from zero 🦖

A guided tour of `src/rexhunter/db.py` and the persistence half of
`src/rexhunter/server.py`, written for someone new to event sourcing. Every terminal
snippet below is real output from a live demo run on 2026-06-13 – nothing is mocked or
hand-typed.

---

## 1. The mental model

Forget code for a minute.

**An event log is a notebook you only ever add lines to.** You never erase a line, never
edit a line, never reorder lines. Each line says *one thing that happened*, in the order
it happened: "Rex started a hunt", "Rex sniffed and found an AI Engineer posting", "Rex
took damage from a broken page". That's the whole data structure. An "event" is one such
line; "append-only" means new lines go at the bottom and old lines are untouchable.

The analogy that makes the invariants click is **a bank account**. Ask yourself: where
does your balance actually live? Not in some "balance" cell that gets overwritten – your
balance is *computed* from the list of every transaction you ever made. The transaction
history is the **truth**. The number in your banking app is a **projection**: a view
*derived* from the truth, recomputable at any time, and never trusted as a source itself.
If the app shows a weird number, nobody panics – they recompute it from the history.

That's exactly RexHunter's two laws:

- **Invariant 1 (write-ahead):** the transaction must be in the ledger *before* the app
  notifies you about it. The log is truth; the notification is a courtesy.
- **Invariant 2 (projection):** everything on screen – the hunt feed, HP bars, run
  status – is computed *from* the log. The screen is the banking app, never the ledger.

### What Stage 0 couldn't do

Stage 0 kept events in a Python list:

```python
events: list[str] = []          # Stage 0 – RAM only
```

A list lives in the process's memory (RAM). When the process dies – crash, `kill -9`,
reboot – RAM is simply gone. Every hunt Rex ever ran: gone. There is no history to
recompute anything from, no way for a browser that reconnects to ask "what did I miss?",
and – the wallet argument – once Stage 4 makes each hunt step a *paid* LLM call, a crash
at step 9 of 10 would force re-buying steps 1–8.

Stage 1 swaps the list for a **SQLite** database: a single ordinary file on disk that a
small, ferociously well-tested library reads and writes for us. Disk survives the process.
That one property is what this whole stage purchases.

---

## 2. The two tables

A SQL database stores data in **tables** – grids with named, typed **columns**, where each
record is a **row**. Stage 1 needs exactly two. Both are created by the `_SCHEMA` string
at the top of `db.py` (`src/rexhunter/db.py:13`), which `connect()` executes every boot –
`CREATE TABLE IF NOT EXISTS` means "create it the first time, silently skip every time
after", so booting is also bootstrapping.

### `runs` – one row per hunt

A *run* is one whole hunting trip: it starts, things happen, it ends somehow.

```sql
CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    territory    TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    ended_at     TEXT,
    outcome      TEXT,
    abort_reason TEXT
);
```

- **`id`** – the run's name, a UUID (a long random string like
  `c2685dd8-e66e-...`, practically guaranteed unique without any coordination).
  `PRIMARY KEY` = "this column identifies the row; no two rows may share a value".
- **`territory`** – where Rex hunted (today always `"mock-gym"`; later, per-board
  territories).
- **`started_at`** – timestamp, stored as ISO-8601 text (`2026-06-13T01:47:25+00:00`).
  `NOT NULL` = "this must be filled in; the database rejects a row without it".
- **`ended_at`**, **`outcome`**, **`abort_reason`** – deliberately *nullable* (allowed to
  be empty). **`NULL` – "no value" – is doing real work here:** a run with
  `outcome IS NULL` is, by definition, *still open*. There is no separate
  "is_running" flag to forget to update. The absence of an outcome IS the running state.
  `outcome` becomes `'completed'`, `'aborted'`, or `'crashed'`; you'll see all of this
  matter in the demo.

### `trajectory_events` – one row per thing that happened

```sql
CREATE TABLE IF NOT EXISTS trajectory_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL REFERENCES runs(id),
    seq        INTEGER NOT NULL,
    type       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, seq)
);
```

- **`id`** – a number the database hands out itself (`AUTOINCREMENT`): 1, 2, 3, … across
  *every* event of *every* run, forever increasing, never reused. Remember this column;
  section 3 is about it.
- **`run_id`** – which hunt this event belongs to. `REFERENCES runs(id)` is a **foreign
  key**: the database itself refuses an event pointing at a run that doesn't exist (we
  switch that enforcement on with `PRAGMA foreign_keys=ON` – a *pragma* is a SQLite
  configuration knob).
- **`seq`** – this event's step number *within its own run*: 0, 1, 2, … restarting for
  every run. The other half of section 3.
- **`type`** – what kind of event (`"sniff"` today; a Pydantic discriminator in Stage 2).
- **`payload`** – the event's content. Per the working contract this stays a plain string
  until Stage 2 gives it typed structure.
- **`created_at`** – when it happened.
- **`UNIQUE (run_id, seq)`** – a constraint: no run may have two events with the same
  step number. This turns invariant 7 ("only one writer per run") from a polite
  convention into something the database physically enforces.

The `CREATE INDEX` line builds an **index** – a lookup structure, like a book's index –
on `(run_id, seq)`, so "give me run X's events in order" is a direct lookup instead of a
scan of the whole table.

---

## 3. The dual cursor – one table, two orderings

This is the design decision most worth owning, so slowly.

First, a word on the word: **cursor** here means *a reader's bookmark* – "I have read up
to here". (Python's database API also has a `cursor` object for fetching query results –
same word, unrelated meaning. The bookmark sense is what matters in this section.)

Every event row carries **two numbers**: the global `id` and the per-run `seq`. Same
table, two orderings, because there are **two different readers asking two different
questions**:

| Cursor | Question it answers | Who reads it |
|---|---|---|
| global `id` | "what happened *anywhere* since I last looked?" | the live SSE feed |
| per-run `seq` | "replay hunt X to me, step by step, in order" | the ghost replayer |

**The analogy: sports television.** The global `id` is the *live ticker* running across
the bottom of the screen – every goal from every match in the league, interleaved, in
wall-clock order. If you look away and come back, you say "show me everything after item
412" – that's the ticker's bookmark. The per-run `seq` is the *match replay* – when you
rewatch one match, you don't want other matches' goals spliced in; you want minute 1,
minute 2, minute 3 *of that match*. Two numbering systems over the same set of goals,
because watching-everything-live and rewatching-one-match are different jobs.

In RexHunter:

- The **SSE feed** (live ticker) runs `WHERE id > :last_seen ORDER BY id`
  (`src/rexhunter/server.py:74`). Its bookmark even survives a dropped browser
  connection: each SSE message carries `id: <global id>`, the browser remembers the last
  one, and on reconnect sends it back in a `Last-Event-ID` header. Resuming is a
  protocol feature, not code we invented.
- The **ghost replayer** (match replay, coming with the UI) runs
  `WHERE run_id = :run ORDER BY seq` – one hunt, in step order, no other runs bleeding in.

Here is the real demo database showing both numbers at once – global `id` 1–5 spans two
runs, while `seq` restarts at 0 when the second run begins:

```
id  run       seq  payload
--  --------  ---  ----------------------------------------------
1   c2685dd8  0    Rex sniffs the air... fresh AI Engineer scent!
2   c2685dd8  1    Rex sniffs the air... fresh AI Engineer scent!
3   c2685dd8  2    Rex sniffs the air... fresh Eval Engineer scen
4   c2685dd8  3    Rex sniffs the air... fresh ML Platform Eng sc
5   2a7f35c4  0    Rex sniffs the air... fresh AI Engineer scent!
```

"Couldn't you derive `seq` by filtering on `run_id` and sorting by `id`?" You could – but
`seq` earns its column three ways:

1. **A gapless promise.** Global `id`s *within one run* have holes wherever other runs
   interleaved. `seq` promises an unbroken 0, 1, 2, … – so a replayer can *assert*
   completeness ("I have steps 0–41, nothing is missing") instead of hoping.
2. **An enforced invariant.** `UNIQUE (run_id, seq)` makes a second concurrent writer to
   the same run a database error, not a silent corruption.
3. **A cheap query.** The `(run_id, seq)` index serves replay directly.

How does `seq` get assigned without a race? Look at `append_event`
(`src/rexhunter/db.py:63`):

```python
cursor = await conn.execute(
    """
    INSERT INTO trajectory_events (run_id, seq, type, payload, created_at)
    SELECT ?, COALESCE(MAX(seq) + 1, 0), ?, ?, ?
    FROM trajectory_events WHERE run_id = ?
    """,
    (run_id, type, payload, _utcnow(), run_id),
)
```

A few things to unpack, SQL and Python both:

- **`INSERT … SELECT`** computes the values *inside* the insert: "find this run's highest
  `seq`, add 1, use that". One statement = atomic: there is no gap between "read the max"
  and "write max+1" where another writer could sneak in.
- **`COALESCE(x, 0)`** means "if x is NULL, use 0". A run's first event has no previous
  max (`MAX(seq)` over zero rows is NULL), so it becomes seq 0.
- **The `?` placeholders** let SQLite substitute values safely – never build SQL with
  f-strings; a payload containing a quote character would break (or hijack) the query.
- **`*` in the Python signature** (`async def append_event(conn, run_id, *, type, payload)`)
  makes everything after it keyword-only: callers must write `type="sniff",
  payload="…"` – impossible to swap them by accident, since both are strings.
- **`cursor.lastrowid`** is how SQLite tells us which global `id` it just assigned – the
  function returns it so a future caller (the broadcast hub) can publish it.

---

## 4. Write-ahead in the code

Invariant 1 says: **commit to the log first, tell the world second.**

A **commit** is the database's "make it permanent" moment. Writes inside a transaction
are provisional – invisible to every other connection and discarded on a crash – until
`commit()` returns; after it returns, the data is on disk and will survive `kill -9`.
Here is the exact line, in `append_event` (`src/rexhunter/db.py:71`):

```python
    cursor = await conn.execute(...)   # the INSERT - provisional
    await conn.commit()                # ← the event becomes durable, HERE
    event_id = cursor.lastrowid
    ...
    return event_id                    # only now can anyone be told about it
```

The function does not return until the commit has happened. So everything downstream of
"the event exists" – the returned `event_id`, any future publish to the broadcast hub –
is *structurally* after durability.

And where is the "publish to the stream" that must come second? In Stage 1, the neat
answer is: **there is no publish step to get wrong yet.** The SSE feed doesn't get
handed events – it *reads the log* (`src/rexhunter/server.py:73-77`):

```python
async with reader.execute(
    "SELECT id, payload FROM trajectory_events WHERE id > ? ORDER BY id",
    (last_seen,),
) as cursor:
    rows = list(await cursor.fetchall())
```

SQLite guarantees other connections only ever see *committed* data. A viewer literally
cannot observe an event before it's durable, because the only window is the log itself.
When the in-process hub arrives in a later stage, the rule becomes a real ordering of two
calls (`commit()` *then* `hub.publish()`), but the contract is set now.

**Why is the order non-negotiable?** Imagine the reverse: publish first, crash before
commit. A viewer's browser now displays event #43. The log has no event #43 – it died
provisional. Their reconnect politely asks "everything after 43" and the server can't
even represent the question; the screen has shown something the truth cannot back; the
account app displayed a transaction the bank has no record of. Invariant 2 (everything
derives from the log) collapses, because something on screen *didn't*. Notice the
asymmetry the design accepts instead: the stream is allowed to **miss** events (a lossy
hub is fine – you recover by re-reading the log) but is never allowed to **invent** them.
Promise less, guarantee it absolutely.

One naming collision to defuse: SQLite's `PRAGMA journal_mode=WAL`
(`src/rexhunter/db.py:43`) puts the *file format* in "Write-Ahead Logging" mode – an
internal SQLite journal that lets readers read while the one writer writes. That is
SQLite applying the same instinct to bytes-on-disk that invariant 1 applies to events.
Same philosophy, two different layers; `busy_timeout=5000` is its sidekick ("if the file
is briefly locked, wait up to 5s instead of throwing").

---

## 5. Live demo – the crash story

Everything below is a real session against a throwaway database
(`REXHUNTER_DB=$TMP/rex.db`), exactly as run. The daemon command:

```bash
REXHUNTER_DB="$TMP/rex.db" uv run uvicorn --app-dir src rexhunter.server:app --port 8768
```

### Act 1 – a hunt writes events, and we catch them on disk

Boot the daemon, wait ~12 seconds (Rex sniffs every 5), and peek at the live stream:

```
$ curl -s --max-time 2 http://127.0.0.1:8768/events
id: 1
data: Rex sniffs the air... fresh AI Engineer scent!

id: 2
data: Rex sniffs the air... fresh AI Engineer scent!
```

That `id:` line is the global cursor riding the SSE protocol – the browser's bookmark
for free. Now the part Stage 0 could never do: open the *file* with the `sqlite3`
command-line tool, while the daemon is still running, and see the same events as rows on
disk:

```
$ sqlite3 -header -column "$TMP/rex.db" "SELECT id, territory, outcome, started_at FROM runs;"
id                                    territory  outcome  started_at
------------------------------------  ---------  -------  --------------------------------
c2685dd8-e66e-42af-b16d-e48f845370c3  mock-gym            2026-06-13T01:47:25.412163+00:00

$ sqlite3 -header -column "$TMP/rex.db" "SELECT id, seq, type, payload FROM trajectory_events;"
id  seq  type   payload
--  ---  -----  ----------------------------------------------
1   0    sniff  Rex sniffs the air... fresh AI Engineer scent!
2   1    sniff  Rex sniffs the air... fresh AI Engineer scent!
```

Two readers (the SSE feed and our sqlite3 shell) reading while the writer writes – that's
WAL mode earning its pragma. And note the run's `outcome` column: empty. **NULL outcome =
hunt still in progress.** Remember it; it's about to matter.

### Act 2 – `kill -9`, the worst-case death

`kill -9` (SIGKILL) is the operating system removing the process mid-instruction. No
cleanup code runs, no "saving your work…" – it is the software equivalent of pulling the
power cord. We kill the daemon, then look at the file it left behind:

```
$ kill -9 <daemon pid>

$ sqlite3 -header -column "$TMP/rex.db" "SELECT id, outcome, ended_at FROM runs;"
id                                    outcome  ended_at
------------------------------------  -------  --------
c2685dd8-e66e-42af-b16d-e48f845370c3

$ sqlite3 -header -column "$TMP/rex.db" "SELECT count(*) AS events_survived FROM trajectory_events;"
events_survived
---------------
4
```

Two things to read off this, one per invariant:

- **Four events survived, not two.** Rex kept hunting after our curl peek ended; every
  one of those appends was committed before anything else happened, so every one is on
  disk. Nothing that was confirmed is lost – *that's invariant 1, demonstrated*. (The
  in-memory list at this exact moment: empty, along with the process.)
- **The run is a lie waiting to be corrected.** Its `outcome` is still NULL – "in
  progress" – but its process is dead. Nobody marked it crashed *because nobody was alive
  to do so*. A dangling open run is precisely what a crash looks like in the data.

### Act 3 – reboot: the crashed-run sweep

Start the daemon again, same database file. Watch its boot log:

```
INFO:     Waiting for application startup.
boot: marked 1 dangling run(s) as crashed
INFO:     Application startup complete.
```

That line is `mark_crashed_runs` (`src/rexhunter/db.py:89`) doing the boot-time sweep:
*"any run still open at boot did not survive its process"* – a single UPDATE of every
`outcome IS NULL` row to `'crashed'`. The logic is beautifully indirect: we never detect
the crash; we detect the *absence of a clean ending*. The state machine on the `runs` row
closes itself:

```
$ sqlite3 -header -column "$TMP/rex.db" "SELECT substr(id,1,8) AS run, outcome, substr(ended_at,12,8) AS ended FROM runs ORDER BY started_at;"
run       outcome  ended
--------  -------  --------
c2685dd8  crashed  01:47:45
2a7f35c4
```

Three details worth savoring:

- The killed run is now `crashed`, and its `ended_at` (01:47:45) is **backfilled from its
  last committed event** – we can't know when the process died, but the log knows the
  last thing it provably did. Derived from events, not invented: invariant 5 in miniature.
- A *new* run (`2a7f35c4`) is already open – the new daemon started hunting immediately,
  with its own NULL outcome.
- The full event table (shown back in section 3) has the old run's 4 events *and* the
  new run's events in one unbroken global sequence: id 5 follows id 4, and the new run's
  `seq` restarts at 0. History accumulated; nothing reset.

### Act 4 – the contrast: a polite death

Finally, `kill -TERM` – the *graceful* signal the daemon is allowed to catch. The
lifespan cancels the hunt task; `rex_loop`'s `except asyncio.CancelledError` branch
(`src/rexhunter/server.py:32`) closes its own run properly before the process exits:

```
$ sqlite3 -header -column "$TMP/rex.db" "SELECT substr(id,1,8) AS run, outcome, abort_reason FROM runs ORDER BY started_at;"
run       outcome  abort_reason
--------  -------  ---------------
c2685dd8  crashed
2a7f35c4  aborted  daemon shutdown

$ sqlite3 "$TMP/rex.db" "PRAGMA journal_mode; PRAGMA integrity_check;"
wal
ok
```

Side by side in one table: the run that died by `kill -9` (swept to `crashed` at next
boot, no reason recorded – the dead leave no notes) and the run that was shut down
cleanly (`aborted`, reason `daemon shutdown`, written by its own dying breath). The
closing `PRAGMA` pair is the working contract's standard health check: the file is in
WAL mode, and its structure is intact – after one murder and one reboot.

---

## Where this goes next

The same scenario runs automatically on every push: `tests/test_stage1_gate.py` spawns a
real writer subprocess, SIGKILLs it mid-append, and asserts everything you just watched.
Stage 2 replaces the string `payload` with typed Pydantic events and hands the loop a
real tool harness – the log underneath it does not change shape.
