"""P2.5: the one-wave eval harness: each gym territory and gym-mimic hunted once into a fresh log,
then scored (eval tier 2). SPENDS on the live path.

NOT a test: pytest must never spend. Three modes:

    PYTHONPATH=src uv run python -m rexhunter.eval_wave --db PATH --baseline   # $0, no key
    ANTHROPIC_API_KEY=... PYTHONPATH=src \
        uv run python -m rexhunter.eval_wave --db PATH --count-only   # FREE pre-flight, $0
    ANTHROPIC_API_KEY=... REXHUNTER_BRAIN=live PYTHONPATH=src \
        uv run python -m rexhunter.eval_wave --db PATH   # the paid wave

Every mode refuses a PATH that exists, or whose -wal or -shm exists: a posting already in the pen
is not penned again, so the score reads a fresh log only. `--baseline` hunts with the scripted
keyword baseline (`board.baseline`): no key, no Anthropic client, no spend. `--count-only` is the
free pre-flight: `count_tokens` on each territory's exact first body, printed, and no hunt. The paid
wave is the daemon's live brain (`brain.select_brain_for` with the gym prompt and seed), opted into
twice (`brain.paid_entry`). A hunting mode runs one hunt per territory through
`scheduler.run_hunts`, never the daemon's scheduler, under `RUN_CEILING_USD` a run and
`WAVE_CEILING_USD` the wave, on the in-process gym over the six boards and gym-mimic, and then
prints the score table (`eval_score`) and the wave's spend.
"""

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx2

from rexhunter import brain, eval_score, hunt_smoke, scheduler, stub
from rexhunter.board.baseline import baseline_brain_for
from rexhunter.board.catalogue import EVAL_BOARDS, EVAL_CATALOGUE, GYM_BOARDS, GYM_CATALOGUE
from rexhunter.board.client import BoardClient
from rexhunter.board.gym import build_gym
from rexhunter.board.tools import GYM_SYSTEM_PROMPT, build_registry, gym_seed
from rexhunter.loop import Brain
from rexhunter.tools import ToolRegistry

# The wave's two ceilings (USD), each checked before every brain attempt: a hunt can pass
# RUN_CEILING_USD by the one call it has in flight, and the wave can pass WAVE_CEILING_USD by the
# one call each of its concurrent hunts has in flight.
RUN_CEILING_USD = 0.40
WAVE_CEILING_USD = 0.75
MAX_ITERATIONS = 50
BOARDS = GYM_BOARDS | EVAL_BOARDS
CATALOGUE = GYM_CATALOGUE | EVAL_CATALOGUE
TERRITORIES: tuple[str, ...] = tuple(BOARDS)  # the six gym territories, then gym-mimic


def refuse_an_existing_log(path: Path) -> None:
    for taken in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
        if taken.exists():
            raise SystemExit(
                f"refusing {taken}: a wave hunts into a fresh log, since a posting already in the "
                "pen is not penned again. Pass a --db that does not exist."
            )


def _gym() -> BoardClient:
    return BoardClient(httpx2.ASGITransport(app=build_gym(CATALOGUE)))


async def hunt_wave(
    path: Path, brain_for: Callable[[str], Brain], registry: ToolRegistry
) -> Sequence[str | None]:
    """One hunt per territory, then done: the one-wave stop."""
    return await scheduler.run_hunts(
        path,
        TERRITORIES,
        brain_for=brain_for,
        registry=registry,
        max_concurrent=stub.MAX_CONCURRENT,
        max_iterations=MAX_ITERATIONS,
        cost_ceiling_usd=RUN_CEILING_USD,
        daemon_spend_ceiling_usd=WAVE_CEILING_USD,
    )


async def run_baseline_wave(path: Path) -> None:
    async with _gym() as board:
        registry = build_registry(board, BOARDS)
        await hunt_wave(path, baseline_brain_for(BOARDS), registry)
    print(await eval_score.score_text(path))


async def run_live_wave(path: Path) -> None:
    async with _gym() as board:
        registry = build_registry(board, BOARDS)
        brain_for, client = brain.select_brain_for(
            registry, system=GYM_SYSTEM_PROMPT, seed=gym_seed
        )
        try:
            await hunt_wave(path, brain_for, registry)
        finally:
            # After the hunts drain: they drove their brain calls on this client.
            if client is not None:
                await client.close()
    print(await eval_score.score_text(path))
    print(f"ceilings: ${RUN_CEILING_USD:.2f} a run, ${WAVE_CEILING_USD:.2f} the wave")


async def count_first_bodies(api_key: str) -> dict[str, int]:
    """The free pre-flight: each territory's exact first body, counted. A non-200 stops it."""
    async with _gym() as board, brain.anthropic_client(api_key, timeout=120.0) as client:
        registry = build_registry(board, BOARDS)
        counts: dict[str, int] = {}
        for territory in TERRITORIES:
            counts[territory] = await hunt_smoke.count_tokens_preflight(
                client,
                registry,
                territory=territory,
                thinking=brain.HUNT_THINKING,
                system=GYM_SYSTEM_PROMPT,
                seed=gym_seed,
            )
    return counts


def _db_path(argv: Sequence[str]) -> Path:
    # No prefixes: the mode is read by its exact name (in `main` and in `paid_entry`), so a prefix
    # argparse took for `--count-only` would run the paid wave.
    parser = argparse.ArgumentParser(
        prog="python -m rexhunter.eval_wave",
        description="Hunt each gym territory and gym-mimic once into a fresh log, then score it.",
        allow_abbrev=False,
    )
    parser.add_argument("--db", type=Path, required=True, help="the fresh log to hunt into")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--count-only", action="store_true", help="the free pre-flight alone")
    mode.add_argument("--baseline", action="store_true", help="the keyword baseline, no spend")
    path: Path = parser.parse_args(argv).db
    return path


def main() -> None:
    argv = sys.argv[1:]
    if "--baseline" in argv:
        # No spend and no Anthropic client, so neither opt-in of the paid path applies.
        path = _db_path(argv)
        refuse_an_existing_log(path)
        asyncio.run(run_baseline_wave(path))
        return
    # The gate the paid entrypoints share comes first, before any argument is read.
    api_key, count_only = brain.paid_entry(before="the pre-flight or wave", paid="wave")
    path = _db_path(argv)
    refuse_an_existing_log(path)
    if count_only:
        counts = asyncio.run(count_first_bodies(api_key))
        for territory, n in counts.items():
            print(f"[count-only] {territory}: input_tokens={n}")
        print("[count-only] validated, NO paid call made.")
        return
    asyncio.run(run_live_wave(path))


if __name__ == "__main__":
    main()
