"""Cost accounting for the brain socket (ADR pillar 5).

Cost is **derived**, not stored: `fold_cost` sums the run's `UsageEvent`s each time the loop needs
it (invariant 5 — no second mutable counter to drift from the log). The loop's breaker calls this
before every brain call and aborts once the total crosses `COST_CEILING_USD`. The price table is an
INPUT to that fold, not stored state: `prices` defaults to `PRICE_PER_MTOK`, and a caller can fold
the same usage under another table (a replay test, under the table its live run was priced at).

Prices are per-million-token **sticker** rates. An unknown model prices at a high fallback, because
over-estimating spend is the safe direction for a guard; never 0, which would silently disable the
guard.

This module sits below the loop (loop → cost → events) and imports nothing from loop/brain, so the
loop can price its log without a circular import.
"""

from collections.abc import Mapping, Sequence

from rexhunter.events import TrajectoryEvent, UsageEvent

# (input $/MTok, output $/MTok) at the sticker rate, the platform.claude.com/docs/en/about-claude/
# pricing prices: claude-fable-5-2 read on 13 Nov 2026 (W2.1b, standard tier, no long-context
# tier), the rest read on 9 Nov 2026 (W2.1).
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-fable-5-2": (8.0, 40.0),
}

# Unknown model → a deliberately-high fallback, at least the dearest row above on each side
# (pinned by tests/test_cost.py). NEVER 0: a 0 price would fold to $0 forever and silently disarm
# the spend guard. An unknown model's cache write and read price at the fallback input price, never
# 0, for the same reason.
_FALLBACK_PRICE_PER_MTOK: tuple[float, float] = (15.0, 75.0)

# The prompt cache's prices as multiples of a model's input price, the platform.claude.com/docs/en/
# build-with-claude/prompt-caching multipliers, read on 27 Nov 2026 (`P5.1`): a write at the
# default 5-minute lifetime, the only one the live brain asks for (`brain.PROMPT_CACHE`), and a
# read per model. A model a caller's table names with no read row reads at 1.0, its plain input
# price, which over-estimates and so is the safe direction for a guard.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER: dict[str, float] = {
    "claude-sonnet-5-5": 0.05,
    "claude-opus-5-5": 0.05,
    "claude-fable-5-2": 0.025,
}


def cost_of(
    usage: UsageEvent, *, prices: Mapping[str, tuple[float, float]] = PRICE_PER_MTOK
) -> float:
    """USD for one brain call's usage, priced by its model in `prices` (high fallback)."""
    if usage.model in prices:
        price_in, price_out = prices[usage.model]
        price_write = price_in * CACHE_WRITE_MULTIPLIER
        price_read = price_in * CACHE_READ_MULTIPLIER.get(usage.model, 1.0)
    else:
        price_in, price_out = _FALLBACK_PRICE_PER_MTOK
        price_write = price_read = price_in
    # The two cache terms come last, so a usage with no cache tokens folds to the float it folded
    # to before `P5.1`, to the last bit (adding 0.0 is exact), and every recorded cost stays put.
    return (
        usage.input_tokens / 1_000_000 * price_in
        + usage.output_tokens / 1_000_000 * price_out
        + usage.cache_creation_input_tokens / 1_000_000 * price_write
        + usage.cache_read_input_tokens / 1_000_000 * price_read
    )


def fold_cost(
    events: Sequence[TrajectoryEvent], *, prices: Mapping[str, tuple[float, float]] = PRICE_PER_MTOK
) -> float:
    """Total USD spent so far, folded from the run's `UsageEvent`s under `prices` (invariant 5).
    Non-usage events are ignored — cost is a projection of the usage sub-log, not a running side
    total."""
    return sum((cost_of(e, prices=prices) for e in events if isinstance(e, UsageEvent)), 0.0)
