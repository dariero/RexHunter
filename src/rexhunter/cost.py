"""Cost accounting for the brain socket (ADR pillar 5).

Cost is **derived**, not stored: `fold_cost` sums the run's `UsageEvent`s each time the loop needs
it (invariant 5 — no second mutable counter to drift from the log). The loop's breaker calls this
before every brain call and aborts once the total crosses `COST_CEILING_USD`. The price table is an
INPUT to that fold, not stored state: `prices` defaults to `PRICE_PER_MTOK`, and a caller can fold
the same usage under another table (a replay test, under the table its live run was priced at).

Prices are per-million-token **sticker** rates. D6 (9 Nov 2026) prices Sonnet 5 at $2/$10, which
"is now the standard price" (it began as an intro discount, ignored here until D6). An unknown model
prices at a high fallback, because over-estimating spend is the safe direction for a guard; never 0,
which would silently disable the guard.

This module sits below the loop (loop → cost → events) and imports nothing from loop/brain, so the
loop can price its log without a circular import.
"""

from collections.abc import Mapping, Sequence

from rexhunter.events import TrajectoryEvent, UsageEvent

# (input $/MTok, output $/MTok) at the sticker rate. Sonnet 5 at $2/$10, the standard price (D6).
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {"claude-sonnet-5": (2.0, 10.0)}

# Unknown model → a deliberately-high fallback (above any current model). NEVER 0: a 0 price would
# fold to $0 forever and silently disarm the spend guard.
_FALLBACK_PRICE_PER_MTOK: tuple[float, float] = (15.0, 75.0)


def cost_of(
    usage: UsageEvent, *, prices: Mapping[str, tuple[float, float]] = PRICE_PER_MTOK
) -> float:
    """USD for one brain call's usage, priced by its model in `prices` (high fallback)."""
    price_in, price_out = prices.get(usage.model, _FALLBACK_PRICE_PER_MTOK)
    return usage.input_tokens / 1_000_000 * price_in + usage.output_tokens / 1_000_000 * price_out


def fold_cost(
    events: Sequence[TrajectoryEvent], *, prices: Mapping[str, tuple[float, float]] = PRICE_PER_MTOK
) -> float:
    """Total USD spent so far, folded from the run's `UsageEvent`s under `prices` (invariant 5).
    Non-usage events are ignored — cost is a projection of the usage sub-log, not a running side
    total."""
    return sum((cost_of(e, prices=prices) for e in events if isinstance(e, UsageEvent)), 0.0)
