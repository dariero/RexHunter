"""Cost accounting for the brain socket (ADR pillar 5).

Cost is **derived**, not stored: `fold_cost` sums the run's `UsageEvent`s each time the loop needs
it (invariant 5 — no second mutable counter to drift from the log). The loop's breaker calls this
before every brain call and aborts once the total crosses `COST_CEILING_USD`.

Prices are per-million-token **sticker** rates. Over-estimating spend is the safe direction for a
guard, so the intro discount ($2/$10 on Sonnet 5 through 2026-08-31) is deliberately ignored, and an
unknown model prices at a high fallback — never 0, which would silently disable the guard.

This module sits below the loop (loop → cost → events) and imports nothing from loop/brain, so the
loop can price its log without a circular import.
"""

from collections.abc import Sequence

from rexhunter.events import TrajectoryEvent, UsageEvent

# (input $/MTok, output $/MTok). Sonnet 5 sticker (P5's only model so far).
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {"claude-sonnet-5": (3.0, 15.0)}

# Unknown model → a deliberately-high fallback (above any current model). NEVER 0: a 0 price would
# fold to $0 forever and silently disarm the spend guard.
_FALLBACK_PRICE_PER_MTOK: tuple[float, float] = (15.0, 75.0)


def cost_of(usage: UsageEvent) -> float:
    """USD for one brain call's usage, priced by its model (sticker rate, high fallback)."""
    price_in, price_out = PRICE_PER_MTOK.get(usage.model, _FALLBACK_PRICE_PER_MTOK)
    return usage.input_tokens / 1_000_000 * price_in + usage.output_tokens / 1_000_000 * price_out


def fold_cost(events: Sequence[TrajectoryEvent]) -> float:
    """Total USD spent so far, folded from the run's `UsageEvent`s (invariant 5). Non-usage events
    are ignored — cost is a projection of the usage sub-log, not a running side total."""
    return sum((cost_of(e) for e in events if isinstance(e, UsageEvent)), 0.0)
