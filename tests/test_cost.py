"""cost.py: the spend fold (invariant 5) and the price table it reads.

The price table is an INPUT to the fold, not stored state: the same UsageEvents fold to a different
spend under a different table, and nothing about the log changes. Offline, no spend.
"""

import pytest

from rexhunter import cost
from rexhunter.events import ToolResultEvent, UsageEvent


def test_fold_cost_prices_usage_under_the_table_it_is_given() -> None:
    # A table that prices claude-sonnet-5 unlike the module table, so the result proves which
    # table the fold read: 1M input @ $1 + 1M output @ $4 = $5.
    table = {"claude-sonnet-5": (1.0, 4.0)}
    usage_in = UsageEvent(model="claude-sonnet-5", input_tokens=1_000_000, output_tokens=0)
    usage_out = UsageEvent(model="claude-sonnet-5", input_tokens=0, output_tokens=1_000_000)
    events = [
        usage_in,
        ToolResultEvent(tool="sniff", raw_request=b"{}", raw_response=b"x"),
        usage_out,
    ]
    assert cost.fold_cost(events, prices=table) == pytest.approx(5.0)
    assert cost.cost_of(usage_in, prices=table) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("model", "usd"),
    [
        ("claude-sonnet-5-5", 12.0),  # $2 + $10
        ("claude-opus-5-5", 24.0),  # $4 + $20
        ("claude-fable-5-1", 60.0),  # $10 + $50
    ],
)
def test_each_listed_model_prices_at_its_sticker(model: str, usd: float) -> None:
    # 1M input + 1M output, so the dollar figure is the two sticker rates added.
    usage = UsageEvent(model=model, input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost.cost_of(usage) == pytest.approx(usd)


def test_an_unknown_model_prices_at_least_the_highest_rate_on_each_side() -> None:
    # The fallback must never under-meter: on EACH side it is at least the dearest model in the
    # table. Two one-sided usages, so a fallback that is high on one side only cannot pass.
    input_only = UsageEvent(model="unknown-model", input_tokens=1_000_000, output_tokens=0)
    output_only = UsageEvent(model="unknown-model", input_tokens=0, output_tokens=1_000_000)
    assert cost.cost_of(input_only) >= max(p_in for p_in, _ in cost.PRICE_PER_MTOK.values())
    assert cost.cost_of(output_only) >= max(p_out for _, p_out in cost.PRICE_PER_MTOK.values())
