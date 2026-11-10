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
