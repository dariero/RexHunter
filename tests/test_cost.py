"""cost.py: the spend fold (invariant 5) and the price table it reads.

The price table is an INPUT to the fold, not stored state: the same UsageEvents fold to a different
spend under a different table, and nothing about the log changes. Offline, no spend.
"""

import pytest

from rexhunter import cost
from rexhunter.events import ToolResultEvent, UsageEvent, decode_event


def test_fold_cost_prices_usage_under_the_table_it_is_given() -> None:
    # A table that prices claude-sonnet-5-5 unlike the module table, so the result proves which
    # table the fold read: 1M input @ $1 + 1M output @ $4 = $5.
    table = {"claude-sonnet-5-5": (1.0, 4.0)}
    usage_in = UsageEvent(model="claude-sonnet-5-5", input_tokens=1_000_000, output_tokens=0)
    usage_out = UsageEvent(model="claude-sonnet-5-5", input_tokens=0, output_tokens=1_000_000)
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
        ("claude-fable-5-2", 48.0),  # $8 + $40
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


@pytest.mark.parametrize(
    ("model", "usd"),
    [
        # 1M input, 2M output, 3M written to the cache at 1.25x input, 4M read from it at the
        # model's read multiplier (the prompt-caching page, read 27 Nov 2026).
        ("claude-sonnet-5-5", 2.0 + 20.0 + 7.5 + 0.4),  # reads at 0.05x
        ("claude-opus-5-5", 4.0 + 40.0 + 15.0 + 0.8),  # reads at 0.05x
        ("claude-fable-5-2", 8.0 + 80.0 + 30.0 + 0.8),  # reads at 0.025x
    ],
)
def test_cache_tokens_fold_at_the_pinned_prices(model: str, usd: float) -> None:
    usage = UsageEvent(
        model=model,
        input_tokens=1_000_000,
        output_tokens=2_000_000,
        cache_creation_input_tokens=3_000_000,
        cache_read_input_tokens=4_000_000,
    )
    events = [usage, ToolResultEvent(tool="sniff", raw_request=b"{}", raw_response=b"x")]
    assert cost.fold_cost(events) == pytest.approx(usd)


def test_an_unknown_models_cache_tokens_price_high() -> None:
    # At the fallback input price, $15/MTok, for a write and a read alike: never 0, and at least the
    # dearest write and read of a listed model.
    write = UsageEvent(
        model="unknown-model",
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=1_000_000,
    )
    read = UsageEvent(
        model="unknown-model", input_tokens=0, output_tokens=0, cache_read_input_tokens=1_000_000
    )
    assert cost.cost_of(write) == pytest.approx(15.0)
    assert cost.cost_of(read) == pytest.approx(15.0)
    listed = list(cost.PRICE_PER_MTOK)
    dearest_write = max(
        cost.cost_of(
            UsageEvent(
                model=m, input_tokens=0, output_tokens=0, cache_creation_input_tokens=1_000_000
            )
        )
        for m in listed
    )
    dearest_read = max(
        cost.cost_of(
            UsageEvent(model=m, input_tokens=0, output_tokens=0, cache_read_input_tokens=1_000_000)
        )
        for m in listed
    )
    assert cost.cost_of(write) >= dearest_write
    assert cost.cost_of(read) >= dearest_read


def test_a_recorded_usage_decodes_with_no_cache_tokens() -> None:
    # A UsageEvent payload as the 26 Nov gym log stored it, for the counts of call_01.json in
    # tests/fixtures/gym_sonnet_5_5_2026-11-26/hunt_f0a5507e-d644-4d7d-938e-4a920c2a07ae/
    # (gym-greenhouse's first call, tests/test_gym_sonnet_5_5_hunt_replay.py:86-87 and :132).
    # Written before the cache counts, it decodes with both at 0, costs what it cost to the last
    # bit, and encodes to its own bytes, so the replays' log digests hold.
    stored = '{"type":"usage","model":"claude-sonnet-5-5","input_tokens":1131,"output_tokens":53}'
    usage = decode_event(stored)
    assert isinstance(usage, UsageEvent)
    assert (usage.cache_creation_input_tokens, usage.cache_read_input_tokens) == (0, 0)
    assert cost.cost_of(usage) == 1131 / 1_000_000 * 2.0 + 53 / 1_000_000 * 10.0
    assert usage.model_dump_json() == stored
