"""P5 Unit 2b — offline replay of the captured live Sonnet 5 response (inv 6, zero network).

`tests/fixtures/smoke_sonnet5.json` is the raw bytes of the ONE paid smoke call. It carries a real
`thinking` sibling block alongside the `tool_use` (Sonnet 5 runs adaptive thinking automatically),
so replaying it proves — on genuine provider output, with no network and no spend — that the 2a
lenient scan skips the thinking and locates the single `tool_use`. A dead run is a pytest fixture
for free (invariant 6); this test could not exist before the paid call created the fixture.
"""

import json
from pathlib import Path

from rexhunter.brain import parse_decision
from rexhunter.loop import ToolCallDecision

FIXTURE = Path(__file__).parent / "fixtures" / "smoke_sonnet5.json"


def test_fixture_carries_a_real_thinking_sibling() -> None:
    payload = json.loads(FIXTURE.read_bytes())
    assert payload["model"] == "claude-sonnet-5"  # the fixture came from the intended model
    types = [block["type"] for block in payload["content"]]
    assert "thinking" in types  # Sonnet 5 adaptive thinking — the lenient-scan case, for real
    assert types.count("tool_use") == 1  # exactly one actionable block


def test_replay_parses_to_a_stable_toolcall_decision() -> None:
    decision = parse_decision(FIXTURE.read_bytes())  # zero network, one boundary (invariant 3)
    assert isinstance(decision, ToolCallDecision)
    assert decision.tool == "sniff"  # the model called the real tool, not a terminal signal
    assert (
        decision.tool_use_id == "toolu_01P2D4vWUcLv9fDKwcbe7jLX"
    )  # the provider's correlation key
    assert "prey" in decision.args  # args survive the boundary as a validated dict
