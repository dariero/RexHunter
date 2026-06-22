from collections.abc import Callable, Sequence

import pytest

from rexhunter.events import TrajectoryEvent
from rexhunter.loop import Brain, Decision


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def scripted_brain() -> Callable[[Sequence[Decision]], Brain]:
    """Factory for a stub Brain (no model, no spend): returns the given decisions in order.

    The loop's failure-injection lever - tests script exactly what the brain decides, so the
    gate can drive raise / hang / unknown-tool deterministically.
    """

    def make(decisions: Sequence[Decision]) -> Brain:
        queue = iter(decisions)

        async def brain(_context: list[TrajectoryEvent]) -> Decision:
            try:
                return next(queue)
            except StopIteration:
                raise AssertionError(
                    "scripted brain exhausted: the loop wanted more decisions"
                ) from None

        return brain

    return make
