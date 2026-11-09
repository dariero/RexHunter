from collections.abc import Callable, Sequence

import anthropic
import httpx2
import pytest

from rexhunter import brain
from rexhunter.events import TrajectoryEvent
from rexhunter.loop import Brain, Decision, ThinkingSink

type MockHandler = Callable[[httpx2.Request], httpx2.Response]
type MockAnthropic = Callable[..., anthropic.AsyncAnthropic]


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

        async def brain(_context: list[TrajectoryEvent], _sink: ThinkingSink) -> Decision:
            try:
                return next(queue)
            except StopIteration:
                raise AssertionError(
                    "scripted brain exhausted: the loop wanted more decisions"
                ) from None

        return brain

    return make


@pytest.fixture
def mock_anthropic() -> MockAnthropic:
    """Factory for the adapter's test seam (W2.0): the PRODUCTION SDK client from
    `brain.anthropic_client` (max_retries=0, the pinned wire, no redirects), over an httpx2
    transport. Pass a handler (wrapped in a `MockTransport`, zero network) or a ready transport
    (e.g. a capturing tee). Only the transport is injected: the factory builds the http client
    itself, exactly as it does for the daemon, so a test sees what the daemon would send and a
    mutant in the factory (say, dropping max_retries=0) reaches the tests."""

    def make(
        handler: MockHandler | httpx2.AsyncBaseTransport, *, api_key: str = "sk-test"
    ) -> anthropic.AsyncAnthropic:
        transport = (
            handler
            if isinstance(handler, httpx2.AsyncBaseTransport)
            else httpx2.MockTransport(handler)
        )
        return brain.anthropic_client(api_key, transport=transport)

    return make
