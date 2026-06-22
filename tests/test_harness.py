"""P2.2 · Unit C — the error taxonomy + the single-attempt execution wrapper.

Two harness primitives the loop builds on:
  - classify(exc) -> bool: retryable (re-try within budget) vs fatal (retrying just burns
    budget). Timeouts and RetryableToolError are retryable; everything else is fatal.
  - execute_once(fn, kwargs, timeout) -> bytes: ONE tool attempt under a deadline, returning
    the raw response bytes (invariant 6). Single attempt by design - the retry *iteration*
    lives in the loop so each attempt appends its own event (invariant 7).

The timeout test is the slice's sharpest hazard: the slow tool MUST block via asyncio.sleep
(cancellable at an await point), never time.sleep (which would wedge the one event loop so the
deadline can never fire), and the test asserts the deadline fired fast - it must not hang.
"""

import asyncio
import time

import pytest
from pydantic import BaseModel, ValidationError

from rexhunter.loop import RetryableToolError, classify, execute_once


@pytest.mark.parametrize(
    "exc,expected",
    [
        pytest.param(TimeoutError(), True, id="timeout-retryable"),
        pytest.param(RetryableToolError("blip"), True, id="marker-retryable"),
        pytest.param(ValueError("bad input"), False, id="valueerror-fatal"),
        pytest.param(RuntimeError("a bug"), False, id="runtime-fatal"),
    ],
)
def test_classify_taxonomy(exc: BaseException, expected: bool) -> None:
    assert classify(exc) is expected


def test_validation_error_is_fatal() -> None:
    # bad args never become good on retry, so a ValidationError must classify fatal.
    class M(BaseModel):
        x: int

    with pytest.raises(ValidationError) as caught:
        M.model_validate({"x": "not-an-int"})
    assert classify(caught.value) is False


@pytest.mark.anyio
async def test_execute_once_returns_raw_response_bytes() -> None:
    async def tool(x: int) -> str:
        return f"got-{x}"

    out = await execute_once(tool, {"x": 5}, timeout_s=1.0)
    assert out == b'"got-5"'  # the return value, JSON-serialised to bytes for the log


@pytest.mark.anyio
async def test_execute_once_serialises_models_and_passes_bytes_through() -> None:
    class Posting(BaseModel):
        title: str

    async def fetch() -> Posting:
        return Posting(title="AI Eng")

    async def raw() -> bytes:
        return b"\x00\xff already-bytes"

    assert await execute_once(fetch, {}, timeout_s=1.0) == b'{"title":"AI Eng"}'
    assert await execute_once(raw, {}, timeout_s=1.0) == b"\x00\xff already-bytes"


@pytest.mark.anyio
async def test_execute_once_surfaces_tool_exceptions() -> None:
    # the wrapper does not swallow - it surfaces the exception for the loop to classify/log.
    async def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await execute_once(boom, {}, timeout_s=1.0)


@pytest.mark.anyio
async def test_execute_once_times_out_fast_without_hanging() -> None:
    async def slow() -> None:
        await asyncio.sleep(10)  # cancellable: the deadline can actually fire

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        await execute_once(slow, {}, timeout_s=0.05)
    assert time.monotonic() - start < 1.0  # fired at the 50ms deadline, did NOT await 10s
