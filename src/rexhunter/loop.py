"""The hand-rolled agent loop & tool harness (ADR pillar 2).

Built in two slices on one file. `P2.2` Unit C lands the harness primitives below - the
error taxonomy and the single-attempt execution wrapper; Unit D adds the decision union and
the `run_hunt` loop that drives them. Keeping both here is deliberate: the loop is the
observability surface, and every state transition is one append at the site of the transition.
"""

import asyncio
import json
from typing import Any

from pydantic import BaseModel

from rexhunter.tools import ToolFn


class RetryableToolError(Exception):
    """A tool failure the loop may re-try within budget (a transient blip, a 429, a 5xx).

    Tools opt INTO retry by raising this (or a subclass). The taxonomy's default is fatal: a
    plain exception is assumed not to fix itself on a second attempt.
    """


def classify(exc: BaseException) -> bool:
    """The retryable-vs-fatal taxonomy. ``True`` = retryable (re-try within budget).

    Retryable: a timeout (the tool may simply have been slow this once) and any
    RetryableToolError. Fatal: everything else - a ValidationError (bad args never become good
    on retry), a bug in the tool, an unknown-tool lookup. Retrying a fatal error only burns
    budget without changing the outcome.
    """
    return isinstance(exc, TimeoutError | RetryableToolError)


def _to_bytes(result: object) -> bytes:
    """Serialise a tool's return value to the raw response bytes the log stores (invariant 6).

    A Pydantic model dumps to its JSON; raw bytes pass straight through (a live adapter's HTTP
    body is already bytes); anything else is JSON-encoded. The bytes are what a ghost replay
    re-feeds, so they must be the exact thing the tool produced.
    """
    if isinstance(result, BaseModel):
        return result.model_dump_json().encode()
    if isinstance(result, bytes):
        return result
    return json.dumps(result).encode()


async def execute_once(fn: ToolFn, kwargs: dict[str, Any], *, timeout_s: float) -> bytes:
    """Run ONE tool attempt under a deadline; return the raw response bytes (invariant 6).

    Single attempt by design - the retry *iteration* lives in `run_hunt` so every attempt
    appends its own event (single writer, invariant 7). A tool that overruns ``timeout_s`` is
    cancelled at the deadline and surfaces as ``TimeoutError`` (asyncio.timeout raises it on
    expiry), which `classify` treats as retryable. The wrapper does not swallow: a tool's own
    exception propagates for the loop to classify and log.
    """
    async with asyncio.timeout(timeout_s):
        result = await fn(**kwargs)
    return _to_bytes(result)
