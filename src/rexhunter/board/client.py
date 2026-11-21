"""The board client (`P2.4`): GET only, the one HTTP method Rex has on a board.

Tiny Arms (invariant 4) holds structurally: `BoardClient` keeps its httpx2 client private and
exposes `get` alone, so no code path builds a POST, the method a board's application endpoint
takes. Its transport is injected and has no default, so the client reaches only what its owner
hands it: in this slice, always the in-process gym (`httpx2.ASGITransport`). It follows no
redirect and reads no proxy or certificate setting from the environment.

`get` turns a board's answer into the loop's tool-error taxonomy, with the body kept (invariant 6):
a 429 or a 5xx, and a transport failure, are retryable, with the board's `retry-after`; any other
status but 200 is fatal.
"""

from collections.abc import Mapping
from types import TracebackType

import httpx2

from rexhunter.board import BOARD_TIMEOUT_S
from rexhunter.loop import FatalToolError, RetryableToolError, parse_retry_after


class BoardClient:
    def __init__(self, transport: httpx2.AsyncBaseTransport) -> None:
        self._http = httpx2.AsyncClient(
            transport=transport,
            follow_redirects=False,
            timeout=BOARD_TIMEOUT_S,
            trust_env=False,
        )

    async def get(self, url: str, params: Mapping[str, str] | None = None) -> bytes:
        """The body of a 200, as received (content-decoded), or the tool error its status is."""
        try:
            response = await self._http.get(url, params=params)
        except httpx2.TransportError as exc:
            raise RetryableToolError(f"board transport failure: {exc!r}") from exc
        status, body = response.status_code, response.content
        if status == 429 or status >= 500:
            raise RetryableToolError(
                f"board answered {status}",
                raw_response=body,
                retry_after=parse_retry_after(response.headers.get("retry-after")),
            )
        if status != 200:
            raise FatalToolError(f"board answered {status}", raw_response=body)
        return body

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> BoardClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
