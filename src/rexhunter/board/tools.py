"""Rex's board tools (`P2.4`): `list_postings` and `read_posting`.

Each tool makes one GET and returns the board's body as it arrived, which the loop stores as the
result's `raw_response` (invariant 6). Before it returns, the tool validates the same bytes
through its board's adapter (invariant 3): a body that fails is a fatal tool error that carries
it. The model never reads those bytes. It reads each result through the tool's view, rendered
again from the stored bytes on every brain call (invariant 5): the page or the posting, typed,
as JSON. `read_posting` is the posting source a catch reference resolves against.
"""

from collections.abc import Callable, Mapping
from typing import Annotated

from pydantic import BaseModel, Field, ValidationError

from rexhunter.board import MAX_PAGES, Board
from rexhunter.board.adapters import ADAPTERS, body_text
from rexhunter.board.client import BoardClient
from rexhunter.events import PostingRef
from rexhunter.loop import FatalToolError
from rexhunter.tools import ToolRegistry

# What a view shows for stored bytes it cannot parse. Unreachable for a result this code wrote,
# whose bytes passed the same parse before they were stored; it keeps a view total, so an older or
# edited log never ends a run at the `<loop>` backstop, and the model is never shown the bytes.
UNREADABLE = '{"error": "the stored board response does not parse"}'


class _ListArgs(BaseModel):
    board: str
    page: int


def _parsed[T](body: bytes, parse: Callable[[str], T]) -> T:
    try:
        return parse(body_text(body))
    except UnicodeDecodeError as exc:
        raise FatalToolError("board body is not valid UTF-8", raw_response=body) from exc
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "the body"
        raise FatalToolError(
            f"board body fails the boundary: {first['type']} at {where}", raw_response=body
        ) from exc


def build_registry(client: BoardClient, boards: Mapping[str, Board]) -> ToolRegistry:
    """A fresh registry holding the two board tools over `client`, for the boards named here."""

    def board_named(name: str) -> Board:
        board = boards.get(name)
        if board is None:
            raise FatalToolError(f"unknown board: {name!r}")  # no request is sent
        return board

    async def list_postings(
        board: str, page: Annotated[int, Field(ge=1, le=MAX_PAGES)] = 1
    ) -> bytes:
        """List one page of a job board's postings: each posting's id, title and location.
        Pages start at 1; `next_page` is the page to ask for next, or null on the last page."""
        config = board_named(board)
        adapter = ADAPTERS[config.ats]
        url, params = adapter.list_request(config, page)
        body = await client.get(url, params)
        _parsed(body, lambda text: adapter.page(text, board, page))
        return body

    async def read_posting(board: str, posting_id: str) -> bytes:
        """Read one posting in full: its title, location, link and description. A posting you
        read here can be named in hunt_complete's catch."""
        config = board_named(board)
        adapter = ADAPTERS[config.ats]
        body = await client.get(adapter.read_url(config, posting_id))
        posting = _parsed(body, lambda text: adapter.posting(text, board))
        if posting.posting_id != posting_id:
            # The pen's key is the reference, so the bytes must be the posting it names.
            raise FatalToolError(
                f"board answered posting {posting.posting_id!r}", raw_response=body
            )
        return body

    def list_view(raw_request: bytes, raw_response: bytes) -> str:
        try:
            args = _ListArgs.model_validate_json(raw_request)
            adapter = ADAPTERS[boards[args.board].ats]
            return adapter.page(body_text(raw_response), args.board, args.page).model_dump_json()
        except ValueError, KeyError:
            return UNREADABLE

    def posting_view(raw_request: bytes, raw_response: bytes) -> str:
        try:
            ref = PostingRef.model_validate_json(raw_request)
            adapter = ADAPTERS[boards[ref.board].ats]
            return adapter.posting(body_text(raw_response), ref.board).model_dump_json()
        except ValueError, KeyError:
            return UNREADABLE

    registry = ToolRegistry()
    registry.tool(list_postings, view=list_view)
    registry.tool(read_posting, view=posting_view, posting_source=True)
    return registry
