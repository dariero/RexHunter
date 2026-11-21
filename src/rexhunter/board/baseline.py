"""The scripted keyword baseline (`P2.4`, D9): the stub daemon's brain on the gym territories.

No model and no spend. It lists page 1 of its board, reads each posting whose title names AI or
machine learning (`matches`), and catches what it read. Each decision is derived from the run's
own log, as a projection is: the list result says what to read, the read results say what is
left, so the brain holds no state between calls. It is a floor for a later eval to score the live
brain against, not a judge of prey.
"""

import re
from collections.abc import Callable, Mapping

from rexhunter.board import Board
from rexhunter.board.adapters import ADAPTERS, body_text
from rexhunter.events import PostingRef, ToolResultEvent
from rexhunter.loop import Brain, Context, Decision, HuntComplete, ThinkingSink, ToolCallDecision

# "AI" as a whole word, in capitals: "AI Engineer" and "Applied AI" match, "Air", "Email" and
# "Domain" do not. "machine learning" in any case.
_AI = re.compile(r"\bAI\b")
_MACHINE_LEARNING = re.compile(r"\bmachine learning\b", re.IGNORECASE)


def matches(title: str) -> bool:
    return bool(_AI.search(title) or _MACHINE_LEARNING.search(title))


def baseline_brain_for(boards: Mapping[str, Board]) -> Callable[[str], Brain]:
    """A `brain_for` for the scheduler: a fresh baseline brain per hunt of a board in `boards`."""

    def brain_for(territory: str) -> Brain:
        adapter = ADAPTERS[boards[territory].ats]

        async def brain(context: Context, _sink: ThinkingSink) -> Decision:
            results = [e for e in context if isinstance(e, ToolResultEvent)]
            listed = next((r for r in results if r.tool == "list_postings"), None)
            if listed is None:
                return ToolCallDecision(tool="list_postings", args={"board": territory, "page": 1})
            page = adapter.page(body_text(listed.raw_response), territory, 1)
            wanted = [p.posting_id for p in page.postings if matches(p.title)]
            read = {
                PostingRef.model_validate_json(r.raw_request).posting_id
                for r in results
                if r.tool == "read_posting"
            }
            for posting_id in wanted:
                if posting_id not in read:
                    ref = PostingRef(board=territory, posting_id=posting_id)
                    return ToolCallDecision(tool="read_posting", args=ref.model_dump())
            return HuntComplete(catch=[PostingRef(board=territory, posting_id=p) for p in wanted])

        return brain

    return brain_for
