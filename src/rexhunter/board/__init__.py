"""The job boards Rex hunts (ADR pillar 2, `P2.4`).

A board is one territory (D2): its `name` is the territory the scheduler hunts and the `board`
argument Rex's tools take. Its `ats` picks the adapter that knows the board's documented shape
(Greenhouse or Lever), `token` is the board's own id in that shape's paths, and `base_url` is
where the board answers. A gym board answers at `http://gym/greenhouse` or `http://gym/lever`,
reached in-process through httpx2's `ASGITransport` (`board.gym`): no port, no network.
"""

from dataclasses import dataclass
from typing import Literal

# The D10 limits. A list page holds PAGE_SIZE postings and a hunt asks for at most MAX_PAGES of
# them; a posting's description reaches the model cut at DESCRIPTION_CAP characters; a board
# request that takes longer than BOARD_TIMEOUT_S is a retryable failure.
PAGE_SIZE = 20
MAX_PAGES = 10
DESCRIPTION_CAP = 8000
BOARD_TIMEOUT_S = 10.0

type Ats = Literal["greenhouse", "lever"]


@dataclass(frozen=True)
class Board:
    name: str
    ats: Ats
    token: str
    base_url: str
