"""The live boards (`P5.1`): Greenhouse and Lever boards on their public job board APIs.

`LIVE_BOARDS` names the boards a live daemon hunts, one territory each, as `GYM_BOARDS` names the
gym's: every one answers over https on a host in `LIVE_HOSTS`. `live_client` is the board client for
them. It sends GET only and follows no redirect, as every board client does, and it refuses a URL
whose scheme is not https, whose host is not in `LIVE_HOSTS`, or that carries userinfo or a port
other than 443, before any request is sent (invariant 4). Its transport is injected, as every board
client's is: the network one comes from `brain.live_transport`, the one place a real transport is
built. The gym client never gets it.
"""

from collections.abc import Mapping

import httpx2

from rexhunter.board import Board
from rexhunter.board.client import BoardClient

# The hosts of the Greenhouse Job Board API and of the Lever Postings API, global and EU, as
# re-read on 27 Nov 2026: public, unauthenticated GETs.
LIVE_HOSTS = frozenset({"boards-api.greenhouse.io", "api.lever.co", "api.eu.lever.co"})

LIVE_BOARDS: Mapping[str, Board] = {}


def live_client(transport: httpx2.AsyncBaseTransport) -> BoardClient:
    return BoardClient(transport, allowed_hosts=LIVE_HOSTS)
