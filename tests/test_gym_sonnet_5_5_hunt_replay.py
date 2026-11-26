"""Offline replay of the first live gym hunts on Claude Sonnet 5.5 (DoD #5, no network).

On 26 Nov 2026 the daemon hunted the six gym territories live (`REXHUNTER_BRAIN=live`,
`claude-sonnet-5-5`, adaptive thinking with `display: "summarized"`, streamed, the gym prompt and
seed, `COST_CEILING_USD=0.20`, `DAEMON_SPEND_CEILING_USD=0.50`, `MAX_ITERATIONS` unset, so 50), and
each brain attempt stored its exchange in the daemon's log. The fixture directory
`tests/fixtures/gym_sonnet_5_5_2026-11-26/` keeps the first run of each territory, each response's
bytes as stored. gym-greenhouse, gym-lever, gym-storm (two 429s from the board, then its list) and
gym-mojibake read and caught four prey each; gym-drift ended at its first list, whose renamed field
fails the board boundary ("tool failure"); gym-pages read fifteen prey and reached its cost ceiling
before a catch. The request bytes are not kept: a later change to a tool's description or view
would make a stored request pin an old wire, so the replay checks each request's shape instead.

Re-driving each run through the adapter, on its recorded model, under the daemon's settings and
over a fresh in-process gym, reproduces its log: the outcome and the run's caps, every event but the
exchanges payload for payload (against a digest of the live log's payloads, with the two fields no
replay can reproduce left out), each exchange's status and response bytes, and the pen.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import anthropic
import httpx2
import pytest

from rexhunter import brain, cost, db
from rexhunter.board.catalogue import GYM_BOARDS, GYM_CATALOGUE
from rexhunter.board.client import BoardClient
from rexhunter.board.gym import build_gym
from rexhunter.board.tools import GYM_SYSTEM_PROMPT, build_registry, gym_seed
from rexhunter.events import (
    BrainExchangeEvent,
    ErrorEvent,
    PreyCapturedEvent,
    ThinkingDelta,
    ToolCallEvent,
    ToolResultEvent,
    TrajectoryEvent,
    UsageEvent,
    decode_event,
)
from rexhunter.loop import run_hunt

pytestmark = pytest.mark.anyio

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gym_sonnet_5_5_2026-11-26"
RECORDED_MODEL = "claude-sonnet-5-5"  # every call was recorded on it (each stream's "model")
# The daemon's settings for these hunts: the two ceilings the live environment set, server.py's
# default MAX_ITERATIONS, and the scheduler's default tool timeout and retry budget. The daemon
# ceiling folds the whole log, which in a replay holds the one run, so it never bites here; it did
# not stop these runs live either.
COST_CEILING_USD = 0.20
DAEMON_SPEND_CEILING_USD = 0.50
MAX_ITERATIONS = 50
TOOL_TIMEOUT_S = 30.0
RETRY_BUDGET = 2
BOARD_TOOLS = ["list_postings", "read_posting"]
# Left out of the pinned digest, because no replay reproduces them: an ErrorEvent's `detail` is a
# traceback that names its machine's paths, and a PreyCapturedEvent's `prey_id` is a fresh uuid4.
UNPINNED: dict[str, set[str]] = {"error": {"detail"}, "prey_captured": {"prey_id"}}

type ToolUse = tuple[str, str, dict[str, Any]]


class Run(NamedTuple):
    run_id: str
    territory: str
    outcome: tuple[str, str | None]
    sha256: dict[str, str]  # each call's stream, pinned: a rewritten fixture fails here first
    tool_uses: list[ToolUse]  # each call's one tool_use, read off its stream
    usage: list[tuple[int, int]]  # each call's (input, output) tokens
    deltas: list[int]  # the ThinkingDeltas relayed during each call
    texts: tuple[int, ...]  # the calls whose turn says something before its tool_use
    tool_errors: dict[int, list[bool]]  # per call, `retryable` of each ErrorEvent its tool wrote
    waits: list[float]  # each tool retry's wait, as conftest's `retry_waits` records it
    spend_usd: float  # the run's fold under today's table
    log_sha256: str  # the live log's payloads but the exchanges', UNPINNED left out, "\n"-joined


RUNS = [
    Run(
        run_id="f0a5507e-d644-4d7d-938e-4a920c2a07ae",
        territory="gym-greenhouse",
        outcome=("completed", None),
        sha256={
            "call_01.json": "dd465b89146123ab3b4c9e66c16127d51b3c8c485b755d871de2463c144f0464",
            "call_02.json": "908ae154c205b9c40f0d4d2c091c233b73f625ba95b4e1e0ea2a80b12f8447dd",
            "call_03.json": "afd4a3654f398d000db0e98fd9f46dd72d7f8128e0c4f2cce56f8badb44933db",
            "call_04.json": "09e3f8f10db29aa4b1bf8e28d20a08e72661bf257162adfd0f46e429fbb86c32",
            "call_05.json": "105b67c4e6af65fdbf6bdf05451d000717e6dd43a00f49765b41de967db949a9",
            "call_06.json": "54aaf0ea56af7fc47b25d3ebcf690306b75fa1a2f95bb1e5a5eb009ab5726509",
        },
        tool_uses=[
            ("toolu_01JMNGDHLFDKvqmvjrkXSt7p", "list_postings", {"board": "gym-greenhouse"}),
            (
                "toolu_01A6KU3AbzySW7Rj1RZLuNHC",
                "read_posting",
                {"board": "gym-greenhouse", "posting_id": "4001001"},
            ),
            (
                "toolu_01TWMdMiRc8bqi7qAP74qobm",
                "read_posting",
                {"board": "gym-greenhouse", "posting_id": "4001002"},
            ),
            (
                "toolu_01JLQFRVjEUD9vEzxcYutNQD",
                "read_posting",
                {"board": "gym-greenhouse", "posting_id": "4001003"},
            ),
            (
                "toolu_01TVNjY48vF8QKmQeh2SwNUQ",
                "read_posting",
                {"board": "gym-greenhouse", "posting_id": "4001004"},
            ),
            (
                "toolu_01WjQkBXmvgkr8p9ep2zZovk",
                "hunt_complete",
                {
                    "catch": [
                        {"board": "gym-greenhouse", "posting_id": "4001001"},
                        {"board": "gym-greenhouse", "posting_id": "4001002"},
                        {"board": "gym-greenhouse", "posting_id": "4001003"},
                        {"board": "gym-greenhouse", "posting_id": "4001004"},
                    ]
                },
            ),
        ],
        usage=[(1131, 53), (1493, 79), (1759, 79), (2030, 79), (2289, 79), (2553, 150)],
        deltas=[0, 0, 0, 0, 0, 0],
        texts=(),
        tool_errors={},
        waits=[],
        spend_usd=0.0277,  # 11255 x $2/M + 519 x $10/M
        log_sha256="84004aaea91eb6e01eea52f9e051a9cd9a1fd31db67431ed26d9049e75454852",
    ),
    Run(
        run_id="b929510c-5b9a-4b7d-b27a-4f4e0b2aadbd",
        territory="gym-lever",
        outcome=("completed", None),
        sha256={
            "call_01.json": "fc407a6114a194838a32053d151ac949d588b0e707f3fa060c0786943d8fbce1",
            "call_02.json": "d83ff02299a16b168d48db07a43ea6b2d1dc2bd17511b728ac0777de4852c5d9",
            "call_03.json": "fc4628aceadd1998b021ba052fa3d70fbd62e0782c5f8943b8709a944acf75ae",
            "call_04.json": "ccb44e0aadb96d74d410422f7fe39d1025ceddea8d53c7ebf8ea8c376369f886",
            "call_05.json": "3259ae691eac49f7831649e454906c38478b919985c053c275dc4a151081d571",
            "call_06.json": "a6fd688d92494c557cec0e6e8f4c8edaf9ddc6db84d66b1b6edc6562545c593e",
        },
        tool_uses=[
            ("toolu_01JqhsMTQXKTYvMgiwZN9bmf", "list_postings", {"board": "gym-lever"}),
            (
                "toolu_01RRjf5CmNq1QUhxJYLi118V",
                "read_posting",
                {"board": "gym-lever", "posting_id": "00005e1f-0000-4000-8000-000000000001"},
            ),
            (
                "toolu_016W6P1m4Q55kHNXpCQNM5ue",
                "read_posting",
                {"board": "gym-lever", "posting_id": "00005e1f-0000-4000-8000-000000000002"},
            ),
            (
                "toolu_019tNJoXKQWNhKe9ELLoePKA",
                "read_posting",
                {"board": "gym-lever", "posting_id": "00005e1f-0000-4000-8000-000000000003"},
            ),
            (
                "toolu_01CXh5nrybDzueMBawAkzapi",
                "read_posting",
                {"board": "gym-lever", "posting_id": "00005e1f-0000-4000-8000-000000000004"},
            ),
            (
                "toolu_01YGw9tUMSb1JBvbhUVViovC",
                "hunt_complete",
                {
                    "catch": [
                        {
                            "board": "gym-lever",
                            "posting_id": "00005e1f-0000-4000-8000-000000000001",
                        },
                        {
                            "board": "gym-lever",
                            "posting_id": "00005e1f-0000-4000-8000-000000000002",
                        },
                        {
                            "board": "gym-lever",
                            "posting_id": "00005e1f-0000-4000-8000-000000000003",
                        },
                        {
                            "board": "gym-lever",
                            "posting_id": "00005e1f-0000-4000-8000-000000000004",
                        },
                    ]
                },
            ),
        ],
        usage=[(1125, 50), (1618, 92), (1908, 92), (2197, 92), (2504, 92), (2802, 238)],
        deltas=[0, 0, 0, 0, 0, 12],
        texts=(),
        tool_errors={},
        waits=[],
        spend_usd=0.030868,  # 12154 x $2/M + 656 x $10/M
        log_sha256="19d71cae7571f4243e0f5b718187a1990b2b164a76ed953ec567d293fafb9173",
    ),
    Run(
        run_id="1303d493-99d6-42a0-8af4-41d2c973396b",
        territory="gym-storm",
        outcome=("completed", None),
        sha256={
            "call_01.json": "8dc76db7b7ff54584a3f0dcfcda683e5b62c032b3d2f76100ad0676549fca0df",
            "call_02.json": "03ffef13c82d3e17ac37f6da10e535da6facdc2142ab418fc56c5e7c452e5702",
            "call_03.json": "19888185bca25b4780db02f675a2ff66644beac4f8d706cc8f34522e1697ec4b",
            "call_04.json": "988b03a07a1f964fc8aef0d312ef291b03a0ffbd697af62bc81dc33b19c5f5a8",
            "call_05.json": "f54b92d4b9a6014c63e88c4f909914a7832da100474be5d04cdd0cdcdf2ed337",
            "call_06.json": "c3073e4a3aa0bdf5308b7109a0b146ed161465e1ceaa4d5e9949bba4d66e562d",
        },
        tool_uses=[
            ("toolu_01UYyqsejcFpmGzbog5ewvAp", "list_postings", {"board": "gym-storm"}),
            (
                "toolu_012Wyw6qpYdpuGa6izw3Uqj4",
                "read_posting",
                {"board": "gym-storm", "posting_id": "00005e20-0000-4000-8000-000000000001"},
            ),
            (
                "toolu_016RD45tqHcyWHzHEo3EjjR6",
                "read_posting",
                {"board": "gym-storm", "posting_id": "00005e20-0000-4000-8000-000000000002"},
            ),
            (
                "toolu_015t3fsS7N8MHvR6F2oJ2sZg",
                "read_posting",
                {"board": "gym-storm", "posting_id": "00005e20-0000-4000-8000-000000000003"},
            ),
            (
                "toolu_01MCeZtDubLoK8KsfHDoNm64",
                "read_posting",
                {"board": "gym-storm", "posting_id": "00005e20-0000-4000-8000-000000000004"},
            ),
            (
                "toolu_01Uhf3n4R7DNAVv3RMawC4YE",
                "hunt_complete",
                {
                    "catch": [
                        {
                            "board": "gym-storm",
                            "posting_id": "00005e20-0000-4000-8000-000000000001",
                        },
                        {
                            "board": "gym-storm",
                            "posting_id": "00005e20-0000-4000-8000-000000000002",
                        },
                        {
                            "board": "gym-storm",
                            "posting_id": "00005e20-0000-4000-8000-000000000003",
                        },
                        {
                            "board": "gym-storm",
                            "posting_id": "00005e20-0000-4000-8000-000000000004",
                        },
                    ]
                },
            ),
        ],
        usage=[(1125, 50), (1610, 91), (1902, 91), (2193, 91), (2502, 91), (2802, 244)],
        deltas=[0, 0, 0, 0, 0, 13],
        texts=(),
        tool_errors={1: [True, True]},
        # The board's retry-after, 2 s: the jitter's bounds on retries 1 and 2, 0.5 and 1 s, are
        # below it.
        waits=[2.0, 2.0],
        spend_usd=0.030848,  # 12134 x $2/M + 658 x $10/M
        log_sha256="add9a4bdde9faa71a91199ac2ced0e1e2309c2022d075bef0c22e2c9870c14d6",
    ),
    Run(
        run_id="19618d54-198e-45dd-b356-29b730da7972",
        territory="gym-mojibake",
        outcome=("completed", None),
        sha256={
            "call_01.json": "90b186b2176d8e7d974f61334aa0c7a269cb5b37f3c1afd7e5acd278c9de26b7",
            "call_02.json": "ee95c69b99a33b8fe9504c8442bf418701914a7012d27d9ae0f45521b330115b",
            "call_03.json": "8f32326c3a7f8c052d4314ffbbb699b3a709d2d90ea4bc4f3db225abff6fd620",
            "call_04.json": "38cf264c811b1c934d1935480cbb0bcb581fa4cb23b91e0ebdcf166029a37833",
            "call_05.json": "bf4c9d1b317a326b2566cd0b2d1137190142505d2b5f18c7af039ac68937e312",
            "call_06.json": "6464cb9a65a3fbf15843dd72724e7a1d091125bbbc94a7cbccb6bd8d2976b3f6",
        },
        tool_uses=[
            ("toolu_01NDiKNimSct79XZKQn8Gtsf", "list_postings", {"board": "gym-mojibake"}),
            (
                "toolu_01LmEAZGzRjK9GNudZshq253",
                "read_posting",
                {"board": "gym-mojibake", "posting_id": "4003001"},
            ),
            (
                "toolu_01CF6jduAV4VYoB6Xyw8QuFw",
                "read_posting",
                {"board": "gym-mojibake", "posting_id": "4003002"},
            ),
            (
                "toolu_019VefNjAvkPqSryFNETir3U",
                "read_posting",
                {"board": "gym-mojibake", "posting_id": "4003003"},
            ),
            (
                "toolu_01WnT6qfiFPbeFofxyeY35s9",
                "read_posting",
                {"board": "gym-mojibake", "posting_id": "4003004"},
            ),
            (
                "toolu_01SuMciTJRcKEaoSumsQhSEx",
                "hunt_complete",
                {
                    "catch": [
                        {"board": "gym-mojibake", "posting_id": "4003001"},
                        {"board": "gym-mojibake", "posting_id": "4003002"},
                        {"board": "gym-mojibake", "posting_id": "4003003"},
                        {"board": "gym-mojibake", "posting_id": "4003004"},
                    ]
                },
            ),
        ],
        usage=[(1129, 52), (1558, 103), (1872, 78), (2174, 78), (2443, 78), (2718, 146)],
        deltas=[0, 0, 0, 0, 0, 0],
        texts=(2,),
        tool_errors={},
        waits=[],
        spend_usd=0.029138,  # 11894 x $2/M + 535 x $10/M
        log_sha256="b68b0f5d0a840183cb96f0e5d6fb04fdc30a803dbd157a6cff7c87c40912f0fc",
    ),
    Run(
        run_id="f726301f-e5ba-485a-918a-29e527009eae",
        territory="gym-drift",
        outcome=("aborted", "tool failure"),
        sha256={
            "call_01.json": "1524f68a30ab7bb4a524ab22a682b3e0afc6c18cad5608a65aaae2b41229a758",
        },
        tool_uses=[
            ("toolu_01Ut6vut3NzdJgfnfWNv44AT", "list_postings", {"board": "gym-drift"}),
        ],
        usage=[(1125, 50)],
        deltas=[0],
        texts=(),
        tool_errors={1: [False]},
        waits=[],
        spend_usd=0.00275,  # 1125 x $2/M + 50 x $10/M
        log_sha256="ef8d8aa3080d2ed8e698e18477f7e13de974833652e80e5c8f36f52d6f23c29a",
    ),
    Run(
        run_id="e1e6b3d5-f943-4a20-b925-faca41f4ce24",
        territory="gym-pages",
        outcome=("aborted", "cost ceiling"),
        sha256={
            "call_01.json": "fe226f885a17816b791a70f93a483ca7d95125c0fb96a5a54b249113cbab15a1",
            "call_02.json": "18193f06f9c9c4c4d6aeacd18cae0164e8937f7d364bb2a70eb9bce5ae327c38",
            "call_03.json": "81b529d1f8bf792a8a024d9fba78024ac3bc4cda17d40e71dd223e4d5b03eed1",
            "call_04.json": "0e4d8a66fc1c5cc41d84384400729d0097e8537c27dd5d53f45e6a05aba029a3",
            "call_05.json": "f0408addebe9d1836a0295c40fc7b26fc3c19bd4cfb7f9e22b53bd3cd5122b5b",
            "call_06.json": "7fa381696ddc3ee4351ccf1d49eb20c69ec40d4e67a265fb150488fbae0dfb22",
            "call_07.json": "cb52f12baf9f52f69d810fadff3c0e852501fb299d0c22c765aad314230fa440",
            "call_08.json": "4ba5419879e8765fec89f0d9cb6ce36059b64eb044ec7eb438bb92b72affa0bf",
            "call_09.json": "2bc6d55bcf0a8ae8540263e2b0c58a185b8972d05f1a9dd76d910fddde655137",
            "call_10.json": "395fbc8cbe271e088682c4b959d17030952a347ec44ec9144224a451df17af2a",
            "call_11.json": "3087205117c1689e991687b394e255635c3c9c29738bd744e8bdc42e70addde5",
            "call_12.json": "e86d19df0d4a506afee73bf1ba0f5c22a1bd26d396af6e69b47c2eeb08666c72",
            "call_13.json": "7582a0d9f9dd27206d812fc6d1d9291a02516eadd3a4ea1928b964c10a5a473d",
            "call_14.json": "df11bb856884b4a214dc67a2a05766f48b10b4f23433cde29a67dd9cadc6ef5b",
            "call_15.json": "70e9d3a0b2b28b0c7910658d7ef422cdbba70170927a1a17b77727f6965d09bb",
            "call_16.json": "e54e63148ee7f80f3e77286dcec64d8aacca8d211f56540135dfae1c961f53d1",
            "call_17.json": "b281f91b1e8f6b6dd1f1dc837718db82991f7ef528ff09910c4676fb1a93b864",
            "call_18.json": "495a0e9e6786d0dde78328bf344afbb24f879072f51371a152370198d743c14f",
        },
        tool_uses=[
            ("toolu_01Tcm4jMWBRoWWfCnCzEeVgS", "list_postings", {"board": "gym-pages"}),
            ("toolu_01Ke25gy6JrSoqj6NU8Vdgdz", "list_postings", {"board": "gym-pages", "page": 2}),
            (
                "toolu_019Zqd4FBRLt69Q1Mie1YsWx",
                "read_posting",
                {"board": "gym-pages", "posting_id": "00005e21-0000-4000-8000-000000000001"},
            ),
            (
                "toolu_011R1oxvkKfzDFbHHM8cDiaP",
                "read_posting",
                {"board": "gym-pages", "posting_id": "00005e21-0000-4000-8000-000000000002"},
            ),
            ("toolu_01Lju8ncY7oxMBWgRZkSCqGC", "list_postings", {"board": "gym-pages", "page": 3}),
            (
                "toolu_01RFJGt2ZNzCgM8v8ZgKFxZX",
                "read_posting",
                {"board": "gym-pages", "posting_id": "00005e21-0000-4000-8000-000000000003"},
            ),
            (
                "toolu_01DzthKtZ99RiBAURYa9Ntma",
                "read_posting",
                {"board": "gym-pages", "posting_id": "00005e21-0000-4000-8000-000000000004"},
            ),
            (
                "toolu_01W2n5vDgWexxPveudpkBmf1",
                "read_posting",
                {"board": "gym-pages", "posting_id": "00005e21-0000-4000-8000-000000000009"},
            ),
            (
                "toolu_016MEddZZ24c12FHUDxAhXDY",
                "read_posting",
                {"board": "gym-pages", "posting_id": "00005e21-0000-4000-8000-00000000000a"},
            ),
            (
                "toolu_01RJ1u74HxJkMbUqcvBqosdY",
                "read_posting",
                {"board": "gym-pages", "posting_id": "00005e21-0000-4000-8000-00000000000b"},
            ),
            (
                "toolu_01HaYbq7oar8fZs6QhddqdxZ",
                "read_posting",
                {"board": "gym-pages", "posting_id": "00005e21-0000-4000-8000-00000000000c"},
            ),
            (
                "toolu_01SiYVixfwiaecYji24dcjdR",
                "read_posting",
                {"board": "gym-pages", "posting_id": "00005e21-0000-4000-8000-000000000011"},
            ),
            (
                "toolu_015XDSqj3811c2zjzERy83eD",
                "read_posting",
                {"board": "gym-pages", "posting_id": "00005e21-0000-4000-8000-000000000012"},
            ),
            (
                "toolu_01CRhqSXp1F2xqV2JmREQgbj",
                "read_posting",
                {"board": "gym-pages", "posting_id": "00005e21-0000-4000-8000-000000000013"},
            ),
            (
                "toolu_016q2oqQ1PxP3RqvijUR14zS",
                "read_posting",
                {"board": "gym-pages", "posting_id": "00005e21-0000-4000-8000-000000000014"},
            ),
            (
                "toolu_01XZ5X5yK1rrV3xoGxWrK43W",
                "read_posting",
                {"board": "gym-pages", "posting_id": "00005e21-0000-4000-8000-000000000019"},
            ),
            (
                "toolu_01JDSeJsBaD2M9fDf6HmL9av",
                "read_posting",
                {"board": "gym-pages", "posting_id": "00005e21-0000-4000-8000-00000000001a"},
            ),
            (
                "toolu_01SwjYA6xkzH6QxjRHsiZbRn",
                "read_posting",
                {"board": "gym-pages", "posting_id": "00005e21-0000-4000-8000-00000000001b"},
            ),
        ],
        usage=[
            (1125, 50),
            (2282, 125),
            (3507, 130),
            (3846, 91),
            (4154, 101),
            (4290, 137),
            (4644, 91),
            (4944, 108),
            (5261, 92),
            (5572, 92),
            (5883, 92),
            (6186, 103),
            (6498, 91),
            (6806, 91),
            (7114, 91),
            (7414, 91),
            (7714, 92),
            (8025, 92),
        ],
        deltas=[0, 16, 9, 0, 10, 11, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        texts=(12,),
        tool_errors={},
        waits=[],
        spend_usd=0.20813,  # 95265 x $2/M + 1760 x $10/M
        log_sha256="73946aef3ff7298f0313b5934bd4de544d570752a958e342752f1428f6739f44",
    ),
]


IDS = [run.territory for run in RUNS]

MockAnthropic = Callable[..., anthropic.AsyncAnthropic]


class Replay(NamedTuple):
    events: list[TrajectoryEvent]
    payloads: list[tuple[str, str]]  # each row's (type, payload) as stored, in seq order
    runs_row: tuple[Any, ...]  # (outcome, abort_reason, cost_ceiling_usd, max_iterations)
    prey: list[tuple[str, str, str]]  # each pen row's (id, board, posting_id), in capture order
    requests: list[httpx2.Request]


def _streams(run: Run) -> list[bytes]:
    return [(FIXTURE_DIR / f"hunt_{run.run_id}" / name).read_bytes() for name in run.sha256]


def _assemble(raw: bytes) -> dict[str, Any]:
    assembler = brain.StreamAssembler()
    for event in brain.iter_sse_events(raw):
        assembler.feed(event)
    return assembler.assembled()


def _pinned_log(payloads: list[tuple[str, str]]) -> str:
    # Each payload goes through the typed model and out again, which gives back the stored text on
    # these logs, so the digest differs from the payloads' own only by the UNPINNED fields.
    return "\n".join(
        decode_event(payload).model_dump_json(exclude=UNPINNED.get(kind))
        for kind, payload in payloads
        if kind != "brain_exchange"
    )


async def _replay(run: Run, db_path: Path, mock_anthropic: MockAnthropic) -> Replay:
    streams = _streams(run)
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, content=streams[len(requests) - 1])

    # A fresh gym per replay: a storm counts down per gym instance, and the recorded gym-storm hunt
    # was the first after the daemon booted, so it met the storm.
    board_client = BoardClient(httpx2.ASGITransport(app=build_gym(GYM_CATALOGUE)))
    async with board_client, mock_anthropic(handler, api_key="replay") as client:
        registry = build_registry(board_client, GYM_BOARDS)
        # The daemon's live brain on the gym (`brain.select_brain_for` with the gym prompt and
        # seed), on the recorded model.
        brain_for = brain.adapter_brain_for(
            client=client,
            model=RECORDED_MODEL,
            registry=registry,
            max_tokens=brain.HUNT_MAX_TOKENS,
            thinking=brain.HUNT_THINKING,
            stream=True,
            system=GYM_SYSTEM_PROMPT,
            seed=gym_seed,
        )
        conn = await db.connect(db_path)
        try:
            run_id = await run_hunt(
                conn,
                territory=run.territory,
                brain=brain_for(run.territory),
                registry=registry,
                tool_timeout_s=TOOL_TIMEOUT_S,
                retry_budget=RETRY_BUDGET,
                max_iterations=MAX_ITERATIONS,
                cost_ceiling_usd=COST_CEILING_USD,
                daemon_spend_ceiling_usd=DAEMON_SPEND_CEILING_USD,
            )
            events = await db.read_events(conn, run_id)
            rows = await conn.execute_fetchall(
                "SELECT type, payload FROM trajectory_events WHERE run_id = ? ORDER BY seq",
                (run_id,),
            )
            caps = await conn.execute_fetchall(
                "SELECT outcome, abort_reason, cost_ceiling_usd, max_iterations FROM runs"
                " WHERE id = ?",
                (run_id,),
            )
            pen = await conn.execute_fetchall(
                "SELECT id, board, posting_id FROM prey WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            )
        finally:
            await conn.close()
    payloads = [(str(row[0]), str(row[1])) for row in rows]
    prey = [(str(row[0]), str(row[1]), str(row[2])) for row in pen]
    return Replay(events, payloads, tuple(next(iter(caps))), prey, requests)


def _expected_types(run: Run) -> list[str]:
    # A call's deltas are written during the call, before its exchange. A board tool then writes its
    # call, an ErrorEvent per failed attempt and, when its last failure was retried, its result; a
    # catch writes one capture per posting; the cost ceiling ends a run with no event.
    types: list[str] = []
    for n, ((_id, name, tool_input), deltas) in enumerate(
        zip(run.tool_uses, run.deltas, strict=True), start=1
    ):
        types += ["thinking_delta"] * deltas + ["brain_exchange", "usage"]
        failures = run.tool_errors.get(n, [])
        if name in BOARD_TOOLS:
            types += ["tool_call"] + ["error"] * len(failures)
            if not failures or failures[-1]:
                types.append("tool_result")
        elif name == "hunt_complete":
            types += ["prey_captured"] * len(tool_input["catch"])
    return types


def test_the_fixture_holds_the_six_recorded_runs_unchanged() -> None:
    assert sorted(run.territory for run in RUNS) == sorted(GYM_BOARDS)  # one run per territory
    assert sorted(p.name for p in FIXTURE_DIR.iterdir()) == sorted(
        f"hunt_{run.run_id}" for run in RUNS
    )
    for run in RUNS:
        run_dir = FIXTURE_DIR / f"hunt_{run.run_id}"
        assert sorted(p.name for p in run_dir.iterdir()) == list(run.sha256)
        for name, digest in run.sha256.items():
            assert hashlib.sha256((run_dir / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("run", RUNS, ids=IDS)
def test_each_stream_assembles_to_the_recorded_turn(run: Run) -> None:
    streams = _streams(run)
    assert len(streams) == len(run.tool_uses) == len(run.usage) == len(run.deltas)
    for n, (raw, (tool_use_id, name, tool_input), (n_in, n_out), deltas) in enumerate(
        zip(streams, run.tool_uses, run.usage, run.deltas, strict=True), start=1
    ):
        turn = _assemble(raw)
        assert turn["model"] == RECORDED_MODEL
        assert turn["stop_reason"] == "tool_use"
        # A call that thought carries one signed thinking block first; a call in `texts` says
        # something before its tool call.
        types = (["thinking"] if deltas else []) + (["text"] if n in run.texts else [])
        assert [block["type"] for block in turn["content"]] == [*types, "tool_use"]
        tool_use = turn["content"][-1]
        assert (tool_use["id"], tool_use["name"], tool_use["input"]) == (
            tool_use_id,
            name,
            tool_input,
        )
        assert (turn["usage"]["input_tokens"], turn["usage"]["output_tokens"]) == (n_in, n_out)


@pytest.mark.parametrize("run", RUNS, ids=IDS)
async def test_the_replay_reproduces_the_live_log(
    run: Run, tmp_path: Path, mock_anthropic: MockAnthropic, retry_waits: list[float]
) -> None:
    replay = await _replay(run, tmp_path / "rex.db", mock_anthropic)
    streams = _streams(run)
    assert len(replay.requests) == len(streams)  # one request per stored attempt, as live
    assert replay.runs_row == (*run.outcome, COST_CEILING_USD, MAX_ITERATIONS)
    assert [kind for kind, _payload in replay.payloads] == _expected_types(run)

    # Every event but the exchanges is the live log's, payload for payload, UNPINNED aside.
    log = _pinned_log(replay.payloads)
    assert hashlib.sha256(log.encode()).hexdigest() == run.log_sha256

    # Each exchange stores its response as served, the fixture's bytes, with its status.
    exchanges = [e for e in replay.events if isinstance(e, BrainExchangeEvent)]
    assert [(x.status, x.raw_response) for x in exchanges] == [(200, raw) for raw in streams]

    # The deltas relayed during each call join to that call's assembled thinking.
    relayed: list[str] = []
    pending: list[str] = []
    for event in replay.events:
        if isinstance(event, ThinkingDelta):
            pending.append(event.text)
        elif isinstance(event, BrainExchangeEvent):
            relayed.append("".join(pending))
            pending = []
    turns = [_assemble(raw)["content"] for raw in streams]
    assert relayed == [
        "".join(block["thinking"] for block in turn if block["type"] == "thinking")
        for turn in turns
    ]

    usage = [e for e in replay.events if isinstance(e, UsageEvent)]
    assert [(u.model, u.input_tokens, u.output_tokens) for u in usage] == [
        (RECORDED_MODEL, n_in, n_out) for n_in, n_out in run.usage
    ]
    assert cost.fold_cost(replay.events) == pytest.approx(run.spend_usd, abs=1e-9)

    # Each board tool use is one ToolCallEvent under the provider's tool_use_id, and each result
    # pairs with its call by that id and the same arguments.
    calls = [e for e in replay.events if isinstance(e, ToolCallEvent)]
    results = [e for e in replay.events if isinstance(e, ToolResultEvent)]
    assert [(c.tool_use_id, c.tool) for c in calls] == [
        (tool_use_id, name) for tool_use_id, name, _input in run.tool_uses if name in BOARD_TOOLS
    ]
    paired = {(c.tool_use_id, c.tool, c.raw_request) for c in calls}
    assert all((r.tool_use_id, r.tool, r.raw_request) in paired for r in results)

    # A failed attempt is an ErrorEvent with its traceback, and a retry waits out the board's
    # `retry-after`.
    errors = [e for e in replay.events if isinstance(e, ErrorEvent)]
    assert [e.retryable for e in errors] == [
        retryable for failures in run.tool_errors.values() for retryable in failures
    ]
    assert all((e.detail or "").startswith("Traceback (most recent call last):") for e in errors)
    assert retry_waits == run.waits

    # The catch reaches the pen: one row per capture, under the capture's own id.
    captures = [e for e in replay.events if isinstance(e, PreyCapturedEvent)]
    catch = [
        ref
        for _id, name, tool_input in run.tool_uses
        if name == "hunt_complete"
        for ref in tool_input["catch"]
    ]
    assert [{"board": c.board, "posting_id": c.posting_id} for c in captures] == catch
    assert replay.prey == [(c.prey_id, c.board, c.posting_id) for c in captures]

    # Each request has the shape and the wire the daemon used on the gym, and call n sends back
    # the n turns before it, each exactly as stored: dicts and key order.
    for n, request in enumerate(replay.requests):
        body = json.loads(request.content)
        assert body["model"] == RECORDED_MODEL
        assert body["system"] == GYM_SYSTEM_PROMPT
        assert [tool["name"] for tool in body["tools"]] == [
            *BOARD_TOOLS,
            "hunt_complete",
            "needs_help",
        ]
        assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
        assert (body["max_tokens"], body["stream"]) == (4096, True)
        assert request.headers.get_list("authorization") == ["Bearer replay"]
        assert request.headers.get_list("anthropic-version") == ["2026-10-21"]
        messages = body["messages"]
        assert messages[0] == {"role": "user", "content": gym_seed(run.territory)}
        assert [m["role"] for m in messages] == ["user", *["assistant", "user"] * n]
        sent = [messages[1 + 2 * k]["content"] for k in range(n)]
        assert sent == turns[:n]
        assert [[list(b) for b in turn] for turn in sent] == [
            [list(b) for b in turn] for turn in turns[:n]
        ]
