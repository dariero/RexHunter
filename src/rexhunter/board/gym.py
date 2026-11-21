"""The gym (ADR pillar 2, `P2.4`): a mock job board in the Greenhouse and Lever shapes, with
injected hazards, served in-process as a FastAPI app.

Each route answers in the shape the board's docs give (read 21 Nov 2026): Greenhouse's job list
and job, Lever's posting list and posting. GET only: any other method on a route is 405, an
unknown path or posting 404. A board's `HazardPlan` bends its answers the way live boards do:
schema drift, a rate-limit storm, mojibake and pagination traps. The gym's answers are a function
of its catalogue, its plans and the order of the requests it got (a storm counts down per gym
instance), so tests build a fresh gym per case.

Each posting carries a `planted` mark, prey or decoy, for a later eval to score a hunt against.
No route serves it: the gym builds every body from the fields the docs name, never from the
dataclass.
"""

import html
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Literal

from fastapi import FastAPI, Response

from rexhunter.board import PAGE_SIZE, Ats

type Drift = Literal["none", "added_field", "renamed_field", "id_as_string", "envelope_renamed"]
type Encoding = Literal["clean", "invalid_utf8", "double_encoded", "bom"]
type Paging = Literal["honest", "ignore_skip", "exact_multiple"]

# The hazards each documented shape has: Greenhouse's job id is an integer and its list an
# envelope with `meta.total`, and it is not paginated; Lever's list is a bare array paged by
# `skip` and `limit`. A plan that names a hazard its board's shape lacks is refused at build.
_GREENHOUSE_ONLY = ("id_as_string", "envelope_renamed")
_UPDATED_AT = "2026-11-01T09:00:00-04:00"


@dataclass(frozen=True)
class HazardPlan:
    drift: Drift = "none"
    storm: int = 0  # answers of `storm_status` before the first 200, counted per gym instance
    storm_status: int = 429
    retry_after: str | None = None  # the storm's `retry-after` header, sent as given
    encoding: Encoding = "clean"
    paging: Paging = "honest"
    meta_total_lie: bool = False


@dataclass(frozen=True)
class GymPosting:
    posting_id: str  # digits on a Greenhouse board, which serves it as an integer
    title: str
    location: str
    team: str
    description: tuple[str, ...]  # paragraphs of plain text
    planted: Literal["prey", "decoy"]


@dataclass(frozen=True)
class GymBoard:
    ats: Ats
    postings: tuple[GymPosting, ...]
    plan: HazardPlan = HazardPlan()


def _refuse(board: GymBoard) -> str | None:
    plan = board.plan
    if board.ats == "greenhouse":
        if any(not posting.posting_id.isdigit() for posting in board.postings):
            return "a Greenhouse posting id is an integer"
        if plan.paging != "honest":
            return "Greenhouse serves its whole list, so it has no paging hazard"
    else:
        if plan.drift in _GREENHOUSE_ONLY or plan.meta_total_lie:
            return "a Lever list is a bare array of postings with string ids"
        if plan.paging == "exact_multiple" and (
            not board.postings or len(board.postings) % PAGE_SIZE
        ):
            return f"an exact_multiple board holds a whole number of {PAGE_SIZE}-posting pages"
    text = "".join(p.title + p.location + p.team + "".join(p.description) for p in board.postings)
    if plan.encoding in ("invalid_utf8", "double_encoded") and text.isascii():
        return "a mojibake board needs a posting with a character outside ASCII"
    if plan.encoding == "invalid_utf8" and any(ord(char) > 0xFF for char in text):
        return "an invalid_utf8 board's text is Latin-1, one byte per character"
    return None


def _encode(body: object, encoding: Encoding) -> bytes:
    text = json.dumps(body, ensure_ascii=False)
    match encoding:
        case "clean":
            return text.encode()
        case "invalid_utf8":
            return text.encode("latin-1")  # "é" goes out as a lone 0xE9
        case "double_encoded":
            return text.encode().decode("latin-1").encode()  # "é" goes out as "Ã©", valid UTF-8
        case "bom":
            return b"\xef\xbb\xbf" + text.encode()


def _drifted(posting: dict[str, Any], drift: Drift, title: str) -> dict[str, Any]:
    match drift:
        case "added_field":
            return {**posting, "remote": True}
        case "renamed_field":
            return {("name" if key == title else key): value for key, value in posting.items()}
        case "id_as_string":
            return {**posting, "id": str(posting["id"])}
        case "none" | "envelope_renamed":
            return posting


def _greenhouse_job(token: str, posting: GymPosting, *, content: bool) -> dict[str, Any]:
    job: dict[str, Any] = {
        "id": int(posting.posting_id),
        "title": posting.title,
        "updated_at": _UPDATED_AT,
        "location": {"name": posting.location},
        "absolute_url": f"http://gym/greenhouse/{token}/jobs/{posting.posting_id}",
    }
    if content:
        # Greenhouse sends the job's HTML converted to entities ("&lt;p&gt;").
        markup = "".join(f"<p>{html.escape(paragraph)}</p>" for paragraph in posting.description)
        job["content"] = html.escape(markup)
    return job


def _lever_posting(site: str, posting: GymPosting) -> dict[str, Any]:
    url = f"http://gym/lever/{site}/{posting.posting_id}"
    return {
        "id": posting.posting_id,
        "text": posting.title,
        "categories": {
            "location": posting.location,
            "commitment": "Full-time",
            "team": posting.team,
            "department": posting.team,
        },
        "hostedUrl": url,
        "applyUrl": f"{url}/apply",
        "description": "".join(f"<p>{html.escape(p)}</p>" for p in posting.description),
        "descriptionPlain": "\n".join(posting.description),
    }


def _json(status: int, body: object, headers: Mapping[str, str] | None = None) -> Response:
    return Response(
        json.dumps(body).encode(),
        status_code=status,
        headers=dict(headers or {}),
        media_type="application/json",
    )


def build_gym(boards: Mapping[str, GymBoard]) -> FastAPI:
    """The gym app over `boards`, keyed by each board's token (Greenhouse) or site (Lever)."""
    for token, board in boards.items():
        reason = _refuse(board)
        if reason is not None:
            raise ValueError(f"gym board {token!r}: {reason}")
    storms = {token: board.plan.storm for token, board in boards.items()}
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def answer(token: str, ats: Ats, build: Callable[[GymBoard], object | None]) -> Response:
        board = boards.get(token)
        if board is None or board.ats != ats:
            return _json(404, {"error": "Not Found"})
        if storms[token] > 0:
            storms[token] -= 1
            plan = board.plan
            headers = {} if plan.retry_after is None else {"retry-after": plan.retry_after}
            phrase = HTTPStatus(plan.storm_status).phrase
            return _json(plan.storm_status, {"error": phrase}, headers)
        body = build(board)
        if body is None:
            return _json(404, {"error": "Not Found"})
        return Response(_encode(body, board.plan.encoding), media_type="application/json")

    def find(board: GymBoard, posting_id: str) -> GymPosting | None:
        return next((p for p in board.postings if p.posting_id == posting_id), None)

    @app.get("/greenhouse/v1/boards/{token}/jobs")
    async def greenhouse_jobs(token: str, content: bool = False) -> Response:
        def build(board: GymBoard) -> object:
            plan = board.plan
            jobs = [
                _drifted(_greenhouse_job(token, p, content=content), plan.drift, "title")
                for p in board.postings
            ]
            total = len(jobs) + 1000 if plan.meta_total_lie else len(jobs)
            envelope = "postings" if plan.drift == "envelope_renamed" else "jobs"
            return {envelope: jobs, "meta": {"total": total}}

        return answer(token, "greenhouse", build)

    @app.get("/greenhouse/v1/boards/{token}/jobs/{job_id}")
    async def greenhouse_job(token: str, job_id: str) -> Response:
        def build(board: GymBoard) -> object | None:
            posting = find(board, job_id)
            if posting is None:
                return None
            job = _greenhouse_job(token, posting, content=True)
            return _drifted(job, board.plan.drift, "title")

        return answer(token, "greenhouse", build)

    @app.get("/lever/v0/postings/{site}")
    async def lever_postings(site: str, skip: int = 0, limit: int | None = None) -> Response:
        def build(board: GymBoard) -> object:
            start = 0 if board.plan.paging == "ignore_skip" else skip
            end = None if limit is None else start + limit
            return [
                _drifted(_lever_posting(site, p), board.plan.drift, "text")
                for p in board.postings[start:end]
            ]

        return answer(site, "lever", build)

    @app.get("/lever/v0/postings/{site}/{posting_id}")
    async def lever_posting(site: str, posting_id: str) -> Response:
        def build(board: GymBoard) -> object | None:
            posting = find(board, posting_id)
            if posting is None:
                return None
            return _drifted(_lever_posting(site, posting), board.plan.drift, "text")

        return answer(site, "lever", build)

    return app
