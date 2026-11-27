"""The Greenhouse and Lever adapters (`P2.4`): each board shape's paths, its boundary, and the
typed postings Rex is shown.

The boundary (invariant 3): a board's body is decoded as strict UTF-8 (a leading byte-order mark
is tolerated) and validated through adapter-private wire models that read only the fields Rex
uses, `Strict*`-typed, and ignore the rest, as the brain's envelope does. A missing, renamed or
retyped field, or a body that is not valid UTF-8, raises (`ValueError`: UnicodeDecodeError or
ValidationError). The wire models read the same fields as `events._posting_title`, so the pen's
label and the posting Rex was shown agree.

The typed postings are what a tool's view renders (invariant 5): re-parsed from the stored bytes
on every brain call, never stored. Paths and fields follow the Greenhouse Job Board API and the
Lever Postings API as re-read on 27 Nov 2026.
"""

import html
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, TypeAdapter

from rexhunter.board import DESCRIPTION_CAP, MAX_PAGES, PAGE_SIZE, Ats, Board


class PostingSummary(BaseModel):
    posting_id: str
    title: str
    location: str


class PostingPage(BaseModel):
    board: str
    page: int
    postings: list[PostingSummary]
    next_page: int | None


class Posting(BaseModel):
    board: str
    posting_id: str
    title: str
    location: str
    url: str
    description: str


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _GhLocation(_Wire):
    name: StrictStr


class _GhJob(_Wire):
    id: StrictInt
    title: StrictStr
    location: _GhLocation
    absolute_url: StrictStr


class _GhJobs(_Wire):
    jobs: list[_GhJob]  # `meta.total` is never read: the items are what the board sent


class _GhJobDetail(_GhJob):
    # The job's HTML converted to entities ("&lt;p&gt;"); the docs' example reads as escaped twice.
    content: StrictStr


class _LvCategories(_Wire):
    location: StrictStr | None = None


class _LvPosting(_Wire):
    id: StrictStr
    text: StrictStr
    categories: _LvCategories
    hostedUrl: StrictStr


class _LvList(_Wire):
    text: StrictStr
    content: StrictStr  # unstyled HTML


class _LvPostingDetail(_LvPosting):
    descriptionPlain: StrictStr
    # Optional, and absent from the gym's postings: absent, null or empty, each adds nothing.
    lists: list[_LvList] | None = None
    additionalPlain: StrictStr | None = None


_LV_POSTINGS = TypeAdapter(list[_LvPosting])


def body_text(body: bytes) -> str:
    """The body as text: strict UTF-8, a leading byte-order mark dropped."""
    return body.decode("utf-8-sig")


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.parts.append(" ")  # a tag ends a word: "<p>a</p><p>b</p>" reads "a b"

    def handle_endtag(self, tag: str) -> None:
        self.parts.append(" ")


def _description(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= DESCRIPTION_CAP:
        return collapsed
    return f"{collapsed[:DESCRIPTION_CAP]} [description cut at {DESCRIPTION_CAP} characters]"


def _plain(markup: str) -> str:
    parser = _Text()
    parser.feed(markup)
    parser.close()
    return "".join(parser.parts)


def _page(board: str, page: int, items: list[PostingSummary], more: bool) -> PostingPage:
    return PostingPage(
        board=board,
        page=page,
        postings=items,
        next_page=page + 1 if more and page < MAX_PAGES else None,
    )


@dataclass(frozen=True)
class Adapter:
    list_request: Callable[[Board, int], tuple[str, dict[str, str]]]
    read_url: Callable[[Board, str], str]
    page: Callable[[str, str, int], PostingPage]  # (body text, board name, page)
    posting: Callable[[str, str], Posting]  # (body text, board name)


def _gh_list_request(board: Board, page: int) -> tuple[str, dict[str, str]]:
    # One unpaginated list ("Returns the list of all job posts"); the page is sliced from it.
    return f"{board.base_url}/v1/boards/{quote(board.token, safe='')}/jobs", {}


def _gh_read_url(board: Board, posting_id: str) -> str:
    token, job = quote(board.token, safe=""), quote(posting_id, safe="")
    return f"{board.base_url}/v1/boards/{token}/jobs/{job}"


def _gh_page(text: str, board: str, page: int) -> PostingPage:
    jobs = _GhJobs.model_validate_json(text).jobs
    start = (page - 1) * PAGE_SIZE
    items = [
        PostingSummary(posting_id=str(job.id), title=job.title, location=job.location.name)
        for job in jobs[start : start + PAGE_SIZE]
    ]
    return _page(board, page, items, len(jobs) > page * PAGE_SIZE)


def _gh_posting(text: str, board: str) -> Posting:
    job = _GhJobDetail.model_validate_json(text)
    # Unescaped until stable, at most twice, then the tags stripped: content escaped once or twice
    # reads as text. A pass over stable text changes nothing, so two passes are that rule; a third
    # level of escaping stays as text.
    markup = html.unescape(html.unescape(job.content))
    return Posting(
        board=board,
        posting_id=str(job.id),
        title=job.title,
        location=job.location.name,
        url=job.absolute_url,
        description=_description(_plain(markup)),
    )


def _lv_list_request(board: Board, page: int) -> tuple[str, dict[str, str]]:
    params = {"mode": "json", "skip": str((page - 1) * PAGE_SIZE), "limit": str(PAGE_SIZE)}
    return f"{board.base_url}/v0/postings/{quote(board.token, safe='')}", params


def _lv_read_url(board: Board, posting_id: str) -> str:
    site, posting = quote(board.token, safe=""), quote(posting_id, safe="")
    return f"{board.base_url}/v0/postings/{site}/{posting}"


def _lv_page(text: str, board: str, page: int) -> PostingPage:
    postings = _LV_POSTINGS.validate_json(text)
    items = [
        PostingSummary(posting_id=p.id, title=p.text, location=p.categories.location or "")
        for p in postings[:PAGE_SIZE]
    ]
    # A full page may have a next one; a board that holds an exact multiple of pages costs one
    # extra call, which returns an empty page.
    return _page(board, page, items, len(postings) >= PAGE_SIZE)


def _lv_text(posting: _LvPostingDetail) -> str:
    # The opening and body, then each list (its text, a newline, its content as plain text), then
    # the closing, joined by blank lines; `_description` folds every break into one space.
    parts = [posting.descriptionPlain]
    parts += [f"{entry.text}\n{_plain(entry.content)}" for entry in posting.lists or ()]
    if posting.additionalPlain:
        parts.append(posting.additionalPlain)
    return "\n\n".join(parts)


def _lv_posting(text: str, board: str) -> Posting:
    posting = _LvPostingDetail.model_validate_json(text)
    return Posting(
        board=board,
        posting_id=posting.id,
        title=posting.text,
        location=posting.categories.location or "",
        url=posting.hostedUrl,
        description=_description(_lv_text(posting)),
    )


ADAPTERS: dict[Ats, Adapter] = {
    "greenhouse": Adapter(_gh_list_request, _gh_read_url, _gh_page, _gh_posting),
    "lever": Adapter(_lv_list_request, _lv_read_url, _lv_page, _lv_posting),
}
