"""Step 7b · the StaticFiles split.

The page shell's CSS/JS move out of the ``_SHELL`` raw string into
``src/rexhunter/static/board.{css,js}``, served byte-for-byte by Starlette's StaticFiles (inside
FastAPI — no new dependency, and SERVING, not building: the no-build rule holds). The split lands
with the current CSS/JS content unchanged, so the skin diff that follows stays readable. The
shell shrinks to a skeleton that links the assets.
"""

import httpx
import pytest

from rexhunter import server

pytestmark = pytest.mark.anyio


async def test_static_assets_served() -> None:
    """board.css / board.js are served with the right content types. No lifespan needed — the
    mount is app-level (the test_live_verdict.py ASGITransport idiom)."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://t"
    ) as client:
        css = await client.get("/static/board.css")
        js = await client.get("/static/board.js")
    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]
    assert css.text.strip()
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]
    assert js.text.strip()


async def test_shell_references_static_assets() -> None:
    """The `/` skeleton links the static assets and carries NO inline <style>/<script> body —
    the CSS/JS live in real files an editor (and its formatter) can work on."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://t"
    ) as client:
        page = await client.get("/")
    body = page.text
    assert '<link rel="stylesheet" href="/static/board.css">' in body
    assert '<script src="/static/board.js"></script>' in body
    assert "<style>" not in body
