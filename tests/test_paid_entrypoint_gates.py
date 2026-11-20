"""The paid entrypoints' gate exits, pinned byte for byte.

smoke, hunt_smoke and hunt_smoke_thinking each refuse to run without ANTHROPIC_API_KEY, and refuse
the paid path unless REXHUNTER_BRAIN=live. Each refusal is a SystemExit whose message is what the
person at the terminal reads, raised before any client is built or any request is made. The exact
messages are pinned so the three gates can share code without a word of them changing.
"""

import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any, NoReturn

import pytest

from rexhunter import brain, hunt_smoke, hunt_smoke_thinking, smoke

KEY_UNSET = "ANTHROPIC_API_KEY is not set — export it before {}."
NOT_LIVE = "refusing to spend: set REXHUNTER_BRAIN=live to run the paid {} (opt-in only)."
# (module, what its key message names, what its refusal names, the paid and free paths it owns)
ENTRYPOINTS: list[tuple[ModuleType, str, str, tuple[str, ...]]] = [
    (smoke, "the smoke call", "smoke", ("run_smoke", "count_tokens")),
    (hunt_smoke, "the pre-flight or hunt", "hunt", ("run_live_hunt", "count_tokens_preflight")),
    (
        hunt_smoke_thinking,
        "the pre-flight or hunt",
        "hunt",
        ("run_live_hunt", "count_tokens_preflight"),
    ),
]
IDS = [module.__name__.removeprefix("rexhunter.") for module, *_ in ENTRYPOINTS]


def _reached(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise AssertionError("the gate let the run reach a client or a paid path")


def _main(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, paths: tuple[str, ...]
) -> Callable[[], None]:
    # Any client, and either path, fails the test: the gate must exit before all of them.
    monkeypatch.setattr(brain, "anthropic_client", _reached)
    for path in paths:
        monkeypatch.setattr(module, path, _reached)
    monkeypatch.delenv("REXHUNTER_BRAIN", raising=False)
    monkeypatch.setattr(sys, "argv", [module.__name__])
    return module.main


@pytest.mark.parametrize(("module", "before", "paid", "paths"), ENTRYPOINTS, ids=IDS)
def test_no_key_exits_before_any_client(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    before: str,
    paid: str,
    paths: tuple[str, ...],
) -> None:
    main = _main(monkeypatch, module, paths)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit) as refused:
        main()
    assert refused.value.code == KEY_UNSET.format(before)


@pytest.mark.parametrize(("module", "before", "paid", "paths"), ENTRYPOINTS, ids=IDS)
def test_the_paid_path_without_live_exits_before_any_client(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    before: str,
    paid: str,
    paths: tuple[str, ...],
) -> None:
    main = _main(monkeypatch, module, paths)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    with pytest.raises(SystemExit) as refused:
        main()
    assert refused.value.code == NOT_LIVE.format(paid)
