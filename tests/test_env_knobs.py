"""The hub's two environment knobs read an empty value as unset, as server.py's knobs do.

A shell's `VAR=` leaves the variable set but empty. server.py's knobs take their default then;
the hub's are read when the module is imported, and an empty one must not fail that import.
"""

import importlib.util

import pytest

import rexhunter.hub as hub_module


@pytest.mark.parametrize(
    ("name", "knob", "default"),
    [
        ("REXHUNTER_VIEWER_QUEUE_MAXSIZE", "VIEWER_QUEUE_MAXSIZE", 256),
        ("REXHUNTER_HEARTBEAT_INTERVAL_S", "HEARTBEAT_INTERVAL_S", 15.0),
    ],
)
def test_an_empty_hub_knob_takes_its_default(
    monkeypatch: pytest.MonkeyPatch, name: str, knob: str, default: float
) -> None:
    # Loaded afresh under another name, so `rexhunter.hub` stays the module the suite imported.
    monkeypatch.setenv(name, "")
    spec = importlib.util.spec_from_file_location("rexhunter_hub_fresh", hub_module.__file__)
    assert spec is not None and spec.loader is not None
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert getattr(fresh, knob) == default
