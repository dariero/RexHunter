"""Runtime pin — RexHunter runs on Python 3.15 only.

pyproject.toml (`requires-python`), .python-version and CI all name 3.15. This test fails
when the interpreter actually running the suite is anything else, such as a stale 3.14 .venv.
"""

import sys


def test_runs_on_python_3_15() -> None:
    assert sys.version_info[:2] == (3, 15)
