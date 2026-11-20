"""Numeric knobs from the environment, for server.py (at lifespan start) and hub.py (at import).

A variable that is set but empty, as `VAR=` in a shell leaves it, reads as unset: the knob takes
its default instead of failing the import or the boot. Any other value that does not parse still
raises. hub.py stays free of the server: this module imports nothing of rexhunter's.
"""

import os


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default
