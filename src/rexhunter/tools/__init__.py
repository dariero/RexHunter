"""The `@rex_tool` registry (ADR pillar 2): one typed signature = schema + validator + handler.

A tool is a plain async function with a Pydantic-typed signature. `@rex_tool` reads that
signature once and builds a single args-model; from that one model come three artifacts that
can never drift apart, because none of them is hand-written:

  - the JSON schema handed to ``brain()`` (P5)  ── ``args_model.model_json_schema()``
  - the validator raw decision args pass through ── ``args_model.model_validate()`` (invariant 3)
  - the handler that executes                    ── the original coroutine

Change the signature and all three move together; there is no second declaration to fall out
of sync. The registry is an instantiable object (tests build isolated registries — no global
state to leak between cases); ``rex_tool`` is the default registry's decorator.
"""

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model

ToolFn = Callable[..., Awaitable[Any]]


class RegisteredTool:
    """A tool bound to the one args-model derived from its signature."""

    def __init__(self, name: str, fn: ToolFn, args_model: type[BaseModel]) -> None:
        self.name = name
        self.fn = fn
        self.args_model = args_model

    @classmethod
    def from_function(cls, fn: ToolFn) -> RegisteredTool:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@rex_tool requires an async function: {fn.__name__!r}")
        hints = get_type_hints(fn)
        fields: dict[str, Any] = {}
        for pname, param in inspect.signature(fn).parameters.items():
            if pname not in hints:
                raise TypeError(
                    f"tool {fn.__name__!r}: parameter {pname!r} lacks a type annotation"
                )
            default = param.default if param.default is not inspect.Parameter.empty else ...
            fields[pname] = (hints[pname], default)
        # extra="forbid": the schema closes (additionalProperties:false) and the validator
        # rejects unknown args — the same boundary strictness the events carry (invariant 3).
        args_model = create_model(
            f"{fn.__name__}_Args",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )
        return cls(name=fn.__name__, fn=fn, args_model=args_model)

    @property
    def json_schema(self) -> dict[str, Any]:
        """Artifact 1 — the schema brain() is handed (P5)."""
        return self.args_model.model_json_schema()

    @property
    def description(self) -> str:
        """The model-facing description brain() presents (P5) — the tool's docstring, so it too
        derives from the one definition rather than a second hand-written declaration."""
        return inspect.getdoc(self.fn) or ""

    def validate(self, args: dict[str, Any]) -> BaseModel:
        """Artifact 2 — the boundary: raw args → typed, or ValidationError. Never executes."""
        return self.args_model.model_validate(args)


class ToolRegistry:
    """A name → RegisteredTool map. Instantiable so tests get an isolated registry."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def tool(self, fn: ToolFn) -> ToolFn:
        """Decorator: register `fn`, deriving its args-model from the signature.

        Returns the function unchanged — the contract lives in the registry, not in a wrapper.
        """
        registered = RegisteredTool.from_function(fn)
        if registered.name in self._tools:
            raise ValueError(f"tool already registered: {registered.name!r}")
        self._tools[registered.name] = registered
        return fn

    def get(self, name: str) -> RegisteredTool:
        """Look up a tool; unknown name raises KeyError (the loop maps it to a fatal event)."""
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"unknown tool: {name!r}") from None

    def registered(self) -> tuple[RegisteredTool, ...]:
        """Every registered tool, in registration order. Provider-agnostic (pillar 2): the P5
        adapter iterates these to present their schemas; no vendor shape leaks in here."""
        return tuple(self._tools.values())

    def __contains__(self, name: object) -> bool:
        return name in self._tools


default_registry = ToolRegistry()
rex_tool = default_registry.tool
