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

from rexhunter.events import PostingRef

ToolFn = Callable[..., Awaitable[Any]]

# What the model is shown for one tool result, from the stored request and response bytes. A
# pure function of those bytes (invariant 5): each brain call renders every result again from
# the log, so the log keeps the bytes as they arrived (invariant 6) and no rendered copy is stored.
type ToolView = Callable[[bytes, bytes], str]


def default_view(raw_request: bytes, raw_response: bytes) -> str:
    """The rendering of a tool that registers no view: the response bytes decoded as UTF-8, with
    any invalid byte shown escaped (`\\xe9`), so a body that is not UTF-8 still renders."""
    return raw_response.decode(errors="backslashreplace")


def _takes_a_posting_ref(args_model: type[BaseModel]) -> bool:
    # The stored request of a posting source is its args model's JSON, so the model must have
    # exactly a PostingRef's required fields and types for that request to validate as one.
    fields = args_model.model_fields
    ref_fields = PostingRef.model_fields
    return fields.keys() == ref_fields.keys() and all(
        fields[name].annotation == ref_fields[name].annotation and fields[name].is_required()
        for name in fields
    )


class RegisteredTool:
    """A tool bound to the one args-model derived from its signature, and the view that renders
    its results for the model. A posting source is a tool whose stored request is a PostingRef:
    its results are what a catch reference resolves against."""

    def __init__(
        self,
        name: str,
        fn: ToolFn,
        args_model: type[BaseModel],
        view: ToolView = default_view,
        posting_source: bool = False,
    ) -> None:
        self.name = name
        self.fn = fn
        self.args_model = args_model
        self.view = view
        self.posting_source = posting_source

    @classmethod
    def from_function(
        cls, fn: ToolFn, *, view: ToolView = default_view, posting_source: bool = False
    ) -> RegisteredTool:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@rex_tool requires an async function: {fn.__name__!r}")
        # include_extras: an `Annotated[int, Field(ge=1)]` keeps its Field, so the limit reaches
        # both the schema and the validator; without it the annotation is stripped to `int`.
        hints = get_type_hints(fn, include_extras=True)
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
        if posting_source and not _takes_a_posting_ref(args_model):
            raise TypeError(
                f"tool {fn.__name__!r}: a posting source takes exactly"
                " (board: str, posting_id: str)"
            )
        return cls(
            name=fn.__name__,
            fn=fn,
            args_model=args_model,
            view=view,
            posting_source=posting_source,
        )

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

    def tool(
        self, fn: ToolFn, *, view: ToolView = default_view, posting_source: bool = False
    ) -> ToolFn:
        """Decorator: register `fn`, deriving its args-model from the signature. `view` renders
        its results for the model (`brain.project_messages`); without one, the bytes are decoded.
        `posting_source` marks a tool whose results a catch reference resolves against; its
        signature must be exactly (board: str, posting_id: str), checked here.

        Returns the function unchanged — the contract lives in the registry, not in a wrapper.
        """
        registered = RegisteredTool.from_function(fn, view=view, posting_source=posting_source)
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
