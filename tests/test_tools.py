"""P2.2 · Unit A — the `@rex_tool` contract.

One typed signature is the single source of truth. The decorator derives ONE Pydantic
args-model from it; three artifacts that can never drift fall out of that one model:
  - the JSON schema (what brain() is handed in P5),
  - the validator (raw decision args in → typed-or-ValidationError, before any execution),
  - the handler (the original coroutine).
These tests pin all three to the same signature, plus the registry's negative contracts
(unknown name, duplicate, sync function, missing annotation). Tools are looked up by
``fn.__name__`` — the registry key IS the function name.
"""

import pytest
from pydantic import ValidationError

from rexhunter.tools import ToolRegistry


def test_schema_is_derived_from_the_signature() -> None:
    reg = ToolRegistry()

    @reg.tool
    async def fetch_posting(board: str, posting_id: int) -> str:
        return f"{board}/{posting_id}"

    schema = reg.get(fetch_posting.__name__).json_schema
    props = schema["properties"]
    assert props["board"]["type"] == "string"
    assert props["posting_id"]["type"] == "integer"
    assert set(schema["required"]) == {"board", "posting_id"}
    # extra="forbid" on the args-model surfaces as a closed schema (and gates the validator).
    assert schema["additionalProperties"] is False


def test_default_arg_is_optional_in_schema() -> None:
    reg = ToolRegistry()

    @reg.tool
    async def search(board: str, limit: int = 10) -> int:
        return limit

    schema = reg.get(search.__name__).json_schema
    assert set(schema["required"]) == {"board"}  # defaulted param is not required
    assert schema["properties"]["limit"]["default"] == 10


@pytest.mark.anyio
async def test_validator_rejects_bad_args_before_any_execution() -> None:
    reg = ToolRegistry()
    called = False

    @reg.tool
    async def record(posting_id: int) -> str:
        nonlocal called
        called = True
        return f"ok-{posting_id}"

    tool = reg.get(record.__name__)
    with pytest.raises(ValidationError):
        tool.validate({"posting_id": "not-an-int"})
    assert called is False  # the validator is the gate; the handler never ran

    # unknown/extra fields are rejected too (extra="forbid" via the shared config).
    with pytest.raises(ValidationError):
        tool.validate({"posting_id": 1, "claws": 3})


@pytest.mark.anyio
async def test_validated_args_drive_the_handler() -> None:
    reg = ToolRegistry()

    @reg.tool
    async def record(posting_id: int) -> str:
        return f"ok-{posting_id}"

    tool = reg.get(record.__name__)
    validated = tool.validate({"posting_id": 7})
    result = await tool.fn(**validated.model_dump())  # the loop's exact call path
    assert result == "ok-7"


def test_unknown_tool_name_raises_keyerror() -> None:
    reg = ToolRegistry()
    assert "ghost" not in reg
    with pytest.raises(KeyError):
        reg.get("ghost")


def test_duplicate_registration_is_rejected() -> None:
    reg = ToolRegistry()

    async def dig(depth: int) -> int:
        return depth

    reg.tool(dig)
    with pytest.raises(ValueError):
        reg.tool(dig)  # same name, second time


def test_sync_function_is_rejected() -> None:
    reg = ToolRegistry()

    def not_async(x: int) -> int:
        return x

    with pytest.raises(TypeError):
        reg.tool(not_async)  # pyright: ignore[reportArgumentType]  # the rejection is the point


def test_unannotated_param_is_rejected() -> None:
    reg = ToolRegistry()

    async def loose(board) -> str:  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
        return "x"

    with pytest.raises(TypeError):
        reg.tool(loose)  # pyright: ignore[reportUnknownArgumentType]
