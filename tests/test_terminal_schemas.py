"""The terminal tools' input schemas carry no class docstring.

`brain.build_tools` derives the `hunt_complete` and `needs_help` input schemas from the HuntComplete
and NeedsHelp models (`model_json_schema()`), and pydantic puts a model's class docstring at the
schema's top level as `description`. Those docstrings are notes for readers of loop.py, about where
the outcome lives and which slice built what, and every request used to show them to the model. The
tool-level description is the text meant for the model, and it stays, as do the properties and any
field description.
"""

from typing import Any

import pytest
from pydantic import BaseModel, Field

from rexhunter import brain, stub


def _tools() -> dict[str, dict[str, Any]]:
    return {tool["name"]: tool for tool in brain.build_tools(stub.build_registry())}


def test_the_terminal_schemas_carry_no_class_docstring() -> None:
    tools = _tools()
    for name in ("hunt_complete", "needs_help"):
        assert "description" not in tools[name]["input_schema"], name


def test_the_terminal_tools_keep_their_description_and_properties() -> None:
    tools = _tools()
    assert tools["hunt_complete"]["description"] == (
        "End the hunt. `catch` is the list of postings worth a verdict."
    )
    assert tools["needs_help"]["description"] == (
        "End the hunt and hand off to a human — Rex is stuck."
    )
    schemas = {
        name: {
            key: value for key, value in tools[name]["input_schema"].items() if key != "description"
        }
        for name in ("hunt_complete", "needs_help")
    }
    assert schemas == {
        "hunt_complete": {
            "$defs": {
                "PostingRef": {
                    "additionalProperties": False,
                    "properties": {"board": {"type": "string"}, "posting_id": {"type": "string"}},
                    "required": ["board", "posting_id"],
                    "type": "object",
                }
            },
            "properties": {
                "catch": {
                    "default": [],
                    "description": (
                        "A reference to each posting you read with read_posting in this hunt and "
                        "judge worth a human's verdict."
                    ),
                    "items": {"$ref": "#/$defs/PostingRef"},
                    "type": "array",
                }
            },
            "type": "object",
        },
        "needs_help": {"properties": {}, "type": "object"},
    }


class _Documented(BaseModel):
    """A note for readers that the model must not see."""

    catch: list[str] = Field(default=[], description="the postings worth a verdict")


def test_a_terminal_field_description_stays(monkeypatch: pytest.MonkeyPatch) -> None:
    # A stand-in with both a class docstring and a field description shows that only the class
    # docstring leaves the schema.
    monkeypatch.setattr(
        brain, "_TERMINAL_TOOLS", (("hunt_complete", _Documented, "End the hunt."),)
    )
    schema = _tools()["hunt_complete"]["input_schema"]
    assert "description" not in schema
    assert schema["properties"]["catch"]["description"] == "the postings worth a verdict"
