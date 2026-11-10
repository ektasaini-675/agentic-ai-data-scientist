import base64

import pytest

import ads.tools  # noqa: F401
from ads.llm.gemini import to_function_declarations, to_gemini_schema
from ads.tools import tool_schemas


def test_schema_conversion_uppercases_types_and_drops_unsupported_keys():
    schema = {
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "the target"},
            "exclude": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["target"],
        "additionalProperties": False,
    }
    out = to_gemini_schema(schema)
    assert out["type"] == "OBJECT"
    assert out["properties"]["exclude"]["items"]["type"] == "STRING"
    assert out["required"] == ["target"]
    assert "additionalProperties" not in out


def test_tools_without_parameters_get_no_schema():
    decls = to_function_declarations(
        [{"name": "profile_dataset", "description": "d", "input_schema": {"type": "object", "properties": {}, "required": []}}]
    )
    assert decls == [{"name": "profile_dataset", "description": "d"}]


def test_every_registered_tool_converts_to_a_gemini_declaration():
    decls = to_function_declarations(tool_schemas())
    assert len(decls) >= 10
    assert all("name" in d and "description" in d for d in decls)
    names = {d["name"] for d in decls}
    assert {"profile_dataset", "train_cv", "detect_leakage"} <= names


def test_messages_convert_with_tool_names_and_thought_signatures():
    pytest.importorskip("google.genai")
    from ads.llm.gemini import to_contents

    signature = base64.b64encode(b"sig").decode()
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "c1", "name": "profile_dataset", "input": {}, "signature": signature},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "Evidence E0001: ok"}]},
    ]
    contents = to_contents(messages)
    assert [c.role for c in contents] == ["user", "model", "user"]
    call_part = contents[1].parts[1]
    assert call_part.function_call.name == "profile_dataset"
    assert call_part.thought_signature == b"sig"
    result_part = contents[2].parts[0]
    assert result_part.function_response.name == "profile_dataset"       # matched back by tool name
    assert result_part.function_response.response == {"result": "Evidence E0001: ok"}


def test_error_results_are_marked_as_errors():
    pytest.importorskip("google.genai")
    from ads.llm.gemini import to_contents

    messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "boom", "is_error": True}]},
    ]
    part = to_contents(messages)[1].parts[0]
    assert part.function_response.response == {"error": "boom"}
