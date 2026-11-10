"""Translation layer between our internal (Anthropic-style) messages and the Gemini API.

Internally everything is content blocks: text, tool_use, tool_result. Gemini uses
Parts (text, function_call, function_response). This module converts both ways,
so the agents and the loop never know which provider they are talking to.
"""
from __future__ import annotations

import base64
import uuid
from typing import Any, Dict, List, Optional

# Gemini's schema dialect only understands this subset of JSON Schema.
_ALLOWED_KEYS = {"type", "description", "properties", "required", "items", "enum"}


def to_gemini_schema(schema: Any) -> Any:
    """JSON Schema -> Gemini schema: keep supported keys, UPPERCASE the type names."""
    if not isinstance(schema, dict):
        return schema
    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _ALLOWED_KEYS:
            continue
        if key == "type" and isinstance(value, str):
            out[key] = value.upper()
        elif key == "properties":
            out[key] = {name: to_gemini_schema(sub) for name, sub in value.items()}
        elif key == "items":
            out[key] = to_gemini_schema(value)
        elif key == "required":
            if value:
                out[key] = list(value)
        else:
            out[key] = value
    return out


def to_function_declarations(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Our tool schemas -> Gemini function declarations. Gemini rejects empty object schemas,
    so tools without parameters get no `parameters` key at all."""
    declarations: List[Dict[str, Any]] = []
    for tool in tools or []:
        decl: Dict[str, Any] = {"name": tool["name"], "description": tool.get("description", "")}
        schema = to_gemini_schema(tool.get("input_schema") or {})
        if schema.get("properties"):
            decl["parameters"] = schema
        declarations.append(decl)
    return declarations


def to_contents(messages: List[Dict[str, Any]]) -> List[Any]:
    """Our messages -> Gemini `Content` objects (needs the google-genai SDK)."""
    from google.genai import types

    contents: List[Any] = []
    tool_names: Dict[str, str] = {}  # tool_use id -> tool name (Gemini matches results by name)

    for message in messages:
        role = "user" if message["role"] == "user" else "model"
        content = message["content"]
        parts: List[Any] = []

        if isinstance(content, str):
            parts.append(types.Part(text=content))
        else:
            for block in content:
                kind = block.get("type")
                if kind == "text":
                    if block.get("text"):
                        parts.append(types.Part(text=block["text"]))
                elif kind == "tool_use":
                    tool_names[block["id"]] = block["name"]
                    kwargs: Dict[str, Any] = {
                        "function_call": types.FunctionCall(name=block["name"], args=block.get("input") or {})
                    }
                    # Gemini 3 models require the model's "thought signature" to be echoed back
                    if block.get("signature"):
                        kwargs["thought_signature"] = base64.b64decode(block["signature"])
                    parts.append(types.Part(**kwargs))
                elif kind == "tool_result":
                    name = tool_names.get(block["tool_use_id"], "unknown_tool")
                    key = "error" if block.get("is_error") else "result"
                    parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(name=name, response={key: block.get("content", "")})
                        )
                    )
        if parts:
            contents.append(types.Content(role=role, parts=parts))
    return contents


def parse_response(resp: Any) -> Optional[List[Dict[str, Any]]]:
    """Gemini response -> our content blocks. Returns None if there is nothing usable
    (empty or blocked response)."""
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        return None
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return None

    batch = uuid.uuid4().hex[:8]  # ids must be unique across the whole conversation
    blocks: List[Dict[str, Any]] = []
    n_calls = 0
    for part in parts:
        if getattr(part, "thought", False):
            continue
        call = getattr(part, "function_call", None)
        if call:
            block: Dict[str, Any] = {
                "type": "tool_use",
                "id": f"call_{batch}_{n_calls}",
                "name": call.name,
                "input": dict(call.args or {}),
            }
            signature = getattr(part, "thought_signature", None)
            if signature:
                block["signature"] = base64.b64encode(signature).decode("ascii")
            blocks.append(block)
            n_calls += 1
        elif getattr(part, "text", None):
            blocks.append({"type": "text", "text": part.text})
    return blocks or None
