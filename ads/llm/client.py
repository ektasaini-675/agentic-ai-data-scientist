"""LLM client with three modes and two providers, so the whole system always runs.

Modes
  live     real API calls (Gemini or Anthropic) with an on-disk response cache
  offline  no key available: `complete` raises LLMUnavailable and agents fall
           back to a deterministic scripted policy (the demo cannot fail)
  fake     replays scripted responses (used by tests; no network, no cost)

Providers (live mode)
  gemini     free tier via Google AI Studio  (GEMINI_API_KEY)
  anthropic  paid                            (ANTHROPIC_API_KEY)

The cache makes runs reproducible and, on a free tier, cheap: an identical
request (model, system prompt, messages, tools) is answered from disk.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ads.config import SETTINGS, resolve_llm_mode, resolve_provider

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"  # alias; set ADS_MODEL to an exact id from AI Studio if needed


class LLMUnavailable(RuntimeError):
    """Raised when no LLM can answer (offline mode, missing key, blocked/empty reply, exhausted fake script)."""


@dataclass
class LLMResponse:
    content: List[Dict[str, Any]]
    stop_reason: str = "end_turn"

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text").strip()

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        return [b for b in self.content if b.get("type") == "tool_use"]

    @classmethod
    def from_text(cls, text: str) -> "LLMResponse":
        return cls(content=[{"type": "text", "text": text}])

    @classmethod
    def from_tool_calls(cls, *calls: Tuple[str, Dict[str, Any]], text: str = "") -> "LLMResponse":
        content: List[Dict[str, Any]] = [{"type": "text", "text": text}] if text else []
        for i, (name, args) in enumerate(calls):
            content.append({"type": "tool_use", "id": f"toolu_fake_{i}", "name": name, "input": args})
        return cls(content=content, stop_reason="tool_use")


FakeItem = Union[LLMResponse, Callable[[str, List[Dict[str, Any]], Optional[List[Dict[str, Any]]]], LLMResponse]]

_RETRYABLE_NAMES = {
    "RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError",
    "OverloadedError", "ServerError", "RemoteProtocolError", "ReadTimeout", "ConnectError",
}
_RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return type(exc).__name__ in _RETRYABLE_NAMES or status in _RETRYABLE_STATUS


class LLMClient:
    def __init__(
        self,
        mode: Optional[str] = None,
        model: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        script: Optional[Sequence[FakeItem]] = None,
        provider: Optional[str] = None,
    ) -> None:
        self.mode = mode or resolve_llm_mode()
        self.provider = provider or (resolve_provider() if self.mode == "live" else None) or "anthropic"
        if model:
            self.model = model
        elif self.provider == "gemini":
            self.model = os.environ.get("ADS_MODEL") or DEFAULT_GEMINI_MODEL
        else:
            self.model = SETTINGS.model
        self.cache_dir = Path(cache_dir) if cache_dir else SETTINGS.cache_dir
        self._script: List[FakeItem] = list(script or [])
        self._sdk: Any = None
        self._last_call = 0.0
        default_gap = "4" if self.provider == "gemini" else "0"  # stay under free-tier requests/minute
        self.min_interval = float(os.environ.get("ADS_MIN_INTERVAL", default_gap))
        self.calls = 0
        self.cache_hits = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    @property
    def available(self) -> bool:
        return self.mode in ("live", "fake")

    # ------------------------------------------------------------------ API ---
    def complete(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        max_tokens = max_tokens or SETTINGS.max_tokens
        if self.mode == "offline":
            raise LLMUnavailable("offline mode: no LLM available")
        if self.mode == "fake":
            return self._next_fake(system, messages, tools)

        key = self._cache_key(system, messages, tools, max_tokens)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.cache_hits += 1
            cached = json.loads(path.read_text(encoding="utf-8"))
            return LLMResponse(content=cached["content"], stop_reason=cached.get("stop_reason", "end_turn"))

        response = self._call_api(system, messages, tools, max_tokens)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"content": response.content, "stop_reason": response.stop_reason}, indent=1),
            encoding="utf-8",
        )
        return response

    # ------------------------------------------------------------- internals ---
    def _next_fake(self, system, messages, tools) -> LLMResponse:
        if not self._script:
            raise LLMUnavailable("fake script exhausted")
        item = self._script.pop(0)
        self.calls += 1
        return item(system, messages, tools) if callable(item) else item

    def _cache_key(self, system, messages, tools, max_tokens) -> str:
        payload = {
            "provider": self.provider,
            "model": self.model,
            "system": system,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "temperature": os.environ.get("ADS_TEMPERATURE"),
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def _call_api(self, system, messages, tools, max_tokens) -> LLMResponse:
        if self.provider == "gemini":
            return self._call_gemini(system, messages, tools, max_tokens)
        return self._call_anthropic(system, messages, tools, max_tokens)

    def _throttle(self) -> None:
        wait = self._last_call + self.min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def _with_retries(self, fn: Callable[[], Any]) -> Any:
        """Retry rate-limit and server errors. Gemini's free tier resets per minute, so back off longer."""
        base = 5 if self.provider == "gemini" else 1
        attempts = 5
        for attempt in range(attempts):
            self._throttle()
            try:
                return fn()
            except Exception as exc:
                if attempt == attempts - 1 or not _retryable(exc):
                    raise
                time.sleep(base * (2 ** attempt))

    # ---- Anthropic ----------------------------------------------------------
    def _anthropic_client(self) -> Any:
        if self._sdk is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
            import anthropic

            self._sdk = anthropic.Anthropic()
        return self._sdk

    def _call_anthropic(self, system, messages, tools, max_tokens) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        temperature = os.environ.get("ADS_TEMPERATURE")
        if temperature:
            kwargs["temperature"] = float(temperature)

        client = self._anthropic_client()
        resp = self._with_retries(lambda: client.messages.create(**kwargs))

        self.calls += 1
        self.usage["input_tokens"] += getattr(resp.usage, "input_tokens", 0)
        self.usage["output_tokens"] += getattr(resp.usage, "output_tokens", 0)

        content: List[Dict[str, Any]] = []
        for block in resp.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input)})
        return LLMResponse(content=content, stop_reason=resp.stop_reason or "end_turn")

    # ---- Gemini -------------------------------------------------------------
    def _gemini_client(self) -> Any:
        if self._sdk is None:
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise LLMUnavailable("GEMINI_API_KEY is not set")
            from google import genai

            self._sdk = genai.Client(api_key=key)
        return self._sdk

    def _call_gemini(self, system, messages, tools, max_tokens) -> LLMResponse:
        from google.genai import types

        from ads.llm.gemini import parse_response, to_contents, to_function_declarations

        # Gemini's thinking tokens count against the output limit, so leave generous headroom
        config_kwargs: Dict[str, Any] = {"system_instruction": system, "max_output_tokens": max(max_tokens, 4096)}
        declarations = to_function_declarations(tools)
        if declarations:
            config_kwargs["tools"] = [types.Tool(function_declarations=declarations)]
            config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
        temperature = os.environ.get("ADS_TEMPERATURE")
        if temperature:
            config_kwargs["temperature"] = float(temperature)
        config = types.GenerateContentConfig(**config_kwargs)

        client = self._gemini_client()
        contents = to_contents(messages)
        resp = self._with_retries(
            lambda: client.models.generate_content(model=self.model, contents=contents, config=config)
        )

        self.calls += 1
        usage = getattr(resp, "usage_metadata", None)
        self.usage["input_tokens"] += getattr(usage, "prompt_token_count", 0) or 0
        self.usage["output_tokens"] += getattr(usage, "candidates_token_count", 0) or 0

        blocks = parse_response(resp)
        if blocks is None:
            raise LLMUnavailable("Gemini returned no usable content (empty or blocked response)")
        has_calls = any(b["type"] == "tool_use" for b in blocks)
        return LLMResponse(content=blocks, stop_reason="tool_use" if has_calls else "end_turn")
