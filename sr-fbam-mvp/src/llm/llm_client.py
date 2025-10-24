"""LLM client abstraction for planlet generation and runtime reasoning."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    model: str = "gpt-5-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    verbosity: str = "medium"
    reasoning_effort: str = "medium"
    max_output_tokens: Optional[int] = None
    timeout: int = 30
    tools: List[Dict[str, Any]] = field(default_factory=list)
    response_format: Optional[Dict[str, Any]] = None
    parallel_tool_calls: bool = True


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response from the LLM."""


class OpenAILLMClient(LLMClient):
    """OpenAI API client with tool-use support."""

    def __init__(self, config: LLMConfig):
        self.config = config
        try:
            import openai

            self.client = openai.OpenAI(api_key=config.api_key, base_url=config.base_url)
        except ImportError as exc:  # pragma: no cover
            raise ImportError("openai package not found. Install with: pip install openai") from exc

    def generate_response(self, messages: List[Dict[str, Any]]) -> str:
        model_name = (self.config.model or "").lower()
        use_responses = any(tool.get("type") == "web_search" for tool in self.config.tools)
        if model_name.startswith(("gpt-4o", "gpt-5", "o1")):
            use_responses = True

        def _to_response_content(content: Any) -> List[Dict[str, Any]]:
            entries = content if isinstance(content, list) else [content]
            normalised: List[Dict[str, Any]] = []
            for entry in entries:
                if isinstance(entry, dict):
                    entry_type = entry.get("type")
                    if entry_type in {"input_text", "input_image"}:
                        normalised.append(entry)
                    elif entry_type == "text":
                        normalised.append({"type": "input_text", "text": entry.get("text", "")})
                    elif entry_type == "image_url":
                        image = entry.get("image_url")
                        url = image.get("url") if isinstance(image, dict) else image
                        if url:
                            normalised.append({"type": "input_image", "image_url": url})
                    else:
                        normalised.append({"type": "input_text", "text": json.dumps(entry)})
                else:
                    normalised.append({"type": "input_text", "text": str(entry)})
            return normalised

        def _to_chat_text(content: Any) -> str:
            if isinstance(content, list):
                return "\n".join(
                    entry.get("text", "") if isinstance(entry, dict) else str(entry)
                    for entry in content
                )
            return str(content)

        def _collect_response_text(response: Any) -> str:
            content = getattr(response, "output_text", None)
            fragments: List[str] = []
            if content:
                fragments.append(content)
            for item in getattr(response, "output", []) or []:
                for piece in getattr(item, "content", []) or []:
                    text = getattr(piece, "text", None)
                    if text:
                        fragments.append(text)
            for item in getattr(response, "content", []) or []:
                text = getattr(item, "text", None)
                if text:
                    fragments.append(text)
            combined = "\n".join(fragment for fragment in fragments if fragment)
            return combined.strip()

        def _invoke_responses() -> str:
            input_messages = [
                {"role": msg["role"], "content": _to_response_content(msg.get("content", ""))}
                for msg in messages
            ]
            kwargs: Dict[str, Any] = {
                "model": self.config.model,
                "input": input_messages,
            }
            if self.config.max_output_tokens is not None:
                kwargs["max_output_tokens"] = self.config.max_output_tokens
            if self.config.tools:
                kwargs["tools"] = self.config.tools
            response = self.client.responses.create(**kwargs)
            text = _collect_response_text(response)
            status = getattr(response, "status", None)
            if text:
                if status and status != "completed":
                    details = getattr(response, "incomplete_details", None)
                    if isinstance(details, dict):
                        reason = details.get("reason")
                    else:
                        reason = getattr(details, "reason", None)
                    print(f"OpenAI Responses warning: status={status} reason={reason}")  # noqa: T201
                return text
            if status and status != "completed":
                details = getattr(response, "incomplete_details", None)
                if isinstance(details, dict):
                    reason = details.get("reason")
                else:
                    reason = getattr(details, "reason", None)
                raise RuntimeError(f"Responses API status={status} reason={reason}")
            raise RuntimeError("Responses API returned no content.")

        def _invoke_chat() -> str:
            chat_messages: List[Dict[str, Any]] = [
                {"role": msg["role"], "content": _to_chat_text(msg.get("content", ""))}
                for msg in messages
            ]
            kwargs = {
                "model": self.config.model,
                "messages": chat_messages,
            }
            if self.config.max_output_tokens is not None:
                kwargs["max_completion_tokens"] = self.config.max_output_tokens
            if self.config.response_format:
                kwargs["response_format"] = self.config.response_format
            if self.config.tools:
                supported_tools = [
                    tool for tool in self.config.tools if tool.get("type") in {"function", "custom"}
                ]
                if supported_tools:
                    kwargs["tools"] = supported_tools
                    kwargs["parallel_tool_calls"] = self.config.parallel_tool_calls

            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            raw_content = getattr(message, "content", "")
            if isinstance(raw_content, list):
                content = "\n".join(
                    entry.get("text", "") if isinstance(entry, dict) else str(entry)
                    for entry in raw_content
                ).strip()
            else:
                content = (raw_content or "").strip()
            if content:
                return content
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                first_call = tool_calls[0]
                arguments = getattr(first_call.function, "arguments", "") if getattr(first_call, "function", None) else ""
                if isinstance(arguments, str) and arguments.strip():
                    return arguments.strip()
            raise RuntimeError("Chat API returned no content.")

        responses_error: Optional[str] = None
        if use_responses:
            try:
                return _invoke_responses()
            except Exception as exc:
                responses_error = str(exc)

        try:
            return _invoke_chat()
        except Exception as exc:  # pragma: no cover
            if responses_error:
                print(f"OpenAI Responses fallback due to: {responses_error}")  # noqa: T201
            print(f"OpenAI API error: {exc}")  # noqa: T201
            return ""


class AnthropicLLMClient(LLMClient):
    """Anthropic Claude API client."""

    def __init__(self, config: LLMConfig):
        self.config = config
        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=config.api_key)
        except ImportError as exc:  # pragma: no cover
            raise ImportError("anthropic package not found. Install with: pip install anthropic") from exc

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        def _coerce_text(content: Any) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("value") or item.get("content")
                        if isinstance(text, str):
                            parts.append(text)
                return "\n".join(parts)
            return str(content)

        system_msg = _coerce_text(next((msg["content"] for msg in messages if msg["role"] == "system"), ""))
        user_msg = _coerce_text(next((msg["content"] for msg in messages if msg["role"] == "user"), ""))
        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_output_tokens,
                temperature=0.0,
                system=system_msg,
                messages=[{"role": "user", "content": user_msg}],
            )
            return response.content[0].text.strip()
        except Exception as exc:  # pragma: no cover
            print(f"Anthropic API error: {exc}")
            return ""


class MockLLMClient(LLMClient):
    """Mock LLM client for testing without API calls."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.response_count = 0

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        self.response_count += 1
        user_content = next((msg.get("content") for msg in messages if msg.get("role") == "user"), "")
        if isinstance(user_content, list):
            texts = [item.get("text", "") for item in user_content if isinstance(item, dict) and "text" in item]
            user_msg = "\n".join(texts)
        else:
            user_msg = user_content or ""
        if "in_battle" in user_msg and "true" in user_msg:
            return "FIGHT"
        if "need_planlet" in user_msg:
            return json.dumps({"planlet": "stub"})
        return "WALK"


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Create an LLM client based on configuration."""

    if config.model.startswith(("gpt-", "o1-")) or config.model == "gpt-5":
        return OpenAILLMClient(config)
    if config.model.startswith("claude-"):
        return AnthropicLLMClient(config)
    if config.model == "mock":
        return MockLLMClient(config)
    raise ValueError(f"Unsupported model: {config.model}")
