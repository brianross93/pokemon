"""LLM client abstraction for planlet generation and runtime reasoning."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    model: str = "gpt-5"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    verbosity: str = "medium"
    reasoning_effort: str = "medium"
    max_output_tokens: int = 1024
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

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_completion_tokens": self.config.max_output_tokens,
        }
        if self.config.tools:
            kwargs["tools"] = self.config.tools
            kwargs["parallel_tool_calls"] = self.config.parallel_tool_calls
        if self.config.response_format:
            kwargs["response_format"] = self.config.response_format

        try:
            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            content = message.content or ""
            return content.strip()
        except Exception as exc:  # pragma: no cover
            print(f"OpenAI API error: {exc}")
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
        system_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
        user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
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
        user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
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
