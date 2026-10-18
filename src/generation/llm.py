"""The LLM behind one tiny interface: generate(system, user) -> str.

Two backends: the Claude API (default) and a local model through Ollama. The rest of the
code only depends on the LLM protocol, so backends can be swapped (or faked in tests).
"""

import json
import urllib.request
from typing import Protocol

import anthropic

from src.config import settings

# Models that accept the server-side refusal fallback ("fallbacks": "default").
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"}


class LLM(Protocol):
    name: str

    def generate(self, system: str, user: str) -> str: ...


class AnthropicLLM:
    """Claude through the official anthropic SDK."""

    def __init__(self, model: str | None = None, effort: str | None = None) -> None:
        self.model = model or settings.anthropic_model
        self.effort = effort or settings.anthropic_effort
        self.name = f"anthropic/{self.model}"
        # The SDK reads ANTHROPIC_API_KEY from the environment; we pass the .env value explicitly.
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)

    def generate(self, system: str, user: str) -> str:
        extra = {}
        if self.model in FALLBACK_MODELS:
            # If a safety classifier declines the request, the API retries it on another
            # model instead of returning nothing.
            extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            # Current Claude models always think before answering; effort sets how much.
            output_config={"effort": self.effort},
            **extra,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"The model declined the request: {response.stop_details}")
        return "".join(block.text for block in response.content if block.type == "text").strip()


class OllamaLLM:
    """A local model served by Ollama (https://ollama.com), via its HTTP API."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.ollama_model
        self.name = f"ollama/{self.model}"

    def generate(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        request = urllib.request.Request(
            f"{settings.ollama_base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read())["message"]["content"].strip()


def get_llm(provider: str | None = None) -> LLM:
    """Return the LLM backend chosen in .env (LLM_PROVIDER), or the one passed in."""
    provider = provider or settings.llm_provider
    if provider == "anthropic":
        return AnthropicLLM()
    if provider == "ollama":
        return OllamaLLM()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (use 'anthropic' or 'ollama')")
