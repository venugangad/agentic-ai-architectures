# Chapter 1: The Provider Abstraction Layer

> **Part I — The Foundation**
> Estimated read time: 35 minutes | Diagrams: 3 | Code examples: 6
> Module built: `core/llm.py`

---

## Overview

Every agentic framework begins with the same design decision, and most get it wrong. They reach for the nearest SDK — `import openai`, or `from anthropic import Anthropic` — and wire it directly into their business logic. It works. Then six months later, the provider changes their pricing structure, deprecates the model their entire prompt strategy was tuned around, or experiences an outage that takes down customer-facing systems for four hours on a Tuesday.

This chapter builds the foundation that prevents all of those outcomes. We are going to design and implement a **Provider Abstraction Layer**: a clean interface that sits between your framework's reasoning logic and every LLM API in the world. When we are done, your agents will never import `openai` or `anthropic` directly. They will talk to a `BaseLlm` — a stable contract — and the plumbing underneath can be swapped, upgraded, or rerouted without touching a single line of agent code.

By the end of this chapter you will have:

- A `BaseLlm` abstract class defining the universal contract every LLM provider must satisfy
- Four working provider adapters: OpenAI, Anthropic, Google Gemini, and Ollama
- A unified `LlmResponse` envelope that normalises every provider's output into one shape
- An `LlmRegistry` that selects and instantiates providers from a YAML config file
- A `RouterLlm` that matches task complexity to model capability automatically
- A running demo where you send the same prompt to all four providers and get back identically-structured responses

Let's build it.

---

## 1.1 The Price of Hardcoded Dependencies

Picture a team at a mid-sized fintech company. In early 2024, they built an internal document summarisation service on top of GPT-4. It worked beautifully. Then three things happened in sequence.

First, OpenAI raised prices for GPT-4 in March 2024 with two weeks' notice. The team's token spend doubled. Because their embedding model, their generation model, and their classification calls all went through the same SDK with the same API key, there was no routing logic to shift cheaper tasks to cheaper models. Every single request, regardless of whether it was a one-line classification or a full-document synthesis, hit the most expensive tier.

Second, OpenAI deprecating `gpt-4-0314` in June 2024 forced a rewrite. The team had tuned their system prompts around that model's specific behavioural quirks. Migrating meant re-running evaluations, updating prompts, and testing edge cases — all under time pressure, because the deprecated model stopped accepting requests on a hard deadline.

Third, an outage in November 2024 took their summarisation service offline for three hours. Their entire call flow went through one endpoint. When that endpoint returned `503`, the service returned nothing. There was no fallback, because the code had no concept that alternatives existed.

This is not a story about bad engineers. It is a story about a perfectly rational short-term decision — reach for the nearest SDK — creating structural fragility that compounds over time. The team was not locked in to OpenAI because they wanted to be. They were locked in because nothing in their architecture made switching possible without a rewrite.

The pattern that prevents this has a name. In the MRKL architecture (Modular Reasoning, Knowledge and Language, Karpas et al., 2022), the LLM is treated not as a monolith that does everything, but as a **Foreman** — a dispatcher that understands goals, decomposes tasks, and routes each sub-task to the most appropriate specialist module. The LLM does not own the execution path. It governs it.

We are going to encode that philosophy into `BaseLlm`. The LLM is a pluggable component, not a load-bearing wall.

---

## 1.2 What the Interface Must Guarantee

Before writing any code, we need to think clearly about what the `BaseLlm` interface must enforce. An interface is a contract: it says what callers can rely on and what implementers must provide. Get the contract wrong and every adapter you write will be subtly broken in ways that only appear under production load.

There are five things every LLM provider, regardless of vendor, must be able to do for this framework:

**1. Generate content asynchronously.** Agents are concurrent systems. Synchronous LLM calls block the event loop and destroy throughput. Every provider adapter must expose an `async` generation method.

**2. Stream responses.** Long outputs — plans, summaries, multi-step reasoning chains — cannot wait for the model to finish before the framework processes the first token. The interface must support streaming via an `AsyncGenerator`.

**3. Report token usage and cost metadata.** Every call to an LLM costs money. The framework must be able to attribute that cost to a specific agent, session, user, or team. Every response must carry token counts and model identity.

**4. Surface its context window limit.** Different models support radically different context sizes: GPT-4o supports 128K tokens, Claude 3.5 Sonnet supports 200K, Gemini 1.5 Pro reaches 1M. The framework needs this information to decide whether to page memory, truncate context, or route to a model with a larger window. Every provider must declare its limit.

**5. Normalise messages into one format.** OpenAI uses `{"role": "user", "content": "..."}`. Anthropic separates `system` from `messages`. Gemini uses a `parts`-based structure. The interface must accept a single, canonical message format and let each adapter translate it internally.

With those requirements clear, let's write the contract.

---

## 1.3 Designing `BaseLlm`: The Universal Contract

We use Python's `abc` module to define `BaseLlm` as an abstract base class. Any class that inherits from it must implement every abstract method — the Python interpreter will refuse to instantiate it otherwise. This is not a convention; it is a compile-time guarantee.

We also define two supporting data classes: `LlmMessage`, which represents a single turn in a conversation, and `LlmResponse`, which is the normalised envelope every provider returns.

```python
# core/llm.py
"""
The Provider Abstraction Layer — core/llm.py
The Agentic Spine: Engineering a Provider-Agnostic AI Framework

This module defines the universal contract that every LLM provider must satisfy.
No agent, tool, or orchestrator in this framework imports a vendor SDK directly.
Everything goes through BaseLlm.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator


class MessageRole(str, Enum):
    """Canonical roles understood by every provider adapter."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"           # used when returning tool call results


@dataclass
class LlmMessage:
    """
    A single turn in a conversation, in our canonical format.
    Provider adapters translate this into their own wire format internally.
    Callers never construct provider-specific message shapes.
    """
    role: MessageRole
    content: str
    name: str | None = None          # for tool messages: the tool's name
    tool_call_id: str | None = None  # for tool result messages


@dataclass
class LlmUsage:
    """
    Token and cost metadata attached to every LlmResponse.
    Enables per-call cost attribution regardless of which provider handled the request.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model_id: str = ""               # exact model string returned by the provider
    provider: str = ""               # "openai" | "anthropic" | "gemini" | "ollama"
    cost_usd: float = 0.0            # calculated by each adapter using its pricing table
    latency_ms: float = 0.0          # wall-clock time for this call


@dataclass
class LlmResponse:
    """
    The normalised envelope returned by every provider adapter.
    Callers receive this shape regardless of which LLM generated the content.
    """
    content: str
    usage: LlmUsage
    finish_reason: str = "stop"      # "stop" | "length" | "tool_calls" | "error"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)  # original provider response


@dataclass
class LlmConfig:
    """
    Generation parameters in a provider-agnostic form.
    Adapters map these to their vendor's specific parameter names.
    """
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    json_mode: bool = False          # request structured JSON output


class BaseLlm(ABC):
    """
    The universal contract every LLM provider adapter must satisfy.

    Design principles:
    - Callers depend on this interface, never on vendor SDKs
    - All I/O is async — agents are concurrent systems
    - Every response carries usage metadata for cost attribution
    - The context window limit is a first-class property, not an afterthought
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The canonical model identifier (e.g. 'gpt-4o', 'claude-3-5-sonnet-20241022')."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """The provider name: 'openai' | 'anthropic' | 'gemini' | 'ollama'."""

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Maximum tokens this model accepts in a single request."""

    @abstractmethod
    async def generate(
        self,
        messages: list[LlmMessage],
        config: LlmConfig | None = None,
    ) -> LlmResponse:
        """
        Send a list of messages and return a normalised LlmResponse.
        This is the primary generation entry point for non-streaming use.
        """

    @abstractmethod
    async def stream(
        self,
        messages: list[LlmMessage],
        config: LlmConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Send a list of messages and yield content tokens as they arrive.
        Use this for long-running generations where latency matters.
        """

    def count_tokens_estimate(self, messages: list[LlmMessage]) -> int:
        """
        Rough token estimate before sending: ~4 characters per token.
        Adapters may override this with provider-specific tokenisers.
        """
        total_chars = sum(len(m.content) for m in messages)
        return total_chars // 4

    def fits_in_context(self, messages: list[LlmMessage]) -> bool:
        """Returns True if the estimated token count fits in this model's context window."""
        return self.count_tokens_estimate(messages) < self.context_window * 0.9
```

Notice what `BaseLlm` does not contain: nothing about API keys, endpoints, SDK imports, or retry logic. Those are implementation details. The contract is about *capability*, not *mechanism*.

The `LlmUsage` dataclass is doing important work. By attaching `provider`, `model_id`, `cost_usd`, and `latency_ms` to every response, we give the framework the raw material for per-call cost attribution. In Chapter 8, when we build the `LlmGateway`, we will aggregate this data into per-tenant dashboards. But the data collection starts here, in every single response, from the first chapter.

---

## 1.4 The Four Provider Adapters

Now we implement `BaseLlm` four times — once per provider. Each adapter's job is to translate our canonical `LlmMessage` list into the vendor's wire format, make the API call, and translate the response back into a `LlmResponse`. The adapter absorbs all the vendor-specific complexity so the rest of the framework never has to.

### The OpenAI Adapter

OpenAI's Chat Completions API is the most familiar to most readers. The wire format maps cleanly to our `LlmMessage` structure. The main adapter work is in normalising the response envelope and calculating cost.

```python
# core/llm.py (continued)

import os
import asyncio

# Provider pricing tables (per 1M tokens, USD) — update as providers change pricing
_OPENAI_PRICING: dict[str, tuple[float, float]] = {
    # model_id: (input_price_per_1m, output_price_per_1m)
    "gpt-4o":                (2.50,  10.00),
    "gpt-4o-mini":           (0.15,   0.60),
    "gpt-4-turbo":           (10.00, 30.00),
    "o1":                    (15.00, 60.00),
    "o1-mini":               (3.00,  12.00),
}


class OpenAiProvider(BaseLlm):
    """
    Adapter for OpenAI's Chat Completions API.
    Translates our canonical LlmMessage format to OpenAI's wire format and back.
    """

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai>=1.30.0")

        self._model = model
        self._client = AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._pricing = _OPENAI_PRICING.get(model, (2.50, 10.00))

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def context_window(self) -> int:
        windows = {"gpt-4o": 128_000, "gpt-4o-mini": 128_000,
                   "gpt-4-turbo": 128_000, "o1": 128_000, "o1-mini": 128_000}
        return windows.get(self._model, 128_000)

    def _to_openai_messages(self, messages: list[LlmMessage]) -> list[dict]:
        """Translate our canonical messages to OpenAI's wire format."""
        result = []
        for m in messages:
            msg: dict[str, Any] = {"role": m.role.value, "content": m.content}
            if m.name:
                msg["name"] = m.name
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            result.append(msg)
        return result

    async def generate(
        self,
        messages: list[LlmMessage],
        config: LlmConfig | None = None,
    ) -> LlmResponse:
        cfg = config or LlmConfig()
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_openai_messages(messages),
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        if cfg.stop_sequences:
            kwargs["stop"] = cfg.stop_sequences
        if cfg.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.monotonic()
        response = await self._client.chat.completions.create(**kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        usage = response.usage
        input_t = usage.prompt_tokens if usage else 0
        output_t = usage.completion_tokens if usage else 0
        cost = (input_t / 1_000_000 * self._pricing[0] +
                output_t / 1_000_000 * self._pricing[1])

        choice = response.choices[0]
        return LlmResponse(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason or "stop",
            usage=LlmUsage(
                input_tokens=input_t,
                output_tokens=output_t,
                total_tokens=input_t + output_t,
                model_id=response.model,
                provider=self.provider,
                cost_usd=cost,
                latency_ms=latency_ms,
            ),
            raw=response.model_dump(),
        )

    async def stream(
        self,
        messages: list[LlmMessage],
        config: LlmConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        cfg = config or LlmConfig()
        async with self._client.chat.completions.stream(
            model=self._model,
            messages=self._to_openai_messages(messages),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
```

### The Anthropic Adapter

Anthropic's Messages API has two structural differences that the adapter must handle. First, the `system` message is not part of the `messages` array — it is a separate top-level field. Second, Anthropic's response structure uses `content` as a list of `ContentBlock` objects, not a simple string.

```python
_ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-5":             (15.00, 75.00),
    "claude-3-5-sonnet-20241022":  ( 3.00, 15.00),
    "claude-3-5-haiku-20241022":   ( 0.80,  4.00),
}


class AnthropicProvider(BaseLlm):
    """
    Adapter for Anthropic's Messages API.
    Key difference from OpenAI: system messages are a separate top-level field,
    not part of the messages array. The adapter handles this translation silently.
    """

    def __init__(self, model: str = "claude-3-5-sonnet-20241022",
                 api_key: str | None = None) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic>=0.25.0")

        self._model = model
        self._client = AsyncAnthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self._pricing = _ANTHROPIC_PRICING.get(model, (3.00, 15.00))

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def context_window(self) -> int:
        return 200_000   # all Claude 3.x models

    def _split_messages(
        self, messages: list[LlmMessage]
    ) -> tuple[str, list[dict]]:
        """
        Anthropic requires system instructions as a separate field.
        Extract all SYSTEM messages and join them; pass the rest as the messages array.
        """
        system_parts: list[str] = []
        chat_messages: list[dict] = []
        for m in messages:
            if m.role == MessageRole.SYSTEM:
                system_parts.append(m.content)
            else:
                chat_messages.append({"role": m.role.value, "content": m.content})
        return "\n\n".join(system_parts), chat_messages

    async def generate(
        self,
        messages: list[LlmMessage],
        config: LlmConfig | None = None,
    ) -> LlmResponse:
        cfg = config or LlmConfig()
        system_text, chat_messages = self._split_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": cfg.max_tokens,
            "messages": chat_messages,
        }
        if system_text:
            kwargs["system"] = system_text
        if cfg.temperature != 1.0:   # Anthropic default is 1.0
            kwargs["temperature"] = cfg.temperature

        start = time.monotonic()
        response = await self._client.messages.create(**kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        input_t = response.usage.input_tokens
        output_t = response.usage.output_tokens
        cost = (input_t / 1_000_000 * self._pricing[0] +
                output_t / 1_000_000 * self._pricing[1])

        # Extract text from Anthropic's ContentBlock list
        text = "".join(
            block.text for block in response.content
            if hasattr(block, "text")
        )

        return LlmResponse(
            content=text,
            finish_reason=response.stop_reason or "stop",
            usage=LlmUsage(
                input_tokens=input_t,
                output_tokens=output_t,
                total_tokens=input_t + output_t,
                model_id=response.model,
                provider=self.provider,
                cost_usd=cost,
                latency_ms=latency_ms,
            ),
        )

    async def stream(
        self,
        messages: list[LlmMessage],
        config: LlmConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        cfg = config or LlmConfig()
        system_text, chat_messages = self._split_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": cfg.max_tokens,
            "messages": chat_messages,
        }
        if system_text:
            kwargs["system"] = system_text

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
```

### The Google Gemini Adapter

Gemini uses a `parts`-based content model. Each message is a `Content` object containing a list of `Part` objects. The adapter also handles Gemini's safety ratings, which can cause a generation to finish without content — a case we surface as `finish_reason="safety"`.

```python
_GEMINI_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-pro":    (1.25,  10.00),
    "gemini-2.5-flash":  (0.075,  0.30),
    "gemini-1.5-pro":    (1.25,   5.00),
    "gemini-1.5-flash":  (0.075,  0.30),
}


class GeminiProvider(BaseLlm):
    """
    Adapter for Google's Gemini API via the google-generativeai SDK.
    Key difference: parts-based content model and separate system_instruction field.
    """

    def __init__(self, model: str = "gemini-2.5-flash",
                 api_key: str | None = None) -> None:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "Install google-generativeai: pip install google-generativeai>=0.7.0"
            )
        key = api_key or os.environ["GOOGLE_API_KEY"]
        genai.configure(api_key=key)
        self._genai = genai
        self._model_name = model
        self._pricing = _GEMINI_PRICING.get(model, (0.075, 0.30))

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "gemini"

    @property
    def context_window(self) -> int:
        windows = {
            "gemini-2.5-pro": 1_048_576,
            "gemini-2.5-flash": 1_048_576,
            "gemini-1.5-pro": 1_048_576,
            "gemini-1.5-flash": 1_048_576,
        }
        return windows.get(self._model_name, 1_048_576)

    def _build_gemini_history(
        self, messages: list[LlmMessage]
    ) -> tuple[str, list[dict]]:
        """Separate system instruction and build Gemini's history format."""
        system_parts: list[str] = []
        history: list[dict] = []
        for m in messages:
            if m.role == MessageRole.SYSTEM:
                system_parts.append(m.content)
            else:
                # Gemini uses "user" and "model" roles (not "assistant")
                gemini_role = "model" if m.role == MessageRole.ASSISTANT else "user"
                history.append({
                    "role": gemini_role,
                    "parts": [{"text": m.content}]
                })
        return "\n".join(system_parts), history

    async def generate(
        self,
        messages: list[LlmMessage],
        config: LlmConfig | None = None,
    ) -> LlmResponse:
        cfg = config or LlmConfig()
        system_instruction, history = self._build_gemini_history(messages)

        model_kwargs: dict[str, Any] = {"model_name": self._model_name}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction

        generation_config = self._genai.GenerationConfig(
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_tokens,
        )
        if cfg.json_mode:
            generation_config.response_mime_type = "application/json"

        model = self._genai.GenerativeModel(**model_kwargs)

        start = time.monotonic()
        # Run synchronous Gemini call in a thread to avoid blocking the event loop
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.generate_content(
                history, generation_config=generation_config
            )
        )
        latency_ms = (time.monotonic() - start) * 1000

        # Gemini may return empty content if safety filters triggered
        text = ""
        finish_reason = "stop"
        if response.candidates:
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason.name.lower()
            if candidate.content and candidate.content.parts:
                text = "".join(
                    p.text for p in candidate.content.parts if hasattr(p, "text")
                )

        input_t = response.usage_metadata.prompt_token_count
        output_t = response.usage_metadata.candidates_token_count
        cost = (input_t / 1_000_000 * self._pricing[0] +
                output_t / 1_000_000 * self._pricing[1])

        return LlmResponse(
            content=text,
            finish_reason=finish_reason,
            usage=LlmUsage(
                input_tokens=input_t,
                output_tokens=output_t,
                total_tokens=input_t + output_t,
                model_id=self._model_name,
                provider=self.provider,
                cost_usd=cost,
                latency_ms=latency_ms,
            ),
        )

    async def stream(
        self,
        messages: list[LlmMessage],
        config: LlmConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        cfg = config or LlmConfig()
        _, history = self._build_gemini_history(messages)
        model = self._genai.GenerativeModel(self._model_name)
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.generate_content(history, stream=True)
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
```

### The Ollama Adapter

Ollama runs LLMs locally via a simple REST API that is deliberately OpenAI-compatible. This means the adapter is the simplest of the four. Its purpose is different: Ollama handles requests that must not touch public APIs — tasks involving personally identifiable information, data residency requirements, or environments without internet access.

```python
class OllamaProvider(BaseLlm):
    """
    Adapter for Ollama (local model server).
    Uses Ollama's OpenAI-compatible REST API via httpx.
    Ideal for privacy-sensitive tasks, airgapped environments, or cost-zero local inference.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ) -> None:
        try:
            import httpx
        except ImportError:
            raise ImportError("Install httpx: pip install httpx>=0.27.0")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=120.0)

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "ollama"

    @property
    def context_window(self) -> int:
        # Ollama models vary; 8K is a safe conservative default
        return 8_192

    def _to_messages(self, messages: list[LlmMessage]) -> list[dict]:
        return [{"role": m.role.value, "content": m.content} for m in messages]

    async def generate(
        self,
        messages: list[LlmMessage],
        config: LlmConfig | None = None,
    ) -> LlmResponse:
        cfg = config or LlmConfig()
        payload = {
            "model": self._model,
            "messages": self._to_messages(messages),
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.max_tokens,
            },
        }
        start = time.monotonic()
        resp = await self._http.post(
            f"{self._base_url}/api/chat", json=payload
        )
        resp.raise_for_status()
        latency_ms = (time.monotonic() - start) * 1000
        data = resp.json()

        input_t = data.get("prompt_eval_count", 0)
        output_t = data.get("eval_count", 0)

        return LlmResponse(
            content=data["message"]["content"],
            finish_reason="stop",
            usage=LlmUsage(
                input_tokens=input_t,
                output_tokens=output_t,
                total_tokens=input_t + output_t,
                model_id=self._model,
                provider=self.provider,
                cost_usd=0.0,   # local inference has no API cost
                latency_ms=latency_ms,
            ),
        )

    async def stream(
        self,
        messages: list[LlmMessage],
        config: LlmConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        cfg = config or LlmConfig()
        payload = {
            "model": self._model,
            "messages": self._to_messages(messages),
            "stream": True,
            "options": {"temperature": cfg.temperature},
        }
        async with self._http.stream(
            "POST", f"{self._base_url}/api/chat", json=payload
        ) as response:
            import json as _json
            async for line in response.aiter_lines():
                if line:
                    chunk = _json.loads(line)
                    if token := chunk.get("message", {}).get("content", ""):
                        yield token
```

---

## 1.5 The `LlmRegistry`: Providers from Config

With four working adapters in place, we need a way to instantiate them without embedding provider selection logic into agent code. The `LlmRegistry` is a central catalogue that reads from a YAML configuration file and returns the right `BaseLlm` instance on demand.

This achieves hot-swap at the config level. Change one line in `providers.yaml`, and every agent in the system starts using a different model — without a deployment, without a restart, without touching agent code.

```python
# core/llm.py (continued)

import yaml
from typing import TYPE_CHECKING

_PROVIDER_CLASSES = {
    "openai":    lambda cfg: OpenAiProvider(
                     model=cfg.get("model", "gpt-4o"),
                     api_key=cfg.get("api_key")),
    "anthropic": lambda cfg: AnthropicProvider(
                     model=cfg.get("model", "claude-3-5-sonnet-20241022"),
                     api_key=cfg.get("api_key")),
    "gemini":    lambda cfg: GeminiProvider(
                     model=cfg.get("model", "gemini-2.5-flash"),
                     api_key=cfg.get("api_key")),
    "ollama":    lambda cfg: OllamaProvider(
                     model=cfg.get("model", "llama3.2"),
                     base_url=cfg.get("base_url", "http://localhost:11434")),
}


class LlmRegistry:
    """
    Central catalogue of available LLM providers.
    Instantiates adapters from a YAML config file.
    Agents call registry.get("alias") — never constructing providers directly.
    """

    def __init__(self) -> None:
        self._providers: dict[str, BaseLlm] = {}

    def register(self, alias: str, provider: BaseLlm) -> None:
        """Register a provider under a short alias."""
        self._providers[alias] = provider

    def get(self, alias: str) -> BaseLlm:
        """Retrieve a registered provider by alias."""
        if alias not in self._providers:
            available = ", ".join(self._providers.keys()) or "none"
            raise KeyError(
                f"Provider '{alias}' not found. "
                f"Registered: {available}"
            )
        return self._providers[alias]

    def list(self) -> list[str]:
        """Return all registered provider aliases."""
        return list(self._providers.keys())

    @classmethod
    def from_config(cls, config_path: str) -> "LlmRegistry":
        """
        Load providers from a YAML configuration file.

        Example providers.yaml:
            providers:
              fast:
                type: gemini
                model: gemini-2.5-flash
              smart:
                type: openai
                model: gpt-4o
              private:
                type: ollama
                model: llama3.2
        """
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        registry = cls()
        for alias, provider_cfg in raw.get("providers", {}).items():
            provider_type = provider_cfg.get("type")
            if provider_type not in _PROVIDER_CLASSES:
                raise ValueError(
                    f"Unknown provider type '{provider_type}' for alias '{alias}'. "
                    f"Valid types: {list(_PROVIDER_CLASSES.keys())}"
                )
            provider = _PROVIDER_CLASSES[provider_type](provider_cfg)
            registry.register(alias, provider)
        return registry
```

The corresponding `providers.yaml` for a typical setup:

```yaml
# providers.yaml — edit this file to change providers without touching agent code
providers:
  fast:
    type: gemini
    model: gemini-2.5-flash        # cheap, fast, large context — for classification/extraction

  smart:
    type: openai
    model: gpt-4o                  # frontier capability — for complex reasoning

  claude:
    type: anthropic
    model: claude-3-5-sonnet-20241022   # strong reasoning + 200K context

  private:
    type: ollama
    model: llama3.2                # local — for PII and sensitive data, zero cost
    base_url: http://localhost:11434

  default:
    type: openai
    model: gpt-4o-mini             # cost-efficient general purpose
```

---

## 1.6 The `RouterLlm`: Matching Tasks to Models

Dynamic model routing is one of the highest-leverage optimisations available in a multi-provider framework. The idea is straightforward: not every task needs the most expensive model. A request to classify a sentiment, extract a date from a string, or check a JSON schema does not need the same model that should handle complex multi-step planning or nuanced long-document analysis.

The `RouterLlm` wraps the `LlmRegistry` and adds a routing layer based on task complexity signals.

```python
class ComplexitySignal(str, Enum):
    """
    Signals for selecting the appropriate model tier.
    Set by the caller or determined automatically from message content.
    """
    LOW = "low"        # → fast, cheap model (classification, extraction, formatting)
    MEDIUM = "medium"  # → balanced model (summarisation, Q&A, code review)
    HIGH = "high"      # → frontier model (planning, long-form reasoning, novel code)
    PRIVATE = "private"# → local model (anything containing PII or sensitive data)


class RouterLlm(BaseLlm):
    """
    A meta-provider that selects the right model based on task complexity.
    This makes model selection a platform policy, not an agent decision.

    Agents call router.generate(messages, config) — the router decides which
    underlying provider handles the request based on complexity routing rules.
    """

    def __init__(
        self,
        registry: LlmRegistry,
        routing_table: dict[ComplexitySignal, str] | None = None,
    ) -> None:
        self._registry = registry
        self._routing_table: dict[ComplexitySignal, str] = routing_table or {
            ComplexitySignal.LOW:     "fast",
            ComplexitySignal.MEDIUM:  "default",
            ComplexitySignal.HIGH:    "smart",
            ComplexitySignal.PRIVATE: "private",
        }

    @property
    def model_id(self) -> str:
        return "router"

    @property
    def provider(self) -> str:
        return "router"

    @property
    def context_window(self) -> int:
        return 1_048_576   # reports the maximum across registered providers

    def _estimate_complexity(self, messages: list[LlmMessage]) -> ComplexitySignal:
        """
        Heuristic complexity estimation from message content.
        Override this in subclasses for ML-based classification.

        Simple heuristics used here:
        - Short messages → LOW
        - Messages containing 'plan', 'design', 'analyse', 'reason' → HIGH
        - Long messages (>500 tokens estimated) → HIGH
        - Default → MEDIUM
        """
        full_text = " ".join(m.content.lower() for m in messages)
        token_estimate = self.count_tokens_estimate(messages)

        high_complexity_keywords = {
            "plan", "design", "architect", "analyse", "analyze",
            "reason", "explain why", "compare and contrast",
            "step by step", "detailed", "comprehensive",
        }

        if any(kw in full_text for kw in high_complexity_keywords):
            return ComplexitySignal.HIGH
        if token_estimate > 500:
            return ComplexitySignal.HIGH
        if token_estimate < 50:
            return ComplexitySignal.LOW
        return ComplexitySignal.MEDIUM

    def _select_provider(
        self,
        complexity: ComplexitySignal,
    ) -> BaseLlm:
        alias = self._routing_table[complexity]
        return self._registry.get(alias)

    async def generate(
        self,
        messages: list[LlmMessage],
        config: LlmConfig | None = None,
        complexity: ComplexitySignal | None = None,
    ) -> LlmResponse:
        signal = complexity or self._estimate_complexity(messages)
        provider = self._select_provider(signal)
        return await provider.generate(messages, config)

    async def stream(
        self,
        messages: list[LlmMessage],
        config: LlmConfig | None = None,
        complexity: ComplexitySignal | None = None,
    ) -> AsyncGenerator[str, None]:
        signal = complexity or self._estimate_complexity(messages)
        provider = self._select_provider(signal)
        async for token in provider.stream(messages, config):
            yield token
```

The `RouterLlm` embodies the MRKL Foreman principle applied at the provider level. The LLM tier appropriate for the task is selected by the platform, not the agent. Agents do not contain logic like "use gpt-4o for complex tasks and gpt-4o-mini for simple ones." They simply call `router.generate()` and the platform makes that decision based on a policy the operations team controls.

---

## 1.7 What You Run: The Demo

Now we put it all together. The demo in `examples/ch01_hello_multimodel/main.py` sends the same prompt to all four providers and prints the response from each — demonstrating that the interface is genuinely uniform regardless of which vendor is serving the request.

```python
# examples/ch01_hello_multimodel/main.py
"""
Chapter 1 Demo — The Provider Abstraction Layer
The Agentic Spine: Engineering a Provider-Agnostic AI Framework

What this demonstrates:
  - One interface (BaseLlm) works identically across four providers
  - Swapping providers requires only a config change, not code changes
  - Every response carries identical usage metadata regardless of provider
  - RouterLlm selects the right provider automatically for simple vs complex tasks

Run:
  cd examples/ch01_hello_multimodel
  cp .env.example .env      # add your API keys
  pip install -r ../../requirements.txt
  python main.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the framework root to path so we can import core.llm
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.llm import (
    AnthropicProvider,
    ComplexitySignal,
    GeminiProvider,
    LlmConfig,
    LlmMessage,
    LlmRegistry,
    MessageRole,
    OllamaProvider,
    OpenAiProvider,
    RouterLlm,
)


SIMPLE_PROMPT = [
    LlmMessage(
        role=MessageRole.SYSTEM,
        content="You are a concise assistant. Answer in one sentence.",
    ),
    LlmMessage(
        role=MessageRole.USER,
        content="What is the capital of France?",
    ),
]

COMPLEX_PROMPT = [
    LlmMessage(
        role=MessageRole.SYSTEM,
        content="You are a senior software architect.",
    ),
    LlmMessage(
        role=MessageRole.USER,
        content=(
            "Design the interface for a provider-agnostic LLM abstraction layer. "
            "What methods must it define, what contracts must it enforce, "
            "and what metadata must every response carry? "
            "Think step by step."
        ),
    ),
]


async def demo_single_provider(name: str, provider, messages) -> None:
    print(f"\n{'─' * 60}")
    print(f"  Provider : {name}")
    print(f"  Model    : {provider.model_id}")
    print(f"  Context  : {provider.context_window:,} tokens")
    print(f"{'─' * 60}")

    config = LlmConfig(temperature=0.3, max_tokens=256)
    response = await provider.generate(messages, config)

    print(f"  Response : {response.content[:200]}...")
    print(f"  Tokens   : {response.usage.input_tokens} in / "
          f"{response.usage.output_tokens} out")
    print(f"  Cost     : ${response.usage.cost_usd:.6f} USD")
    print(f"  Latency  : {response.usage.latency_ms:.0f} ms")


async def demo_router(registry: LlmRegistry) -> None:
    print(f"\n{'═' * 60}")
    print("  RouterLlm — automatic complexity-based dispatch")
    print(f"{'═' * 60}")

    router = RouterLlm(registry)

    for label, messages, signal in [
        ("Simple task (→ fast model)", SIMPLE_PROMPT, ComplexitySignal.LOW),
        ("Complex task (→ smart model)", COMPLEX_PROMPT, ComplexitySignal.HIGH),
    ]:
        print(f"\n  Task: {label}")
        response = await router.generate(messages, complexity=signal)
        print(f"  Handled by  : {response.usage.provider} / {response.usage.model_id}")
        print(f"  Cost        : ${response.usage.cost_usd:.6f} USD")
        print(f"  Tokens out  : {response.usage.output_tokens}")


async def main() -> None:
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   The Agentic Spine — Chapter 1: Provider Abstraction    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # --- Direct provider demos ---
    providers_to_test = []

    if os.environ.get("OPENAI_API_KEY"):
        providers_to_test.append(("OpenAI / GPT-4o Mini",
                                   OpenAiProvider("gpt-4o-mini")))
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers_to_test.append(("Anthropic / Claude 3.5 Haiku",
                                   AnthropicProvider("claude-3-5-haiku-20241022")))
    if os.environ.get("GOOGLE_API_KEY"):
        providers_to_test.append(("Google / Gemini 2.5 Flash",
                                   GeminiProvider("gemini-2.5-flash")))

    # Ollama if running locally
    try:
        import httpx
        async with httpx.AsyncClient() as c:
            await c.get("http://localhost:11434/api/tags", timeout=2.0)
        providers_to_test.append(("Ollama / Llama 3.2 (local)",
                                   OllamaProvider("llama3.2")))
    except Exception:
        print("\n  ℹ  Ollama not detected — skipping local model demo")

    if not providers_to_test:
        print("\n  ⚠  No API keys found. Set at least one of:")
        print("     OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY")
        return

    print(f"\n  Testing {len(providers_to_test)} provider(s) with identical interface...\n")

    for name, provider in providers_to_test:
        await demo_single_provider(name, provider, SIMPLE_PROMPT)

    # --- Registry + Router demo ---
    if len(providers_to_test) >= 2:
        registry = LlmRegistry()
        # Register whatever providers we have
        for alias, (_, provider) in zip(
            ["fast", "smart", "default"], providers_to_test
        ):
            registry.register(alias, provider)

        await demo_router(registry)

    print("\n✅  Chapter 1 complete. core/llm.py is your provider abstraction layer.")
    print("    No agent code will ever import openai or anthropic directly.")
    print("    Next → Chapter 2: The Core Agent Backbone (core/agent.py)\n")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 1.8 Practical Patterns and Anti-Patterns

Before moving to Chapter 2, let us be explicit about the patterns this layer enforces and the mistakes it prevents.

### Patterns That Work

**Always route through the registry.** Never instantiate a provider adapter directly in agent code. If you find yourself writing `OpenAiProvider(model="gpt-4o")` inside a class that is not `LlmRegistry`, something is wrong. The registry is the single construction point for provider instances. This keeps your agent code decoupled from the choice of provider.

**Keep vendor-specific parameters out of `LlmConfig`.** `LlmConfig` carries `temperature`, `max_tokens`, `top_p`, and `json_mode`. Those are universal concepts. Anthropic's `top_k`, OpenAI's `frequency_penalty`, and Gemini's `candidate_count` are provider-specific and should not leak into the config object. If an adapter needs a specific parameter, it handles that internally.

**Treat cost as a first-class concern from day one.** Every `LlmResponse` carries `cost_usd`. This is not optional metadata — it is the raw material for the cost attribution and budget enforcement we build in Chapter 8. Log it, aggregate it, attribute it to sessions and users. The absence of cost data is a decision that compounds painfully at scale.

**Build your exit cost into the interface today.** The `BaseLlm` interface is your stable internal API. Your codebase calls `provider.generate()`. If you need to swap from managed OpenAI to a self-hosted model six months from now, only the `LlmRegistry` configuration changes — not the 200 call sites in your agents and tools. That is the difference between a configuration change and a six-month rewrite.

### Anti-Patterns to Avoid

**Importing vendor SDKs outside of adapter classes.** The rule is absolute: `import openai` belongs only inside `OpenAiProvider`. `from anthropic import Anthropic` belongs only inside `AnthropicProvider`. Anywhere else is a leak that re-introduces the coupling we are eliminating.

**Using model-specific behaviour as a feature.** The temptation to say "we're using Claude because it's better at format X" should be treated with suspicion. If your agent depends on a provider-specific behavioural quirk, you have a hidden coupling. Capabilities should be surfaced through tools and prompts, not through reliance on undocumented model behaviours that can change between versions.

**Skipping the streaming interface.** It is easy to implement only `generate()` and defer `stream()` to later. Do not. Streaming is how agents handle long outputs without blocking the event loop, and retrofitting streaming onto non-streaming code is significantly more expensive than building it from the start.

**Ignoring context window limits.** Every provider adapter declares its `context_window`. The orchestration layer — starting in Chapter 5 — will use this to decide whether to page memory or refuse a request. If you hardcode `BaseLlm.context_window` to return `999_999` in your adapter because it is convenient, you will get silent failures in production when a model receives more tokens than it can process.

---

## Summary

You have built the first and most fundamental module of the framework: `core/llm.py`. Everything else in this book is built on top of it.

- **`BaseLlm`** defines the universal contract — generate, stream, report usage, declare context window
- **Four adapters** translate that contract into the wire format of OpenAI, Anthropic, Gemini, and Ollama — absorbing all vendor-specific complexity so the rest of the framework never has to
- **`LlmResponse`** and **`LlmUsage`** normalise every provider's output into one shape, including cost metadata for attribution
- **`LlmRegistry`** loads providers from a YAML config file, giving you hot-swap at the configuration level
- **`RouterLlm`** makes model selection a platform policy — simple tasks go to cheap models, complex tasks go to frontier models, private tasks go local

The key architectural insight from this chapter: the model is a pluggable component, not a load-bearing wall. It was a foreman — it dispatches; it does not execute. The architecture supplies the machinery; the model supplies the policy within that machinery.

From this point forward, no module in this framework will ever import a vendor SDK directly. They will all call `BaseLlm`.

**Key takeaways:**

- Single-provider architectures create three compounding failure modes: outage fragility, deprecation exposure, and cost rigidity
- A proper abstraction layer is not a 30-line wrapper that re-exports the vendor SDK — it is an opinionated capability layer where call sites use stable verbs, not provider endpoints
- Provider adapters absorb all vendor-specific complexity: Anthropic's `system` field, Gemini's `parts` structure, Ollama's local endpoint
- Cost metadata belongs in every response, from the first call — not as an afterthought added when the CFO asks questions
- The `RouterLlm` encodes the MRKL Foreman principle at the provider level: task complexity determines model tier, as a platform policy, not an agent decision

---

## Further Reading

- Karpas et al. (2022), *MRKL Systems: A Modular, Neuro-Symbolic Architecture that Combines Language Models, External Knowledge Sources and Discrete Reasoning* — the foundational paper for the Foreman model
- Nowaczyk (2025), *Architectures for Building Agentic AI* (arXiv:2512.09458) — the Spine architecture and why reliability is an architectural property
- Anthropic (2026), *Effective Context Engineering for AI Agents* — the context window as a managed resource
- AI Gateway Architecture Guide, Maxim AI (2026) — production patterns for multi-provider routing at scale

---

*Next: Chapter 2 — The Core Agent Backbone. We build `core/agent.py`: the `BaseAgent` class, the `run_async` execution cycle, lifecycle hooks, and the scratchpad pattern for managing agent state.*
