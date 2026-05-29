# tests/conftest.py — shared fixtures for all chapters
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

# ── MockLlmProvider ────────────────────────────────────────────────────────────

@dataclass
class MockLlmResponse:
    content:       str
    input_tokens:  int = 10
    output_tokens: int = 20
    model:         str = "mock"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class MockLlmProvider:
    """
    Deterministic LLM provider for tests.
    Queued responses are returned in order; falls back to default_response.
    """

    def __init__(self, default_response: str = "Mock LLM response."):
        self._queue: list[str] = []
        self._default = default_response
        self.calls: list[list[Any]] = []   # captured call history

    def queue(self, *responses: str) -> MockLlmProvider:
        """Pre-load responses to be returned in order."""
        self._queue.extend(responses)
        return self

    async def complete(self, messages: list[Any], **kwargs: Any) -> MockLlmResponse:
        self.calls.append(list(messages))
        content = self._queue.pop(0) if self._queue else self._default
        return MockLlmResponse(content=content)

    @property
    def name(self) -> str:
        return "mock"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm() -> MockLlmProvider:
    return MockLlmProvider()


@pytest.fixture
def mock_llm_factory():
    """Factory for mock LLMs with pre-queued responses."""
    def _make(*responses: str) -> MockLlmProvider:
        return MockLlmProvider().queue(*responses)
    return _make
