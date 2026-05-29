# tests/ch08/test_gateway.py — Chapter 8: Production LLM Gateway
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from core.gateway import (
    BudgetError,
    CallBudget,
    CallRecord,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    CostTracker,
    GatewayConfig,
    LlmGateway,
    RateLimitError,
    RetryPolicy,
    TokenBucketRateLimiter,
)

# ── CallBudget ────────────────────────────────────────────────────────────────

class TestCallBudget:
    def test_check_passes_within_limits(self):
        budget = CallBudget(max_tokens=1000, max_cost_usd=1.0)
        budget.check(estimated_tokens=100, estimated_cost=0.10)  # should not raise

    def test_check_raises_on_token_exceed(self):
        budget = CallBudget(max_tokens=100, max_cost_usd=10.0)
        budget.tokens_used = 90
        with pytest.raises(BudgetError) as exc_info:
            budget.check(estimated_tokens=20, estimated_cost=0.01)
        assert exc_info.value.limit_type == "tokens"

    def test_check_raises_on_cost_exceed(self):
        budget = CallBudget(max_tokens=10000, max_cost_usd=0.50)
        budget.cost_used_usd = 0.45
        with pytest.raises(BudgetError) as exc_info:
            budget.check(estimated_tokens=10, estimated_cost=0.10)
        assert exc_info.value.limit_type == "cost"

    def test_record_accumulates(self):
        budget = CallBudget(max_tokens=10000, max_cost_usd=10.0)
        budget.record(tokens=100, cost=0.05)
        budget.record(tokens=200, cost=0.10)
        assert budget.tokens_used == 300
        assert abs(budget.cost_used_usd - 0.15) < 1e-9


# ── CostTracker ───────────────────────────────────────────────────────────────

class TestCostTracker:
    def test_estimate_cost(self):
        tracker = CostTracker()
        cost = tracker.estimate_cost(input_tokens=1000, output_tokens=500)
        assert cost > 0

    def test_sink_called_on_record(self):
        received = []
        tracker = CostTracker(sink=received.append)
        rec = CallRecord(
            call_id="c1", model="gpt-4o",
            input_tokens=100, output_tokens=50,
            cost_usd=0.01, latency_ms=200,
            timestamp=time.time(), success=True,
        )
        tracker.record(rec)
        assert len(received) == 1
        assert received[0].call_id == "c1"

    def test_get_summary_empty(self):
        tracker = CostTracker()
        summary = tracker.get_summary()
        assert summary["total_calls"] == 0
        assert summary["total_cost_usd"] == 0.0


# ── TokenBucketRateLimiter ────────────────────────────────────────────────────

class TestTokenBucketRateLimiter:
    def test_acquire_succeeds_within_capacity(self):
        limiter = TokenBucketRateLimiter(
            requests_per_minute=60,
            tokens_per_minute=100_000,
        )
        # Should not raise for a small request
        asyncio.get_event_loop().run_until_complete(
            limiter.acquire(estimated_tokens=100, wait=False)
        )

    def test_acquire_raises_when_bucket_empty_no_wait(self):
        limiter = TokenBucketRateLimiter(
            requests_per_minute=1,   # only 1 req/min
            tokens_per_minute=100_000,
        )
        loop = asyncio.new_event_loop()
        # First request drains the bucket
        loop.run_until_complete(limiter.acquire(estimated_tokens=1, wait=False))
        # Second request should fail immediately
        with pytest.raises(RateLimitError):
            loop.run_until_complete(limiter.acquire(estimated_tokens=1, wait=False))
        loop.close()


# ── CircuitBreaker ────────────────────────────────────────────────────────────

class TestCircuitBreaker:
    def _make_cb(self, threshold: int = 3, recovery: float = 30.0, success_threshold: int = 2):
        return CircuitBreaker(
            failure_threshold=threshold,
            recovery_timeout_seconds=recovery,
            success_threshold=success_threshold,
        )

    def test_starts_closed(self):
        cb = self._make_cb()
        assert cb.state == CircuitState.CLOSED

    async def test_opens_after_threshold_failures(self):
        cb = self._make_cb(threshold=3)

        async def failing():
            raise ValueError("boom")

        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.call(failing)

        assert cb.state == CircuitState.OPEN

    async def test_raises_circuit_open_when_open(self):
        cb = self._make_cb(threshold=1)

        async def failing():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            await cb.call(failing)

        with pytest.raises(CircuitOpenError):
            await cb.call(failing)

    async def test_transitions_half_open_after_timeout(self):
        cb = self._make_cb(threshold=1, recovery=0.01)

        async def failing():
            raise ValueError()

        with pytest.raises(ValueError):
            await cb.call(failing)

        assert cb.state == CircuitState.OPEN
        await asyncio.sleep(0.02)
        # Trigger the transition check
        cb._maybe_transition()
        assert cb.state == CircuitState.HALF_OPEN

    async def test_closes_after_success_threshold(self):
        cb = self._make_cb(threshold=1, recovery=0.01, success_threshold=2)

        async def failing():
            raise ValueError()

        async def succeeding():
            return "ok"

        with pytest.raises(ValueError):
            await cb.call(failing)

        await asyncio.sleep(0.02)
        cb._maybe_transition()

        # Two successes → CLOSED
        await cb.call(succeeding)
        await cb.call(succeeding)
        assert cb.state == CircuitState.CLOSED


# ── RetryPolicy ───────────────────────────────────────────────────────────────

class TestRetryPolicy:
    async def test_succeeds_on_first_try(self):
        policy = RetryPolicy(max_retries=3, base_delay_seconds=0.0)

        async def fn():
            return "ok"

        result = await policy.execute(fn)
        assert result == "ok"

    async def test_retries_on_transient_error(self):
        policy = RetryPolicy(max_retries=3, base_delay_seconds=0.0)
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "ok"

        result = await policy.execute(fn)
        assert result == "ok"
        assert call_count == 3

    async def test_never_retries_budget_error(self):
        policy = RetryPolicy(max_retries=3, base_delay_seconds=0.0)
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            raise BudgetError("over budget", limit_type="tokens", limit_value=100)

        with pytest.raises(BudgetError):
            await policy.execute(fn)

        assert call_count == 1  # never retried

    async def test_never_retries_circuit_open_error(self):
        policy = RetryPolicy(max_retries=3, base_delay_seconds=0.0)
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            raise CircuitOpenError("circuit open", opens_at=time.time())

        with pytest.raises(CircuitOpenError):
            await policy.execute(fn)

        assert call_count == 1


# ── LlmGateway integration ────────────────────────────────────────────────────

class TestLlmGateway:
    async def test_complete_returns_response(self, mock_llm):
        mock_llm.queue("Hello from gateway!")
        gateway = LlmGateway(provider=mock_llm, config=GatewayConfig())
        from core.llm import Message
        response = await gateway.complete([Message(role="user", content="hi")])
        assert response.content == "Hello from gateway!"

    async def test_budget_error_raised_before_llm_call(self):
        from core.llm import Message
        config = GatewayConfig(max_tokens_per_call=1)  # tiny limit
        mock = MagicMock()
        gateway = LlmGateway(provider=mock, config=config)
        # Should raise BudgetError without calling the provider
        with pytest.raises(BudgetError):
            await gateway.complete([Message(role="user", content="x" * 1000)])
        mock.complete.assert_not_called()
