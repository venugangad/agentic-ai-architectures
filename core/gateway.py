# core/gateway.py
"""
The Production LLM Gateway — core/gateway.py
The Agentic Spine: Engineering a Provider-Agnostic AI Framework

Config:          GatewayConfig (all tunable parameters)
Errors:          BudgetError · RateLimitError · CircuitOpenError
Budget:          CallBudget (per-session/per-request envelope)
Cost:            CallRecord · CostTracker (pluggable sink)
Rate limiting:   TokenBucketRateLimiter (requests/min + tokens/min)
Circuit breaker: CircuitBreaker (CLOSED → OPEN → HALF_OPEN → CLOSED)
Retry:           RetryPolicy (exponential backoff + full jitter)
Gateway:         LlmGateway (composes all layers; LlmProvider interface)
Sessions:        RedisSessionService · PostgresSessionService

Built in Chapter 8: The Production LLM Gateway
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


# ── Errors ────────────────────────────────────────────────────────────────────


class BudgetError(Exception):
    """A call would exceed a configured token or cost budget."""
    def __init__(self, message: str, limit_type: str, limit_value: float) -> None:
        super().__init__(message)
        self.limit_type = limit_type
        self.limit_value = limit_value


class RateLimitError(Exception):
    """Rate-limit capacity is exhausted."""
    def __init__(self, message: str, retry_after_seconds: float = 0.0) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class CircuitOpenError(Exception):
    """The circuit breaker is open — downstream calls are blocked."""
    def __init__(self, message: str, opens_at: float = 0.0) -> None:
        super().__init__(message)
        self.opens_at = opens_at


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class GatewayConfig:
    """
    All tunable gateway parameters in one place.
    Override at the composition root; nothing downstream needs to change.
    """
    # Per-call limits
    max_tokens_per_call: int = 8_000
    max_cost_per_call_usd: float = 0.50

    # Token bucket rate limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 100_000

    # Circuit breaker
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    success_threshold: int = 2

    # Retry (exponential backoff + jitter)
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    retry_on: tuple[type[Exception], ...] = field(
        default_factory=lambda: (RateLimitError, TimeoutError, ConnectionError)
    )

    # Cost (USD per 1M tokens — update from provider pricing page)
    input_cost_per_million: float = 0.15   # gpt-4o-mini default
    output_cost_per_million: float = 0.60


# ── Budget ────────────────────────────────────────────────────────────────────


@dataclass
class CallBudget:
    """
    Per-session or per-request spend envelope.
    Attach to AgentContext.metadata["budget"] and pass to gateway.complete().
    """
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    tokens_used: int = 0
    cost_used_usd: float = 0.0

    def check(self, estimated_tokens: int, estimated_cost: float) -> None:
        """Raise BudgetError if this call would exceed the budget."""
        if self.max_tokens and self.tokens_used + estimated_tokens > self.max_tokens:
            raise BudgetError(
                f"Token budget: {self.tokens_used + estimated_tokens} > {self.max_tokens}",
                limit_type="tokens",
                limit_value=float(self.max_tokens),
            )
        if self.max_cost_usd and self.cost_used_usd + estimated_cost > self.max_cost_usd:
            raise BudgetError(
                f"Cost budget: ${self.cost_used_usd + estimated_cost:.4f} > ${self.max_cost_usd:.4f}",
                limit_type="cost_usd",
                limit_value=self.max_cost_usd,
            )

    def record(self, tokens: int, cost: float) -> None:
        self.tokens_used += tokens
        self.cost_used_usd += cost


# ── Cost Tracking ─────────────────────────────────────────────────────────────


@dataclass
class CallRecord:
    """Complete accounting record for one LLM call."""
    call_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: float
    user_id: str | None = None
    session_id: str | None = None
    agent_name: str | None = None
    success: bool = True
    error_type: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CostTracker:
    """
    Accumulates cost across calls with a pluggable sink.
    sink: callable(record: CallRecord) — sync or async.
    Examples: Prometheus gauge, Datadog metric, database writer, logging.
    """

    def __init__(
        self,
        input_cost_per_million: float,
        output_cost_per_million: float,
        sink: Any | None = None,
    ) -> None:
        self._input_cpm = input_cost_per_million
        self._output_cpm = output_cost_per_million
        self._sink = sink
        self._records: list[CallRecord] = []
        self._lock = asyncio.Lock()
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._total_calls: int = 0

    def estimate_cost(self, input_tokens: int, output_tokens: int = 0) -> float:
        return (
            input_tokens * self._input_cpm / 1_000_000
            + output_tokens * self._output_cpm / 1_000_000
        )

    async def record(self, rec: CallRecord) -> None:
        async with self._lock:
            self._records.append(rec)
            self._total_cost += rec.cost_usd
            self._total_tokens += rec.total_tokens
            self._total_calls += 1

        log.info(
            "gateway call_id=%s model=%s tokens=%d cost=$%.5f latency=%dms ok=%s",
            rec.call_id, rec.model, rec.total_tokens,
            rec.cost_usd, rec.latency_ms, rec.success,
        )
        if self._sink:
            try:
                if asyncio.iscoroutinefunction(self._sink):
                    await self._sink(rec)
                else:
                    self._sink(rec)
            except Exception as exc:
                log.error("CostTracker sink error: %s", exc)

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "total_cost_usd": round(self._total_cost, 6),
            "avg_cost_per_call": round(
                self._total_cost / max(self._total_calls, 1), 6
            ),
        }

    def get_records(
        self,
        since: float | None = None,
        user_id: str | None = None,
    ) -> list[CallRecord]:
        recs = self._records
        if since:
            recs = [r for r in recs if r.timestamp >= since]
        if user_id:
            recs = [r for r in recs if r.user_id == user_id]
        return recs


# ── Rate Limiter ──────────────────────────────────────────────────────────────


class TokenBucketRateLimiter:
    """
    Dual token bucket: requests/minute and tokens/minute.
    Continuous refill (not fixed windows) prevents burst-at-boundary attacks.
    Thread-safe via asyncio.Lock.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100_000,
    ) -> None:
        self._rpm = requests_per_minute
        self._tpm = tokens_per_minute
        self._req_tokens: float = float(requests_per_minute)
        self._tok_tokens: float = float(tokens_per_minute)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 0, wait: bool = True) -> float:
        """
        Reserve capacity. Returns seconds waited (0 if none).
        Raises RateLimitError if wait=False and capacity unavailable.
        """
        async with self._lock:
            self._refill()
            wait_time = self._compute_wait(estimated_tokens)
            if wait_time > 0 and not wait:
                raise RateLimitError(
                    f"Rate limit: retry in {wait_time:.1f}s",
                    retry_after_seconds=wait_time,
                )
            self._req_tokens = max(0.0, self._req_tokens - 1.0)
            self._tok_tokens = max(0.0, self._tok_tokens - estimated_tokens)

        if wait_time > 0:
            await asyncio.sleep(wait_time)
        return wait_time

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._req_tokens = min(
            float(self._rpm), self._req_tokens + self._rpm * elapsed / 60.0
        )
        self._tok_tokens = min(
            float(self._tpm), self._tok_tokens + self._tpm * elapsed / 60.0
        )
        self._last_refill = now

    def _compute_wait(self, tokens: int) -> float:
        req_wait = 0.0
        tok_wait = 0.0
        if self._req_tokens < 1.0:
            req_wait = (1.0 - self._req_tokens) / (self._rpm / 60.0)
        if tokens > 0 and self._tok_tokens < tokens:
            tok_wait = (tokens - self._tok_tokens) / (self._tpm / 60.0)
        return max(req_wait, tok_wait)


# ── Circuit Breaker ───────────────────────────────────────────────────────────


class CircuitState(str, Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    CLOSED → OPEN after failure_threshold consecutive failures.
    OPEN   → HALF_OPEN after recovery_timeout_seconds.
    HALF_OPEN → CLOSED after success_threshold probe successes.
    HALF_OPEN → OPEN on any probe failure (timer reset).
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
        success_threshold: int = 2,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._success_threshold = success_threshold
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        async with self._lock:
            await self._maybe_transition()
            if self._state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit open — recovery in {self._recovery_timeout:.0f}s",
                    opens_at=self._opened_at + self._recovery_timeout,
                )
        try:
            result = await fn(*args, **kwargs)
            await self._on_success()
            return result
        except Exception:
            await self._on_failure()
            raise

    async def _maybe_transition(self) -> None:
        if (self._state == CircuitState.OPEN
                and time.monotonic() - self._opened_at >= self._recovery_timeout):
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0
            log.info("CircuitBreaker: OPEN → HALF_OPEN")

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    log.info("CircuitBreaker: HALF_OPEN → CLOSED")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def _on_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                log.warning("CircuitBreaker: HALF_OPEN → OPEN (probe failed)")
            elif (self._state == CircuitState.CLOSED
                  and self._failure_count >= self._failure_threshold):
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                log.warning(
                    "CircuitBreaker: CLOSED → OPEN after %d failures", self._failure_count
                )

    def get_metrics(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "opened_at": self._opened_at if self._state != CircuitState.CLOSED else None,
        }


# ── Retry Policy ──────────────────────────────────────────────────────────────


class RetryPolicy:
    """
    Exponential backoff with full jitter.
    delay = random(0, min(max_delay, base * 2^attempt))

    Full jitter prevents thundering herd on recovery.
    BudgetError and CircuitOpenError are never retried.
    Retry-After from RateLimitError is honoured.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 30.0,
        retry_on: tuple[type[Exception], ...] | None = None,
    ) -> None:
        self._max_retries = max_retries
        self._base = base_delay_seconds
        self._max = max_delay_seconds
        self._retry_on = retry_on or (RateLimitError, TimeoutError, ConnectionError)

    async def execute(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await fn(*args, **kwargs)
            except (CircuitOpenError, BudgetError):
                raise
            except Exception as exc:
                if not isinstance(exc, self._retry_on):
                    raise
                last_exc = exc
                if attempt == self._max_retries:
                    break
                delay = self._jitter(attempt)
                if isinstance(exc, RateLimitError) and exc.retry_after_seconds > delay:
                    delay = exc.retry_after_seconds
                log.warning(
                    "RetryPolicy: attempt %d/%d failed (%s), %.2fs wait",
                    attempt + 1, self._max_retries, type(exc).__name__, delay,
                )
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    def _jitter(self, attempt: int) -> float:
        import random
        cap = min(self._max, self._base * (2 ** attempt))
        return random.uniform(0, cap)


# ── LlmGateway ────────────────────────────────────────────────────────────────


class LlmGateway:
    """
    Production wrapper for any LlmProvider.

    Layers (in order):
      1. BudgetGuard    — pre-call token/cost check (no I/O)
      2. RateLimiter    — token bucket, may wait
      3. CircuitBreaker — fail-fast when provider is degraded
      4. RetryPolicy    — exponential backoff with jitter
      5. CostTracker    — record tokens, cost, latency

    Same interface as LlmProvider from Chapter 1.
    Replace provider with gateway at the composition root — no agent changes needed.
    """

    def __init__(
        self,
        provider: Any,
        config: GatewayConfig | None = None,
        cost_sink: Any | None = None,
    ) -> None:
        self._provider = provider
        self._cfg = config or GatewayConfig()
        self._rate_limiter = TokenBucketRateLimiter(
            requests_per_minute=self._cfg.requests_per_minute,
            tokens_per_minute=self._cfg.tokens_per_minute,
        )
        self._circuit = CircuitBreaker(
            failure_threshold=self._cfg.failure_threshold,
            recovery_timeout_seconds=self._cfg.recovery_timeout_seconds,
            success_threshold=self._cfg.success_threshold,
        )
        self._retry = RetryPolicy(
            max_retries=self._cfg.max_retries,
            base_delay_seconds=self._cfg.base_delay_seconds,
            max_delay_seconds=self._cfg.max_delay_seconds,
            retry_on=self._cfg.retry_on,
        )
        self._tracker = CostTracker(
            input_cost_per_million=self._cfg.input_cost_per_million,
            output_cost_per_million=self._cfg.output_cost_per_million,
            sink=cost_sink,
        )

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        budget: CallBudget | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        # ① Budget pre-check
        est_in = self._estimate_tokens(messages)
        est_cost = self._tracker.estimate_cost(est_in, max_tokens)
        if budget:
            budget.check(est_in + max_tokens, est_cost)
        if est_in + max_tokens > self._cfg.max_tokens_per_call:
            raise BudgetError(
                f"~{est_in + max_tokens} tokens exceeds per-call limit {self._cfg.max_tokens_per_call}",
                limit_type="tokens_per_call",
                limit_value=float(self._cfg.max_tokens_per_call),
            )

        # ② Rate limit
        await self._rate_limiter.acquire(estimated_tokens=est_in)

        # ③+④ Circuit + Retry
        call_id = str(uuid.uuid4())[:8]
        start = time.monotonic()

        async def _call() -> Any:
            return await self._provider.complete(
                messages, temperature=temperature, max_tokens=max_tokens, **kwargs
            )

        try:
            response = await self._retry.execute(
                lambda: self._circuit.call(_call)
            )
        except Exception as exc:
            await self._tracker.record(CallRecord(
                call_id=call_id,
                model=getattr(self._provider, "model", "unknown"),
                input_tokens=0, output_tokens=0, cost_usd=0.0,
                latency_ms=(time.monotonic() - start) * 1000,
                timestamp=time.time(),
                user_id=user_id, session_id=session_id, agent_name=agent_name,
                success=False, error_type=type(exc).__name__,
            ))
            raise

        # ⑤ Cost tracking
        in_tok = getattr(response, "input_tokens", 0)
        out_tok = getattr(response, "output_tokens", 0)
        cost = self._tracker.estimate_cost(in_tok, out_tok)
        await self._tracker.record(CallRecord(
            call_id=call_id,
            model=getattr(self._provider, "model", "unknown"),
            input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost,
            latency_ms=(time.monotonic() - start) * 1000,
            timestamp=time.time(),
            user_id=user_id, session_id=session_id, agent_name=agent_name,
        ))
        if budget:
            budget.record(in_tok + out_tok, cost)
        return response

    def get_metrics(self) -> dict[str, Any]:
        return {
            "cost": self._tracker.get_summary(),
            "circuit": self._circuit.get_metrics(),
        }

    @staticmethod
    def _estimate_tokens(messages: list[dict]) -> int:
        total = sum(len(m.get("content", "")) for m in messages)
        return max(1, total // 4)


# ── Redis Session Service ─────────────────────────────────────────────────────


class RedisSessionService:
    """
    Redis-backed SessionService.
    Key schema: {prefix}:{app}:{user}:{sid}  →  JSON string
    Index key:  {prefix}:index:{app}:{user}  →  Redis SET of session_ids

    Requires: pip install redis[asyncio]
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "agentspine:session",
        session_ttl_seconds: int = 86400 * 30,
    ) -> None:
        import redis.asyncio as aioredis  # type: ignore
        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        self._prefix = prefix
        self._ttl = session_ttl_seconds

    def _key(self, app: str, uid: str, sid: str) -> str:
        return f"{self._prefix}:{app}:{uid}:{sid}"

    def _idx(self, app: str, uid: str) -> str:
        return f"{self._prefix}:index:{app}:{uid}"

    async def create_session(
        self, user_id: str, app_name: str = "default",
        session_id: str | None = None, initial_state: dict | None = None,
        metadata: dict | None = None,
    ) -> Any:
        from storage.session import SessionState, SessionExistsError
        sid = session_id or str(uuid.uuid4())
        key = self._key(app_name, user_id, sid)
        if await self._redis.exists(key):
            raise SessionExistsError(sid)
        session = SessionState(
            session_id=sid, user_id=user_id, app_name=app_name,
            state=dict(initial_state or {}), metadata=dict(metadata or {}),
        )
        await self._redis.set(key, json.dumps(session.to_dict()), ex=self._ttl)
        await self._redis.sadd(self._idx(app_name, user_id), sid)
        return session

    async def get_session(
        self, session_id: str, user_id: str, app_name: str = "default"
    ) -> Any:
        from storage.session import SessionState
        data = await self._redis.get(self._key(app_name, user_id, session_id))
        if data is None:
            return None
        try:
            return SessionState.from_dict(json.loads(data))
        except (json.JSONDecodeError, KeyError) as exc:
            log.error("RedisSessionService: corrupt %s: %s", session_id, exc)
            return None

    async def update_session(self, session: Any) -> Any:
        from storage.session import SessionNotFoundError
        key = self._key(session.app_name, session.user_id, session.session_id)
        if not await self._redis.exists(key):
            raise SessionNotFoundError(session.session_id)
        session.updated_at = time.time()
        await self._redis.set(key, json.dumps(session.to_dict()), ex=self._ttl)
        return session

    async def delete_session(
        self, session_id: str, user_id: str, app_name: str = "default"
    ) -> None:
        await self._redis.delete(self._key(app_name, user_id, session_id))
        await self._redis.srem(self._idx(app_name, user_id), session_id)

    async def list_sessions(self, user_id: str, app_name: str = "default") -> list[Any]:
        sids = await self._redis.smembers(self._idx(app_name, user_id))
        sessions = []
        for sid in sids:
            s = await self.get_session(sid, user_id, app_name)
            if s:
                sessions.append(s)
        return sessions

    async def get_or_create_session(
        self, user_id: str, session_id: str | None,
        app_name: str = "default", initial_state: dict | None = None,
    ) -> Any:
        if session_id:
            s = await self.get_session(session_id, user_id, app_name)
            if s:
                return s
        return await self.create_session(user_id=user_id, app_name=app_name,
                                          session_id=session_id, initial_state=initial_state)


# ── Postgres Session Service ──────────────────────────────────────────────────


class PostgresSessionService:
    """
    PostgreSQL-backed SessionService with asyncpg connection pool.

    Schema (run once):
        CREATE TABLE sessions (
            session_id  TEXT NOT NULL,
            user_id     TEXT NOT NULL,
            app_name    TEXT NOT NULL DEFAULT 'default',
            data        JSONB NOT NULL,
            created_at  DOUBLE PRECISION NOT NULL,
            updated_at  DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (app_name, user_id, session_id)
        );
        CREATE INDEX ON sessions (app_name, user_id);

    Requires: pip install asyncpg
    """

    def __init__(self, dsn: str, min_size: int = 2, max_size: int = 10) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Any = None

    async def _pool_(self) -> Any:
        if self._pool is None:
            import asyncpg  # type: ignore
            self._pool = await asyncpg.create_pool(
                self._dsn, min_size=self._min_size, max_size=self._max_size
            )
        return self._pool

    async def create_session(
        self, user_id: str, app_name: str = "default",
        session_id: str | None = None, initial_state: dict | None = None,
        metadata: dict | None = None,
    ) -> Any:
        from storage.session import SessionState, SessionExistsError
        sid = session_id or str(uuid.uuid4())
        session = SessionState(
            session_id=sid, user_id=user_id, app_name=app_name,
            state=dict(initial_state or {}), metadata=dict(metadata or {}),
        )
        pool = await self._pool_()
        try:
            await pool.execute(
                """INSERT INTO sessions(session_id,user_id,app_name,data,created_at,updated_at)
                   VALUES($1,$2,$3,$4::jsonb,$5,$6)""",
                sid, user_id, app_name,
                json.dumps(session.to_dict()),
                session.created_at, session.updated_at,
            )
        except Exception as exc:
            s = str(exc).lower()
            if "unique" in s or "duplicate" in s:
                raise SessionExistsError(sid)
            raise
        return session

    async def get_session(
        self, session_id: str, user_id: str, app_name: str = "default"
    ) -> Any:
        from storage.session import SessionState
        pool = await self._pool_()
        row = await pool.fetchrow(
            "SELECT data FROM sessions WHERE app_name=$1 AND user_id=$2 AND session_id=$3",
            app_name, user_id, session_id,
        )
        if row is None:
            return None
        try:
            return SessionState.from_dict(json.loads(row["data"]))
        except (json.JSONDecodeError, KeyError) as exc:
            log.error("PostgresSessionService: corrupt %s: %s", session_id, exc)
            return None

    async def update_session(self, session: Any) -> Any:
        from storage.session import SessionNotFoundError
        session.updated_at = time.time()
        pool = await self._pool_()
        result = await pool.execute(
            """UPDATE sessions SET data=$4::jsonb, updated_at=$5
               WHERE app_name=$1 AND user_id=$2 AND session_id=$3""",
            session.app_name, session.user_id, session.session_id,
            json.dumps(session.to_dict()), session.updated_at,
        )
        if result == "UPDATE 0":
            raise SessionNotFoundError(session.session_id)
        return session

    async def delete_session(
        self, session_id: str, user_id: str, app_name: str = "default"
    ) -> None:
        pool = await self._pool_()
        await pool.execute(
            "DELETE FROM sessions WHERE app_name=$1 AND user_id=$2 AND session_id=$3",
            app_name, user_id, session_id,
        )

    async def list_sessions(self, user_id: str, app_name: str = "default") -> list[Any]:
        from storage.session import SessionState
        pool = await self._pool_()
        rows = await pool.fetch(
            "SELECT data FROM sessions WHERE app_name=$1 AND user_id=$2 ORDER BY updated_at DESC",
            app_name, user_id,
        )
        sessions = []
        for row in rows:
            try:
                sessions.append(SessionState.from_dict(json.loads(row["data"])))
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions

    async def get_or_create_session(
        self, user_id: str, session_id: str | None,
        app_name: str = "default", initial_state: dict | None = None,
    ) -> Any:
        if session_id:
            s = await self.get_session(session_id, user_id, app_name)
            if s:
                return s
        return await self.create_session(user_id=user_id, app_name=app_name,
                                          session_id=session_id, initial_state=initial_state)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
