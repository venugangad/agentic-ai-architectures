# CLAUDE.md — agnostic-agent-framework
> Context file for AI-assisted development · Read before generating any code

---

## Project Purpose

Provider-agnostic agentic AI framework built chapter-by-chapter in *The Agentic Spine*.

**Core invariant:** All LLM calls go through `LlmProvider` (defined in `core/llm.py`).
Never import `openai`, `anthropic`, `google.generativeai`, or any provider SDK
directly from `core/`, `tools/`, `storage/`, `knowledge/`, `orchestration/`, or `monitoring/`.
Only `core/llm.py` and its concrete provider subclasses may import provider SDKs.

---

## Architecture Rules (Non-Negotiable)

1. **Composition root only** — Wire all dependencies in `examples/` or `tests/`. Never inside library modules. No module instantiates its own dependencies.

2. **No globals or module-level singletons** — Every dependency is injected. No `_instance` caching at module level.

3. **Protocols over ABCs** — Use `typing.Protocol` for cross-module contracts. `core/llm.py`'s `LlmProvider` is a `Protocol`. Other modules use `isinstance(x, LlmProvider)` with `@runtime_checkable`.

4. **Async throughout** — Every I/O method must be `async def`. Sync wrappers are explicitly forbidden. Use `asyncio.to_thread()` for CPU-bound work.

5. **Layer discipline** — Dependencies only flow downward:
   ```
   orchestration/  →  core/, tools/, storage/, knowledge/, monitoring/
   monitoring/     →  (no imports from other layers)
   knowledge/      →  core/
   storage/        →  (no imports from other layers)
   tools/          →  core/
   core/           →  (no cross-imports within core/ except llm.py)
   ```
   Never import sideways (e.g., `orchestration/router.py` must not import from `orchestration/orchestrator.py`).

6. **No silent failures** — Every exception must be caught explicitly and re-raised with context, or logged + re-raised. Never `except: pass`.

---

## Chapter Mapping

| Module | Chapter | Key Classes |
|---|---|---|
| `core/llm.py` | 1 | `LlmProvider` (Protocol), `OpenAIProvider`, `AnthropicProvider`, `GeminiProvider`, `MockLlmProvider`, `Message`, `LlmResponse` |
| `core/agent.py` | 2 | `Agent`, `AgentEvent`, `EventKind` |
| `tools/registry.py` | 3 | `ToolRegistry`, `ToolSpec`, `ToolResult` |
| `tools/mcp_connector.py` | 3 | `MCPConnector` (stub — not yet implemented) |
| `orchestration/router.py` | 4 | `SemanticRouter`, `RouteConfig` |
| `storage/session.py` | 5 | `SessionService` (Protocol), `Session`, `InMemorySessionService`, `FileSessionService`, `RedisSessionService`, `PostgresSessionService` |
| `knowledge/rag.py` | 6 | `HybridRetriever`, `VectorIndex`, `BM25Index`, `TextChunker`, `EmbeddingProvider`, `Chunk`, `RetrievalResult` |
| `knowledge/memory_service.py` | 6 | `MemoryService`, `MemoryTier`, `MemoryEntry` |
| `core/planner.py` | 7 | `BasePlanner`, `LinearPlanner`, `ReActPlanner`, `TreeOfThoughtPlanner`, `SagaPlanner`, `Plan`, `PlanStep`, `create_planner` |
| `core/gateway.py` | 8 | `LlmGateway`, `GatewayConfig`, `CircuitBreaker`, `CircuitState`, `TokenBucketRateLimiter`, `CostTracker`, `CallBudget`, `CallRecord`, `RetryPolicy` |
| `monitoring/safety.py` | 9 | `SafetyMonitor`, `SafetyPolicy`, `PIIDetector`, `PromptInjectionDetector`, `ContentFilter`, `HumanCheckpoint`, `SafetyViolation`, `SafetyError` |
| `monitoring/telemetry.py` | 9 | `TelemetryCollector`, `Span`, `Trace`, `SpanKind`, `MetricsRegistry`, `StructuredLogger`, `StdoutExporter`, `PrometheusExporter` |
| `orchestration/orchestrator.py` | 10 | `SupervisorAgent`, `AgentRegistry`, `AgentCard`, `A2ATask`, `A2AResult`, `TaskState`, `agent_worker`, `tool_worker` |

---

## Testing Requirements

- Every new module needs a `tests/chNN/test_<module>.py`.
- Use `MockLlmProvider` from `tests/conftest.py` — **never call real LLMs in tests**.
- `pytest-asyncio` is configured with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed.
- Coverage gates: 80% overall, 90% for `core/`.
- Test both the happy path AND the failure path for every public method.
- For circuit breaker, rate limiter, and budget tests: test state transitions explicitly.

---

## Common Commands

```bash
# Editable install for development (OpenAI provider + RAG + telemetry + dev tools)
pip install -e ".[dev,openai,rag,telemetry]"

# Run all tests
pytest

# Run one chapter's tests
pytest tests/ch09/ -v

# Run with coverage
pytest --cov=. --cov-report=term-missing

# Lint
ruff check .

# Type-check
mypy . --ignore-missing-imports

# Build distribution
python -m build

# Verify wheel installs clean
python -m venv /tmp/test-env
/tmp/test-env/bin/pip install dist/*.whl
/tmp/test-env/bin/python -c "import orchestration.orchestrator; print('OK')"
```

---

## Adding a New LLM Provider

1. Open `core/llm.py`
2. Add a new class that satisfies `LlmProvider` Protocol:
   ```python
   class MyProvider:
       async def complete(self, messages: list[Message], **kwargs) -> LlmResponse: ...
       @property
       def name(self) -> str: return "my-provider"
   ```
3. Add a test in `tests/ch01/test_llm.py` using `MockLlmProvider` to verify the interface.
4. Update `pyproject.toml` optional-dependencies to add the provider's SDK.

## Adding a New Session Backend

1. Open `storage/session.py`
2. Implement the `SessionService` Protocol (all 6 methods: create, get, update, delete, list, get_or_create)
3. Add the backend to the composition root switch in `examples/`
4. Add tests in `tests/ch05/`

## Adding a New Planner Strategy

1. Open `core/planner.py`
2. Subclass `BasePlanner`, implement `plan(goal, context) -> Plan`
3. Register in `create_planner()` factory
4. Add to `tests/ch07/test_planner.py`

---

## Known Stubs (Not Yet Implemented)

| File | Status | Priority |
|---|---|---|
| `tools/mcp_connector.py` | Stub — 3 lines | High |
| Streaming variant of `LlmProvider.complete()` | Not started | High |
| `knowledge/graph_rag.py` | Not started | Medium |
| `tests/evals/` | Not started | Medium |
