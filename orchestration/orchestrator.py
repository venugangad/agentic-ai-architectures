# orchestration/orchestrator.py — Chapter 10: Multi-Agent Orchestration & Shipping
# Part of: The Agentic Spine companion repository
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


# ── A2A data model ────────────────────────────────────────────────────────────

class TaskState(str, Enum):
    SUBMITTED  = "submitted"
    WORKING    = "working"
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"


@dataclass
class AgentCard:
    """
    A2A-compatible description of an agent's capabilities.
    Published at /.well-known/agent.json in HTTP deployments.
    """
    agent_id:     str
    name:         str
    description:  str
    version:      str = "1.0.0"
    capabilities: list[str] = field(default_factory=list)
    input_modes:  list[str] = field(default_factory=lambda: ["text"])
    output_modes: list[str] = field(default_factory=lambda: ["text"])
    endpoint:     str | None = None    # None = in-process call
    metadata:     dict[str, Any] = field(default_factory=dict)

    def matches(self, capability: str) -> bool:
        """Case-insensitive substring capability check."""
        cap_lower = capability.lower()
        return any(cap_lower in c.lower() for c in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agentId":      self.agent_id,
            "name":         self.name,
            "description":  self.description,
            "version":      self.version,
            "capabilities": self.capabilities,
            "inputModes":   self.input_modes,
            "outputModes":  self.output_modes,
            "endpoint":     self.endpoint,
        }


@dataclass
class A2ATask:
    """Task submitted by supervisor to a worker agent."""
    task_id:     str
    capability:  str
    instruction: str
    context:     dict[str, Any] = field(default_factory=dict)
    session_id:  str | None = None
    created_at:  float = field(default_factory=time.time)
    state:       TaskState = TaskState.SUBMITTED


@dataclass
class A2AResult:
    """Result returned by a worker agent to the supervisor."""
    task_id:    str
    agent_id:   str
    state:      TaskState
    output:     str | None = None
    error:      str | None = None
    metadata:   dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


# ── Worker function type alias ────────────────────────────────────────────────

# The only contract between supervisor and workers.
# Can wrap an Agent, a plain async function, or an HTTP proxy.
WorkerFn = Callable[[A2ATask], Awaitable[A2AResult]]


# ── AgentRegistry ────────────────────────────────────────────────────────────

@dataclass
class _WorkerEntry:
    card:    AgentCard
    execute: WorkerFn


class AgentRegistry:
    """
    In-process directory of worker agents.

    In production: back this with Redis or a service-mesh registry.
    Workers register on startup; supervisor queries by capability.

    Usage:
        registry = AgentRegistry()
        registry.register(
            AgentCard("flight-001", "FlightAgent", "Book flights",
                      capabilities=["flight_search", "book_flight"]),
            agent_worker(flight_agent),
        )
        cards = registry.find("flight_search")
    """

    def __init__(self) -> None:
        self._workers: dict[str, _WorkerEntry] = {}

    def register(self, card: AgentCard, execute: WorkerFn) -> None:
        self._workers[card.agent_id] = _WorkerEntry(card=card, execute=execute)

    def find(self, capability: str) -> list[AgentCard]:
        """Return all cards whose capabilities match the query (substring, case-insensitive)."""
        return [
            entry.card
            for entry in self._workers.values()
            if entry.card.matches(capability)
        ]

    def get_worker(self, agent_id: str) -> WorkerFn | None:
        entry = self._workers.get(agent_id)
        return entry.execute if entry else None

    def list_cards(self) -> list[AgentCard]:
        return [e.card for e in self._workers.values()]

    def unregister(self, agent_id: str) -> None:
        self._workers.pop(agent_id, None)


# ── Supervisor system prompt ──────────────────────────────────────────────────

_SUPERVISOR_SYSTEM = """\
You are a supervisor coordinating a fleet of specialist agents.
Given a goal and a list of available workers (their names and capabilities),
produce a JSON plan: a list of sub-tasks, each with:
  { "task_id": "<string>", "capability": "<string>", "instruction": "<string>", "depends_on": ["<task_id>", ...] }

Rules:
- Use ONLY capabilities that appear in the worker list.
- Set depends_on to [] for tasks that can run in parallel immediately.
- Set depends_on to the task_ids of tasks that must complete first.
- Keep instructions specific and self-contained.
- task_id values must be unique strings (e.g., "t1", "t2", ...).
- Return ONLY valid JSON — no prose, no markdown fences.
"""


# ── SupervisorAgent ───────────────────────────────────────────────────────────

class SupervisorAgent:
    """
    Coordinates a fleet of worker agents via the A2A protocol.

    Workflow:
      1. Plan  — ask the LLM to decompose the goal into a JSON sub-task list
      2. Schedule — topological ordering respecting depends_on
      3. Dispatch — route each task to the matching worker; run ready tasks in parallel
      4. Synthesise — ask the LLM to compose a final answer from all sub-task results

    Usage:
        supervisor = SupervisorAgent(llm=gateway, registry=registry, max_workers=5)
        answer = await supervisor.run(goal, session_id="sess-001")
    """

    def __init__(
        self,
        llm:        Any,                 # LlmProvider (core/llm.py)
        registry:   AgentRegistry,
        name:       str = "supervisor",
        max_workers: int = 5,
    ) -> None:
        self._llm      = llm
        self._registry = registry
        self._name     = name
        self._sem      = asyncio.Semaphore(max_workers)

    async def run(
        self,
        goal:       str,
        session_id: str | None = None,
        context:    dict[str, Any] | None = None,
    ) -> str:
        ctx  = context or {}
        plan = await self._plan(goal)
        results: dict[str, A2AResult] = {}
        await self._execute_plan(plan, results, session_id, ctx)
        return await self._synthesise(goal, results)

    # ── planning ──────────────────────────────────────────────────────────────

    async def _plan(self, goal: str) -> list[dict[str, Any]]:
        from core.llm import Message  # type: ignore[import]
        workers_desc = "\n".join(
            f"- {c.name}: {', '.join(c.capabilities)}"
            for c in self._registry.list_cards()
        )
        messages = [
            Message(role="system", content=_SUPERVISOR_SYSTEM),
            Message(role="user", content=(
                f"Goal: {goal}\n\n"
                f"Available workers:\n{workers_desc}\n\n"
                f"Produce the JSON plan."
            )),
        ]
        response = await self._llm.complete(messages)
        return self._parse_plan(response.content)

    @staticmethod
    def _parse_plan(text: str) -> list[dict[str, Any]]:
        clean = re.sub(r"```(?:json)?|```", "", text).strip()
        try:
            data = json.loads(clean)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "tasks" in data:
                return data["tasks"]
        except json.JSONDecodeError:
            pass
        return []

    # ── parallel execution with topological dependency resolution ─────────────

    async def _execute_plan(
        self,
        plan:       list[dict[str, Any]],
        results:    dict[str, A2AResult],
        session_id: str | None,
        context:    dict[str, Any],
    ) -> None:
        # Normalise task_ids
        tasks_by_id: dict[str, dict[str, Any]] = {}
        for i, step in enumerate(plan):
            tid = step.get("task_id") or f"task-{i}"
            step["task_id"] = tid
            tasks_by_id[tid] = step

        completed: set[str] = set()
        pending = list(tasks_by_id.keys())

        while pending:
            # Frontier: tasks whose deps are all satisfied
            ready = [
                tid for tid in pending
                if all(dep in completed for dep in tasks_by_id[tid].get("depends_on", []))
            ]
            if not ready:
                raise RuntimeError(
                    f"Circular dependency or unsatisfiable depends_on. "
                    f"Pending={pending}, Completed={completed}"
                )
            await asyncio.gather(*(
                self._dispatch_one(tasks_by_id[tid], results, session_id, context)
                for tid in ready
            ))
            completed.update(ready)
            for tid in ready:
                pending.remove(tid)

    async def _dispatch_one(
        self,
        step:       dict[str, Any],
        results:    dict[str, A2AResult],
        session_id: str | None,
        context:    dict[str, Any],
    ) -> None:
        capability  = step.get("capability", "")
        instruction = step.get("instruction", "")
        task_id     = step["task_id"]

        cards = self._registry.find(capability)
        if not cards:
            results[task_id] = A2AResult(
                task_id=task_id, agent_id=self._name,
                state=TaskState.FAILED,
                error=f"No worker registered for capability: {capability!r}",
            )
            return

        worker_fn = self._registry.get_worker(cards[0].agent_id)
        if not worker_fn:
            results[task_id] = A2AResult(
                task_id=task_id, agent_id=cards[0].agent_id,
                state=TaskState.FAILED,
                error="Worker card found but executor missing.",
            )
            return

        # Enrich context: inject prior results for depends_on tasks
        enriched = dict(context)
        for dep_id in step.get("depends_on", []):
            if dep_id in results and results[dep_id].output:
                enriched[f"result_{dep_id}"] = results[dep_id].output

        task = A2ATask(
            task_id=task_id,
            capability=capability,
            instruction=instruction,
            context=enriched,
            session_id=session_id,
            state=TaskState.SUBMITTED,
        )

        t0 = time.time()
        async with self._sem:
            try:
                result = await worker_fn(task)
            except Exception as exc:
                result = A2AResult(
                    task_id=task_id,
                    agent_id=cards[0].agent_id,
                    state=TaskState.FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                )
        result.latency_ms = (time.time() - t0) * 1_000
        results[task_id]  = result

    # ── synthesis ─────────────────────────────────────────────────────────────

    async def _synthesise(
        self,
        goal:    str,
        results: dict[str, A2AResult],
    ) -> str:
        from core.llm import Message  # type: ignore[import]
        results_text = "\n\n".join(
            f"[{tid}] {'✓' if r.state == TaskState.COMPLETED else '✗'}: "
            f"{r.output or r.error}"
            for tid, r in results.items()
        )
        messages = [
            Message(role="system", content=(
                "You are a supervisor synthesising results from specialist agents. "
                "Compose a clear, concise final answer from the sub-task results below. "
                "If any sub-task failed, acknowledge the gap and work around it."
            )),
            Message(role="user", content=(
                f"Original goal: {goal}\n\n"
                f"Sub-task results:\n{results_text}\n\n"
                f"Compose the final answer."
            )),
        ]
        response = await self._llm.complete(messages)
        return response.content


# ── Worker factories ──────────────────────────────────────────────────────────

def agent_worker(agent: Any) -> WorkerFn:
    """
    Wrap a core.agent.Agent as a WorkerFn.

    Usage:
        registry.register(card, agent_worker(my_agent))
    """
    async def _execute(task: A2ATask) -> A2AResult:
        try:
            output = await agent.run(task.instruction, session_id=task.session_id)
            return A2AResult(
                task_id=task.task_id,
                agent_id=agent.name,
                state=TaskState.COMPLETED,
                output=output,
            )
        except Exception as exc:
            return A2AResult(
                task_id=task.task_id,
                agent_id=getattr(agent, "name", "unknown"),
                state=TaskState.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )
    return _execute


def tool_worker(name: str, fn: Callable[..., Awaitable[Any]]) -> WorkerFn:
    """
    Wrap a single async function as a lightweight WorkerFn.
    Useful for simple capability tasks that don't need a full agent.

    The task.context dict is unpacked as keyword arguments to fn.

    Usage:
        registry.register(card, tool_worker("price_lookup", price_lookup_fn))
    """
    async def _execute(task: A2ATask) -> A2AResult:
        try:
            result = await fn(**task.context)
            return A2AResult(
                task_id=task.task_id,
                agent_id=name,
                state=TaskState.COMPLETED,
                output=str(result),
            )
        except Exception as exc:
            return A2AResult(
                task_id=task.task_id,
                agent_id=name,
                state=TaskState.FAILED,
                error=str(exc),
            )
    return _execute
