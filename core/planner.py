# core/planner.py
"""
Strategic Planning — core/planner.py
The Agentic Spine: Engineering a Provider-Agnostic AI Framework

Data model:    PlanStep, Plan
Interface:     BasePlanner (abstract)
Strategies:    LinearPlanner · ReActPlanner · TreeOfThoughtPlanner · SagaPlanner
Integration:   PlanningAgent (BaseAgent subclass emitting AgentEvent stream)
Factory:       create_planner(strategy, ...)

Built in Chapter 7: Strategic Planning
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.agent import AgentContext, AgentEvent

log = logging.getLogger(__name__)


# ── Data Model ───────────────────────────────────────────────────────────────


@dataclass
class PlanStep:
    """
    One atomic step in a plan.

    tool_name=None means a reasoning/thought step with no tool call.
    depends_on: step_ids that must complete before this step can run.
    compensate_with: tool to call if this step must be rolled back (Saga).
    """
    step_id: str
    description: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    compensate_with: str | None = None
    compensate_args: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"          # pending | running | done | failed | compensated
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_ready(self, completed_ids: set[str]) -> bool:
        """True when all dependency steps are done."""
        return all(dep in completed_ids for dep in self.depends_on)


@dataclass
class Plan:
    """An ordered (possibly parallel) collection of PlanSteps aimed at a goal."""
    plan_id: str
    goal: str
    steps: list[PlanStep]
    strategy: str                    # "linear" | "react" | "tot" | "saga"
    created_at: float = field(default_factory=time.time)
    status: str = "pending"         # pending | running | done | failed | compensating
    metadata: dict[str, Any] = field(default_factory=dict)

    def pending_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == "pending"]

    def completed_step_ids(self) -> set[str]:
        return {s.step_id for s in self.steps if s.status == "done"}

    def failed_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == "failed"]

    def ready_steps(self) -> list[PlanStep]:
        """Steps pending and unblocked by dependencies."""
        done = self.completed_step_ids()
        return [s for s in self.pending_steps() if s.is_ready(done)]

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "strategy": self.strategy,
            "status": self.status,
            "steps": [
                {
                    "step_id": s.step_id,
                    "description": s.description,
                    "tool_name": s.tool_name,
                    "status": s.status,
                    "depends_on": s.depends_on,
                    "error": s.error,
                }
                for s in self.steps
            ],
        }


# ── BasePlanner ───────────────────────────────────────────────────────────────


class _NoLlm:
    """Placeholder used by planners that need no LLM (Linear, Saga)."""
    async def complete(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("This planner does not use an LLM.")


class BasePlanner(ABC):
    """
    Abstract base for all planning strategies.

    Subclasses must implement plan().
    execute() provides a default sequential executor with compensation.
    """

    def __init__(self, llm_provider: Any, registry: Any | None = None) -> None:
        self._llm = llm_provider
        self._registry = registry

    @abstractmethod
    async def plan(self, goal: str, context: dict[str, Any]) -> Plan:
        """Produce a Plan for the given goal."""
        ...

    async def execute(self, plan: Plan) -> Plan:
        """Execute sequentially; compensate on failure."""
        plan.status = "running"
        log.info("Executing plan %s (%s, %d steps)", plan.plan_id, plan.strategy, len(plan.steps))
        for step in plan.steps:
            if plan.status == "failed":
                break
            await self._execute_step(step, plan)
        plan.status = "done" if not plan.failed_steps() else "failed"
        if plan.status == "failed":
            await self._compensate(plan)
        return plan

    async def _execute_step(self, step: PlanStep, plan: Plan) -> None:
        step.status = "running"
        if step.tool_name and self._registry:
            result = await self._registry.dispatch(step.tool_name, step.tool_args)
            if result.get("status") == "error":
                step.status = "failed"
                step.error = result.get("message", "unknown error")
                plan.status = "failed"
                log.error("Step %s failed: %s", step.step_id, step.error)
            else:
                step.status = "done"
                step.result = result.get("result")
        else:
            step.status = "done"   # reasoning step

    async def _compensate(self, plan: Plan) -> None:
        plan.status = "compensating"
        completed = [s for s in reversed(plan.steps)
                     if s.status == "done" and s.compensate_with]
        for step in completed:
            log.info("Compensating %s via %s", step.step_id, step.compensate_with)
            try:
                if self._registry:
                    await self._registry.dispatch(step.compensate_with, step.compensate_args)
                step.status = "compensated"
            except Exception as exc:
                log.error("Compensation for %s failed: %s", step.step_id, exc)
        plan.status = "failed"


# ── LinearPlanner ─────────────────────────────────────────────────────────────


class LinearPlanner(BasePlanner):
    """
    Builds a Plan from a caller-defined list of tool calls.
    No LLM calls for planning. Deterministic and auditable.

    Use for: known workflows, ETL pipelines, fixed multi-step forms.
    """

    def __init__(
        self,
        steps: list[dict[str, Any]],
        llm_provider: Any | None = None,
        registry: Any | None = None,
    ) -> None:
        super().__init__(llm_provider or _NoLlm(), registry)
        self._step_specs = steps

    async def plan(self, goal: str, context: dict[str, Any]) -> Plan:
        steps = []
        for i, spec in enumerate(self._step_specs):
            steps.append(PlanStep(
                step_id=f"step_{i+1}",
                description=spec.get("description", spec.get("tool_name", f"step {i+1}")),
                tool_name=spec.get("tool_name"),
                tool_args=spec.get("args", {}),
                depends_on=spec.get("depends_on", [f"step_{i}"] if i > 0 else []),
                compensate_with=spec.get("compensate_with"),
                compensate_args=spec.get("compensate_args", {}),
            ))
        return Plan(
            plan_id=str(uuid.uuid4())[:8],
            goal=goal, steps=steps, strategy="linear",
        )


# ── ReActPlanner ──────────────────────────────────────────────────────────────


_REACT_SYSTEM = """\
You are a planning agent that solves goals step by step.

At each step respond with valid JSON matching this schema:
{{
  "thought":   "your reasoning about the goal and what to do next",
  "action":    "tool_name OR 'DONE' if goal achieved OR 'STUCK' if no progress possible",
  "args":      {{"key": "value"}},
  "reasoning": "why this action over alternatives"
}}

Rules:
- DONE: include "summary" key in args with what was accomplished.
- STUCK: include "reason" key in args explaining why you cannot proceed.
- Only use tools from the available list.
- Do not repeat a failed action with identical arguments.
- Maximum {max_steps} steps.

Available tools:
{tools}
"""


class ReActPlanner(BasePlanner):
    """
    Interleaved plan-and-execute ReAct loop.
    The LLM sees each tool result before deciding the next step.

    Use for: open-ended tasks, multi-hop tool reasoning, adaptive workflows.
    """

    def __init__(
        self,
        llm_provider: Any,
        registry: Any,
        max_steps: int = 20,
        temperature: float = 0.2,
    ) -> None:
        super().__init__(llm_provider, registry)
        self._max_steps = max_steps
        self._temperature = temperature

    async def plan(self, goal: str, context: dict[str, Any]) -> Plan:
        tools_desc = (
            self._registry.to_description_block()
            if self._registry else "No tools available."
        )
        system = _REACT_SYSTEM.format(tools=tools_desc, max_steps=self._max_steps)
        messages: list[dict] = [{"role": "user", "content": f"Goal: {goal}"}]

        plan = Plan(
            plan_id=str(uuid.uuid4())[:8],
            goal=goal, steps=[], strategy="react", status="running",
        )

        for iteration in range(self._max_steps):
            step_id = f"react_{iteration + 1}"
            resp = await self._llm.complete(
                [{"role": "system", "content": system}] + messages,
                temperature=self._temperature,
                max_tokens=512,
            )
            raw = resp.content.strip()
            parsed = self._parse_json(raw)

            if parsed is None:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user",
                                 "content": "Your response was not valid JSON. Respond with JSON only."})
                continue

            thought = parsed.get("thought", "")
            action = parsed.get("action", "STUCK")
            args = parsed.get("args", {})

            step = PlanStep(
                step_id=step_id,
                description=thought[:200],
                tool_name=action if action not in ("DONE", "STUCK") else None,
                tool_args=args,
                depends_on=[f"react_{iteration}"] if iteration > 0 else [],
                metadata={"reasoning": parsed.get("reasoning", "")},
            )
            plan.steps.append(step)

            if action == "DONE":
                step.status = "done"
                step.result = args.get("summary", "Goal achieved.")
                plan.status = "done"
                log.info("ReActPlanner: DONE after %d steps", iteration + 1)
                break

            if action == "STUCK":
                step.status = "failed"
                step.error = args.get("reason", "No available action can make progress.")
                plan.status = "failed"
                log.warning("ReActPlanner: STUCK: %s", step.error)
                break

            if self._registry:
                result = await self._registry.dispatch(action, args)
                if result.get("status") == "error":
                    step.status = "failed"
                    step.error = result.get("message", "tool error")
                    observation = f"ERROR: {step.error}"
                else:
                    step.status = "done"
                    step.result = result.get("result")
                    observation = f"Result: {step.result}"
            else:
                step.status = "done"
                observation = "No registry — step marked done."

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            plan.status = "failed"
            log.warning("ReActPlanner: max_steps (%d) reached", self._max_steps)

        return plan

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        import json
        import re
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        return None


# ── TreeOfThoughtPlanner ──────────────────────────────────────────────────────


@dataclass
class ThoughtNode:
    node_id: str
    thought: str
    score: float = 0.0
    parent_id: str | None = None
    children: list[ThoughtNode] = field(default_factory=list)
    is_terminal: bool = False
    action: str | None = None
    action_args: dict[str, Any] = field(default_factory=dict)


_TOT_GENERATE_SYSTEM = """\
You are solving a problem using tree-of-thought reasoning.

Problem: {goal}
Current reasoning path:
{path}

Generate {n_branches} DIFFERENT candidate next thoughts (different approaches/angles).
Respond with a JSON array:
[
  {{"thought": "...", "action": "tool_name or DONE or STUCK", "args": {{}}, "reasoning": "..."}},
  ...
]
"""

_TOT_SCORE_SYSTEM = """\
Rate each candidate thought for solving this problem:
{goal}

Candidates:
{candidates}

Score each 0-10: 10=solves it, 7-9=strong, 4-6=plausible, 1-3=weak, 0=wrong.
Respond with a JSON array of scores in order: [score1, score2, ...]
"""


class TreeOfThoughtPlanner(BasePlanner):
    """
    Beam search over a tree of LLM-generated thoughts.
    At each depth: generate → score → prune → expand best.

    Use for: complex reasoning, multiple viable approaches, accuracy > speed.
    """

    def __init__(
        self,
        llm_provider: Any,
        registry: Any | None = None,
        n_branches: int = 3,
        beam_width: int = 2,
        max_depth: int = 5,
        score_threshold: float = 5.0,
        temperature: float = 0.7,
    ) -> None:
        super().__init__(llm_provider, registry)
        self._n_branches = n_branches
        self._beam_width = beam_width
        self._max_depth = max_depth
        self._score_threshold = score_threshold
        self._temperature = temperature

    async def plan(self, goal: str, context: dict[str, Any]) -> Plan:
        root = ThoughtNode(node_id="root", thought=f"Goal: {goal}", score=10.0)
        beam: list[tuple[ThoughtNode, str]] = [(root, f"Goal: {goal}")]
        best_terminal: ThoughtNode | None = None

        for depth in range(self._max_depth):
            next_beam: list[tuple[ThoughtNode, str]] = []
            for node, path in beam:
                candidates = await self._generate_thoughts(goal, path)
                scores = await self._score_thoughts(goal, [c["thought"] for c in candidates])
                for cand, score in zip(candidates, scores):
                    child = ThoughtNode(
                        node_id=f"d{depth}_{len(next_beam)}",
                        thought=cand["thought"],
                        score=score,
                        parent_id=node.node_id,
                        action=cand.get("action"),
                        action_args=cand.get("args", {}),
                        is_terminal=cand.get("action") == "DONE",
                    )
                    node.children.append(child)
                    if child.is_terminal and score >= self._score_threshold:
                        if best_terminal is None or score > best_terminal.score:
                            best_terminal = child
                    if score >= self._score_threshold:
                        child_path = path + f"\n→ [{score:.1f}] {cand['thought']}"
                        next_beam.append((child, child_path))

            if best_terminal:
                break
            if not next_beam:
                log.warning("ToT: all branches below threshold at depth %d", depth)
                break
            next_beam.sort(key=lambda x: x[0].score, reverse=True)
            beam = next_beam[:self._beam_width]

        return self._tree_to_plan(goal, root, best_terminal)

    async def _generate_thoughts(self, goal: str, path: str) -> list[dict]:
        import json
        prompt = _TOT_GENERATE_SYSTEM.format(
            goal=goal, path=path, n_branches=self._n_branches
        )
        resp = await self._llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=self._temperature, max_tokens=800,
        )
        try:
            return json.loads(resp.content)
        except Exception:
            import re
            m = re.search(r'\[.*\]', resp.content, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
        return [{"thought": f"Direct attempt: {goal}", "action": "DONE",
                 "args": {"summary": "direct approach"}}]

    async def _score_thoughts(self, goal: str, thoughts: list[str]) -> list[float]:
        import json
        if not thoughts:
            return []
        candidates_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(thoughts))
        prompt = _TOT_SCORE_SYSTEM.format(goal=goal, candidates=candidates_text)
        resp = await self._llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=64,
        )
        try:
            scores = json.loads(resp.content)
            return [float(s) for s in scores[:len(thoughts)]]
        except Exception:
            import re
            nums = re.findall(r'\d+(?:\.\d+)?', resp.content)
            return [float(n) for n in nums[:len(thoughts)]] or [5.0] * len(thoughts)

    def _tree_to_plan(self, goal: str, root: ThoughtNode, terminal: ThoughtNode | None) -> Plan:
        steps: list[PlanStep] = []
        if terminal:
            path_nodes: list[ThoughtNode] = []
            node = terminal
            while node.parent_id:
                path_nodes.insert(0, node)
                node = self._find_node(root, node.parent_id) or root
            for i, n in enumerate(path_nodes):
                steps.append(PlanStep(
                    step_id=f"tot_{i+1}",
                    description=n.thought[:200],
                    tool_name=n.action if n.action not in (None, "DONE", "STUCK") else None,
                    tool_args=n.action_args,
                    depends_on=[f"tot_{i}"] if i > 0 else [],
                    metadata={"score": n.score, "node_id": n.node_id},
                ))
        return Plan(
            plan_id=str(uuid.uuid4())[:8],
            goal=goal, steps=steps, strategy="tot",
            status="pending" if steps else "failed",
            metadata={"terminal_score": terminal.score if terminal else 0},
        )

    def _find_node(self, root: ThoughtNode, node_id: str) -> ThoughtNode | None:
        if root.node_id == node_id:
            return root
        for child in root.children:
            found = self._find_node(child, node_id)
            if found:
                return found
        return None


# ── SagaPlanner ───────────────────────────────────────────────────────────────


@dataclass
class SagaStep:
    """Forward action + compensation action for Saga pattern."""
    name: str
    tool_name: str
    tool_args: dict[str, Any]
    compensate_tool: str | None = None
    compensate_args: dict[str, Any] = field(default_factory=dict)
    compensate_args_from_result: dict[str, str] = field(default_factory=dict)
    # Maps compensation arg key → JSONPath in forward result (e.g. "$.booking_ref")


class SagaPlanner(BasePlanner):
    """
    Linear steps with automatic LIFO compensation on failure.
    compensate_args_from_result extracts server-generated IDs at runtime.

    Use for: payments, reservations, any workflow with real-world side effects.
    """

    def __init__(
        self,
        saga_steps: list[SagaStep],
        llm_provider: Any | None = None,
        registry: Any | None = None,
    ) -> None:
        super().__init__(llm_provider or _NoLlm(), registry)
        self._saga_steps = saga_steps

    async def plan(self, goal: str, context: dict[str, Any]) -> Plan:
        steps = []
        for i, ss in enumerate(self._saga_steps):
            steps.append(PlanStep(
                step_id=f"saga_{i+1}",
                description=ss.name,
                tool_name=ss.tool_name,
                tool_args=ss.tool_args,
                depends_on=[f"saga_{i}"] if i > 0 else [],
                compensate_with=ss.compensate_tool,
                compensate_args=ss.compensate_args.copy(),
                metadata={"compensate_from_result": ss.compensate_args_from_result},
            ))
        return Plan(
            plan_id=str(uuid.uuid4())[:8],
            goal=goal, steps=steps, strategy="saga",
        )

    async def execute(self, plan: Plan) -> Plan:
        """Sequential execution with LIFO compensation stack."""
        plan.status = "running"
        completed: list[PlanStep] = []

        for step in plan.steps:
            step.status = "running"
            log.info("Saga: executing %s (%s)", step.step_id, step.description)
            if self._registry and step.tool_name:
                try:
                    result = await self._registry.dispatch(step.tool_name, step.tool_args)
                    if result.get("status") == "error":
                        step.status = "failed"
                        step.error = result.get("message", "tool error")
                    else:
                        step.status = "done"
                        step.result = result.get("result")
                        self._enrich_compensate_args(step)
                        completed.append(step)
                except Exception as exc:
                    step.status = "failed"
                    step.error = str(exc)
            else:
                step.status = "done"
                completed.append(step)

            if step.status == "failed":
                log.error("Saga: step %s failed — %s", step.step_id, step.error)
                plan.status = "failed"
                await self._saga_compensate(completed)
                return plan

        plan.status = "done"
        log.info("Saga: all %d steps completed", len(plan.steps))
        return plan

    async def _saga_compensate(self, completed: list[PlanStep]) -> None:
        log.info("Saga: compensating %d completed steps", len(completed))
        for step in reversed(completed):
            if not step.compensate_with:
                continue
            log.info("Saga: compensating %s via %s", step.step_id, step.compensate_with)
            try:
                if self._registry:
                    await self._registry.dispatch(step.compensate_with, step.compensate_args)
                step.status = "compensated"
            except Exception as exc:
                log.error("Saga: compensation for %s failed: %s", step.step_id, exc)

    @staticmethod
    def _enrich_compensate_args(step: PlanStep) -> None:
        from_result = step.metadata.get("compensate_from_result", {})
        if not from_result or step.result is None:
            return
        result = step.result if isinstance(step.result, dict) else {}
        for comp_key, result_path in from_result.items():
            keys = result_path.lstrip("$.").split(".")
            value: Any = result
            for k in keys:
                value = value.get(k) if isinstance(value, dict) else None
            if value is not None:
                step.compensate_args[comp_key] = value


# ── PlanningAgent ─────────────────────────────────────────────────────────────


class PlanningAgent:
    """
    Wraps any BasePlanner into the AgentEvent stream.
    Emits THOUGHT (plan summary) → TOOL_CALL/TOOL_RESULT per step → FINAL.
    """

    def __init__(
        self,
        planner: BasePlanner,
        name: str = "planning_agent",
    ) -> None:
        self._planner = planner
        self.name = name

    async def run_async(self, context: AgentContext) -> AsyncGenerator[AgentEvent, None]:
        return self._run(context)

    async def _run(self, context: AgentContext) -> AsyncGenerator[AgentEvent, None]:
        from core.agent import AgentEvent, EventType
        goal = context.user_message

        yield AgentEvent(
            event_type=EventType.THOUGHT,
            content=f"Planning for goal: {goal}",
            agent_name=self.name,
            data={"phase": "planning"},
        )

        plan = await self._planner.plan(goal, context.session_state)

        yield AgentEvent(
            event_type=EventType.THOUGHT,
            content=f"Plan ready: {len(plan.steps)} steps, strategy={plan.strategy}",
            agent_name=self.name,
            data={"plan": plan.to_dict()},
        )

        for step in plan.steps:
            if step.tool_name:
                yield AgentEvent(
                    event_type=EventType.TOOL_CALL,
                    content=f"Executing: {step.description}",
                    agent_name=self.name,
                    data={"tool": step.tool_name, "args": step.tool_args,
                          "step_id": step.step_id},
                )

            await self._planner._execute_step(step, plan)

            if step.tool_name:
                yield AgentEvent(
                    event_type=EventType.TOOL_RESULT,
                    content=str(step.result)[:200] if step.result else (step.error or ""),
                    agent_name=self.name,
                    data={"step_id": step.step_id, "status": step.status,
                          "result": step.result, "error": step.error},
                )

        if plan.failed_steps():
            failed = plan.failed_steps()[0]
            summary = f"Plan failed at '{failed.description}': {failed.error}"
            if any(s.status == "compensated" for s in plan.steps):
                summary += " (compensation applied)"
        else:
            results = [str(s.result) for s in plan.steps if s.result]
            summary = f"Plan completed. {'; '.join(results[:3])}"

        yield AgentEvent(
            event_type=EventType.FINAL,
            content=summary,
            agent_name=self.name,
            data={"plan_status": plan.status, "plan": plan.to_dict()},
        )


# ── Factory ───────────────────────────────────────────────────────────────────


def create_planner(
    strategy: str,
    llm_provider: Any,
    registry: Any | None = None,
    **kwargs: Any,
) -> BasePlanner:
    """
    Instantiate the right planner.

    strategy: "linear" | "react" | "tot" | "saga"
    kwargs are forwarded to the planner constructor.

    Selection guide:
        "linear" — fixed steps, zero LLM planning overhead, fastest
        "react"  — open-ended, steps emerge from tool results, medium cost
        "tot"    — complex reasoning, multiple hypotheses, highest quality
        "saga"   — linear with side-effect compensation, any failure rolls back
    """
    strategies: dict[str, type] = {
        "linear": LinearPlanner,
        "react":  ReActPlanner,
        "tot":    TreeOfThoughtPlanner,
        "saga":   SagaPlanner,
    }
    cls = strategies.get(strategy)
    if cls is None:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Choose from: {', '.join(strategies)}"
        )
    return cls(llm_provider=llm_provider, registry=registry, **kwargs)
