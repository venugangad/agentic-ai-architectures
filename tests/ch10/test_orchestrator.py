# tests/ch10/test_orchestrator.py — Chapter 10: Multi-Agent Orchestration
from __future__ import annotations

from collections.abc import Callable

import pytest

from orchestration.orchestrator import (
    A2AResult,
    A2ATask,
    AgentCard,
    AgentRegistry,
    SupervisorAgent,
    TaskState,
    agent_worker,
    tool_worker,
)

# ── AgentCard ────────────────────────────────────────────────────────────────

class TestAgentCard:
    def test_matches_exact(self):
        card = AgentCard("a1", "Agent", "desc", capabilities=["flight_search"])
        assert card.matches("flight_search")

    def test_matches_substring(self):
        card = AgentCard("a1", "Agent", "desc", capabilities=["airline_flight_booking"])
        assert card.matches("flight")

    def test_matches_case_insensitive(self):
        card = AgentCard("a1", "Agent", "desc", capabilities=["FlightSearch"])
        assert card.matches("flightsearch")

    def test_no_match(self):
        card = AgentCard("a1", "Agent", "desc", capabilities=["hotel_search"])
        assert not card.matches("flight")

    def test_to_dict_has_required_fields(self):
        card = AgentCard("a1", "FlightAgent", "Books flights",
                         capabilities=["flight_search"])
        d = card.to_dict()
        assert d["agentId"] == "a1"
        assert d["name"] == "FlightAgent"
        assert "flight_search" in d["capabilities"]


# ── AgentRegistry ─────────────────────────────────────────────────────────────

class TestAgentRegistry:
    def _make_worker(self, response: str = "ok"):
        async def _fn(task: A2ATask) -> A2AResult:
            return A2AResult(task_id=task.task_id, agent_id="test",
                             state=TaskState.COMPLETED, output=response)
        return _fn

    def test_register_and_find(self):
        registry = AgentRegistry()
        card = AgentCard("f1", "FlightAgent", "Flights", capabilities=["flight_search"])
        registry.register(card, self._make_worker())
        results = registry.find("flight_search")
        assert len(results) == 1
        assert results[0].agent_id == "f1"

    def test_find_returns_empty_for_no_match(self):
        registry = AgentRegistry()
        assert registry.find("unknown_capability") == []

    def test_find_returns_multiple_matches(self):
        registry = AgentRegistry()
        registry.register(
            AgentCard("a1", "AgentA", "d", capabilities=["travel"]),
            self._make_worker()
        )
        registry.register(
            AgentCard("a2", "AgentB", "d", capabilities=["travel_booking"]),
            self._make_worker()
        )
        results = registry.find("travel")
        assert len(results) == 2

    def test_get_worker_returns_callable(self):
        registry = AgentRegistry()
        fn = self._make_worker()
        card = AgentCard("a1", "A", "d", capabilities=["task"])
        registry.register(card, fn)
        retrieved = registry.get_worker("a1")
        assert callable(retrieved)

    def test_get_worker_none_for_unknown(self):
        registry = AgentRegistry()
        assert registry.get_worker("nonexistent") is None

    def test_unregister_removes_agent(self):
        registry = AgentRegistry()
        card = AgentCard("a1", "A", "d", capabilities=["task"])
        registry.register(card, self._make_worker())
        registry.unregister("a1")
        assert registry.find("task") == []

    def test_list_cards(self):
        registry = AgentRegistry()
        for i in range(3):
            registry.register(
                AgentCard(f"a{i}", f"Agent{i}", "d", capabilities=[f"cap{i}"]),
                self._make_worker()
            )
        assert len(registry.list_cards()) == 3


# ── Worker factories ──────────────────────────────────────────────────────────

class TestWorkerFactories:
    async def test_agent_worker_completed_on_success(self):
        class FakeAgent:
            name = "fake"
            async def run(self, msg, session_id=None):
                return f"response to: {msg}"

        worker = agent_worker(FakeAgent())
        task = A2ATask("t1", "cap", "do something")
        result = await worker(task)
        assert result.state == TaskState.COMPLETED
        assert "response to" in result.output

    async def test_agent_worker_failed_on_exception(self):
        class FailingAgent:
            name = "failing"
            async def run(self, msg, session_id=None):
                raise RuntimeError("agent crashed")

        worker = agent_worker(FailingAgent())
        task = A2ATask("t1", "cap", "do something")
        result = await worker(task)
        assert result.state == TaskState.FAILED
        assert "RuntimeError" in result.error

    async def test_tool_worker_completed_on_success(self):
        async def my_tool(value: str) -> str:
            return f"processed: {value}"

        worker = tool_worker("my_tool", my_tool)
        task = A2ATask("t1", "cap", "instruction", context={"value": "hello"})
        result = await worker(task)
        assert result.state == TaskState.COMPLETED
        assert "processed: hello" in result.output

    async def test_tool_worker_failed_on_exception(self):
        async def broken_tool(**kwargs):
            raise ValueError("broken")

        worker = tool_worker("broken", broken_tool)
        task = A2ATask("t1", "cap", "instruction", context={})
        result = await worker(task)
        assert result.state == TaskState.FAILED


# ── SupervisorAgent ───────────────────────────────────────────────────────────

def _scripted_llm(plan_json: str, synthesis: str = "Final answer."):
    """Returns a mock LLM that returns plan_json on first call, synthesis on second."""
    from dataclasses import dataclass

    @dataclass
    class FakeResp:
        content: str
        input_tokens: int = 10
        output_tokens: int = 20

    class ScriptedLLM:
        def __init__(self):
            self._calls = 0
        async def complete(self, messages, **kw):
            self._calls += 1
            return FakeResp(content=plan_json if self._calls == 1 else synthesis)

    return ScriptedLLM()


def _instant_worker(output: str) -> Callable:
    async def _fn(task: A2ATask) -> A2AResult:
        return A2AResult(task_id=task.task_id, agent_id="w",
                         state=TaskState.COMPLETED, output=output)
    return _fn


class TestSupervisorAgent:
    async def test_run_single_task(self):
        plan = '[{"task_id":"t1","capability":"search","instruction":"Find X","depends_on":[]}]'
        llm = _scripted_llm(plan, synthesis="Here is what I found.")

        registry = AgentRegistry()
        registry.register(
            AgentCard("w1", "Worker", "d", capabilities=["search"]),
            _instant_worker("Search result: X found."),
        )

        supervisor = SupervisorAgent(llm=llm, registry=registry)
        result = await supervisor.run("Find X")
        assert "Here is what I found." in result

    async def test_parallel_tasks_both_executed(self):
        plan = (
            '[{"task_id":"t1","capability":"task_a","instruction":"Do A","depends_on":[]},'
            ' {"task_id":"t2","capability":"task_b","instruction":"Do B","depends_on":[]}]'
        )
        llm = _scripted_llm(plan, synthesis="Combined result.")

        executed = []

        async def worker_a(task):
            executed.append("a")
            return A2AResult(task_id=task.task_id, agent_id="a",
                             state=TaskState.COMPLETED, output="A done")

        async def worker_b(task):
            executed.append("b")
            return A2AResult(task_id=task.task_id, agent_id="b",
                             state=TaskState.COMPLETED, output="B done")

        registry = AgentRegistry()
        registry.register(AgentCard("a", "A", "d", capabilities=["task_a"]), worker_a)
        registry.register(AgentCard("b", "B", "d", capabilities=["task_b"]), worker_b)

        supervisor = SupervisorAgent(llm=llm, registry=registry)
        await supervisor.run("Do A and B")
        assert set(executed) == {"a", "b"}

    async def test_depends_on_sequenced(self):
        plan = (
            '[{"task_id":"t1","capability":"step1","instruction":"Step 1","depends_on":[]},'
            ' {"task_id":"t2","capability":"step2","instruction":"Step 2","depends_on":["t1"]}]'
        )
        llm = _scripted_llm(plan, synthesis="Steps done.")

        order = []

        async def w1(task):
            order.append(1)
            return A2AResult(task_id=task.task_id, agent_id="w1",
                             state=TaskState.COMPLETED, output="step1 result")

        async def w2(task):
            order.append(2)
            # Should have context from t1
            assert "result_t1" in task.context
            return A2AResult(task_id=task.task_id, agent_id="w2",
                             state=TaskState.COMPLETED, output="step2 result")

        registry = AgentRegistry()
        registry.register(AgentCard("w1", "W1", "d", capabilities=["step1"]), w1)
        registry.register(AgentCard("w2", "W2", "d", capabilities=["step2"]), w2)

        supervisor = SupervisorAgent(llm=llm, registry=registry)
        await supervisor.run("Do step 1 then step 2")
        assert order == [1, 2]

    async def test_missing_worker_produces_failed_result(self):
        plan = '[{"task_id":"t1","capability":"unknown_cap","instruction":"X","depends_on":[]}]'
        results_seen = {}

        class CapturingSynthesisLlm:
            calls = 0
            async def complete(self, messages, **kw):
                from dataclasses import dataclass
                @dataclass
                class R:
                    content: str
                    input_tokens: int = 10
                    output_tokens: int = 10
                self.calls += 1
                if self.calls == 1:
                    return R(plan)
                # Capture what was passed to synthesis
                results_seen["synthesis_input"] = messages[-1].content
                return R("Handled gracefully.")

        registry = AgentRegistry()  # no workers registered
        supervisor = SupervisorAgent(llm=CapturingSynthesisLlm(), registry=registry)
        await supervisor.run("Do something unknown")
        # Synthesis should have received the failure
        assert "✗" in results_seen.get("synthesis_input", "")

    async def test_circular_dependency_raises(self):
        plan = (
            '[{"task_id":"t1","capability":"cap","instruction":"A","depends_on":["t2"]},'
            ' {"task_id":"t2","capability":"cap","instruction":"B","depends_on":["t1"]}]'
        )
        llm = _scripted_llm(plan)

        async def worker(task):
            return A2AResult(task_id=task.task_id, agent_id="w",
                             state=TaskState.COMPLETED, output="ok")

        registry = AgentRegistry()
        registry.register(AgentCard("w", "W", "d", capabilities=["cap"]), worker)

        supervisor = SupervisorAgent(llm=llm, registry=registry)
        with pytest.raises(RuntimeError, match="[Cc]ircular"):
            await supervisor.run("Circular task")

    async def test_parse_plan_handles_invalid_json(self):
        result = SupervisorAgent._parse_plan("not json at all {{{{")
        assert result == []

    async def test_parse_plan_strips_markdown_fences(self):
        json_with_fences = '```json\n[{"task_id":"t1","capability":"c","instruction":"x","depends_on":[]}]\n```'
        result = SupervisorAgent._parse_plan(json_with_fences)
        assert len(result) == 1
        assert result[0]["task_id"] == "t1"
