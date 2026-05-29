# core/agent.py
"""
The Core Agent Backbone — core/agent.py
The Agent Circuit: Engineering a Provider-Agnostic AI Framework

This module defines the universal agent contract and the LlmAgent implementation.
All orchestration, routing, and planning code in this framework calls BaseAgent —
never a specific agent implementation directly.

Built in Chapter 2: The Core Agent Backbone
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ──────────────────────────────────────────────
# Event stream — what agents produce
# ──────────────────────────────────────────────

class EventType(str, Enum):
    """Every emission from an agent run is typed."""
    TEXT_CHUNK    = "text_chunk"    # streaming partial content
    TEXT_FINAL    = "text_final"    # complete response text
    TOOL_CALL     = "tool_call"     # agent is requesting a tool execution
    TOOL_RESULT   = "tool_result"   # tool execution result returned
    THOUGHT       = "thought"       # agent's internal reasoning trace (ReAct)
    ERROR         = "error"         # recoverable error with message
    FINAL         = "final"         # run is complete; this is the last event


@dataclass
class AgentEvent:
    """
    A single emission from an agent's run_async generator.
    Callers iterate over these events — they never get a single blocking return value.
    This enables streaming, observability hooks, and mid-run intervention.
    """
    type: EventType
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    agent_name: str = ""
    timestamp: float = field(default_factory=time.time)
    run_id: str = ""

    def is_final(self) -> bool:
        return self.type == EventType.FINAL

    def is_tool_call(self) -> bool:
        return self.type == EventType.TOOL_CALL

    def is_text(self) -> bool:
        return self.type in (EventType.TEXT_CHUNK, EventType.TEXT_FINAL)


# ──────────────────────────────────────────────
# Scratchpad — per-run working memory
# ──────────────────────────────────────────────

@dataclass
class AgentScratchpad:
    """
    Ephemeral working memory for one agent run.
    Created fresh at the start of run_async, discarded when it returns.
    Holds intermediate tool results and step count for budget enforcement.
    """
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    thoughts: list[str] = field(default_factory=list)
    steps: int = 0
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def add_tool_result(self, tool_name: str, result: Any) -> None:
        self.tool_results.append({
            "tool": tool_name,
            "result": result,
            "step": self.steps,
        })

    def add_thought(self, thought: str) -> None:
        self.thoughts.append(thought)
        self.steps += 1

    def to_context_string(self) -> str:
        """Render the scratchpad as a string for injection into context."""
        if not self.tool_results and not self.thoughts:
            return ""
        parts = []
        for item in self.tool_results:
            parts.append(
                f"[Tool: {item['tool']}] → {json.dumps(item['result'], default=str)}"
            )
        return "\n".join(parts)


# ──────────────────────────────────────────────
# AgentContext — the execution environment
# ──────────────────────────────────────────────

@dataclass
class AgentContext:
    """
    The complete execution environment passed into every agent run.
    Contains the user's input, session identity, and runtime configuration.
    Agents read from context; they do not own it.
    """
    # Identity
    user_id: str
    session_id: str
    app_name: str = "default"

    # The current user message
    user_message: str = ""

    # Session state (persisted across runs — managed by SessionService in Ch5)
    session_state: dict[str, Any] = field(default_factory=dict)

    # Runtime limits
    max_steps: int = 20          # max tool-call iterations per run
    max_tokens: int = 4096       # max tokens for the final response

    # Metadata
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────
# Callback types for lifecycle hooks
# ──────────────────────────────────────────────

BeforeAgentCallback = Callable[[AgentContext], Awaitable[AgentContext | None]]
AfterAgentCallback  = Callable[[AgentContext, AgentEvent], Awaitable[None]]
BeforeToolCallback  = Callable[[str, dict], Awaitable[dict | None]]
AfterToolCallback   = Callable[[str, dict, Any], Awaitable[Any]]


# ──────────────────────────────────────────────
# BaseAgent — the universal contract
# ──────────────────────────────────────────────

class BaseAgent(ABC):
    """
    The universal contract every agent in this framework must satisfy.

    Design principles:
    - All execution is async and streaming (AsyncGenerator[AgentEvent])
    - Agents are composable: sub_agents allows hierarchical assembly
    - Lifecycle hooks attach observability and security without coupling
    - Agents do not own their LLM instance — they receive it via context or registry
    """

    def __init__(
        self,
        name: str,
        description: str,
        sub_agents: list[BaseAgent] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.sub_agents: list[BaseAgent] = sub_agents or []
        self.parent_agent: BaseAgent | None = None

        # Lifecycle hooks — attached after construction
        self._before_agent: list[BeforeAgentCallback] = []
        self._after_agent:  list[AfterAgentCallback]  = []
        self._before_tool:  list[BeforeToolCallback]  = []
        self._after_tool:   list[AfterToolCallback]   = []

        # Set parent reference on sub-agents
        for sub in self.sub_agents:
            sub.parent_agent = self

    # ── Hook registration ──

    def add_before_agent(self, fn: BeforeAgentCallback) -> BaseAgent:
        """Register a pre-run hook. Returns self for chaining."""
        self._before_agent.append(fn)
        return self

    def add_after_agent(self, fn: AfterAgentCallback) -> BaseAgent:
        self._after_agent.append(fn)
        return self

    def add_before_tool(self, fn: BeforeToolCallback) -> BaseAgent:
        self._before_tool.append(fn)
        return self

    def add_after_tool(self, fn: AfterToolCallback) -> BaseAgent:
        self._after_tool.append(fn)
        return self

    # ── Agent hierarchy ──

    def find_agent(self, name: str) -> BaseAgent | None:
        """Depth-first search for a named agent in this agent's hierarchy."""
        if self.name == name:
            return self
        for sub in self.sub_agents:
            found = sub.find_agent(name)
            if found:
                return found
        return None

    def clone(self) -> BaseAgent:
        """
        Create a fresh instance with the same configuration.
        Used by the ParallelAgent (Chapter 7) to spawn independent workers.
        Subclasses must override if they hold non-copyable state.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement clone() "
            "to be used in parallel execution."
        )

    # ── The public execution interface ──

    async def run_async(
        self,
        context: AgentContext,
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        The public entry point for running this agent.
        Applies all before/after hooks and delegates to _run_async_impl.
        Callers iterate over AgentEvent objects — never block for a final answer.
        """
        # Run before-agent hooks; any hook can return a modified context
        # or return None to abort (treated as a guardrail block)
        for hook in self._before_agent:
            result = await hook(context)
            if result is None:
                yield AgentEvent(
                    type=EventType.ERROR,
                    content="Run blocked by before_agent hook.",
                    agent_name=self.name,
                    run_id=context.run_id,
                )
                return
            context = result

        # Delegate to the implementation
        final_event: AgentEvent | None = None
        async for event in self._run_async_impl(context):
            event.agent_name = self.name
            event.run_id = context.run_id
            yield event
            if event.is_final():
                final_event = event

        # Run after-agent hooks with the final event
        if final_event:
            for hook in self._after_agent:
                await hook(context, final_event)

    @abstractmethod
    async def _run_async_impl(
        self,
        context: AgentContext,
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        The agent's unique logic. Subclasses implement this.
        Must yield AgentEvent objects, ending with EventType.FINAL.
        """


# ──────────────────────────────────────────────
# Reusable lifecycle hooks
# ──────────────────────────────────────────────

class HookExamples:
    """
    Reusable lifecycle hooks for common concerns.
    Attach these to any agent without modifying the agent's code.
    """

    @staticmethod
    def make_logging_hook(logger=None) -> BeforeAgentCallback:
        """
        Logs every agent invocation with user ID, session, and message.
        Attach to any agent to get structured execution audit logs.
        """
        import logging
        _log = logger or logging.getLogger("agent.audit")

        async def hook(context: AgentContext) -> AgentContext:
            _log.info(
                "agent_run_start",
                extra={
                    "user_id":    context.user_id,
                    "session_id": context.session_id,
                    "run_id":     context.run_id,
                    "app":        context.app_name,
                    "message_len": len(context.user_message),
                }
            )
            return context

        return hook

    @staticmethod
    def make_max_length_guardrail(max_chars: int = 4000) -> BeforeAgentCallback:
        """
        Blocks requests that exceed a maximum input length.
        Returns None (blocking) if the input is too long.
        Attach to public-facing agents to prevent prompt-stuffing attacks.
        """
        async def hook(context: AgentContext) -> AgentContext | None:
            if len(context.user_message) > max_chars:
                return None   # abort the run
            return context

        return hook

    @staticmethod
    def make_tool_param_validator(
        session_user_key: str = "user_id"
    ) -> BeforeToolCallback:
        """
        Enforces least-privilege on tool calls: the user_id in tool arguments
        must match the session's user_id. Prevents one user calling a tool
        that modifies another user's data.
        """
        async def hook(tool_name: str, arguments: dict) -> dict | None:
            # Return None to block the tool call; return (possibly modified)
            # arguments dict to allow it
            if "user_id" in arguments:
                # In practice, the session user_id would be threaded through context
                # Chapter 5 wires this properly
                pass
            return arguments

        return hook

    @staticmethod
    def make_cost_tracker() -> AfterAgentCallback:
        """
        Accumulates cost from every LlmUsage event during a run.
        In Chapter 8, this feeds the LlmGateway's per-tenant cost dashboard.
        """
        total_cost: list[float] = [0.0]   # mutable closure

        async def hook(context: AgentContext, event: AgentEvent) -> None:
            cost = event.data.get("cost_usd", 0.0)
            total_cost[0] += cost
            context.metadata["total_cost_usd"] = total_cost[0]

        return hook


# ──────────────────────────────────────────────
# AgentPersona — the agent constitution
# ──────────────────────────────────────────────

@dataclass
class AgentPersona:
    """
    The agent's constitution — assembled into a system prompt at run time.
    Separating the persona components makes them independently testable and auditable.
    """
    role: str
    goal: str
    constraints: list[str] = field(default_factory=list)
    reasoning_style: str = "step_by_step"   # "step_by_step" | "direct" | "react"
    output_format: str = ""
    custom_instructions: str = ""

    def to_system_prompt(self, tool_descriptions: str = "") -> str:
        """Render the persona as a system prompt string."""
        parts = [
            f"## Role\n{self.role}",
            f"## Goal\n{self.goal}",
        ]
        if self.constraints:
            constraint_list = "\n".join(f"- {c}" for c in self.constraints)
            parts.append(f"## Constraints\n{constraint_list}")
        if tool_descriptions:
            parts.append(f"## Available Tools\n{tool_descriptions}")
        if self.reasoning_style == "react":
            parts.append(
                "## Reasoning Instructions\n"
                "For each step, first output your Thought (what you are "
                "going to do and why), then take an Action (a tool call or "
                "a direct response). After each tool result, output a new "
                "Thought before your next Action."
            )
        elif self.reasoning_style == "step_by_step":
            parts.append(
                "## Reasoning Instructions\n"
                "Think step by step before responding. "
                "Break complex problems into clear sub-steps."
            )
        if self.output_format:
            parts.append(f"## Output Format\n{self.output_format}")
        if self.custom_instructions:
            parts.append(f"## Additional Instructions\n{self.custom_instructions}")

        return "\n\n".join(parts)


# ──────────────────────────────────────────────
# LlmAgent — the concrete ReAct implementation
# ──────────────────────────────────────────────

from core.llm import BaseLlm, LlmConfig, LlmMessage, MessageRole


class LlmAgent(BaseAgent):
    """
    A concrete BaseAgent implementation that uses BaseLlm for reasoning.
    Implements the ReAct (Reason + Act) pattern: the agent alternates between
    generating Thoughts (reasoning) and taking Actions (tool calls or responses).

    This is the primary agent type in the framework. Orchestrators (Chapter 10),
    planners (Chapter 7), and tools (Chapter 3) are all composed around LlmAgent.
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm: BaseLlm,
        persona: AgentPersona | None = None,
        tools: list[Any] | None = None,     # list[BaseTool] — typed fully in Ch3
        sub_agents: list[BaseAgent] | None = None,
        config: LlmConfig | None = None,
    ) -> None:
        super().__init__(name=name, description=description, sub_agents=sub_agents)
        self._llm = llm
        self._persona = persona or AgentPersona(
            role=f"You are {name}, a capable AI agent.",
            goal="Complete the user's request accurately and concisely.",
        )
        self._tools: list[Any] = tools or []
        self._config = config or LlmConfig(temperature=0.3, max_tokens=2048)

    def clone(self) -> LlmAgent:
        return LlmAgent(
            name=self.name,
            description=self.description,
            llm=self._llm,
            persona=self._persona,
            tools=list(self._tools),
            config=self._config,
        )

    def _build_tool_descriptions(self) -> str:
        """Render all tools as a string for the system prompt."""
        if not self._tools:
            return ""
        lines = []
        for tool in self._tools:
            lines.append(f"- **{tool.name}**: {tool.description}")
            if hasattr(tool, "parameters_schema"):
                lines.append(f"  Parameters: {json.dumps(tool.parameters_schema)}")
        return "\n".join(lines)

    def _build_messages(
        self,
        context: AgentContext,
        scratchpad: AgentScratchpad,
    ) -> list[LlmMessage]:
        """
        Assemble the full message list for one LLM invocation.
        Draws from: persona (system prompt), session history, scratchpad, user message.
        """
        messages: list[LlmMessage] = []

        # System prompt from persona
        system_text = self._persona.to_system_prompt(
            tool_descriptions=self._build_tool_descriptions()
        )
        messages.append(LlmMessage(role=MessageRole.SYSTEM, content=system_text))

        # Session history (condensed — full implementation in Chapter 5)
        history = context.session_state.get("history", [])
        for turn in history[-6:]:   # last 3 turns (6 messages) to bound context
            messages.append(LlmMessage(
                role=MessageRole.USER if turn["role"] == "user" else MessageRole.ASSISTANT,
                content=turn["content"],
            ))

        # Scratchpad — tool results from this run's previous steps
        scratch_text = scratchpad.to_context_string()
        if scratch_text:
            messages.append(LlmMessage(
                role=MessageRole.ASSISTANT,
                content=f"[Previous steps in this run]\n{scratch_text}",
            ))

        # Current user message
        messages.append(LlmMessage(
            role=MessageRole.USER,
            content=context.user_message,
        ))

        return messages

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict,
        scratchpad: AgentScratchpad,
    ) -> tuple[Any, AgentEvent]:
        """
        Run before_tool hooks, execute a tool, run after_tool hooks, record result.
        Returns the result and an AgentEvent for the caller to yield.
        """
        # Before-tool hooks (validation, permission checks)
        current_args = arguments
        for hook in self._before_tool:
            result = await hook(tool_name, current_args)
            if result is None:
                error_msg = f"Tool call '{tool_name}' blocked by before_tool hook."
                return None, AgentEvent(
                    type=EventType.ERROR,
                    content=error_msg,
                    data={"tool": tool_name, "blocked": True},
                )
            current_args = result

        # Find and execute the tool
        tool = next((t for t in self._tools if t.name == tool_name), None)
        if tool is None:
            error_msg = f"Unknown tool: '{tool_name}'. Available: {[t.name for t in self._tools]}"
            return None, AgentEvent(type=EventType.ERROR, content=error_msg)

        try:
            tool_result = await tool.execute(current_args)
        except Exception as exc:
            tool_result = {"error": str(exc), "tool": tool_name}

        # After-tool hooks (sanitisation, logging)
        final_result = tool_result
        for hook in self._after_tool:
            final_result = await hook(tool_name, current_args, final_result)

        # Record in scratchpad
        scratchpad.add_tool_result(tool_name, final_result)

        event = AgentEvent(
            type=EventType.TOOL_RESULT,
            content=json.dumps(final_result, default=str),
            data={"tool": tool_name, "arguments": current_args, "result": final_result},
        )
        return final_result, event

    async def _run_async_impl(
        self,
        context: AgentContext,
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        The ReAct execution loop.

        Each iteration:
          1. Build messages from context + scratchpad
          2. Call LLM — get response (text or tool call)
          3. If text → yield FINAL and stop
          4. If tool call → execute tool, yield TOOL_RESULT, loop
          5. If step limit reached → yield FINAL with partial result
        """
        scratchpad = AgentScratchpad(run_id=context.run_id)
        start_time = time.monotonic()

        for step in range(context.max_steps):
            messages = self._build_messages(context, scratchpad)

            # Check context fits in the model's window
            if not self._llm.fits_in_context(messages):
                yield AgentEvent(
                    type=EventType.ERROR,
                    content=(
                        f"Context exceeds {self._llm.model_id} window "
                        f"({self._llm.context_window:,} tokens). "
                        "Consider reducing session history or using a larger-context model."
                    ),
                )
                return

            # Invoke the LLM
            llm_config = LlmConfig(
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
            response = await self._llm.generate(messages, llm_config)
            scratchpad.add_thought(response.content[:200])   # record reasoning

            if response.tool_calls:
                # Model wants to call a tool
                for tool_call in response.tool_calls:
                    t_name = tool_call.get("function", {}).get("name", "")
                    t_args_raw = tool_call.get("function", {}).get("arguments", "{}")
                    try:
                        t_args = json.loads(t_args_raw) if isinstance(t_args_raw, str) else t_args_raw
                    except json.JSONDecodeError:
                        t_args = {}

                    # Yield the tool call event for observability
                    yield AgentEvent(
                        type=EventType.TOOL_CALL,
                        content=f"Calling {t_name}({t_args})",
                        data={"tool": t_name, "arguments": t_args, "step": step},
                    )

                    # Execute the tool
                    _, result_event = await self._execute_tool(t_name, t_args, scratchpad)
                    yield result_event

            else:
                # Model produced a direct response — the run is complete
                elapsed_ms = (time.monotonic() - start_time) * 1000

                # Emit thought if we have one (ReAct trace)
                if scratchpad.thoughts and len(scratchpad.thoughts) > 1:
                    yield AgentEvent(
                        type=EventType.THOUGHT,
                        content=scratchpad.thoughts[-1],
                        data={"steps_taken": scratchpad.steps},
                    )

                yield AgentEvent(
                    type=EventType.FINAL,
                    content=response.content,
                    data={
                        "steps_taken": scratchpad.steps,
                        "elapsed_ms": elapsed_ms,
                        "cost_usd": response.usage.cost_usd,
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "model_id": response.usage.model_id,
                        "provider": response.usage.provider,
                    },
                )
                return

        # Step limit reached without a final response
        yield AgentEvent(
            type=EventType.FINAL,
            content=(
                f"Step limit ({context.max_steps}) reached. "
                "Partial results are in the scratchpad. "
                "Consider increasing max_steps or decomposing the task."
            ),
            data={
                "steps_taken": scratchpad.steps,
                "partial_results": scratchpad.tool_results,
                "limit_reached": True,
            },
        )


# ──────────────────────────────────────────────
# Convenience runner
# ──────────────────────────────────────────────

async def run_agent(
    agent: BaseAgent,
    user_message: str,
    user_id: str = "anon",
    session_id: str | None = None,
    app_name: str = "default",
    session_state: dict[str, Any] | None = None,
    max_steps: int = 20,
    verbose: bool = False,
) -> str:
    """
    Convenience function: run an agent and return its final text response.
    Prints intermediate events if verbose=True.
    Use this for single-shot calls in scripts and examples.
    For production streaming use, iterate over agent.run_async() directly.
    """
    context = AgentContext(
        user_id=user_id,
        session_id=session_id or str(uuid.uuid4()),
        app_name=app_name,
        user_message=user_message,
        session_state=session_state or {},
        max_steps=max_steps,
    )

    final_content = ""
    async for event in agent.run_async(context):
        if verbose:
            if event.type == EventType.THOUGHT:
                print(f"  💭 Thought: {event.content[:120]}...")
            elif event.type == EventType.TOOL_CALL:
                print(f"  🔧 Tool call: {event.content}")
            elif event.type == EventType.TOOL_RESULT:
                print(f"  ✅ Tool result: {event.content[:80]}...")
            elif event.type == EventType.ERROR:
                print(f"  ❌ Error: {event.content}")
        if event.is_final():
            final_content = event.content

    return final_content
