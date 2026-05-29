# orchestration/router.py
"""
Intelligence through Routing — orchestration/router.py
The Agentic Spine: Engineering a Provider-Agnostic AI Framework

Router types:
  AgentRouter      — keyword/regex rules, priority-ordered dispatch
  SequentialRouter — fixed pipeline, output chaining
  LlmRouter        — LLM intent classification, confidence threshold
  HybridRouter     — keyword-first + LLM fallback (production default)

Built in Chapter 4: Intelligence through Routing
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.agent import AgentContext, AgentEvent, BaseAgent, EventType
from core.llm import LlmConfig, LlmMessage, MessageRole

log = logging.getLogger(__name__)


class MatchStrategy(str, Enum):
    KEYWORD = "keyword"
    REGEX   = "regex"
    ALWAYS  = "always"
    NEVER   = "never"


@dataclass
class RouteCondition:
    strategy: MatchStrategy = MatchStrategy.KEYWORD
    keywords: list[str] = field(default_factory=list)
    pattern: str = ""
    case_sensitive: bool = False

    def matches(self, text: str) -> bool:
        if self.strategy == MatchStrategy.ALWAYS:
            return True
        if self.strategy == MatchStrategy.NEVER:
            return False
        check = text if self.case_sensitive else text.lower()
        if self.strategy == MatchStrategy.KEYWORD:
            kws = self.keywords if self.case_sensitive else [k.lower() for k in self.keywords]
            return any(kw in check for kw in kws)
        if self.strategy == MatchStrategy.REGEX:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            return bool(re.search(self.pattern, text, flags))
        return False


@dataclass
class RouteRule:
    name: str
    condition: RouteCondition
    target_agent: str
    priority: int = 0
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches(self, text: str) -> bool:
        return self.condition.matches(text)


@dataclass
class RouteMatch:
    rule_name: str
    target_agent: str
    strategy: str
    confidence: float = 1.0
    reasoning: str = ""

    def as_event_data(self) -> dict:
        return {
            "rule": self.rule_name,
            "target": self.target_agent,
            "strategy": self.strategy,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


class AgentRouter(BaseAgent):
    """Routes via keyword/regex rules. Priority-ordered; first match wins."""

    def __init__(self, name, description, rules, agents, fallback_agent=None):
        super().__init__(name=name, description=description, sub_agents=list(agents.values()))
        self._rules = sorted(rules, key=lambda r: -r.priority)
        self._agents = agents
        self._fallback_agent = fallback_agent

    def add_rule(self, rule: RouteRule) -> AgentRouter:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: -r.priority)
        return self

    def register_agent(self, name: str, agent: BaseAgent) -> AgentRouter:
        self._agents[name] = agent
        self.sub_agents.append(agent)
        agent.parent_agent = self
        return self

    def _classify(self, text: str) -> RouteMatch | None:
        for rule in self._rules:
            if rule.matches(text):
                return RouteMatch(rule_name=rule.name, target_agent=rule.target_agent,
                                  strategy=rule.condition.strategy.value, confidence=1.0)
        return None

    async def _run_async_impl(self, context: AgentContext) -> AsyncGenerator[AgentEvent, None]:
        match = self._classify(context.user_message)
        if match is None:
            if self._fallback_agent:
                match = RouteMatch(rule_name="fallback", target_agent=self._fallback_agent, strategy="fallback")
            else:
                yield AgentEvent(type=EventType.ERROR,
                                 content=f"No routing rule matched. Request: '{context.user_message[:80]}'",
                                 data={"rules_evaluated": len(self._rules)})
                return

        target = self._agents.get(match.target_agent)
        if target is None:
            yield AgentEvent(type=EventType.ERROR,
                             content=f"Target agent '{match.target_agent}' not registered.",
                             data=match.as_event_data())
            return

        yield AgentEvent(type=EventType.THOUGHT,
                         content=(f"Routing to '{match.target_agent}' "
                                  f"[rule: {match.rule_name}, strategy: {match.strategy}, "
                                  f"confidence: {match.confidence:.2f}]"),
                         data=match.as_event_data())

        async for event in target.run_async(context):
            yield event


class SequentialRouter(BaseAgent):
    """Runs a fixed pipeline; each agent's output feeds the next."""

    def __init__(self, name, description, pipeline, stop_on_error=True):
        super().__init__(name=name, description=description, sub_agents=list(pipeline))
        self._pipeline = pipeline
        self._stop_on_error = stop_on_error

    async def _run_async_impl(self, context: AgentContext) -> AsyncGenerator[AgentEvent, None]:
        current_message = context.user_message
        context.session_state["original_message"] = current_message

        for idx, agent in enumerate(self._pipeline):
            is_last = (idx == len(self._pipeline) - 1)
            yield AgentEvent(type=EventType.THOUGHT,
                             content=f"Pipeline stage {idx+1}/{len(self._pipeline)}: '{agent.name}'",
                             data={"stage": idx+1, "total_stages": len(self._pipeline), "agent": agent.name})

            stage_ctx = AgentContext(
                user_id=context.user_id, session_id=context.session_id,
                app_name=context.app_name, user_message=current_message,
                session_state=context.session_state, max_steps=context.max_steps,
                max_tokens=context.max_tokens, run_id=context.run_id, metadata=context.metadata,
            )
            stage_final = ""
            had_error = False
            async for event in agent.run_async(stage_ctx):
                if not event.is_final():
                    yield event
                else:
                    stage_final = event.content
                    if event.data.get("error"):
                        had_error = True
                if is_last and event.is_final():
                    yield event

            if not is_last:
                if had_error and self._stop_on_error:
                    yield AgentEvent(type=EventType.FINAL, content=stage_final,
                                     data={"pipeline_aborted": True, "aborted_at_stage": idx+1})
                    return
                current_message = stage_final or current_message


_CLASSIFY_SYSTEM = """\
You are a routing classifier. Respond with ONLY a JSON object:
{{"agent": "<agent_name>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}

Available agents:
{agent_descriptions}

Choose the single best agent. If unclear, choose the fallback agent.
"""


class LlmRouter(BaseAgent):
    """Routes via LLM intent classification. Uses fast/cheap model at temp=0."""

    def __init__(self, name, description, classification_llm, agents,
                 fallback_agent=None, confidence_threshold=0.5, classification_config=None):
        super().__init__(name=name, description=description, sub_agents=[a for a, _ in agents.values()])
        self._classification_llm = classification_llm
        self._agents = {k: v[0] for k, v in agents.items()}
        self._agent_descriptions = {k: v[1] for k, v in agents.items()}
        self._fallback_agent = fallback_agent
        self._confidence_threshold = confidence_threshold
        self._classification_config = classification_config or LlmConfig(temperature=0.0, max_tokens=128)

    def _build_classification_prompt(self) -> str:
        lines = []
        for name, desc in self._agent_descriptions.items():
            tag = " [FALLBACK]" if name == self._fallback_agent else ""
            lines.append(f"- {name}{tag}: {desc}")
        return "\n".join(lines)

    async def _classify(self, user_message: str) -> RouteMatch:
        system_prompt = _CLASSIFY_SYSTEM.format(
            agent_descriptions=self._build_classification_prompt()
        )
        messages = [
            LlmMessage(role=MessageRole.SYSTEM, content=system_prompt),
            LlmMessage(role=MessageRole.USER, content=user_message),
        ]
        try:
            response = await self._classification_llm.generate(messages, self._classification_config)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\n?", "", raw).rstrip("`").strip()
            parsed = json.loads(raw)
            agent_name = parsed.get("agent", self._fallback_agent)
            confidence = float(parsed.get("confidence", 0.0))
            reasoning = parsed.get("reasoning", "")
            if agent_name not in self._agents:
                agent_name = self._fallback_agent
                confidence = 0.0
            if confidence < self._confidence_threshold and self._fallback_agent:
                agent_name = self._fallback_agent
            return RouteMatch(rule_name="llm_classification", target_agent=agent_name,
                              strategy="llm", confidence=confidence, reasoning=reasoning)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            fallback = self._fallback_agent or next(iter(self._agents))
            return RouteMatch(rule_name="parse_error_fallback", target_agent=fallback,
                              strategy="fallback", confidence=0.0, reasoning=f"Parse error: {exc}")

    async def _run_async_impl(self, context: AgentContext) -> AsyncGenerator[AgentEvent, None]:
        match = await self._classify(context.user_message)
        yield AgentEvent(type=EventType.THOUGHT,
                         content=(f"LLM routing → '{match.target_agent}' "
                                  f"(confidence: {match.confidence:.2f}) | {match.reasoning}"),
                         data=match.as_event_data())
        target = self._agents.get(match.target_agent)
        if target is None:
            yield AgentEvent(type=EventType.ERROR,
                             content=f"LlmRouter: target '{match.target_agent}' not found.",
                             data=match.as_event_data())
            return
        async for event in target.run_async(context):
            yield event


class HybridRouter(BaseAgent):
    """Keyword rules first; LLM classification fallback. Production default."""

    def __init__(self, name, description, rule_router: AgentRouter, llm_router: LlmRouter):
        super().__init__(name=name, description=description, sub_agents=[rule_router, llm_router])
        self._rule_router = rule_router
        self._llm_router = llm_router

    async def _run_async_impl(self, context: AgentContext) -> AsyncGenerator[AgentEvent, None]:
        match = self._rule_router._classify(context.user_message)
        if match is not None:
            async for event in self._rule_router.run_async(context):
                yield event
        else:
            yield AgentEvent(type=EventType.THOUGHT,
                             content="No keyword rule matched — escalating to LLM classification.",
                             data={"stage": "hybrid_escalation"})
            async for event in self._llm_router.run_async(context):
                yield event
