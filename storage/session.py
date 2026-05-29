# storage/session.py
"""
State and Session Management — storage/session.py
The Agentic Spine: Engineering a Provider-Agnostic AI Framework

Data model:    SessionState, ConversationTurn
Interface:     SessionService (abstract)
Backends:      InMemorySessionService, FileSessionService
Lifecycle:     SessionRunner (load → inject → run → persist)

Built in Chapter 5: State and Session Management
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    agent_name: str = ""
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp,
                "agent_name": self.agent_name, "turn_id": self.turn_id, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationTurn":
        return cls(role=d["role"], content=d["content"], timestamp=d.get("timestamp", time.time()),
                   agent_name=d.get("agent_name", ""), turn_id=d.get("turn_id", str(uuid.uuid4())[:8]),
                   metadata=d.get("metadata", {}))


@dataclass
class SessionState:
    """
    Complete persisted state for one conversation session.
    Four concerns: history (→LLM), state (→agents), metadata (→ops), events (→telemetry).
    """
    session_id: str
    user_id: str
    app_name: str = "default"
    history: list[ConversationTurn] = field(default_factory=list)
    max_history_turns: int = 50
    state: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turn_count: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def add_user_turn(self, content: str, metadata: dict | None = None) -> ConversationTurn:
        turn = ConversationTurn(role="user", content=content, metadata=metadata or {})
        self.history.append(turn)
        self.turn_count += 1
        self.updated_at = time.time()
        self._trim_history()
        return turn

    def add_assistant_turn(self, content: str, agent_name: str = "", metadata: dict | None = None) -> ConversationTurn:
        turn = ConversationTurn(role="assistant", content=content, agent_name=agent_name, metadata=metadata or {})
        self.history.append(turn)
        self.updated_at = time.time()
        self._trim_history()
        return turn

    def get_recent_history(self, n_turns: int | None = None) -> list[ConversationTurn]:
        if n_turns is None:
            return list(self.history)
        return self.history[-n_turns:]

    def to_messages_list(self, max_turns: int = 20) -> list[dict[str, str]]:
        recent = self.get_recent_history(max_turns * 2)
        return [{"role": t.role, "content": t.content} for t in recent]

    def _trim_history(self) -> None:
        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns:]

    def set(self, key: str, value: Any) -> None:
        self.state[key] = value
        self.updated_at = time.time()

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def update_state(self, delta: dict[str, Any]) -> None:
        self.state.update(delta)
        self.updated_at = time.time()

    def accumulate_cost(self, cost_usd: float, tokens: int = 0) -> None:
        self.total_cost_usd += cost_usd
        self.total_tokens += tokens
        self.updated_at = time.time()

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        self.events.append({"type": event_type, "timestamp": time.time(), **data})

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id, "user_id": self.user_id, "app_name": self.app_name,
            "history": [t.to_dict() for t in self.history], "max_history_turns": self.max_history_turns,
            "state": self.state, "created_at": self.created_at, "updated_at": self.updated_at,
            "turn_count": self.turn_count, "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens, "metadata": self.metadata, "events": self.events,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionState":
        session = cls(
            session_id=d["session_id"], user_id=d["user_id"], app_name=d.get("app_name", "default"),
            max_history_turns=d.get("max_history_turns", 50), state=d.get("state", {}),
            created_at=d.get("created_at", time.time()), updated_at=d.get("updated_at", time.time()),
            turn_count=d.get("turn_count", 0), total_cost_usd=d.get("total_cost_usd", 0.0),
            total_tokens=d.get("total_tokens", 0), metadata=d.get("metadata", {}),
            events=d.get("events", []),
        )
        session.history = [ConversationTurn.from_dict(t) for t in d.get("history", [])]
        return session


class SessionNotFoundError(Exception):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session '{session_id}' not found.")
        self.session_id = session_id


class SessionExistsError(Exception):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session '{session_id}' already exists.")
        self.session_id = session_id


class SessionService(ABC):
    """Backend-agnostic session persistence interface."""

    @abstractmethod
    async def create_session(self, user_id: str, app_name: str = "default",
                             session_id: str | None = None, initial_state: dict | None = None,
                             metadata: dict | None = None) -> SessionState: ...

    @abstractmethod
    async def get_session(self, session_id: str, user_id: str, app_name: str = "default") -> SessionState | None: ...

    @abstractmethod
    async def update_session(self, session: SessionState) -> SessionState: ...

    @abstractmethod
    async def delete_session(self, session_id: str, user_id: str, app_name: str = "default") -> None: ...

    @abstractmethod
    async def list_sessions(self, user_id: str, app_name: str = "default") -> list[SessionState]: ...

    async def get_or_create_session(self, user_id: str, session_id: str | None,
                                    app_name: str = "default", initial_state: dict | None = None) -> SessionState:
        if session_id:
            session = await self.get_session(session_id, user_id, app_name)
            if session is not None:
                return session
        return await self.create_session(user_id=user_id, app_name=app_name,
                                         session_id=session_id, initial_state=initial_state)


class InMemorySessionService(SessionService):
    """In-process dict-backed storage. Zero deps. Use for dev/tests."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, dict[str, SessionState]]] = {}

    def _get_store(self, app_name: str, user_id: str) -> dict[str, SessionState]:
        return self._store.setdefault(app_name, {}).setdefault(user_id, {})

    async def create_session(self, user_id, app_name="default", session_id=None,
                             initial_state=None, metadata=None) -> SessionState:
        sid = session_id or str(uuid.uuid4())
        store = self._get_store(app_name, user_id)
        if sid in store:
            raise SessionExistsError(sid)
        session = SessionState(session_id=sid, user_id=user_id, app_name=app_name,
                               state=dict(initial_state or {}), metadata=dict(metadata or {}))
        store[sid] = session
        return session

    async def get_session(self, session_id, user_id, app_name="default") -> SessionState | None:
        return self._get_store(app_name, user_id).get(session_id)

    async def update_session(self, session: SessionState) -> SessionState:
        store = self._get_store(session.app_name, session.user_id)
        if session.session_id not in store:
            raise SessionNotFoundError(session.session_id)
        session.updated_at = time.time()
        store[session.session_id] = session
        return session

    async def delete_session(self, session_id, user_id, app_name="default") -> None:
        self._get_store(app_name, user_id).pop(session_id, None)

    async def list_sessions(self, user_id, app_name="default") -> list[SessionState]:
        return list(self._get_store(app_name, user_id).values())

    def session_count(self) -> int:
        return sum(len(s) for u in self._store.values() for s in u.values())


class FileSessionService(SessionService):
    """JSON-file-backed storage. Single-server persistence. Layout: {base}/{app}/{user}/{id}.json"""

    def __init__(self, base_dir: str | Path = ".sessions") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, app: str, uid: str, sid: str) -> Path:
        return self.base_dir / app / uid / f"{sid}.json"

    def _dir(self, app: str, uid: str) -> Path:
        d = self.base_dir / app / uid
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def create_session(self, user_id, app_name="default", session_id=None,
                             initial_state=None, metadata=None) -> SessionState:
        sid = session_id or str(uuid.uuid4())
        path = self._path(app_name, user_id, sid)
        if path.exists():
            raise SessionExistsError(sid)
        session = SessionState(session_id=sid, user_id=user_id, app_name=app_name,
                               state=dict(initial_state or {}), metadata=dict(metadata or {}))
        self._dir(app_name, user_id)
        path.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")
        return session

    async def get_session(self, session_id, user_id, app_name="default") -> SessionState | None:
        path = self._path(app_name, user_id, session_id)
        if not path.exists():
            return None
        try:
            return SessionState.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError) as exc:
            log.error("FileSessionService: failed to read %s: %s", path, exc)
            return None

    async def update_session(self, session: SessionState) -> SessionState:
        path = self._path(session.app_name, session.user_id, session.session_id)
        if not path.exists():
            raise SessionNotFoundError(session.session_id)
        session.updated_at = time.time()
        self._dir(session.app_name, session.user_id)
        path.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")
        return session

    async def delete_session(self, session_id, user_id, app_name="default") -> None:
        path = self._path(app_name, user_id, session_id)
        if path.exists():
            path.unlink()

    async def list_sessions(self, user_id, app_name="default") -> list[SessionState]:
        d = self.base_dir / app_name / user_id
        if not d.exists():
            return []
        sessions = []
        for f in sorted(d.glob("*.json")):
            try:
                sessions.append(SessionState.from_dict(json.loads(f.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, KeyError) as exc:
                log.warning("FileSessionService: skipping %s: %s", f, exc)
        return sessions


# ── SessionRunner — complete lifecycle wrapper ──

from core.agent import AgentContext, AgentEvent, BaseAgent, EventType


class SessionRunner:
    """
    Manages the full session lifecycle for one agent interaction.
    load → inject history+state → run → write-back → append turn → persist
    """

    def __init__(self, service: SessionService, agent: BaseAgent,
                 app_name: str = "default", max_history_turns_in_context: int = 10) -> None:
        self._service = service
        self._agent = agent
        self._app_name = app_name
        self._max_history = max_history_turns_in_context

    async def run(self, user_id: str, session_id: str | None, user_message: str,
                  max_steps: int = 20, context_metadata: dict | None = None):
        """Process one user message. Yields AgentEvent objects."""
        session = await self._service.get_or_create_session(
            user_id=user_id, session_id=session_id, app_name=self._app_name)
        session.add_user_turn(user_message)

        context = AgentContext(
            user_id=user_id, session_id=session.session_id, app_name=self._app_name,
            user_message=user_message,
            session_state={"history": session.to_messages_list(self._max_history), **session.state},
            max_steps=max_steps, metadata=context_metadata or {},
        )

        final_content = final_agent = ""
        final_cost = final_tokens = 0

        async for event in self._agent.run_async(context):
            yield event
            if event.is_final():
                final_content = event.content
                final_agent = event.agent_name
                final_cost = event.data.get("cost_usd", 0.0)
                final_tokens = (event.data.get("input_tokens", 0) + event.data.get("output_tokens", 0))
                session.update_state({k: v for k, v in context.session_state.items() if k != "history"})

        if final_content:
            session.add_assistant_turn(final_content, agent_name=final_agent,
                                       metadata={"cost_usd": final_cost, "tokens": final_tokens})
        session.accumulate_cost(final_cost, final_tokens)
        await self._service.update_session(session)

    async def run_and_collect(self, user_id: str, session_id: str | None,
                              user_message: str, max_steps: int = 20) -> str:
        final = ""
        async for event in self.run(user_id, session_id, user_message, max_steps):
            if event.is_final():
                final = event.content
        return final

    async def get_session(self, user_id: str, session_id: str) -> SessionState | None:
        return await self._service.get_session(session_id, user_id, self._app_name)

    async def get_history(self, user_id: str, session_id: str,
                          n_turns: int | None = None) -> list[ConversationTurn]:
        session = await self.get_session(user_id, session_id)
        return session.get_recent_history(n_turns) if session else []
