# tools/registry.py
"""
The Action Surface — tools/registry.py
The Agent Circuit: Engineering a Provider-Agnostic AI Framework

This module defines the universal tool contract, the FunctionTool wrapper,
the central ToolRegistry, and the McpConnector for external tool ecosystems.
All agent tool use in this framework goes through ToolRegistry —
never through direct function calls embedded in agent code.

Built in Chapter 3: Building the Action Surface
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
import types
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# BaseTool — the universal tool contract
# ──────────────────────────────────────────────

class BaseTool(ABC):
    """
    The contract every tool in the framework must satisfy.

    The three-part contract:
      1. schema   — tell the LLM what you can do and what you need
      2. validate — refuse to run with bad arguments
      3. execute  — run deterministically, return a JSON-serialisable result
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name. Used as the dispatch key."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-paragraph description for the LLM."""

    @property
    @abstractmethod
    def parameters_schema(self) -> dict:
        """JSON Schema describing the tool's arguments."""

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> Any:
        """
        Run the tool with the given (already-validated) arguments.
        Must return a JSON-serialisable result.
        Must never raise — catch internal errors and return structured objects.
        """

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Validate arguments against this tool's schema.
        Returns the (possibly coerced) arguments if valid.
        Raises ToolValidationError if invalid.
        """
        schema = self.parameters_schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        missing = [r for r in required if r not in arguments]
        if missing:
            raise ToolValidationError(
                tool=self.name,
                message=f"Missing required arguments: {missing}",
                received=arguments,
            )

        unknown = [k for k in arguments if k not in properties]
        if unknown:
            log.warning("tool %s received unknown arguments: %s (ignored)", self.name, unknown)
            arguments = {k: v for k, v in arguments.items() if k in properties}

        return arguments

    def to_llm_schema(self) -> dict:
        """Return this tool in OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    async def safe_execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Validate then execute, wrapping any exception in a structured error dict.
        This is the method ToolRegistry calls — never execute() directly.
        """
        start = time.monotonic()
        try:
            validated = self.validate(arguments)
            result = await self.execute(validated)
            elapsed_ms = (time.monotonic() - start) * 1000
            return {
                "status": "ok",
                "result": result,
                "tool": self.name,
                "elapsed_ms": round(elapsed_ms, 1),
            }
        except ToolValidationError as exc:
            return {
                "status": "error",
                "error_type": "validation",
                "message": exc.message,
                "tool": self.name,
            }
        except Exception as exc:
            log.exception("tool %s raised an unhandled error", self.name)
            return {
                "status": "error",
                "error_type": "execution",
                "message": str(exc),
                "tool": self.name,
            }


class ToolValidationError(Exception):
    def __init__(self, tool: str, message: str, received: dict) -> None:
        super().__init__(message)
        self.tool = tool
        self.message = message
        self.received = received


# ──────────────────────────────────────────────
# Type → JSON Schema mapping
# ──────────────────────────────────────────────

_PYTHON_TYPE_TO_JSON: dict[Any, str] = {
    int:   "integer",
    float: "number",
    str:   "string",
    bool:  "boolean",
    list:  "array",
    dict:  "object",
}


def _type_to_json_schema(annotation: Any) -> dict:
    """Convert a Python type annotation to a JSON Schema fragment."""
    origin = getattr(annotation, "__origin__", None)

    if origin is list:
        args = getattr(annotation, "__args__", (str,))
        return {"type": "array", "items": _type_to_json_schema(args[0])}

    if origin is dict:
        return {"type": "object"}

    if origin is types.UnionType or str(origin) in ("<class 'typing.Union'>", "typing.Union"):
        args = [a for a in getattr(annotation, "__args__", ()) if a is not type(None)]
        if args:
            return _type_to_json_schema(args[0])

    return {"type": _PYTHON_TYPE_TO_JSON.get(annotation, "string")}


def _extract_param_descriptions(docstring: str | None) -> dict[str, str]:
    """Parse a Google-style docstring to extract parameter descriptions."""
    if not docstring:
        return {}
    descriptions: dict[str, str] = {}
    in_args = False
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
            in_args = True
            continue
        if in_args:
            if stripped and not stripped.startswith(" ") and stripped.endswith(":"):
                in_args = False
                continue
            if ":" in stripped and not stripped.startswith("Returns") and not stripped.startswith("Raises"):
                param, _, desc = stripped.partition(":")
                descriptions[param.strip()] = desc.strip()
    return descriptions


# ──────────────────────────────────────────────
# FunctionTool — wrap any callable
# ──────────────────────────────────────────────

class FunctionTool(BaseTool):
    """
    Wraps any Python function (sync or async) into a BaseTool.
    Schema is auto-generated from type annotations and docstring.
    Sync functions run in a thread executor to avoid blocking.
    """

    def __init__(
        self,
        fn: Callable,
        name: str | None = None,
        description: str | None = None,
        override_schema: dict | None = None,
    ) -> None:
        self._fn = fn
        self._name = name or fn.__name__
        self._description = description or (inspect.getdoc(fn) or "").split("\n\n")[0].strip()
        self._schema = override_schema or self._build_schema()

    def _build_schema(self) -> dict:
        sig = inspect.signature(self._fn)
        try:
            hints = get_type_hints(self._fn)
        except Exception:
            hints = {}

        param_descriptions = _extract_param_descriptions(inspect.getdoc(self._fn))
        properties: dict[str, dict] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            annotation = hints.get(param_name, str)
            prop = _type_to_json_schema(annotation)
            desc = param_descriptions.get(param_name, "")
            if desc:
                prop["description"] = desc
            properties[param_name] = prop
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> dict:
        return self._schema

    async def execute(self, arguments: dict[str, Any]) -> Any:
        if asyncio.iscoroutinefunction(self._fn):
            return await self._fn(**arguments)
        else:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self._fn(**arguments))

    @classmethod
    def from_function(
        cls,
        fn: Callable,
        name: str | None = None,
        description: str | None = None,
    ) -> FunctionTool:
        return cls(fn=fn, name=name, description=description)

    @classmethod
    def register(cls, fn: Callable) -> FunctionTool:
        """Decorator: convert a function to a FunctionTool in place."""
        return cls(fn=fn)


# ──────────────────────────────────────────────
# ToolRegistry — discovery, schema export, dispatch
# ──────────────────────────────────────────────

class ToolRegistry:
    """
    Central registry for all tools available to agents.
    Maintains a name → tool mapping, exports LLM schemas, and dispatches by name.
    """

    def __init__(self, namespace: str = "default") -> None:
        self._tools: dict[str, BaseTool] = {}
        self.namespace = namespace

    def register(self, tool: BaseTool) -> ToolRegistry:
        if tool.name in self._tools:
            raise ToolRegistryError(
                f"Tool '{tool.name}' is already registered in namespace '{self.namespace}'. "
                "Use deregister() first or choose a unique name."
            )
        self._tools[tool.name] = tool
        log.debug("registered tool: %s (namespace: %s)", tool.name, self.namespace)
        return self

    def register_function(
        self,
        fn: Callable,
        name: str | None = None,
        description: str | None = None,
    ) -> ToolRegistry:
        return self.register(FunctionTool.from_function(fn, name=name, description=description))

    def deregister(self, name: str) -> ToolRegistry:
        self._tools.pop(name, None)
        return self

    def merge(self, other: ToolRegistry, prefix: str = "") -> ToolRegistry:
        """Merge another registry, optionally prefixing tool names."""
        for tool in other.tools():
            wrapped = _PrefixedTool(tool, prefix) if prefix else tool
            if wrapped.name not in self._tools:
                self._tools[wrapped.name] = wrapped
        return self

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise ToolNotFoundError(name, list(self._tools.keys()))
        return self._tools[name]

    def tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def to_llm_tools(self, names: list[str] | None = None) -> list[dict]:
        """Export tools as an LLM-compatible schema list."""
        tools = [self.get(n) for n in names] if names else self.tools()
        return [t.to_llm_schema() for t in tools]

    def to_description_block(self) -> str:
        """Render all tools as plain text for providers without native tool calling."""
        lines = ["Available tools:\n"]
        for tool in self.tools():
            lines.append(f"**{tool.name}**")
            lines.append(f"  {tool.description}")
            required = tool.parameters_schema.get("required", [])
            props = tool.parameters_schema.get("properties", {})
            for param, spec in props.items():
                req_marker = " (required)" if param in required else " (optional)"
                desc = spec.get("description", "")[:50]
                lines.append(f"  - {param} [{spec.get('type', 'any')}]{req_marker}: {desc}")
            lines.append("")
        return "\n".join(lines)

    async def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Find, validate, and execute a tool by name. Never raises."""
        try:
            tool = self.get(tool_name)
        except ToolNotFoundError as exc:
            return {
                "status": "error",
                "error_type": "not_found",
                "message": str(exc),
                "tool": tool_name,
            }
        return await tool.safe_execute(arguments)


class ToolRegistryError(Exception):
    """Raised on invalid registry operations."""


class ToolNotFoundError(Exception):
    def __init__(self, name: str, available: list[str]) -> None:
        super().__init__(
            f"Tool '{name}' not found in registry. Available tools: {available}"
        )
        self.name = name
        self.available = available


class _PrefixedTool(BaseTool):
    """Wraps a BaseTool with a name prefix. Used by ToolRegistry.merge()."""

    def __init__(self, tool: BaseTool, prefix: str) -> None:
        self._tool = tool
        self._prefix = prefix

    @property
    def name(self) -> str:
        return f"{self._prefix}{self._tool.name}"

    @property
    def description(self) -> str:
        return self._tool.description

    @property
    def parameters_schema(self) -> dict:
        return self._tool.parameters_schema

    async def execute(self, arguments: dict[str, Any]) -> Any:
        return await self._tool.execute(arguments)


# ──────────────────────────────────────────────
# MCP integration
# ──────────────────────────────────────────────

@dataclass
class McpServerConfig:
    """
    Configuration for an MCP server connection.

    Examples:
        McpServerConfig(
            name="github", transport="stdio",
            command=["npx", "-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."},
        )
        McpServerConfig(
            name="filesystem", transport="stdio",
            command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
    """
    name: str
    transport: str = "stdio"
    command: list[str] = field(default_factory=list)
    url: str = ""
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0


class McpTool(BaseTool):
    """A BaseTool that delegates execution to an MCP server tool."""

    def __init__(
        self,
        name: str,
        description: str,
        schema: dict,
        call_fn: Callable,
    ) -> None:
        self._name = name
        self._description = description
        self._schema = schema
        self._call_fn = call_fn

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> dict:
        return self._schema

    async def execute(self, arguments: dict[str, Any]) -> Any:
        return await self._call_fn(self._name, arguments)


class McpConnector:
    """
    Connects to one MCP server and exposes its tools as a ToolRegistry.

    Usage:
        async with McpConnector(config) as connector:
            mcp_registry = await connector.load_tools()
            main_registry.merge(mcp_registry, prefix="github.")
    """

    def __init__(self, config: McpServerConfig) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0

    async def __aenter__(self) -> McpConnector:
        await self._connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self._disconnect()

    async def _connect(self) -> None:
        if self.config.transport == "stdio":
            await self._connect_stdio()
        elif self.config.transport == "sse":
            await self._connect_sse()
        else:
            raise ValueError(f"Unknown MCP transport: {self.config.transport!r}")

    async def _connect_stdio(self) -> None:
        import os
        env = {**os.environ, **self.config.env}
        self._process = await asyncio.create_subprocess_exec(
            *self.config.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        await self._send({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agentic-spine", "version": "0.1.0"},
            },
        })
        await self._recv()
        await self._send({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        })

    async def _connect_sse(self) -> None:
        log.info("SSE transport connecting to %s", self.config.url)

    async def _disconnect(self) -> None:
        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except TimeoutError:
                self._process.kill()

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _send(self, message: dict) -> None:
        line = json.dumps(message) + "\n"
        if self._process and self._process.stdin:
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()

    async def _recv(self) -> dict:
        if self._process and self._process.stdout:
            line = await asyncio.wait_for(
                self._process.stdout.readline(),
                timeout=self.config.timeout_seconds,
            )
            return json.loads(line.decode().strip())
        return {}

    async def _call_tool(self, tool_name: str, arguments: dict) -> Any:
        req_id = self._next_id()
        await self._send({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        })
        response = await self._recv()
        result = response.get("result", {})
        content = result.get("content", [])
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and first.get("type") == "text":
                return first.get("text", "")
        return content

    async def load_tools(self) -> ToolRegistry:
        """Fetch the server's tool listing and return a populated ToolRegistry."""
        req_id = self._next_id()
        await self._send({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/list",
            "params": {},
        })
        response = await self._recv()
        server_tools = response.get("result", {}).get("tools", [])

        registry = ToolRegistry(namespace=self.config.name)
        for tool_spec in server_tools:
            tool = McpTool(
                name=tool_spec["name"],
                description=tool_spec.get("description", ""),
                schema=tool_spec.get("inputSchema", {"type": "object", "properties": {}}),
                call_fn=self._call_tool,
            )
            registry.register(tool)
            log.debug("loaded MCP tool: %s from server %s", tool.name, self.config.name)

        log.info(
            "McpConnector: loaded %d tools from server '%s'",
            len(registry), self.config.name,
        )
        return registry
