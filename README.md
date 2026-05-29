# agnostic-agent-framework

> Companion repository for **"The Agentic Spine"**  
> *Engineering a Provider-Agnostic AI Framework from Scratch — in Running Python*

---

## What This Repo Is

This repo is built **chapter by chapter** alongside the book. Each chapter adds one module to the framework. By Chapter 10, you have a complete, installable, production-grade agentic platform that your organisation owns entirely.

**You are not configuring someone else's framework. You are building your own.**

---

## The Framework At A Glance

```
agnostic-agent-framework/
├── core/
│   ├── llm.py           ← Ch1: BaseLlm + OpenAI, Anthropic, Gemini, Ollama adapters
│   ├── agent.py         ← Ch2: BaseAgent, LlmAgent, lifecycle hooks
│   ├── planner.py       ← Ch7: BasePlanner, ReActPlanner, ToTPlanner
│   └── gateway.py       ← Ch8: LlmGateway — routing, fallback, caching, cost
├── tools/
│   ├── registry.py      ← Ch3: FunctionTool, ToolRegistry, versioning
│   └── mcp_connector.py ← Ch3: MCPToolset — the USB-C for AI tools
├── orchestration/
│   ├── router.py        ← Ch4: ToolRouter — semantic + deterministic dispatch
│   ├── workflows.py     ← Ch7: Sequential, Parallel, Loop, Saga patterns
│   └── orchestrator.py  ← Ch10: Supervisor, A2A protocol, AgentRegistry
├── storage/
│   └── session.py       ← Ch5: SessionService, multi-tenancy, event log
├── knowledge/
│   ├── rag.py           ← Ch6: VectorStore, GraphRagConnector, AgenticRAG
│   └── memory_service.py← Ch6: BaseMemoryService, tiered memory, provenance
├── monitoring/
│   ├── safety.py        ← Ch9: SafetyMonitor, guardrails, HITL hooks
│   └── telemetry.py     ← Ch9: Telemetry, structured traces, cost tracking
├── examples/
│   ├── ch01_hello_multimodel/   ← Switch GPT-4o ↔ Claude ↔ Gemini in one line
│   ├── ch02_base_agent/         ← Configure any agent, any LLM, any tools
│   ├── ch03_tool_registry/      ← Register any function as an MCP tool
│   ├── ch04_router/             ← Intent-based dispatch
│   ├── ch05_sessions/           ← Multi-tenant state + checkpoint recovery
│   ├── ch06_rag_complete/       ← Standalone RAG system (vector + graph + agentic)
│   ├── ch07_planner/            ← ReAct + ToT from a config file
│   ├── ch08_gateway/            ← Unified gateway with fallback chains
│   ├── ch09_safety/             ← Guardrails + HITL approval flows
│   └── ch10_full_framework/     ← The complete assembled system
├── tests/
│   └── ch*/                     ← pytest suite, one folder per chapter
├── setup.py
├── pyproject.toml
└── requirements.txt
```

---

## Quick Start (after Chapter 10)

```bash
git clone https://github.com/your-org/agnostic-agent-framework
cd agnostic-agent-framework
pip install -e .

# Run the full framework demo
cd examples/ch10_full_framework
cp .env.example .env   # add your API keys
python main.py
```

---

## Chapter-by-Chapter Progress

| Chapter | Module | Status |
|---|---|---|
| 1 | `core/llm.py` | ⬜ |
| 2 | `core/agent.py` | ⬜ |
| 3 | `tools/registry.py` | ⬜ |
| 4 | `orchestration/router.py` | ⬜ |
| 5 | `storage/session.py` | ⬜ |
| 6 | `knowledge/rag.py` | ⬜ |
| 7 | `core/planner.py` | ⬜ |
| 8 | `core/gateway.py` | ⬜ |
| 9 | `monitoring/` | ⬜ |
| 10 | `orchestration/orchestrator.py` + `setup.py` | ⬜ |

---

## Design Principles

1. **Provider-agnostic by design** — `BaseLlm` is the only LLM interface any module ever touches
2. **Interface-first** — abstract contracts defined before any implementation
3. **Runnable at every chapter** — each chapter example works independently
4. **Zero vendor lock-in** — swap OpenAI for Anthropic with one config line
5. **Observable by default** — every action generates a structured trace
6. **Multi-tenant safe** — namespace isolation baked into session management
7. **Incrementally composable** — each module works standalone or as part of the full stack
