# monitoring/telemetry.py — Chapter 9: Safety, Guardrails, and Telemetry
# Part of: The Agentic Spine companion repository
from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Iterator, Protocol, runtime_checkable


# ── Span / Trace data model ───────────────────────────────────────────────────

class SpanKind(str, Enum):
    AGENT   = "agent"
    TOOL    = "tool"
    LLM     = "llm"
    PLANNER = "planner"
    ROUTER  = "router"
    SAFETY  = "safety"


@dataclass
class SpanEvent:
    name:       str
    timestamp:  float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    span_id:    str
    trace_id:   str
    name:       str
    kind:       SpanKind
    parent_id:  str | None = None
    start_time: float = field(default_factory=time.time)
    end_time:   float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events:     list[SpanEvent] = field(default_factory=list)
    status:     str = "ok"           # "ok" | "error"
    error_msg:  str | None = None

    # ── convenience ───────────────────────────────────────────────────────────

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1_000

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append(SpanEvent(name=name, attributes=attributes or {}))

    def record_error(self, exc: Exception) -> None:
        self.status    = "error"
        self.error_msg = f"{type(exc).__name__}: {exc}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id":    self.span_id,
            "trace_id":   self.trace_id,
            "parent_id":  self.parent_id,
            "name":       self.name,
            "kind":       self.kind.value,
            "start_time": self.start_time,
            "end_time":   self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": [
                {"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes}
                for e in self.events
            ],
            "status":    self.status,
            "error_msg": self.error_msg,
        }


@dataclass
class Trace:
    trace_id:   str
    root_span:  Span
    spans:      list[Span] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def all_spans(self) -> list[Span]:
        return [self.root_span] + self.spans

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id":   self.trace_id,
            "created_at": self.created_at,
            "spans":      [s.to_dict() for s in self.all_spans()],
        }


# ── ContextVar for async-safe parent propagation ──────────────────────────────

_current_span_var: contextvars.ContextVar[Span | None] = \
    contextvars.ContextVar("current_span", default=None)


# ── TelemetryCollector ────────────────────────────────────────────────────────

class TelemetryCollector:
    """
    Manages traces and spans. Thread-safe via ContextVar per asyncio.Task.

    Usage:
        collector = TelemetryCollector()
        trace = collector.start_trace("my_agent", kind=SpanKind.AGENT)
        try:
            async with collector.async_span("llm_call", SpanKind.LLM) as span:
                span.set_attribute("model", "gpt-4o")
                result = await llm.complete(...)
        finally:
            collector.end_trace(trace)
    """

    def __init__(self) -> None:
        self._traces: dict[str, Trace] = {}

    # ── trace lifecycle ───────────────────────────────────────────────────────

    def start_trace(
        self,
        name: str,
        kind: SpanKind = SpanKind.AGENT,
        attributes: dict[str, Any] | None = None,
    ) -> Trace:
        trace_id = str(uuid.uuid4())
        root = Span(
            span_id=str(uuid.uuid4()),
            trace_id=trace_id,
            name=name,
            kind=kind,
            attributes=attributes or {},
        )
        trace = Trace(trace_id=trace_id, root_span=root)
        self._traces[trace_id] = trace
        _current_span_var.set(root)
        return trace

    def end_trace(self, trace: Trace) -> None:
        if trace.root_span.end_time is None:
            trace.root_span.end_time = time.time()
        _current_span_var.set(None)

    # ── synchronous span context manager ─────────────────────────────────────

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.AGENT,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        parent = _current_span_var.get()
        trace  = self._traces.get(parent.trace_id) if parent else None

        s = Span(
            span_id=str(uuid.uuid4()),
            trace_id=parent.trace_id if parent else str(uuid.uuid4()),
            name=name,
            kind=kind,
            parent_id=parent.span_id if parent else None,
            attributes=attributes or {},
        )
        if trace:
            trace.spans.append(s)

        token = _current_span_var.set(s)
        try:
            yield s
        except Exception as exc:
            s.record_error(exc)
            raise
        finally:
            s.end_time = time.time()
            _current_span_var.reset(token)

    # ── async span context manager ────────────────────────────────────────────

    @asynccontextmanager
    async def async_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.AGENT,
        attributes: dict[str, Any] | None = None,
    ) -> AsyncIterator[Span]:
        parent = _current_span_var.get()
        trace  = self._traces.get(parent.trace_id) if parent else None

        s = Span(
            span_id=str(uuid.uuid4()),
            trace_id=parent.trace_id if parent else str(uuid.uuid4()),
            name=name,
            kind=kind,
            parent_id=parent.span_id if parent else None,
            attributes=attributes or {},
        )
        if trace:
            trace.spans.append(s)

        token = _current_span_var.set(s)
        try:
            yield s
        except Exception as exc:
            s.record_error(exc)
            raise
        finally:
            s.end_time = time.time()
            _current_span_var.reset(token)

    # ── helpers ───────────────────────────────────────────────────────────────

    def current_span(self) -> Span | None:
        return _current_span_var.get()

    def get_trace(self, trace_id: str) -> Trace | None:
        return self._traces.get(trace_id)

    def flush(self) -> list[Trace]:
        """Return all collected traces and clear the in-memory store."""
        traces = list(self._traces.values())
        self._traces.clear()
        return traces


# ── Metrics ───────────────────────────────────────────────────────────────────

class MetricKind(str, Enum):
    COUNTER   = "counter"
    GAUGE     = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class MetricPoint:
    name:      str
    kind:      MetricKind
    value:     float
    labels:    dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsRegistry:
    """
    In-process counter / gauge / histogram aggregator.

    Usage:
        metrics = MetricsRegistry()
        metrics.inc("agent_turns", agent="travel", status="ok")
        metrics.observe("turn_latency_ms", 120.0, agent="travel")
        metrics.set_gauge("active_sessions", 42)
        points = metrics.snapshot()   # export all
    """

    def __init__(self) -> None:
        self._counters:   dict[str, float]       = defaultdict(float)
        self._gauges:     dict[str, float]        = {}
        self._histograms: dict[str, list[float]]  = defaultdict(list)

    # ── write ─────────────────────────────────────────────────────────────────

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Increment a counter."""
        self._counters[self._key(name, labels)] += value

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        """Set a point-in-time gauge value."""
        self._gauges[self._key(name, labels)] = value

    def observe(self, name: str, value: float, **labels: str) -> None:
        """Record one histogram observation (e.g., latency in ms)."""
        self._histograms[self._key(name, labels)].append(value)

    # ── read ──────────────────────────────────────────────────────────────────

    def counter(self, name: str, **labels: str) -> float:
        return self._counters.get(self._key(name, labels), 0.0)

    def gauge(self, name: str, **labels: str) -> float | None:
        return self._gauges.get(self._key(name, labels))

    def histogram_percentile(self, name: str, p: float, **labels: str) -> float | None:
        """Return the p-th percentile (0–100) of recorded observations."""
        values = self._histograms.get(self._key(name, labels))
        if not values:
            return None
        sorted_vals = sorted(values)
        idx = max(0, int(len(sorted_vals) * p / 100) - 1)
        return sorted_vals[idx]

    def snapshot(self) -> list[MetricPoint]:
        """Export all current values as a flat list of MetricPoints."""
        points: list[MetricPoint] = []

        for key, v in self._counters.items():
            name, labels = self._parse_key(key)
            points.append(MetricPoint(name=name, kind=MetricKind.COUNTER,
                                      value=v, labels=labels))

        for key, v in self._gauges.items():
            name, labels = self._parse_key(key)
            points.append(MetricPoint(name=name, kind=MetricKind.GAUGE,
                                      value=v, labels=labels))

        for key, vals in self._histograms.items():
            name, labels = self._parse_key(key)
            if vals:
                for p, suffix in [(50, "p50"), (95, "p95"), (99, "p99")]:
                    pval = self.histogram_percentile(name, p, **labels)
                    if pval is not None:
                        points.append(MetricPoint(
                            name=f"{name}.{suffix}",
                            kind=MetricKind.HISTOGRAM,
                            value=pval,
                            labels=labels,
                        ))
        return points

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _key(name: str, labels: dict[str, str]) -> str:
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}" if label_str else name

    @staticmethod
    def _parse_key(key: str) -> tuple[str, dict[str, str]]:
        if "{" not in key:
            return key, {}
        name, rest = key.split("{", 1)
        label_str  = rest.rstrip("}")
        labels: dict[str, str] = {}
        for pair in label_str.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                labels[k] = v
        return name, labels


# ── Structured Logger ─────────────────────────────────────────────────────────

class StructuredLogger:
    """
    JSON-line logger that automatically attaches trace_id and span_id
    from the active ContextVar span.

    Usage:
        logger = StructuredLogger("agentspine", collector=collector)
        logger.info("tool_called", tool="search_flights", session_id=sid)
        # emits: {"ts":..., "level":"INFO", "msg":"tool_called",
        #          "trace_id":"...", "span_id":"...", "tool":"search_flights"}
    """

    def __init__(
        self,
        name: str,
        collector: TelemetryCollector | None = None,
        level: int = logging.INFO,
    ) -> None:
        self._log = logging.getLogger(name)
        self._log.setLevel(level)
        self._collector = collector
        if not self._log.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(message)s"))
            self._log.addHandler(h)

    def _emit(self, level: int, msg: str, **extra: Any) -> None:
        span = self._collector.current_span() if self._collector else None
        record: dict[str, Any] = {
            "ts":       time.time(),
            "level":    logging.getLevelName(level),
            "msg":      msg,
            "trace_id": span.trace_id if span else None,
            "span_id":  span.span_id  if span else None,
        }
        record.update(extra)
        self._log.log(level, json.dumps(record, default=str))

    def debug(self, msg: str, **kw: Any) -> None:
        self._emit(logging.DEBUG, msg, **kw)

    def info(self, msg: str, **kw: Any) -> None:
        self._emit(logging.INFO, msg, **kw)

    def warning(self, msg: str, **kw: Any) -> None:
        self._emit(logging.WARNING, msg, **kw)

    def error(self, msg: str, **kw: Any) -> None:
        self._emit(logging.ERROR, msg, **kw)


# ── Exporter protocol + built-in implementations ──────────────────────────────

@runtime_checkable
class TelemetryExporter(Protocol):
    def export_trace(self, trace: Trace) -> None: ...
    def export_metrics(self, points: list[MetricPoint]) -> None: ...


class StdoutExporter:
    """
    Default exporter: write JSON-line records to stdout.
    Zero external dependencies; works with any log aggregation system.
    """

    def export_trace(self, trace: Trace) -> None:
        print(json.dumps({"_type": "trace", **trace.to_dict()}, default=str))

    def export_metrics(self, points: list[MetricPoint]) -> None:
        for pt in points:
            print(json.dumps({
                "_type": "metric",
                "name":   pt.name,
                "kind":   pt.kind.value,
                "value":  pt.value,
                "labels": pt.labels,
                "ts":     pt.timestamp,
            }, default=str))


class PrometheusExporter:
    """
    Formats metric snapshot as Prometheus text exposition format.
    Wire the get_metrics_text() return value to your /metrics HTTP handler.

    Traces are silently dropped — Prometheus handles only metrics.
    Use an OTLP exporter alongside for traces.
    """

    def __init__(self) -> None:
        self._latest: list[MetricPoint] = []

    def export_trace(self, trace: Trace) -> None:
        pass  # Prometheus does not ingest traces

    def export_metrics(self, points: list[MetricPoint]) -> None:
        self._latest = points

    def get_metrics_text(self) -> str:
        lines: list[str] = []
        for pt in self._latest:
            label_str = ",".join(f'{k}="{v}"' for k, v in pt.labels.items())
            metric    = f"{pt.name}{{{label_str}}}" if label_str else pt.name
            lines.append(f"# TYPE {pt.name} {pt.kind.value}")
            lines.append(f"{metric} {pt.value} {int(pt.timestamp * 1_000)}")
        return "\n".join(lines) + "\n"
