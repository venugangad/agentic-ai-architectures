# tests/ch09/test_telemetry.py — Chapter 9: Telemetry
from __future__ import annotations

import asyncio
import json
import logging

import pytest

from monitoring.telemetry import (
    MetricKind,
    MetricPoint,
    MetricsRegistry,
    PrometheusExporter,
    SpanKind,
    StructuredLogger,
    TelemetryCollector,
)

# ── TelemetryCollector ────────────────────────────────────────────────────────

class TestTelemetryCollector:
    def test_start_trace_creates_root_span(self):
        collector = TelemetryCollector()
        trace = collector.start_trace("my_agent", kind=SpanKind.AGENT)
        assert trace.root_span.name == "my_agent"
        assert trace.root_span.kind == SpanKind.AGENT
        assert trace.root_span.parent_id is None
        assert trace.root_span.trace_id == trace.trace_id

    def test_end_trace_sets_end_time(self):
        collector = TelemetryCollector()
        trace = collector.start_trace("test")
        collector.end_trace(trace)
        assert trace.root_span.end_time is not None
        assert trace.root_span.duration_ms is not None
        assert trace.root_span.duration_ms >= 0

    def test_span_context_manager_creates_child(self):
        collector = TelemetryCollector()
        trace = collector.start_trace("root")

        with collector.span("child_op", kind=SpanKind.TOOL) as child:
            child.set_attribute("tool", "search")

        assert len(trace.spans) == 1
        assert trace.spans[0].name == "child_op"
        assert trace.spans[0].parent_id == trace.root_span.span_id
        assert trace.spans[0].attributes["tool"] == "search"

    def test_span_records_error_on_exception(self):
        collector = TelemetryCollector()
        trace = collector.start_trace("root")

        with pytest.raises(ValueError), collector.span("failing_op") as _span:
            raise ValueError("test error")

        assert trace.spans[0].status == "error"
        assert "ValueError" in trace.spans[0].error_msg

    def test_span_end_time_always_set(self):
        collector = TelemetryCollector()
        collector.start_trace("root")

        with pytest.raises(RuntimeError), collector.span("op") as span:
            raise RuntimeError("boom")

        assert span.end_time is not None

    async def test_async_span_creates_child(self):
        collector = TelemetryCollector()
        trace = collector.start_trace("root")

        async with collector.async_span("async_op", kind=SpanKind.LLM) as child:
            child.set_attribute("model", "gpt-4o")
            await asyncio.sleep(0)

        assert len(trace.spans) == 1
        assert trace.spans[0].kind == SpanKind.LLM

    async def test_nested_spans_correct_parent_chain(self):
        collector = TelemetryCollector()
        trace = collector.start_trace("root")

        with collector.span("level1") as l1, collector.span("level2") as l2:
            pass

        assert l1.parent_id == trace.root_span.span_id
        assert l2.parent_id == l1.span_id

    def test_current_span_returns_none_outside_trace(self):
        collector = TelemetryCollector()
        assert collector.current_span() is None

    def test_flush_returns_and_clears(self):
        collector = TelemetryCollector()
        collector.start_trace("t1")
        collector.start_trace("t2")
        traces = collector.flush()
        assert len(traces) == 2
        assert len(collector.flush()) == 0

    def test_span_to_dict_has_required_fields(self):
        collector = TelemetryCollector()
        trace = collector.start_trace("test")
        collector.end_trace(trace)
        d = trace.root_span.to_dict()
        for field in ["span_id", "trace_id", "name", "kind", "start_time",
                      "end_time", "duration_ms", "attributes", "status"]:
            assert field in d


# ── MetricsRegistry ───────────────────────────────────────────────────────────

class TestMetricsRegistry:
    def test_inc_counter(self):
        m = MetricsRegistry()
        m.inc("requests")
        m.inc("requests")
        assert m.counter("requests") == 2.0

    def test_inc_with_labels(self):
        m = MetricsRegistry()
        m.inc("requests", status="ok")
        m.inc("requests", status="error")
        assert m.counter("requests", status="ok") == 1.0
        assert m.counter("requests", status="error") == 1.0

    def test_set_gauge(self):
        m = MetricsRegistry()
        m.set_gauge("active_sessions", 42)
        assert m.gauge("active_sessions") == 42

    def test_gauge_overwrite(self):
        m = MetricsRegistry()
        m.set_gauge("queue_depth", 10)
        m.set_gauge("queue_depth", 5)
        assert m.gauge("queue_depth") == 5

    def test_histogram_percentile(self):
        m = MetricsRegistry()
        for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            m.observe("latency_ms", v)
        p50 = m.histogram_percentile("latency_ms", 50)
        p95 = m.histogram_percentile("latency_ms", 95)
        assert p50 is not None
        assert p95 is not None
        assert p50 <= p95

    def test_histogram_none_when_empty(self):
        m = MetricsRegistry()
        assert m.histogram_percentile("empty_metric", 50) is None

    def test_snapshot_contains_all_types(self):
        m = MetricsRegistry()
        m.inc("calls")
        m.set_gauge("active", 3)
        m.observe("latency", 100.0)
        m.observe("latency", 200.0)

        points = m.snapshot()
        kinds = {p.kind for p in points}
        assert MetricKind.COUNTER in kinds
        assert MetricKind.GAUGE in kinds
        assert MetricKind.HISTOGRAM in kinds

    def test_snapshot_includes_percentiles(self):
        m = MetricsRegistry()
        for v in range(100):
            m.observe("latency", float(v))
        points = m.snapshot()
        names = {p.name for p in points}
        assert "latency.p50" in names
        assert "latency.p95" in names
        assert "latency.p99" in names

    def test_label_isolation(self):
        m = MetricsRegistry()
        m.inc("calls", agent="flight")
        m.inc("calls", agent="hotel")
        assert m.counter("calls", agent="flight") == 1.0
        assert m.counter("calls", agent="hotel") == 1.0
        assert m.counter("calls") == 0.0  # unlabelled key is separate


# ── StructuredLogger ──────────────────────────────────────────────────────────

class TestStructuredLogger:
    def _capture_logger(self, name: str = "test") -> tuple[StructuredLogger, list[str]]:
        lines: list[str] = []
        logger = StructuredLogger(name)
        # Replace handler with a capturing one
        log = logging.getLogger(name)
        log.handlers.clear()
        handler = logging.StreamHandler(
            type("S", (), {"write": lambda s, m: lines.append(m), "flush": lambda s: None})()
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        return logger, lines

    def test_emits_json(self):
        logger, lines = self._capture_logger("json_test")
        logger.info("test message", key="value")
        assert len(lines) >= 1
        parsed = json.loads(lines[0])
        assert parsed["msg"] == "test message"
        assert parsed["key"] == "value"
        assert "ts" in parsed
        assert "level" in parsed

    def test_includes_trace_context_when_active(self):
        collector = TelemetryCollector()
        trace = collector.start_trace("root")
        logger, lines = self._capture_logger("trace_test")
        logger._collector = collector
        logger.info("inside span")
        parsed = json.loads(lines[0])
        assert parsed["trace_id"] == trace.trace_id
        collector.end_trace(trace)


# ── Exporters ─────────────────────────────────────────────────────────────────

class TestPrometheusExporter:
    def test_get_metrics_text_format(self):
        exporter = PrometheusExporter()
        points = [
            MetricPoint(name="requests_total", kind=MetricKind.COUNTER,
                        value=42.0, labels={"status": "ok"}),
        ]
        exporter.export_metrics(points)
        text = exporter.get_metrics_text()
        assert "requests_total" in text
        assert "42.0" in text
        assert 'status="ok"' in text

    def test_export_trace_is_noop(self):
        exporter = PrometheusExporter()
        collector = TelemetryCollector()
        trace = collector.start_trace("test")
        collector.end_trace(trace)
        # Should not raise
        exporter.export_trace(trace)
