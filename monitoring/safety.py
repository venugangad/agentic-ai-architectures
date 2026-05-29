# monitoring/safety.py — Chapter 9: Safety, Guardrails, and Telemetry
# Part of: The Agent Circuit companion repository
from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ── severity & policy ─────────────────────────────────────────────────────────

class Severity(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
    BLOCK  = "block"   # always raises; never just records


@dataclass
class SafetyPolicy:
    # PII
    detect_pii_in_input:  bool = True
    detect_pii_in_output: bool = True
    pii_severity: Severity = Severity.HIGH

    # Prompt injection
    detect_injection: bool = True
    injection_severity: Severity = Severity.BLOCK

    # Content filter
    blocked_output_patterns: list[str] = field(default_factory=list)
    content_filter_severity: Severity = Severity.BLOCK

    # Budget guard (mirrors GatewayConfig but scoped to safety layer)
    max_tool_calls_per_turn: int = 20
    max_turns_per_session:   int = 100

    # HITL
    require_human_approval_for: list[str] = field(default_factory=list)
    hitl_timeout_seconds: float = 3600.0


@dataclass
class SafetyViolation:
    violation_id:    str
    kind:            str        # "pii" | "injection" | "content" | "budget" | "hitl"
    severity:        Severity
    location:        str        # "input" | "output" | "tool_call"
    detail:          str
    matched_pattern: str | None
    agent_name:      str
    session_id:      str | None
    timestamp:       float = field(default_factory=time.time)

    def is_blocking(self) -> bool:
        return self.severity == Severity.BLOCK


# ── detectors ────────────────────────────────────────────────────────────────

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email":       re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "phone_us":    re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    "ssn":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|"
        r"6(?:011|5[0-9]{2})[0-9]{12})\b"
    ),
    "ipv4":        re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
    "aws_key":     re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+(instructions?|rules?)", re.I),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(?!an?\s+agent)", re.I),
    re.compile(r"new\s+(system\s+)?prompt\s*:", re.I),
    re.compile(r"<\s*system\s*>", re.I),
    re.compile(r"\[\s*system\s*\]", re.I),
    re.compile(r"act\s+as\s+(?:if\s+you\s+(?:are|were)|a\s+)", re.I),
    re.compile(r"forget\s+(everything|all)\s+(you('ve|\s+have)\s+)?(?:been\s+)?told", re.I),
    re.compile(r"developer\s+mode\s*(?:enabled|on|activated)", re.I),
    re.compile(r"DAN\s+mode", re.I),
    re.compile(r"jailbreak", re.I),
]


class PIIDetector:
    def __init__(self, patterns: dict[str, re.Pattern[str]] | None = None):
        self._patterns = patterns or _PII_PATTERNS

    def scan(self, text: str) -> list[tuple[str, str]]:
        """Return list of (kind, matched_value) for every PII hit."""
        hits: list[tuple[str, str]] = []
        for kind, pattern in self._patterns.items():
            for match in pattern.finditer(text):
                hits.append((kind, match.group()))
        return hits

    def redact(self, text: str) -> str:
        """Replace every PII hit with [REDACTED:<kind>]."""
        for kind, pattern in self._patterns.items():
            text = pattern.sub(f"[REDACTED:{kind.upper()}]", text)
        return text


class PromptInjectionDetector:
    def __init__(self, patterns: list[re.Pattern[str]] | None = None):
        self._patterns = patterns or _INJECTION_PATTERNS

    def scan(self, text: str) -> list[str]:
        """Return list of matched pattern strings."""
        hits: list[str] = []
        for pattern in self._patterns:
            m = pattern.search(text)
            if m:
                hits.append(m.group())
        return hits


class ContentFilter:
    def __init__(self, blocked_patterns: list[str]):
        self._compiled = [re.compile(p, re.I) for p in blocked_patterns]

    def scan(self, text: str) -> list[str]:
        hits: list[str] = []
        for pattern in self._compiled:
            m = pattern.search(text)
            if m:
                hits.append(m.group())
        return hits


# ── SafetyError ───────────────────────────────────────────────────────────────

class SafetyError(Exception):
    def __init__(self, violation: SafetyViolation):
        super().__init__(f"[{violation.severity}] {violation.kind}: {violation.detail}")
        self.violation = violation


# ── SafetyMonitor ────────────────────────────────────────────────────────────

class SafetyMonitor:
    """
    Composition-root wrapper that checks inputs and outputs against policy.

    Usage:
        monitor = SafetyMonitor(policy, on_violation=my_sink)
        monitor.check_input(user_msg, agent_name="travel", session_id=sid)
        response = await agent.run(user_msg)
        monitor.check_output(response, agent_name="travel", session_id=sid)
    """

    def __init__(
        self,
        policy: SafetyPolicy | None = None,
        on_violation: Callable[[SafetyViolation], None] | None = None,
    ):
        self._policy       = policy or SafetyPolicy()
        self._on_violation = on_violation
        self._pii          = PIIDetector()
        self._injection    = PromptInjectionDetector()
        self._content      = ContentFilter(self._policy.blocked_output_patterns)

    # ── public API ────────────────────────────────────────────────────────────

    def check_input(
        self,
        text: str,
        agent_name: str = "",
        session_id: str | None = None,
    ) -> list[SafetyViolation]:
        violations: list[SafetyViolation] = []
        p = self._policy

        if p.detect_injection:
            for hit in self._injection.scan(text):
                violations.append(self._make(
                    "injection", p.injection_severity,
                    "input", f"Injection pattern detected: {hit!r}", hit,
                    agent_name, session_id,
                ))

        if p.detect_pii_in_input:
            for kind, value in self._pii.scan(text):
                violations.append(self._make(
                    "pii", p.pii_severity,
                    "input", f"PII detected: {kind}={value!r}", value,
                    agent_name, session_id,
                ))

        self._dispatch(violations)
        return violations

    def check_output(
        self,
        text: str,
        agent_name: str = "",
        session_id: str | None = None,
    ) -> list[SafetyViolation]:
        violations: list[SafetyViolation] = []
        p = self._policy

        if p.detect_pii_in_output:
            for kind, value in self._pii.scan(text):
                violations.append(self._make(
                    "pii", p.pii_severity,
                    "output", f"PII in output: {kind}={value!r}", value,
                    agent_name, session_id,
                ))

        for pattern in self._content.scan(text):
            violations.append(self._make(
                "content", p.content_filter_severity,
                "output", f"Blocked content matched: {pattern!r}", pattern,
                agent_name, session_id,
            ))

        self._dispatch(violations)
        return violations

    # ── internals ─────────────────────────────────────────────────────────────

    def _dispatch(self, violations: list[SafetyViolation]) -> None:
        for v in violations:
            if self._on_violation:
                self._on_violation(v)
            if v.is_blocking():
                raise SafetyError(v)

    def _make(
        self,
        kind: str, severity: Severity,
        location: str, detail: str, pattern: str | None,
        agent_name: str, session_id: str | None,
    ) -> SafetyViolation:
        return SafetyViolation(
            violation_id=str(uuid.uuid4()),
            kind=kind, severity=severity,
            location=location, detail=detail,
            matched_pattern=pattern,
            agent_name=agent_name, session_id=session_id,
        )


# ── Human-in-the-Loop checkpoint ─────────────────────────────────────────────

class CheckpointPausedError(Exception):
    """Raised when a checkpoint needs human approval before proceeding."""
    def __init__(self, checkpoint_id: str, description: str, payload: dict[str, Any]):
        super().__init__(f"Checkpoint {checkpoint_id!r} awaiting approval: {description}")
        self.checkpoint_id = checkpoint_id
        self.description   = description
        self.payload       = payload


class ApprovalStatus(str, Enum):
    PENDING   = "pending"
    APPROVED  = "approved"
    REJECTED  = "rejected"
    TIMED_OUT = "timed_out"


@dataclass
class CheckpointRecord:
    checkpoint_id: str
    description:   str
    payload:       dict[str, Any]
    status:        ApprovalStatus = ApprovalStatus.PENDING
    reviewer_note: str | None = None
    created_at:    float = field(default_factory=time.time)
    resolved_at:   float | None = None


class HumanCheckpoint:
    """
    Suspend-and-resume gate for high-stakes tool calls.

    First call  → raises CheckpointPausedError (stack unwinds, plan saved).
    Second call → returns CheckpointRecord if APPROVED, raises SafetyError if REJECTED.

    Usage:
        checkpoint = HumanCheckpoint(timeout_seconds=3600)
        # wire as pre-call hook:
        registry.register("wire_transfer", fn, pre_call=checkpoint.gate("Wire transfer"))
    """

    def __init__(
        self,
        store: dict[str, CheckpointRecord] | None = None,
        timeout_seconds: float = 3600.0,
    ):
        # In production: back with Redis or Postgres via SessionService
        self._store   = store if store is not None else {}
        self._timeout = timeout_seconds

    def gate(self, description: str) -> Callable[[dict[str, Any]], None]:
        """Return a pre-call hook that pauses if no approval exists yet."""
        def _hook(tool_args: dict[str, Any]) -> None:
            self.wait_for_approval(description, tool_args)
        return _hook

    def wait_for_approval(
        self,
        description: str,
        payload: dict[str, Any],
        checkpoint_id: str | None = None,
    ) -> CheckpointRecord:
        cid      = checkpoint_id or str(uuid.uuid4())
        existing = self._store.get(cid)

        if existing:
            if existing.status == ApprovalStatus.APPROVED:
                return existing
            if existing.status == ApprovalStatus.REJECTED:
                raise SafetyError(SafetyViolation(
                    violation_id=cid, kind="hitl",
                    severity=Severity.BLOCK, location="tool_call",
                    detail=f"Rejected by reviewer: {existing.reviewer_note}",
                    matched_pattern=None, agent_name="", session_id=None,
                ))
            if time.time() - existing.created_at > self._timeout:
                existing.status = ApprovalStatus.TIMED_OUT
                raise SafetyError(SafetyViolation(
                    violation_id=cid, kind="hitl",
                    severity=Severity.BLOCK, location="tool_call",
                    detail="Checkpoint timed out waiting for human approval.",
                    matched_pattern=None, agent_name="", session_id=None,
                ))

        # First visit — create record and pause
        record = CheckpointRecord(
            checkpoint_id=cid,
            description=description,
            payload=payload,
        )
        self._store[cid] = record
        raise CheckpointPausedError(cid, description, payload)

    def resolve(
        self,
        checkpoint_id: str,
        approved: bool,
        reviewer_note: str | None = None,
    ) -> CheckpointRecord:
        record = self._store.get(checkpoint_id)
        if not record:
            raise KeyError(f"Unknown checkpoint: {checkpoint_id!r}")
        record.status        = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        record.reviewer_note = reviewer_note
        record.resolved_at   = time.time()
        return record
