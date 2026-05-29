# knowledge/memory_service.py
"""
Three-Tier Memory Service — knowledge/memory_service.py
The Agent Circuit: Engineering a Provider-Agnostic AI Framework

Tiers:     MemoryTier (SHORT_TERM · WORKING · LONG_TERM)
Model:     MemoryEntry (content, tier, TTL, importance, recency)
Service:   MemoryService (store, recall, summarise_and_promote)

SHORT_TERM: in-process dict · 1-hour TTL · term-overlap search
WORKING:    dict + VectorIndex · 7-day TTL · semantic search
LONG_TERM:  VectorIndex only · no expiry · knowledge base

Built in Chapter 6: Knowledge Retrieval: Agentic RAG
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from knowledge.rag import HybridRetriever, RetrievalRequest


class MemoryTier(str, Enum):
    SHORT_TERM = "short_term"
    WORKING    = "working"
    LONG_TERM  = "long_term"


@dataclass
class MemoryEntry:
    entry_id: str
    content: str
    tier: MemoryTier
    user_id: str
    session_id: str | None
    created_at: float
    importance: float = 0.5          # 0.0–1.0; drives retention + ordering
    metadata: dict[str, Any] = field(default_factory=dict)
    expires_at: float | None = None  # None = never expires

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class MemoryService:
    """
    Three-tier memory manager backed by HybridRetriever.

    Agents call store() / recall() instead of the retriever directly.
    The service handles tier routing, TTL enforcement, and importance-weighted
    recall ordering.

    Usage
    -----
        memory = MemoryService(retriever)

        # Inside agent — store a fact
        await memory.store(
            content="User prefers window seats",
            tier=MemoryTier.SHORT_TERM,
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            importance=0.8,
        )

        # Inside agent — recall relevant context
        entries = await memory.recall(
            query="seat preference",
            user_id=ctx.user_id,
            session_id=ctx.session_id,
        )
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        short_term_ttl_seconds: float = 3600,       # 1 hour
        working_ttl_seconds: float = 86400 * 7,     # 1 week
    ) -> None:
        self._retriever = retriever
        self._short_ttl = short_term_ttl_seconds
        self._working_ttl = working_ttl_seconds
        self._short_term: dict[str, MemoryEntry] = {}
        self._working: dict[str, MemoryEntry] = {}

    # ── Store ───────────────────────────────────────────────────────────

    async def store(
        self,
        content: str,
        tier: MemoryTier,
        user_id: str,
        session_id: str | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """
        Store a memory entry in the requested tier.
        WORKING and LONG_TERM entries are also indexed in the vector store.
        """
        entry = MemoryEntry(
            entry_id=str(uuid.uuid4()),
            content=content,
            tier=tier,
            user_id=user_id,
            session_id=session_id,
            created_at=time.time(),
            importance=importance,
            metadata=metadata or {},
            expires_at=self._ttl_for(tier),
        )

        if tier == MemoryTier.SHORT_TERM:
            self._short_term[entry.entry_id] = entry

        elif tier == MemoryTier.WORKING:
            self._working[entry.entry_id] = entry
            await self._retriever.ingest(
                content=content,
                document_id=entry.entry_id,
                metadata={
                    "tier": tier.value,
                    "user_id": user_id,
                    "session_id": session_id or "",
                    "created_at": entry.created_at,
                    **(metadata or {}),
                },
            )

        elif tier == MemoryTier.LONG_TERM:
            await self._retriever.ingest(
                content=content,
                document_id=entry.entry_id,
                metadata={
                    "tier": tier.value,
                    "user_id": user_id,
                    "created_at": entry.created_at,
                    **(metadata or {}),
                },
            )

        return entry

    # ── Recall ──────────────────────────────────────────────────────────

    async def recall(
        self,
        query: str,
        user_id: str,
        tiers: list[MemoryTier] | None = None,
        top_k: int = 5,
        session_id: str | None = None,
    ) -> list[MemoryEntry]:
        """
        Retrieve relevant memory entries across requested tiers.
        Results are ordered by importance × recency_weight (descending).
        """
        active_tiers = tiers or list(MemoryTier)
        results: list[MemoryEntry] = []

        if MemoryTier.SHORT_TERM in active_tiers:
            results.extend(self._search_short_term(query, user_id, session_id))

        vector_tiers = [t for t in active_tiers if t != MemoryTier.SHORT_TERM]
        if vector_tiers:
            tier_values = {t.value for t in vector_tiers}
            retrieval_results = await self._retriever.retrieve(
                RetrievalRequest(
                    query=query,
                    top_k=top_k * 2,
                    metadata_filter={"user_id": user_id},
                )
            )
            for rr in retrieval_results:
                tier_val = rr.chunk.metadata.get("tier", "")
                if tier_val in tier_values:
                    results.append(MemoryEntry(
                        entry_id=rr.chunk.document_id,
                        content=rr.chunk.content,
                        tier=MemoryTier(tier_val),
                        user_id=user_id,
                        session_id=rr.chunk.metadata.get("session_id"),
                        created_at=rr.chunk.metadata.get("created_at", time.time()),
                        importance=rr.score,
                    ))

        results.sort(
            key=lambda e: e.importance * self._recency_weight(e),
            reverse=True,
        )
        return results[:top_k]

    # ── Promotion ───────────────────────────────────────────────────────

    async def summarise_and_promote(
        self,
        session_id: str,
        user_id: str,
        summary: str,
        importance: float = 0.7,
    ) -> MemoryEntry:
        """
        Promote a session summary from SHORT_TERM to WORKING memory.
        Purges all short-term entries for the session.
        Called by SessionRunner at session end when a summary is available.
        """
        self._purge_session_short_term(session_id)
        return await self.store(
            content=summary,
            tier=MemoryTier.WORKING,
            user_id=user_id,
            session_id=session_id,
            importance=importance,
            metadata={"summary_of_session": session_id},
        )

    async def add_to_knowledge_base(
        self,
        content: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Convenience: store directly to LONG_TERM (permanent knowledge base)."""
        return await self.store(
            content=content,
            tier=MemoryTier.LONG_TERM,
            user_id=user_id,
            importance=1.0,
            metadata=metadata or {},
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    def _search_short_term(
        self, query: str, user_id: str, session_id: str | None
    ) -> list[MemoryEntry]:
        import re
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        matches: list[MemoryEntry] = []
        for entry in list(self._short_term.values()):
            if entry.user_id != user_id:
                continue
            if session_id and entry.session_id != session_id:
                continue
            if entry.is_expired():
                continue
            terms = set(re.findall(r'\b\w+\b', entry.content.lower()))
            overlap = len(query_terms & terms) / max(len(query_terms), 1)
            if overlap > 0:
                entry.importance = overlap
                matches.append(entry)
        return sorted(matches, key=lambda e: e.importance, reverse=True)

    def _ttl_for(self, tier: MemoryTier) -> float | None:
        if tier == MemoryTier.SHORT_TERM:
            return time.time() + self._short_ttl
        if tier == MemoryTier.WORKING:
            return time.time() + self._working_ttl
        return None   # LONG_TERM never expires

    def _recency_weight(self, entry: MemoryEntry) -> float:
        """Exponential decay: weight halves every ~7 hours."""
        age_hours = (time.time() - entry.created_at) / 3600
        return 1.0 / (1.0 + age_hours * 0.1)

    def _purge_session_short_term(self, session_id: str) -> None:
        self._short_term = {
            eid: e for eid, e in self._short_term.items()
            if e.session_id != session_id
        }
