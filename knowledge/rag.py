# knowledge/rag.py
"""
Knowledge Retrieval: Agentic RAG — knowledge/rag.py
The Agentic Spine: Engineering a Provider-Agnostic AI Framework

Data model:    Chunk, RetrievalResult, RetrievalRequest
Chunking:      TextChunker (sentence-aware sliding window)
Embedding:     EmbeddingProvider (Protocol) · MockEmbeddingProvider · OpenAIEmbeddingProvider
Indexing:      VectorIndex (cosine similarity) · BM25Index (sparse keyword)
Fusion:        _reciprocal_rank_fusion (RRF)
Re-ranking:    Reranker (Protocol) · SimpleReranker · CrossEncoderReranker
Pipeline:      HybridRetriever (ingest → search → rerank → format_context)

Built in Chapter 6: Knowledge Retrieval: Agentic RAG
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)


# ── Data Model ──────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """
    The atomic unit of the retrieval system.
    One chunk = one passage that can stand alone as context.
    """
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any]
    embedding: list[float] | None = None
    chunk_index: int = 0
    token_count: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            # embeddings intentionally omitted — too large for serialisation
        }


@dataclass
class RetrievalResult:
    """One retrieved passage with its relevance signal."""
    chunk: Chunk
    score: float
    retrieval_method: str          # "dense" | "sparse" | "hybrid" | "reranked"
    rank: int = 0

    @property
    def content(self) -> str:
        return self.chunk.content

    def to_context_string(self, include_metadata: bool = True) -> str:
        """Render as a block the LLM can read."""
        parts = [f"[Source {self.rank + 1}]"]
        if include_metadata:
            if src := self.chunk.metadata.get("source"):
                parts.append(f"({src})")
        parts.append(self.chunk.content)
        return " ".join(parts)


@dataclass
class RetrievalRequest:
    """Everything needed to execute one retrieval call."""
    query: str
    top_k: int = 5
    score_threshold: float = 0.0
    metadata_filter: dict[str, Any] | None = None
    search_mode: str = "hybrid"    # "dense" | "sparse" | "hybrid"
    rerank: bool = True


# ── Chunking ─────────────────────────────────────────────────────────────────


class TextChunker:
    """
    Sentence-aware sliding-window chunker.

    Parameters
    ----------
    chunk_size:     target token count per chunk (approximated by word count)
    overlap:        tokens shared between adjacent chunks
    min_chunk_size: discard tail chunks smaller than this
    """

    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 80,
        min_chunk_size: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " "]

    def chunk_document(
        self,
        content: str,
        document_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        meta = metadata or {}
        sentences = self._split_into_sentences(content)
        windows = self._build_windows(sentences)
        chunks = []
        for idx, window_text in enumerate(windows):
            token_est = len(window_text.split())
            if token_est < self.min_chunk_size:
                continue
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                content=window_text.strip(),
                metadata={**meta, "chunk_index": idx},
                chunk_index=idx,
                token_count=token_est,
            ))
        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _build_windows(self, sentences: list[str]) -> list[str]:
        windows: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = len(sent.split())
            if current_tokens + sent_tokens > self.chunk_size and current:
                windows.append(" ".join(current))
                overlap_buffer: list[str] = []
                ob_tokens = 0
                for s in reversed(current):
                    st = len(s.split())
                    if ob_tokens + st > self.overlap:
                        break
                    overlap_buffer.insert(0, s)
                    ob_tokens += st
                current = overlap_buffer[:]
                current_tokens = ob_tokens
            current.append(sent)
            current_tokens += sent_tokens

        if current:
            windows.append(" ".join(current))
        return windows


# ── Embedding ─────────────────────────────────────────────────────────────────


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Protocol: any object with these three members can embed text.
    Implementations: MockEmbeddingProvider, OpenAIEmbeddingProvider,
    or any sentence-transformers / Cohere wrapper.
    """
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, query: str) -> list[float]: ...

    @property
    def dimension(self) -> int: ...


class MockEmbeddingProvider:
    """
    Deterministic hash-based embeddings. Zero network calls.
    Semantic similarity is NOT meaningful — for unit tests only.
    """

    def __init__(self, dimension: int = 128) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    async def embed_query(self, query: str) -> list[float]:
        return self._hash_embed(query)

    def _hash_embed(self, text: str) -> list[float]:
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        vec: list[float] = []
        for i in range(0, min(len(h), self._dimension * 4), 4):
            chunk = h[i:i + 4]
            val = int.from_bytes(chunk.ljust(4, b'\x00'), 'big') / 2 ** 32
            vec.append(val * 2 - 1)
        while len(vec) < self._dimension:
            vec.append(0.0)
        mag = sum(x * x for x in vec) ** 0.5
        return [x / mag for x in vec] if mag > 0 else vec


class OpenAIEmbeddingProvider:
    """
    Production embeddings via OpenAI text-embedding API.
    Requires: pip install openai
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        import openai  # type: ignore
        self._client = openai.AsyncOpenAI()
        self._model = model
        self._dim = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }.get(model, 1536)

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in resp.data]

    async def embed_query(self, query: str) -> list[float]:
        results = await self.embed_texts([query])
        return results[0]


# ── Vector Index ──────────────────────────────────────────────────────────────


class VectorIndex:
    """
    In-memory cosine similarity index.
    Thread-safe reads; async write lock on upsert/delete.
    Swap for Chroma, Weaviate, or pgvector at the composition root.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, chunks: list[Chunk]) -> None:
        async with self._lock:
            for chunk in chunks:
                if chunk.embedding is None:
                    raise ValueError(
                        f"Chunk {chunk.chunk_id} has no embedding — call embed() first"
                    )
                self._chunks[chunk.chunk_id] = chunk
        log.debug("VectorIndex: upserted %d chunks (total=%d)", len(chunks), len(self._chunks))

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        candidates = list(self._chunks.values())
        if metadata_filter:
            candidates = [c for c in candidates if self._matches_filter(c, metadata_filter)]
        if not candidates:
            return []

        scored: list[tuple[float, Chunk]] = []
        for chunk in candidates:
            score = self._cosine(query_embedding, chunk.embedding)  # type: ignore[arg-type]
            if score >= score_threshold:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievalResult(chunk=ch, score=sc, retrieval_method="dense", rank=i)
            for i, (sc, ch) in enumerate(scored[:top_k])
        ]

    async def delete_document(self, document_id: str) -> int:
        async with self._lock:
            before = len(self._chunks)
            self._chunks = {
                cid: c for cid, c in self._chunks.items()
                if c.document_id != document_id
            }
            return before - len(self._chunks)

    async def count(self) -> int:
        return len(self._chunks)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    @staticmethod
    def _matches_filter(chunk: Chunk, f: dict[str, Any]) -> bool:
        for k, v in f.items():
            if chunk.metadata.get(k) != v:
                return False
        return True


# ── BM25 Index ────────────────────────────────────────────────────────────────


class BM25Index:
    """
    In-memory BM25 (Robertson-Zaragoza 2009) sparse retrieval.
    k1=1.5, b=0.75 are standard defaults.
    Scores are normalised to [0, 1] relative to best result for fusion compatibility.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._term_doc_freq: dict[str, int] = {}
        self._doc_term_freqs: list[dict[str, int]] = []
        self._avg_doc_len: float = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._doc_term_freqs = [self._tokenize_freq(c.content) for c in chunks]
        self._term_doc_freq = {}
        for tf in self._doc_term_freqs:
            for term in tf:
                self._term_doc_freq[term] = self._term_doc_freq.get(term, 0) + 1
        lengths = [sum(tf.values()) for tf in self._doc_term_freqs]
        self._avg_doc_len = sum(lengths) / len(lengths) if lengths else 1.0

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        query_terms = list(self._tokenize_freq(query).keys())
        N = len(self._chunks)
        if N == 0 or not query_terms:
            return []

        scores: list[tuple[float, int]] = []
        for doc_idx, chunk in enumerate(self._chunks):
            if metadata_filter and not VectorIndex._matches_filter(chunk, metadata_filter):
                continue
            score = 0.0
            tf = self._doc_term_freqs[doc_idx]
            doc_len = sum(tf.values())
            for term in query_terms:
                if term not in tf:
                    continue
                n_t = self._term_doc_freq.get(term, 0)
                idf = math.log((N - n_t + 0.5) / (n_t + 0.5) + 1)
                tf_val = tf[term]
                numerator = tf_val * (self.k1 + 1)
                denominator = tf_val + self.k1 * (
                    1 - self.b + self.b * doc_len / self._avg_doc_len
                )
                score += idf * (numerator / denominator)
            if score > score_threshold:
                scores.append((score, doc_idx))

        scores.sort(reverse=True)
        max_score = scores[0][0] if scores else 1.0
        return [
            RetrievalResult(
                chunk=self._chunks[idx],
                score=round(sc / max_score, 4),
                retrieval_method="sparse",
                rank=i,
            )
            for i, (sc, idx) in enumerate(scores[:top_k])
        ]

    @staticmethod
    def _tokenize_freq(text: str) -> dict[str, int]:
        import re
        tokens = re.findall(r'\b[a-z0-9_\-\.]+\b', text.lower())
        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        return freq


# ── RRF Fusion ────────────────────────────────────────────────────────────────


def _reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievalResult]:
    """
    Combine multiple ranked lists with Reciprocal Rank Fusion.
    Score(d) = Σ weight_i / (k + rank_i(d))
    Scores are normalised to [0, 1] relative to the top result.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for result_list, weight in zip(ranked_lists, weights):
        for rank, result in enumerate(result_list):
            cid = result.chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + weight * (1.0 / (k + rank + 1))
            chunk_map[cid] = result.chunk

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    max_score = rrf_scores[sorted_ids[0]] if sorted_ids else 1.0
    return [
        RetrievalResult(
            chunk=chunk_map[cid],
            score=round(rrf_scores[cid] / max_score, 4),
            retrieval_method="hybrid",
            rank=i,
        )
        for i, cid in enumerate(sorted_ids)
    ]


# ── Re-ranking ────────────────────────────────────────────────────────────────


class Reranker(Protocol):
    """Any object with this shape can rerank a result list."""
    async def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]: ...


class SimpleReranker:
    """
    Term-overlap heuristic reranker. No LLM calls.
    Combined score = 0.7 × retrieval_score + 0.3 × query_term_overlap.
    """

    async def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        import re
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        rescored: list[tuple[float, RetrievalResult]] = []
        for result in results:
            chunk_terms = set(re.findall(r'\b\w+\b', result.chunk.content.lower()))
            overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            combined = 0.7 * result.score + 0.3 * overlap
            rescored.append((combined, result))
        rescored.sort(reverse=True)
        return [
            RetrievalResult(
                chunk=r.chunk, score=round(sc, 4),
                retrieval_method="reranked", rank=i,
            )
            for i, (sc, r) in enumerate(rescored[:top_k])
        ]


class CrossEncoderReranker:
    """
    LLM-based reranker: scores (query, passage) pairs 0-10 and reranks.
    For production: replace with a dedicated cross-encoder model
    (e.g. ms-marco-MiniLM-L-6-v2 via sentence-transformers).
    """

    def __init__(self, llm_provider: Any, batch_size: int = 5) -> None:
        self._llm = llm_provider
        self._batch_size = batch_size

    async def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        if not results:
            return []
        scored: list[tuple[float, RetrievalResult]] = []
        for result in results:
            score = await self._score_pair(query, result.chunk.content)
            scored.append((score, result))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievalResult(
                chunk=r.chunk,
                score=round(sc / 10.0, 3),
                retrieval_method="reranked",
                rank=i,
            )
            for i, (sc, r) in enumerate(scored[:top_k])
        ]

    async def _score_pair(self, query: str, passage: str) -> float:
        import re
        prompt = (
            f"Rate how relevant this passage is for answering the query.\n"
            f"Query: {query}\n"
            f"Passage: {passage[:400]}\n\n"
            f"Respond with ONLY a number from 0 to 10. "
            f"0=completely irrelevant, 10=perfectly answers the query."
        )
        try:
            resp = await self._llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=4,
            )
            nums = re.findall(r'\d+(?:\.\d+)?', resp.content)
            return float(nums[0]) if nums else 5.0
        except Exception:
            return 5.0


# ── HybridRetriever ───────────────────────────────────────────────────────────


class HybridRetriever:
    """
    Complete RAG pipeline: one method for ingestion, one for retrieval.

    Lifecycle
    ---------
        retriever = HybridRetriever(embedding_provider=OpenAIEmbeddingProvider())
        await retriever.ingest(content="...", document_id="doc1", metadata={...})
        results  = await retriever.retrieve(RetrievalRequest(query="...", top_k=5))
        block    = HybridRetriever.format_context(results)

    Ingest once at startup; query many times per request.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        chunker: TextChunker | None = None,
        reranker: Any | None = None,           # Reranker Protocol
        chunk_size: int = 400,
        overlap: int = 80,
    ) -> None:
        self._embedder = embedding_provider
        self._chunker = chunker or TextChunker(chunk_size=chunk_size, overlap=overlap)
        self._reranker: Any = reranker or SimpleReranker()
        self._vector_index = VectorIndex()
        self._bm25_index = BM25Index()
        self._all_chunks: list[Chunk] = []

    # ── Ingestion ──────────────────────────────────────────────────────

    async def ingest(
        self,
        content: str,
        document_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Chunk → embed (batched) → index.
        Returns number of chunks indexed.
        """
        chunks = self._chunker.chunk_document(content, document_id, metadata)
        if not chunks:
            log.warning("ingest: no chunks produced for document_id=%s", document_id)
            return 0

        texts = [c.content for c in chunks]
        embeddings = await self._embedder.embed_texts(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        await self._vector_index.upsert(chunks)
        self._all_chunks.extend(chunks)
        self._bm25_index.index(self._all_chunks)

        log.info("ingest: document_id=%s → %d chunks", document_id, len(chunks))
        return len(chunks)

    async def delete_document(self, document_id: str) -> int:
        removed = await self._vector_index.delete_document(document_id)
        self._all_chunks = [c for c in self._all_chunks if c.document_id != document_id]
        self._bm25_index.index(self._all_chunks)
        return removed

    # ── Retrieval ──────────────────────────────────────────────────────

    async def retrieve(self, request: RetrievalRequest) -> list[RetrievalResult]:
        """Execute full pipeline: embed query → search → fuse → rerank."""
        if await self._vector_index.count() == 0:
            return []

        query_embedding = await self._embedder.embed_query(request.query)
        over_fetch = request.top_k * 3

        dense_results: list[RetrievalResult] = []
        sparse_results: list[RetrievalResult] = []

        if request.search_mode in ("dense", "hybrid"):
            dense_results = await self._vector_index.search(
                query_embedding,
                top_k=over_fetch,
                score_threshold=request.score_threshold,
                metadata_filter=request.metadata_filter,
            )

        if request.search_mode in ("sparse", "hybrid"):
            sparse_results = self._bm25_index.search(
                request.query,
                top_k=over_fetch,
                score_threshold=request.score_threshold,
                metadata_filter=request.metadata_filter,
            )

        if request.search_mode == "hybrid":
            fused = _reciprocal_rank_fusion(
                [dense_results, sparse_results],
                weights=[0.6, 0.4],
            )
        elif request.search_mode == "dense":
            fused = dense_results
        else:
            fused = sparse_results

        if request.rerank and fused:
            fused = await self._reranker.rerank(request.query, fused, request.top_k)
        else:
            fused = fused[:request.top_k]

        for i, r in enumerate(fused):
            r.rank = i
        return fused

    # ── Formatting ─────────────────────────────────────────────────────

    @staticmethod
    def format_context(
        results: list[RetrievalResult],
        max_chars: int = 4000,
        include_scores: bool = False,
    ) -> str:
        """
        Render results into a context block for LLM injection.
        Truncates at max_chars to protect the context window.
        """
        if not results:
            return ""
        parts: list[str] = []
        total = 0
        for r in results:
            block = r.to_context_string()
            if include_scores:
                block += f" [score={r.score:.3f}]"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n---\n\n".join(parts)
