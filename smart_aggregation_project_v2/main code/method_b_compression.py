"""
Method B: Query-Aware Compression (Content-Based Aggregation)
=============================================================

Paradigm: COMPRESSION -- Instead of selecting chunks from a candidate pool,
use an LLM to extract only the query-relevant sentences from each retrieved
chunk and pack the distilled content into the final context window.

Pipeline:
  Stage 1:  Multi-strategy dense retrieval  ->  120 candidates  (same as Method A)
  Stage 2:  Cross-encoder reranking         ->  top-20 candidates
  Stage 3:  Per-chunk LLM compression       ->  extract relevant sentences only
  Stage 4:  Budget-aware packing            ->  fill up to token_budget tokens

Key differences vs. Method A (Smart Aggregation):
  - Keeps MORE source passages in context (up to budget) rather than exactly k=10
  - Token budget is filled with compressed extracts, not whole chunks
  - Adds LLM latency but reduces redundancy and irrelevant text
  - Compression ratio per chunk is query-dependent, not fixed

References:
  [1] Lewis et al., RAG (NeurIPS 2020)
  [2] Jimeno Yepes et al., arXiv:2402.05131 (2024)
  [3] Nogueira & Cho, arXiv:1901.04085 (2019)

Dependencies:
  pip install sentence-transformers faiss-cpu transformers torch
  pip install anthropic   # for LLM compression; swap for any chat API
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Reuse shared data models from the project
# ---------------------------------------------------------------------------
from models import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Compression result dataclass
# ---------------------------------------------------------------------------
@dataclass
class CompressedChunk:
    """A chunk that has been compressed to query-relevant content only."""
    source_chunk_id: str
    doc_id: str
    original_tokens: int
    compressed_text: str
    compressed_tokens: int
    compression_ratio: float          # original / compressed
    relevance_score: float            # cross-encoder score before compression


# ---------------------------------------------------------------------------
# LLM Compressor (pluggable backend)
# ---------------------------------------------------------------------------
class LLMCompressor:
    """
    Compresses a chunk to only the sentences relevant to the query.

    Backend is swappable:
      - 'anthropic'   : uses claude-haiku via Anthropic SDK (fast + cheap)
      - 'hf'          : uses a local HuggingFace seq2seq model (offline)
      - 'extractive'  : sentence-level cosine similarity (no API needed)
    """

    SYSTEM_PROMPT = (
        "You are a precise information extractor. "
        "Given a document chunk and a question, output ONLY the sentences "
        "from the chunk that are directly relevant to answering the question. "
        "Copy sentences verbatim. Do not add commentary. "
        "If no sentence is relevant, output: NO_RELEVANT_CONTENT"
    )

    def __init__(self, backend: str = "extractive", model: str = None, embedder=None):
        """
        Args:
            backend:  'anthropic' | 'hf' | 'extractive'
            model:    model name/path (backend-specific)
            embedder: a SentenceTransformer / project Embedder instance
                      (required for 'extractive'; used via .encode() or .embed_batch())
        """
        self.backend = backend
        self.model_name = model
        self.embedder = embedder
        self._client = None
        self._hf_pipe = None
        self._setup()

    def _setup(self):
        if self.backend == "anthropic":
            import anthropic  # pip install anthropic
            self._client = anthropic.Anthropic()
            if self.model_name is None:
                self.model_name = "claude-haiku-4-5-20251001"

        elif self.backend == "hf":
            from transformers import pipeline
            model = self.model_name or "facebook/bart-large-cnn"
            self._hf_pipe = pipeline(
                "summarization", model=model,
                max_length=200, min_length=20, truncation=True
            )

        elif self.backend == "extractive":
            # Uses cosine similarity between query embedding and sentence embeddings.
            # If no embedder is provided at construction time we defer setup until
            # compress() is first called, so the caller can inject one later.
            pass
        else:
            raise ValueError(
                f"Unknown backend: {self.backend!r}. "
                "Choose 'anthropic', 'hf', or 'extractive'."
            )

    def _get_embedder(self):
        """Return embedder, lazy-loading a default if none was provided."""
        if self.embedder is not None:
            return self.embedder
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self.embedder

    def compress(self, query: str, chunk_text: str,
                 relevance_threshold: float = 0.45) -> str:
        """
        Return only the query-relevant portion of chunk_text.

        Args:
            query:                The user question.
            chunk_text:           Full chunk content.
            relevance_threshold:  (extractive only) min cosine sim to keep a sentence.

        Returns:
            Compressed text (may be empty string if nothing is relevant).
        """
        if self.backend == "anthropic":
            return self._compress_anthropic(query, chunk_text)
        elif self.backend == "hf":
            return self._compress_hf(query, chunk_text)
        else:
            return self._compress_extractive(query, chunk_text, relevance_threshold)

    def _compress_anthropic(self, query: str, chunk_text: str) -> str:
        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=512,
            system=self.SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Question: {query}\n\nChunk:\n{chunk_text}"
            }]
        )
        result = response.content[0].text.strip()
        return "" if result == "NO_RELEVANT_CONTENT" else result

    def _compress_hf(self, query: str, chunk_text: str) -> str:
        input_text = f"Question: {query} | Context: {chunk_text}"
        result = self._hf_pipe(input_text)[0]["summary_text"]
        return result.strip()

    def _compress_extractive(self, query: str, chunk_text: str,
                              threshold: float) -> str:
        """
        Sentence-level extractive compression using embedding cosine similarity.
        No API calls required -- runs entirely with the bi-encoder.
        """
        sentences = re.split(r'(?<=[.!?])\s+', chunk_text.strip())
        sentences = [s.strip() for s in sentences if len(s.split()) >= 4]
        if not sentences:
            return chunk_text  # unsplittable: return as-is

        emb = self._get_embedder()

        # Support both project Embedder (embed_batch) and SentenceTransformer (encode)
        if hasattr(emb, 'encode'):
            query_emb = emb.encode([query], normalize_embeddings=True)
            sent_embs = emb.encode(sentences, normalize_embeddings=True)
        else:
            # Project Embedder -- use embed_batch
            query_emb = emb.embed_batch([query])
            nq = np.linalg.norm(query_emb, axis=1, keepdims=True)
            query_emb = query_emb / (nq + 1e-8)
            sent_embs = emb.embed_batch(sentences)
            ns = np.linalg.norm(sent_embs, axis=1, keepdims=True)
            sent_embs = sent_embs / (ns + 1e-8)

        similarities = (query_emb @ sent_embs.T).flatten()
        relevant = [s for s, sim in zip(sentences, similarities) if sim >= threshold]
        return " ".join(relevant)


# ---------------------------------------------------------------------------
# Main Method B class
# ---------------------------------------------------------------------------
class QueryAwareCompression:
    """
    Method B: Compression-based token-constrained aggregation for RAG.

    After retrieval and reranking, each surviving chunk is compressed by an LLM
    to retain only query-relevant sentences.  Compressed extracts are greedily
    packed into a token budget (default 4 096 tokens) in score order.

    This implements the COMPRESSION paradigm for the three-paradigm comparison:
      Method A -- Selection (Smart Aggregation / MMR)
      Method B -- Compression (this class)
      Method C -- Clustering (method_c_clustering.py)
    """

    DEFAULT_CONFIG = {
        "top_k_per_strategy":   30,    # chunks retrieved per strategy (Stage 1)
        "rerank_top_k":         20,    # candidates passed to compressor (Stage 2)
        "token_budget":       4096,    # max tokens in final context (Stage 4)
        "min_compressed_tokens": 10,   # discard compressions shorter than this
        "compression_threshold": 0.45, # extractive similarity threshold
    }

    def __init__(
        self,
        embedder,
        vector_store,
        cross_encoder=None,
        compressor: Optional[LLMCompressor] = None,
        config: Optional[Dict] = None,
    ):
        """
        Args:
            embedder:      Project Embedder bi-encoder (shared with Method A).
            vector_store:  Project VectorStore with per-chunk strategy labels.
            cross_encoder: Project CrossEncoderReranker (optional; falls back
                           to bi-encoder score ordering if None).
            compressor:    LLMCompressor instance.  If None, defaults to
                           'extractive' mode using the same embedder.
            config:        Override any key in DEFAULT_CONFIG.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.cross_encoder = cross_encoder
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        if compressor is None:
            self.compressor = LLMCompressor(backend="extractive", embedder=embedder)
        else:
            self.compressor = compressor

        self.stats: Dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(self, query: str, verbose: bool = True) -> List[CompressedChunk]:
        """
        Run the full compression pipeline and return packed compressed chunks.

        Returns:
            List of CompressedChunk objects whose combined token count
            does not exceed config['token_budget'].
        """
        if verbose:
            print(f"\n{'='*70}")
            print("METHOD B -- QUERY-AWARE COMPRESSION PIPELINE")
            print(f"{'='*70}")
            print(f"Query: {query}\n")

        total_start = time.time()

        # Stage 1: multi-strategy retrieval
        t = time.time()
        candidates = self._stage1_retrieval(query)
        self.stats["stage1_time"] = time.time() - t
        if verbose:
            print(f"[OK] Stage 1: Multi-Strategy Retrieval -> {len(candidates)} candidates "
                  f"({self.stats['stage1_time']:.2f}s)")

        # Stage 2: cross-encoder reranking
        t = time.time()
        reranked = self._stage2_reranking(query, candidates)
        self.stats["stage2_time"] = time.time() - t
        if verbose:
            print(f"[OK] Stage 2: Cross-Encoder Reranking -> top {len(reranked)} "
                  f"({self.stats['stage2_time']:.2f}s)")

        # Stage 3: per-chunk LLM compression
        t = time.time()
        compressed = self._stage3_compress(query, reranked)
        self.stats["stage3_time"] = time.time() - t
        if verbose:
            avg_ratio = (
                sum(c.compression_ratio for c in compressed) / len(compressed)
                if compressed else 0.0
            )
            print(f"[OK] Stage 3: LLM Compression -> {len(compressed)} non-empty extracts "
                  f"(avg ratio {avg_ratio:.1f}x) ({self.stats['stage3_time']:.2f}s)")

        # Stage 4: budget-aware greedy packing
        t = time.time()
        packed = self._stage4_pack(compressed)
        self.stats["stage4_time"] = time.time() - t
        total_tokens = sum(c.compressed_tokens for c in packed)
        if verbose:
            print(f"[OK] Stage 4: Budget Packing -> {len(packed)} extracts, "
                  f"{total_tokens} tokens ({self.stats['stage4_time']:.2f}s)")
            print(f"\n{'='*70}")
            print(f"COMPLETE in {time.time() - total_start:.2f}s | "
                  f"Docs covered: {len({c.doc_id for c in packed})}")
            print(f"{'='*70}\n")

        self.stats["total_time"] = time.time() - total_start
        self.stats["output_tokens"] = total_tokens
        self.stats["n_chunks"] = len(packed)
        return packed

    def get_context_string(self, packed: List[CompressedChunk]) -> str:
        """Concatenate compressed extracts into a single LLM context string."""
        parts = []
        for i, cc in enumerate(packed, 1):
            parts.append(
                f"[{i}] (doc={cc.doc_id}, score={cc.relevance_score:.3f})\n"
                f"{cc.compressed_text}"
            )
        return "\n\n".join(parts)

    def get_statistics(self) -> Dict:
        return self.stats.copy()

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------
    def _stage1_retrieval(self, query: str) -> List[RetrievalResult]:
        """Same as Method A Stage 1: retrieve top-K per strategy."""
        query_emb = self.embedder.embed_single(query)
        strategies = ["fixed-128", "fixed-256", "fixed-512", "element-based"]
        all_results: List[RetrievalResult] = []
        for strategy in strategies:
            results = self.vector_store.search(
                query_emb,
                k=self.config["top_k_per_strategy"],
                strategy_filter=strategy,
            )
            all_results.extend(results)
        return all_results

    def _stage2_reranking(
        self, query: str, candidates: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Cross-encoder reranking; falls back to bi-encoder score if no CE."""
        if self.cross_encoder is None:
            ranked = sorted(candidates, key=lambda x: x.score, reverse=True)
            return ranked[: self.config["rerank_top_k"]]

        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.cross_encoder.predict(pairs)
        for result, score in zip(candidates, scores):
            result.score = float(score)

        ranked = sorted(candidates, key=lambda x: x.score, reverse=True)
        return ranked[: self.config["rerank_top_k"]]

    def _stage3_compress(
        self, query: str, reranked: List[RetrievalResult]
    ) -> List[CompressedChunk]:
        """
        For each reranked chunk, call the compressor and build a CompressedChunk.
        Chunks that compress to nothing are discarded.
        """
        compressed_chunks: List[CompressedChunk] = []
        threshold = self.config["compression_threshold"]

        for result in reranked:
            chunk = result.chunk
            compressed_text = self.compressor.compress(
                query, chunk.text, relevance_threshold=threshold
            )
            if not compressed_text.strip():
                continue

            compressed_tokens = len(compressed_text.split())
            original_tokens = chunk.tokens or len(chunk.text.split())

            if compressed_tokens < self.config["min_compressed_tokens"]:
                continue

            ratio = original_tokens / max(compressed_tokens, 1)
            compressed_chunks.append(
                CompressedChunk(
                    source_chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    original_tokens=original_tokens,
                    compressed_text=compressed_text,
                    compressed_tokens=compressed_tokens,
                    compression_ratio=ratio,
                    relevance_score=result.score,
                )
            )
        return compressed_chunks

    def _stage4_pack(self, compressed: List[CompressedChunk]) -> List[CompressedChunk]:
        """
        Greedy budget packing: add compressed extracts in descending relevance
        order until the token budget is exhausted.
        """
        budget = self.config["token_budget"]
        packed: List[CompressedChunk] = []
        used_tokens = 0

        ordered = sorted(compressed, key=lambda c: c.relevance_score, reverse=True)

        for cc in ordered:
            if used_tokens + cc.compressed_tokens > budget:
                continue  # try next (smaller) extract
            packed.append(cc)
            used_tokens += cc.compressed_tokens

        return packed


# ---------------------------------------------------------------------------
# Evaluation helper (mirrors Method A evaluation interface)
# ---------------------------------------------------------------------------
def evaluate_method_b(
    method_b: QueryAwareCompression,
    queries: List[str],
    ground_truth_doc_ids: List[str],
    verbose: bool = False,
) -> Dict:
    """
    Evaluate Method B on a list of queries.

    A query is correct if the ground-truth doc_id appears in any returned
    CompressedChunk (mirrors Method A's document-level accuracy metric).

    Returns:
        dict with keys: accuracy, avg_output_tokens, avg_n_chunks, avg_latency_s
    """
    assert len(queries) == len(ground_truth_doc_ids)

    correct = 0
    total_tokens = 0
    total_chunks = 0
    total_time = 0.0

    for i, (query, gt_doc_id) in enumerate(zip(queries, ground_truth_doc_ids)):
        packed = method_b.retrieve(query, verbose=False)
        stats = method_b.get_statistics()

        retrieved_docs = {cc.doc_id for cc in packed}
        if gt_doc_id in retrieved_docs:
            correct += 1

        total_tokens += stats.get("output_tokens", 0)
        total_chunks += stats.get("n_chunks", 0)
        total_time += stats.get("total_time", 0.0)

        if verbose:
            status = "[OK]" if gt_doc_id in retrieved_docs else "[X]"
            print(f"[{i+1:3d}] {status}  docs={retrieved_docs}  "
                  f"tokens={stats.get('output_tokens', 0)}")

    n = len(queries)
    return {
        "accuracy": correct / n,
        "avg_output_tokens": total_tokens / n,
        "avg_n_chunks": total_chunks / n,
        "avg_latency_s": total_time / n,
        "n_queries": n,
        "n_correct": correct,
    }
