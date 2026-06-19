"""
Method C: Cluster-Based Aggregation (Structure-Based Diversity)
===============================================================

Paradigm: CLUSTERING -- After multi-strategy retrieval, group the candidate
pool into K semantic clusters using KMeans, then select the highest-scoring
representative from each cluster.  This guarantees global topical diversity
(each cluster = one distinct aspect of the corpus) rather than the greedy,
local diversity provided by MMR in Method A.

Pipeline:
  Stage 1:  Multi-strategy dense retrieval  ->  120 candidates  (shared)
  Stage 2:  Greedy deduplication            ->  ~70 unique chunks (shared)
  Stage 3:  KMeans clustering               ->  K clusters
  Stage 4:  Per-cluster top-1 selection     ->  K final chunks

Key differences vs. Method A (MMR):
  - MMR is greedy: diversity emerges iteratively, depends on insertion order.
  - KMeans: partition embedding space globally before selection.
  - MMR can get trapped near a local high-score region; KMeans cannot.
  - Method C is faster at Stage 4 (O(n·K·I) KMeans vs O(n·k²) MMR).
  - Method C produces exactly K chunks (user-tunable cluster count).

References:
  [1] Lewis et al., RAG (NeurIPS 2020)
  [2] Jimeno Yepes et al., arXiv:2402.05131 (2024)
  [3] Carbonell & Goldstein, MMR (ACM SIGIR 1998)
  [4] Arthur & Vassilvitskii, k-means++ (SODA 2007)

Dependencies:
  pip install sentence-transformers faiss-cpu scikit-learn numpy
"""

from __future__ import annotations

import time
import warnings as _warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning
    _warnings.filterwarnings("ignore", category=ConvergenceWarning)
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

# ---------------------------------------------------------------------------
# Reuse shared data models from the project
# ---------------------------------------------------------------------------
from models import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Cluster selection result
# ---------------------------------------------------------------------------
@dataclass
class ClusterResult:
    """A representative chunk chosen from one KMeans cluster."""
    cluster_id: int
    chunk: Chunk
    score: float                    # cross-encoder or bi-encoder score
    cluster_size: int               # total candidates in the cluster
    centroid_distance: float        # L2 distance to cluster centroid


# ---------------------------------------------------------------------------
# Main Method C class
# ---------------------------------------------------------------------------
class ClusterBasedAggregation:
    """
    Method C: KMeans cluster-based diversity selection for RAG.

    The embedding space of retrieved candidates is partitioned into K clusters.
    The highest-scoring chunk in each cluster is selected as the representative.
    This enforces global topical coverage: if there are K=10 distinct semantic
    groups in the candidate pool, exactly one chunk from each group is returned.

    Comparison to Method A MMR:
      - MMR selects greedily -- later selections depend on what was chosen before.
      - KMeans partitions the space first, then selects -- globally optimal spread.
      - In practice, clustering tends to produce lower redundancy at the cost of
        slightly lower per-chunk relevance scores.
    """

    DEFAULT_CONFIG = {
        "top_k_per_strategy":  30,    # chunks per strategy in Stage 1
        "dedup_threshold":     0.85,  # cosine sim threshold for deduplication
        "n_clusters":          10,    # K -- number of final chunks returned
        "kmeans_init":         "k-means++",  # 'k-means++' or 'random'
        "kmeans_n_init":       10,    # number of KMeans random restarts
        "kmeans_max_iter":     300,   # max KMeans iterations
        "use_cross_encoder":   True,  # score candidates with CE before selection
        "ce_batch_size":       32,    # cross-encoder batch size
    }

    def __init__(
        self,
        embedder,
        vector_store,
        cross_encoder=None,
        config: Optional[Dict] = None,
    ):
        """
        Args:
            embedder:      Project Embedder bi-encoder (shared with Method A).
            vector_store:  Project VectorStore with strategy labels.
            cross_encoder: Optional project CrossEncoderReranker for accurate scoring.
            config:        Override any key in DEFAULT_CONFIG.
        """
        if not _SKLEARN_OK:
            raise ImportError(
                "scikit-learn is required for Method C. "
                "Install with: pip install scikit-learn"
            )
        self.embedder = embedder
        self.vector_store = vector_store
        self.cross_encoder = cross_encoder
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.stats: Dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(self, query: str, verbose: bool = True) -> List[ClusterResult]:
        """
        Run the full cluster-based pipeline and return one representative
        per cluster.

        Returns:
            List of ClusterResult objects (length == n_clusters, or fewer
            if the candidate pool is smaller than n_clusters).
        """
        if verbose:
            print(f"\n{'='*70}")
            print("METHOD C -- CLUSTER-BASED AGGREGATION PIPELINE")
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

        # Stage 2: deduplication (same greedy cosine sim as Method A)
        t = time.time()
        unique = self._stage2_dedup(candidates)
        self.stats["stage2_time"] = time.time() - t
        if verbose:
            print(f"[OK] Stage 2: Deduplication -> {len(unique)} unique "
                  f"(removed {len(candidates) - len(unique)}) "
                  f"({self.stats['stage2_time']:.2f}s)")

        # Optional: cross-encoder scoring before clustering
        if self.cross_encoder and self.config["use_cross_encoder"]:
            t = time.time()
            unique = self._score_with_cross_encoder(query, unique)
            self.stats["ce_time"] = time.time() - t
            if verbose:
                print(f"[OK] CE Scoring -> {len(unique)} chunks rescored "
                      f"({self.stats['ce_time']:.2f}s)")

        # Stage 3: KMeans clustering on embeddings
        t = time.time()
        cluster_labels, centroids = self._stage3_cluster(unique)
        self.stats["stage3_time"] = time.time() - t
        actual_k = len(np.unique(cluster_labels))
        if verbose:
            print(f"[OK] Stage 3: KMeans Clustering -> {actual_k} clusters "
                  f"({self.stats['stage3_time']:.2f}s)")

        # Stage 4: select best chunk per cluster
        t = time.time()
        representatives = self._stage4_select(unique, cluster_labels, centroids)
        self.stats["stage4_time"] = time.time() - t
        if verbose:
            print(f"[OK] Stage 4: Cluster Representatives -> {len(representatives)} chunks "
                  f"({self.stats['stage4_time']:.2f}s)")
            total_t = time.time() - total_start
            print(f"\n{'='*70}")
            print(f"COMPLETE in {total_t:.2f}s | "
                  f"Docs covered: {len({r.chunk.doc_id for r in representatives})}")
            print(f"{'='*70}\n")

        self.stats["total_time"] = time.time() - total_start
        self.stats["n_chunks"] = len(representatives)
        return representatives

    def get_statistics(self) -> Dict:
        return self.stats.copy()

    def get_chunks(self, results: List[ClusterResult]) -> List[Chunk]:
        """Convenience: extract raw Chunk objects from ClusterResult list."""
        return [r.chunk for r in results]

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------
    def _stage1_retrieval(self, query: str) -> List[RetrievalResult]:
        """Multi-strategy dense retrieval (identical to Methods A and B)."""
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

    def _stage2_dedup(self, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Greedy pairwise cosine-similarity deduplication.
        Identical to Method A Stage 2 -- keeps higher-scoring chunk when two
        chunks exceed the similarity threshold.
        """
        if len(candidates) <= 1:
            return candidates

        embeddings = np.array(
            [c.chunk.embedding for c in candidates], dtype=np.float32
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_norm = embeddings / (norms + 1e-8)
        sim = emb_norm @ emb_norm.T
        tau = self.config["dedup_threshold"]

        keep = set(range(len(candidates)))
        for i in range(len(candidates)):
            if i not in keep:
                continue
            for j in range(i + 1, len(candidates)):
                if j not in keep:
                    continue
                if sim[i, j] > tau:
                    if candidates[i].score >= candidates[j].score:
                        keep.discard(j)
                    else:
                        keep.discard(i)
                        break

        return [candidates[k] for k in sorted(keep)]

    def _score_with_cross_encoder(
        self, query: str, chunks: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Rescore all unique chunks with the cross-encoder in batches."""
        texts = [c.chunk.text for c in chunks]
        batch_size = self.config["ce_batch_size"]
        scores: List[float] = []

        for start in range(0, len(texts), batch_size):
            batch_pairs = [(query, t) for t in texts[start: start + batch_size]]
            batch_scores = self.cross_encoder.predict(batch_pairs)
            scores.extend(batch_scores.tolist())

        for result, score in zip(chunks, scores):
            result.score = float(score)

        return chunks

    def _stage3_cluster(
        self, unique: List[RetrievalResult]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        KMeans clustering on L2-normalised chunk embeddings.

        Returns:
            cluster_labels:  int array of shape (n_candidates,)
            centroids:       float array of shape (K, emb_dim)
        """
        embeddings = np.array(
            [c.chunk.embedding for c in unique], dtype=np.float32
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_norm = embeddings / (norms + 1e-8)

        K = min(self.config["n_clusters"], len(unique))

        km = KMeans(
            n_clusters=K,
            init=self.config["kmeans_init"],
            n_init=self.config["kmeans_n_init"],
            max_iter=self.config["kmeans_max_iter"],
            random_state=42,
        )
        labels = km.fit_predict(emb_norm)
        return labels, km.cluster_centers_

    def _stage4_select(
        self,
        unique: List[RetrievalResult],
        cluster_labels: np.ndarray,
        centroids: np.ndarray,
    ) -> List[ClusterResult]:
        """
        For each cluster, select the candidate with the highest cross-encoder
        (or bi-encoder) score as the cluster representative.
        """
        embeddings = np.array(
            [c.chunk.embedding for c in unique], dtype=np.float32
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_norm = embeddings / (norms + 1e-8)

        unique_labels = np.unique(cluster_labels)
        representatives: List[ClusterResult] = []

        for label in unique_labels:
            indices = np.where(cluster_labels == label)[0]
            cluster_candidates = [(i, unique[i]) for i in indices]
            best_i, best_result = max(cluster_candidates, key=lambda x: x[1].score)

            # Centroid distance for diagnostics
            centroid = centroids[label]
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
            dist = float(np.linalg.norm(emb_norm[best_i] - centroid_norm))

            representatives.append(
                ClusterResult(
                    cluster_id=int(label),
                    chunk=best_result.chunk,
                    score=best_result.score,
                    cluster_size=len(indices),
                    centroid_distance=dist,
                )
            )

        representatives.sort(key=lambda r: r.score, reverse=True)
        return representatives


# ---------------------------------------------------------------------------
# Cluster diversity analyser
# ---------------------------------------------------------------------------
def compute_inter_cluster_diversity(results: List[ClusterResult]) -> Dict:
    """
    Compute pairwise cosine similarity statistics among selected representatives.
    Lower avg similarity -> higher diversity.

    Returns dict with keys: mean_sim, max_sim, min_sim, std_sim
    """
    if len(results) < 2:
        return {"mean_sim": 0.0, "max_sim": 0.0, "min_sim": 0.0, "std_sim": 0.0}

    embs = np.array([r.chunk.embedding for r in results], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_norm = embs / (norms + 1e-8)
    sim_matrix = embs_norm @ embs_norm.T

    n = len(results)
    triu = np.array([sim_matrix[i, j] for i in range(n) for j in range(i + 1, n)])

    return {
        "mean_sim": float(triu.mean()),
        "max_sim":  float(triu.max()),
        "min_sim":  float(triu.min()),
        "std_sim":  float(triu.std()),
    }


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_method_c(
    method_c: ClusterBasedAggregation,
    queries: List[str],
    ground_truth_doc_ids: List[str],
    verbose: bool = False,
) -> Dict:
    """
    Evaluate Method C on a query list using document-level retrieval accuracy.

    Returns:
        dict: accuracy, avg_latency_s, avg_inter_chunk_similarity,
              avg_cluster_size_std, n_queries, n_correct
    """
    assert len(queries) == len(ground_truth_doc_ids)

    correct = 0
    total_time = 0.0
    total_diversity = 0.0
    total_cluster_size_std = 0.0

    for i, (query, gt_doc_id) in enumerate(zip(queries, ground_truth_doc_ids)):
        results = method_c.retrieve(query, verbose=False)
        stats = method_c.get_statistics()

        retrieved_docs = {r.chunk.doc_id for r in results}
        if gt_doc_id in retrieved_docs:
            correct += 1

        diversity = compute_inter_cluster_diversity(results)
        total_diversity += diversity["mean_sim"]

        cluster_sizes = np.array([r.cluster_size for r in results], dtype=float)
        total_cluster_size_std += (
            float(cluster_sizes.std()) if len(cluster_sizes) > 1 else 0.0
        )
        total_time += stats.get("total_time", 0.0)

        if verbose:
            status = "[OK]" if gt_doc_id in retrieved_docs else "[X]"
            print(
                f"[{i+1:3d}] {status}  "
                f"docs={retrieved_docs}  "
                f"mean_sim={diversity['mean_sim']:.3f}  "
                f"latency={stats.get('total_time', 0):.2f}s"
            )

    n = len(queries)
    return {
        "accuracy": correct / n,
        "avg_latency_s": total_time / n,
        "avg_inter_chunk_similarity": total_diversity / n,
        "avg_cluster_size_std": total_cluster_size_std / n,
        "n_queries": n,
        "n_correct": correct,
    }
