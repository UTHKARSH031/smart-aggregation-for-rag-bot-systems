"""
Smart Aggregation for RAG Systems - CORE IMPLEMENTATION
========================================================

This is THE innovation: Successfully implementing multi-strategy aggregation
that the original paper (Jimeno et al., 2024) abandoned due to token limits.

4-Step Pipeline:
1. Multi-Strategy Retrieval (120 chunks)
2. Deduplication (70 unique chunks)
3. Cross-Encoder Reranking (20 best chunks)
4. Diversity Selection via MMR (10 final chunks)
"""

import numpy as np
from typing import List, Dict, Tuple
import time

from models import Chunk, RetrievalResult

class SmartAggregation:
    """
    Smart Aggregation Pipeline - The Core Innovation
    
    Solves: Original paper achieved 84.4% with aggregation but abandoned it
    Our Solution: Filter 120+ chunks -> 10 perfect chunks intelligently
    """
    
    def __init__(
        self,
        embedder,
        vector_store,
        cross_encoder=None,
        config: Dict = None
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.cross_encoder = cross_encoder
        
        self.config = {
            'top_k_per_strategy': 30,
            'dedup_threshold': 0.85,
            'rerank_top_k': 20,
            'mmr_lambda': 0.8,
            'final_k': 10
        }
        if config:
            self.config.update(config)
        
        self.stats = {}
    
    def retrieve(self, query: str, verbose: bool = True) -> List[Chunk]:
        """
        Main method - Run the full 4-step pipeline
        
        Args:
            query: User question
            verbose: Print progress
            
        Returns:
            List of 10 final chunks
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"SMART AGGREGATION PIPELINE")
            print(f"{'='*70}")
            print(f"Query: {query}\n")
        
        start_time = time.time()
        
        t1 = time.time()
        candidates = self._step1_multi_strategy_retrieval(query)
        self.stats['step1_time'] = time.time() - t1
        if verbose:
            print(f"[OK] Step 1: Multi-Strategy Retrieval -> {len(candidates)} candidates ({self.stats['step1_time']:.2f}s)")
        
        t2 = time.time()
        unique = self._step2_deduplication(candidates)
        self.stats['step2_time'] = time.time() - t2
        if verbose:
            removed = len(candidates) - len(unique)
            print(f"[OK] Step 2: Deduplication -> {len(unique)} unique (removed {removed}) ({self.stats['step2_time']:.2f}s)")
        
        t3 = time.time()
        reranked = self._step3_reranking(query, unique)
        self.stats['step3_time'] = time.time() - t3
        if verbose:
            print(f"[OK] Step 3: Cross-Encoder Reranking -> Top {len(reranked)} ({self.stats['step3_time']:.2f}s)")
        
        t4 = time.time()
        final = self._step4_diversity_selection(query, reranked)
        self.stats['step4_time'] = time.time() - t4
        
        self.stats['total_time'] = time.time() - start_time
        
        if verbose:
            print(f"[OK] Step 4: Diversity Selection (MMR) -> {len(final)} final chunks ({self.stats['step4_time']:.2f}s)")
            print(f"\n{'='*70}")
            print(f"COMPLETE: {len(final)} chunks in {self.stats['total_time']:.2f}s")
            print(f"{'='*70}\n")
        
        return final
    
    def _step1_multi_strategy_retrieval(self, query: str) -> List[RetrievalResult]:
        """
        STEP 1: Multi-Strategy Retrieval
        
        Retrieve top-K chunks from EACH chunking strategy.
        This gives us diverse perspectives on what's relevant.
        """
                     
        query_emb = self.embedder.embed_single(query)
        
        strategies = ["fixed-128", "fixed-256", "fixed-512", "element-based"]
        all_results = []
        
        for strategy in strategies:
            results = self.vector_store.search(
                query_emb,
                k=self.config['top_k_per_strategy'],
                strategy_filter=strategy
            )
            all_results.extend(results)
        
        return all_results
    
    def _step2_deduplication(self, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        STEP 2: Deduplication
        
        Remove chunks with >85% embedding cosine similarity.
        Keeps the one with higher retrieval score.
        """
        if len(candidates) <= 1:
            return candidates
        
        embeddings = np.array([c.chunk.embedding for c in candidates])
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / (norms + 1e-8)
        
        similarity_matrix = embeddings_norm @ embeddings_norm.T
        
        to_keep = set(range(len(candidates)))
        
        for i in range(len(candidates)):
            if i not in to_keep:
                continue
            for j in range(i + 1, len(candidates)):
                if j not in to_keep:
                    continue
                
                if similarity_matrix[i, j] > self.config['dedup_threshold']:
                                                                     
                    if candidates[i].score >= candidates[j].score:
                        to_keep.discard(j)
                    else:
                        to_keep.discard(i)
                        break
        
        unique = [candidates[i] for i in sorted(to_keep)]
        return unique
    
    def _step3_reranking(self, query: str, chunks: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        STEP 3: Cross-Encoder Reranking
        
        Use a more powerful model to re-score chunks.
        Cross-encoder sees query + chunk TOGETHER (more accurate than bi-encoder).
        """
        if self.cross_encoder is None:
                                                                   
            sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
            return sorted_chunks[:self.config['rerank_top_k']]
        
        pairs = [(query, c.chunk.text) for c in chunks]
        
        rerank_scores = self.cross_encoder.predict(pairs)
        
        for i, chunk_result in enumerate(chunks):
            chunk_result.score = float(rerank_scores[i])
        
        reranked = sorted(chunks, key=lambda x: x.score, reverse=True)
        return reranked[:self.config['rerank_top_k']]
    
    def _step4_diversity_selection(
        self, 
        query: str, 
        chunks: List[RetrievalResult]
    ) -> List[Chunk]:
        """
        STEP 4: Diversity-Aware Selection (MMR)
        
        Select final K chunks that balance:
        - High relevance (good score)
        - High diversity (cover different aspects)
        
        Uses Maximal Marginal Relevance (MMR) algorithm
        (Carbonell & Goldstein, 1998).
        
        MMR Formula:
        Score = lambda × Relevance - (1 - lambda) × Max_Similarity_to_Selected
        
        Where lambda = 0.7 favours relevance, (1-lambda) = 0.3 penalises redundancy
        """
        if len(chunks) <= self.config['final_k']:
            return [c.chunk for c in chunks]
        
        embeddings = np.array([c.chunk.embedding for c in chunks])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / (norms + 1e-8)
        
        selected_indices = []
        remaining_indices = list(range(len(chunks)))
        
        first_idx = 0                               
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        lambda_param = self.config['mmr_lambda']
        
        # Normalize relevance scores to [0, 1] using Min-Max scaling
        raw_scores = np.array([c.score for c in chunks])
        if len(raw_scores) > 1 and raw_scores.max() > raw_scores.min():
            norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
        else:
            norm_scores = np.ones_like(raw_scores)
            
        while len(selected_indices) < self.config['final_k'] and remaining_indices:
            best_mmr_score = -float('inf')
            best_idx = None
            
            for idx in remaining_indices:
                # Use normalized relevance for MMR math
                relevance = norm_scores[idx]
                
                selected_embs = embeddings_norm[selected_indices]
                current_emb = embeddings_norm[idx:idx+1]
                similarities = (current_emb @ selected_embs.T).flatten()
                max_sim = similarities.max()
                
                # Standard MMR: lambda * relevance - (1-lambda) * redundancy
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        final_chunks = [chunks[i].chunk for i in selected_indices]
        return final_chunks
    
    def _get_strategy_counts(self, results: List[RetrievalResult]) -> Dict[str, int]:
        """Helper: Count chunks per strategy"""
        counts = {}
        for r in results:
            strategy = r.chunk.strategy
            counts[strategy] = counts.get(strategy, 0) + 1
        return counts
    
    def get_statistics(self) -> Dict:
        """Return timing statistics from last retrieval"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Clear statistics"""
        self.stats = {}
