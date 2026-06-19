"""
Embedding and Vector Storage
==============================

Handles:
1. Converting text to vectors (embeddings)
2. Storing vectors for fast retrieval (FAISS)
"""

import numpy as np
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

class Embedder:
    """
    Converts text to numerical vectors
    Uses sentence-transformers library
    """
    
    def __init__(self, model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"):
        """
        Args:
            model_name: Which embedding model to use
            
        Popular models:
        - "multi-qa-mpnet-base-dot-v1" - Original paper used this (768 dim)
        - "all-MiniLM-L6-v2" - Faster, smaller (384 dim)
        - "all-mpnet-base-v2" - Good quality (768 dim)
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[OK] Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text (for queries)"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Embed multiple texts efficiently
        
        Args:
            texts: List of texts to embed
            batch_size: Process N texts at once (faster than one-by-one)
            show_progress: Show progress bar
            
        Returns:
            Array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings

class CrossEncoderReranker:
    """
    Cross-Encoder for reranking
    
    Difference from bi-encoder:
    - Bi-encoder: Embeds query and chunks separately, compares vectors
    - Cross-encoder: Sees query + chunk TOGETHER, more accurate
    
    Trade-off: Slower but better quality
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            model_name: Cross-encoder model
            
        Popular models:
        - "cross-encoder/ms-marco-MiniLM-L-6-v2" - Fast, good (default)
        - "cross-encoder/ms-marco-electra-base" - Slower, better
        """
        print(f"Loading cross-encoder: {model_name}...")
        self.model = CrossEncoder(model_name)
        print(f"[OK] Cross-encoder loaded")
    
    def predict(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """
        Score query-chunk pairs
        
        Args:
            pairs: List of (query, chunk_text) tuples
            
        Returns:
            Array of relevance scores (0-1)
        """
        scores = self.model.predict(pairs, show_progress_bar=False)
        return scores

class VectorStore:
    """
    Vector database using FAISS
    
    Stores chunk embeddings and enables fast similarity search.
    FAISS = Facebook AI Similarity Search (industry standard)
    """
    
    def __init__(self, embedding_dim: int):
        """
        Args:
            embedding_dim: Dimension of embeddings (usually 768)
        """
        self.embedding_dim = embedding_dim
        
        self.index = faiss.IndexFlatIP(embedding_dim)                      
        
        self.chunks = []                         
        self.id_to_idx = {}                         
        
        print(f"[OK] Vector store initialized (dim={embedding_dim})")
    
    def add_chunks(self, chunks: List[dict], embeddings: np.ndarray):
        """
        Add chunks to the vector store
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: Numpy array of shape (len(chunks), embedding_dim)
        """
                                                    
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / (norms + 1e-8)
        
        self.index.add(embeddings_normalized.astype('float32'))
        
        start_idx = len(self.chunks)
        for i, chunk_dict in enumerate(chunks):
                                    
            chunk_dict['embedding'] = embeddings_normalized[i]
            self.chunks.append(chunk_dict)
            self.id_to_idx[chunk_dict['chunk_id']] = start_idx + i
        
        print(f"[OK] Added {len(chunks)} chunks. Total: {len(self.chunks)}")
    
    def _build_result(self, idx: int, dist: float, rank: int):
        """Build a RetrievalResult from a chunk index, distance, and rank."""
        from models import RetrievalResult, Chunk

        chunk_dict = self.chunks[idx]
        chunk = Chunk(
            chunk_id=chunk_dict['chunk_id'],
            text=chunk_dict['text'],
            doc_id=chunk_dict['doc_id'],
            strategy=chunk_dict['strategy'],
            tokens=chunk_dict['tokens'],
            embedding=chunk_dict['embedding']
        )
        return RetrievalResult(chunk=chunk, score=float(dist), rank=rank)

    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 10,
        strategy_filter: Optional[str] = None
    ) -> List:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            strategy_filter: Only search chunks from this strategy
            
        Returns:
            List of RetrievalResult objects
        """
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_norm = query_norm.reshape(1, -1).astype('float32')
        
        if strategy_filter:
            strategy_indices = [
                i for i, c in enumerate(self.chunks) 
                if c['strategy'] == strategy_filter
            ]
            
            if not strategy_indices:
                return []
            
            strategy_embeddings = np.array([
                self.chunks[i]['embedding'] for i in strategy_indices
            ]).astype('float32')
            
            subset_index = faiss.IndexFlatIP(self.embedding_dim)
            subset_index.add(strategy_embeddings)
            distances, indices = subset_index.search(query_norm, k)
            
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(strategy_indices):
                    original_idx = strategy_indices[idx]
                    results.append(self._build_result(original_idx, dist, i))
        else:
            distances, indices = self.index.search(query_norm, k)
            
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.chunks):
                    results.append(self._build_result(idx, dist, i))
        
        return results
    
    def get_stats(self) -> dict:
        """Get statistics about stored chunks"""
        strategy_counts = {}
        for chunk in self.chunks:
            strategy = chunk['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            'total_chunks': len(self.chunks),
            'embedding_dim': self.embedding_dim,
            'by_strategy': strategy_counts
        }
