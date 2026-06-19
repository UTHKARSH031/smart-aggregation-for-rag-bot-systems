"""
Data Models for Smart Aggregation
==================================

Shared data classes used across the pipeline.
Extracted to avoid circular imports between embeddings.py and smart_aggregation.py.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Chunk:
    """Single document chunk"""
    chunk_id: str
    text: str
    doc_id: str
    strategy: str
    tokens: int
    embedding: np.ndarray = None


@dataclass
class RetrievalResult:
    """Chunk with retrieval score"""
    chunk: Chunk
    score: float
    rank: int = 0
