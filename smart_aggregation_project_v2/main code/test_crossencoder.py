"""
Tests for Cross-Encoder Reranking
==================================

Verifies that:
- Cross-encoder loads and produces scores
- Relevant pairs score higher than irrelevant ones
- Score outputs are valid floats
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


@pytest.fixture(scope="module")
def cross_encoder():
    """Load cross-encoder once for all tests in this module."""
    from embeddings import CrossEncoderReranker
    return CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


@pytest.fixture
def sample_pairs():
    """Sample query-document pairs for testing."""
    return [
        ("What is 3M revenue?", "3M's revenue is 30 billion dollars."),
        ("What is 3M revenue?", "Apples are delicious."),
        ("What is 3M revenue?", "3M is a great company."),
    ]


class TestCrossEncoder:
    """Cross-encoder scoring tests."""

    def test_predict_returns_scores(self, cross_encoder, sample_pairs):
        """Scores should be returned for each pair."""
        scores = cross_encoder.predict(sample_pairs)
        assert len(scores) == len(sample_pairs)

    def test_scores_are_floats(self, cross_encoder, sample_pairs):
        """Each score should be a valid number."""
        scores = cross_encoder.predict(sample_pairs)
        for score in scores:
            assert np.isfinite(float(score)), f"Score {score} is not finite"

    def test_relevant_pair_scores_highest(self, cross_encoder, sample_pairs):
        """The directly relevant pair should score highest."""
        scores = cross_encoder.predict(sample_pairs)
        # "3M's revenue is 30 billion dollars." should be most relevant
        assert scores[0] == max(scores), (
            f"Expected pair 0 (relevant) to score highest, "
            f"but scores were: {scores}"
        )

    def test_irrelevant_pair_scores_lowest(self, cross_encoder, sample_pairs):
        """The irrelevant pair should score lowest."""
        scores = cross_encoder.predict(sample_pairs)
        # "Apples are delicious." should be least relevant
        assert scores[1] == min(scores), (
            f"Expected pair 1 (irrelevant) to score lowest, "
            f"but scores were: {scores}"
        )
