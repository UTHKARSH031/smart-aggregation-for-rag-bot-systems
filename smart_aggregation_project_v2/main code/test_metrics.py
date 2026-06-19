"""
Tests for Evaluation Metrics
==============================

Verifies that:
- Retrieval metrics compute correctly
- Answer metrics handle edge cases
- NDCG, MAP, precision, recall, F1 are numerically correct
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from evaluation import RAGEvaluator


@pytest.fixture
def evaluator():
    return RAGEvaluator()


class TestRetrievalMetrics:
    """Tests for retrieval evaluation."""

    def test_perfect_retrieval(self, evaluator):
        """All retrieved docs are relevant."""

        class MockChunk:
            def __init__(self, doc_id):
                self.doc_id = doc_id

        retrieved = [MockChunk("doc1"), MockChunk("doc1"), MockChunk("doc1"),
                     MockChunk("doc1"), MockChunk("doc1")]
        ground_truth = ["doc1"]
        metrics = evaluator.evaluate_retrieval(retrieved, ground_truth, k_values=[5])

        assert metrics["precision@5"] == 1.0
        assert metrics["recall@5"] == 1.0
        assert metrics["ndcg@5"] == 1.0

    def test_no_relevant_docs(self, evaluator):
        """No retrieved docs are relevant."""

        class MockChunk:
            def __init__(self, doc_id):
                self.doc_id = doc_id

        retrieved = [MockChunk("doc2"), MockChunk("doc3"), MockChunk("doc4"),
                     MockChunk("doc5"), MockChunk("doc6")]
        ground_truth = ["doc1"]
        metrics = evaluator.evaluate_retrieval(retrieved, ground_truth, k_values=[5])

        assert metrics["precision@5"] == 0.0
        assert metrics["recall@5"] == 0.0
        assert metrics["ndcg@5"] == 0.0

    def test_partial_retrieval(self, evaluator):
        """Some retrieved docs are relevant."""

        class MockChunk:
            def __init__(self, doc_id):
                self.doc_id = doc_id

        retrieved = [MockChunk("doc1"), MockChunk("doc2"), MockChunk("doc1"),
                     MockChunk("doc3"), MockChunk("doc4")]
        ground_truth = ["doc1"]
        metrics = evaluator.evaluate_retrieval(retrieved, ground_truth, k_values=[5])

        assert metrics["precision@5"] == 2 / 5
        assert metrics["recall@5"] == 1.0  # doc1 is found
        assert metrics["ndcg@5"] > 0.0

    def test_empty_ground_truth(self, evaluator):
        """Empty ground truth should return zero metrics gracefully."""

        class MockChunk:
            def __init__(self, doc_id):
                self.doc_id = doc_id

        retrieved = [MockChunk("doc1")]
        metrics = evaluator.evaluate_retrieval(retrieved, [], k_values=[5])

        assert metrics["recall@5"] == 0.0
        assert metrics["ndcg@5"] == 0.0
        assert metrics["map"] == 0.0


class TestAnswerMetrics:
    """Tests for answer evaluation."""

    def test_exact_match(self, evaluator):
        metrics = evaluator.evaluate_answer("$750 million", "$750 million")
        assert metrics["exact_match"] == 1.0

    def test_exact_match_case_insensitive(self, evaluator):
        metrics = evaluator.evaluate_answer("Apple Inc", "apple inc")
        assert metrics["exact_match"] == 1.0

    def test_no_match(self, evaluator):
        metrics = evaluator.evaluate_answer("completely different", "answer text")
        assert metrics["exact_match"] == 0.0

    def test_token_f1_partial_overlap(self, evaluator):
        metrics = evaluator.evaluate_answer("the revenue was high", "the revenue was very high")
        assert 0.0 < metrics["token_f1"] < 1.0

    def test_token_f1_identical(self, evaluator):
        metrics = evaluator.evaluate_answer("hello world", "hello world")
        assert metrics["token_f1"] == 1.0

    def test_rouge_l_identical(self, evaluator):
        metrics = evaluator.evaluate_answer("the quick brown fox", "the quick brown fox")
        assert metrics["rouge_l"] == 1.0

    def test_rouge_l_no_overlap(self, evaluator):
        metrics = evaluator.evaluate_answer("alpha beta", "gamma delta")
        assert metrics["rouge_l"] == 0.0
