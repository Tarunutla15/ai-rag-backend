"""Unit tests for eval/metrics (no network)."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eval.metrics.retrieval import (
    aggregate_retrieval_metrics,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    per_query_document_metrics,
    per_query_retrieval_metrics,
    precision_at_k,
    recall_at_k,
)
from eval.metrics.latency import summarize_latencies


def test_precision_recall_hit_rate():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    assert precision_at_k(retrieved, relevant, 3) == 2 / 3
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert hit_rate_at_k(retrieved, relevant, 1) == 1.0
    assert hit_rate_at_k(["x", "y"], relevant, 5) == 0.0


def test_mrr():
    assert mrr(["x", "b", "a"], {"a"}) == 1 / 3
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_ndcg():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c"}
    score = ndcg_at_k(retrieved, relevant, 3)
    assert 0.0 < score <= 1.0


def test_per_query_aggregate():
    rows = [
        per_query_retrieval_metrics(["a", "b"], {"a"}, [1, 2]),
        per_query_retrieval_metrics(["c", "a"], {"a", "c"}, [1, 2]),
    ]
    agg = aggregate_retrieval_metrics(rows)
    assert "mrr" in agg
    assert agg["precision@1"] == 1.0  # both queries hit at rank 1


def test_document_level_metrics():
    metrics = per_query_document_metrics(
        retrieved_ids=["c1", "c2"],
        retrieved_doc_ids=["doc-a", "doc-b"],
        relevant_document_ids={"doc-a"},
        k_values=[1, 2],
    )
    assert metrics["hit_rate@1"] == 1.0
    assert metrics["precision@2"] == 0.5


def test_latency_summary():
    stats = summarize_latencies([10.0, 20.0, 30.0, 40.0, 50.0])
    assert stats.count == 5
    assert stats.mean_ms == 30.0
    assert stats.p50_ms == 30.0
    assert stats.p95_ms >= stats.p50_ms
