"""Check benchmark metric reporting and missing-score reasons."""

import numpy as np
import pandas as pd

import ne_aligned_utils.benchmark.scoring as benchmark_metrics


def test_score_aggregate_cache_can_compute_only_requested_metrics(monkeypatch):
    cache = {
        "dataset": np.asarray("toy"),
        "methods": np.asarray(["joint-pca"]),
        "seeds": np.asarray([0]),
        "labels_available": np.asarray(True),
        "batch_ids": np.asarray([0, 0, 1, 1]),
        "labels0": np.asarray([[0, 1]]),
        "labels1": np.asarray([[0, 1]]),
        "joint_pca_embeddings": np.asarray([[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]]),
    }
    monkeypatch.setattr(benchmark_metrics, "compute_neighbor_purity", lambda *_args, **_kwargs: 0.5)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("an unrequested metric was computed")

    for name in (
        "compute_silhouette_index",
        "compute_silhouette_score",
        "compute_ilisi",
        "compute_kbet",
        "compute_knn_other_batch_fraction",
        "compute_kmeans_cluster_scores",
    ):
        monkeypatch.setattr(benchmark_metrics, name, unexpected)

    result = benchmark_metrics.score_aggregate_cache(
        cache, metric_names=("neighborhood_purity_score",)
    )

    assert result["neighborhood_purity_score"].tolist() == [0.5]
    assert "silhouette_score" not in result


def test_metric_summary_keeps_backends_separate():
    rows = pd.DataFrame(
        {
            "dataset": ["toy"] * 4,
            "method": ["ot"] * 4,
            "backend": ["tsne", "tsne", "umap", "umap"],
            "seed": [0, 1, 0, 1],
            "neighborhood_purity_score": [0.0, 0.5, 0.5, 1.0],
        }
    )

    summary = benchmark_metrics.summarize_metric_rows(rows).set_index("backend")

    assert summary["neighborhood_purity_score_mean"].to_dict() == {"tsne": 0.25, "umap": 0.75}


def test_metric_summary_accepts_legacy_rows_without_backend():
    rows = pd.DataFrame(
        {
            "dataset": ["toy", "toy"],
            "method": ["ot", "ot"],
            "seed": [0, 1],
            "neighborhood_purity_score": [0.0, 0.5],
        }
    )

    summary = benchmark_metrics.summarize_metric_rows(rows)

    assert "backend" not in summary.columns
    assert summary["neighborhood_purity_score_mean"].tolist() == [0.25]
