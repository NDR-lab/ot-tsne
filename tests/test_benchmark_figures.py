"""Per-dataset and pooled metric grids, and the pooled summary behind the second."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ne_aligned_utils import config
from ne_aligned_utils.paper import layout, registry
from ne_aligned_utils.paper.figures import benchmark
from ne_aligned_utils.paper.rendering import presentation
from ne_aligned_utils.paper.selection.benchmark import reported_metric_rows
from ne_aligned_utils.paper.summaries.metrics import pool_datasets, summarise_seeds

METHODS = ("ot", "ot-knndiag", "multimap", "seurat-50", "seurat-2d", "joint-pca", "joint")


def _summary(datasets=("nCoV", "lupus")):
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        [
            dict(
                dataset=dataset,
                method_key=key,
                method_label=registry.METHODS[key].label,
                family=registry.METHODS[key].family,
                seed=seed,
                **{metric: float(rng.uniform(0.2, 0.9)) for metric in registry.METRICS},
            )
            for dataset in datasets
            for key in METHODS
            for seed in range(3)
        ]
    )
    return summarise_seeds(frame, metrics=list(registry.METRICS))


def test_pooled_summary_averages_dataset_means_and_spreads_across_datasets():
    summary = _summary()
    pooled = pool_datasets(summary, datasets=("nCoV", "lupus"))
    ot = summary[summary["method_key"] == "ot"]
    row = pooled.set_index("method_key").loc["ot"]
    assert row["kbet_mean"] == pytest.approx(ot["kbet_mean"].mean())
    assert row["kbet_sd"] == pytest.approx(ot["kbet_mean"].std(ddof=1))
    assert row["n"] == 2 and set(pooled["dataset"]) == {"pooled"}


def test_pooled_summary_refuses_a_method_missing_from_a_dataset():
    summary = _summary()
    gap = summary[~((summary["dataset"] == "lupus") & (summary["method_key"] == "joint"))]
    with pytest.raises(ValueError, match="joint"):
        pool_datasets(gap, datasets=("nCoV", "lupus"))
    with pytest.raises(ValueError, match="no summarised rows"):
        pool_datasets(summary, datasets=("nCoV", "PBMC"))


def test_summary_configuration_filter_applies_to_every_method():
    frame = pd.DataFrame(
        dict(
            backend=["tsne", config.NO_BACKEND, "umap", "tsne", "tsne", "tsne"],
            rel_reg=[1.0, np.nan, 1.0, 0.1, 1.0, 1.0],
            effective_k=[15.0, np.nan, 15.0, 15.0, 30.0, np.nan],
        )
    )
    selected = reported_metric_rows(
        frame, backend="tsne", no_backend=config.NO_BACKEND, rel_reg=1.0, effective_k=15.0
    )
    assert list(selected.index) == [0, 1, 5]


@pytest.mark.parametrize("pooled", (False, True))
def test_grids_draw_four_family_columns_and_save(tmp_path, pooled):
    summary = _summary()
    ctx = presentation.benchmark_render_context(
        tmp_path, method_keys=METHODS, datasets=("nCoV", "lupus")
    )
    try:
        with layout.paper_style():
            if pooled:
                path = benchmark.pooled_metric_grid(
                    pool_datasets(summary, datasets=("nCoV", "lupus")), "pooled", render_context=ctx
                )
            else:
                path = benchmark.dataset_metric_grid(
                    summary,
                    "grid",
                    render_context=ctx,
                    x="kbet_mean",
                    y="purity_mean",
                    x_label="kBET",
                    y_label="Purity",
                )
    finally:
        plt.close("all")
    assert path.exists() and path.suffix == ".png"


def test_empty_requested_dataset_is_rejected(tmp_path):
    ctx = presentation.benchmark_render_context(tmp_path, method_keys=["ot"], datasets=("missing",))
    with pytest.raises(ValueError, match="no summary rows"):
        benchmark.dataset_metric_grid(
            pd.DataFrame(dict(dataset=["other"])),
            "test",
            render_context=ctx,
            x="kbet_mean",
            y="purity_mean",
            x_label="kBET",
            y_label="Purity",
        )
