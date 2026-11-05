"""Receiving-side, unweighted cross-batch k-occurrence diagnostics.

Shared kNN and OT's row/column top-k union are undirected supports. Their
receiving coverage is structurally one for positive k, even with uneven degrees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GRAPH_LABELS = {
    "knn_0_to_1": "kNN 0→1",
    "knn_1_to_0": "kNN 1→0",
    "mutual_knn": "Mutual kNN",
    "shared_knn": "Shared kNN",
    "sparse_ot": "Sparsified OT",
}


def cross_batch_occurrences(forward, backward, ot_rows, ot_columns):
    """Count incoming edges using the same selections as the purity diagnostics.

    forward/backward contain local opposite-batch neighbor indices. OT supports
    are rectangular row/column top-k masks, *before* their union. No within-batch
    edge, affinity weight, or outgoing-only row is included in receiving counts.
    """
    n0, n1 = len(forward), len(backward)
    forward, backward = np.asarray(forward), np.asarray(backward)
    if not (forward.ndim == backward.ndim == 2 and forward.shape[1] == backward.shape[1]):
        raise ValueError("Neighbor lists must be matrices with the same k")
    if (
        np.any(forward < 0)
        or np.any(forward >= n1)
        or np.any(backward < 0)
        or np.any(backward >= n0)
    ):
        raise ValueError("Neighbor index is outside its receiving batch")
    if np.shape(ot_rows) != (n0, n1) or np.shape(ot_columns) != (n0, n1):
        raise ValueError("OT supports must have shape (n0, n1)")
    outgoing0, outgoing1 = np.zeros((n0, n1), bool), np.zeros((n0, n1), bool)
    outgoing0[np.arange(n0)[:, None], forward] = True
    outgoing1[backward, np.arange(n1)[:, None]] = True
    counts = {
        ("knn_0_to_1", 1): outgoing0.sum(axis=0),
        ("knn_1_to_0", 0): outgoing1.sum(axis=1),
    }
    for graph, support in (
        ("mutual_knn", outgoing0 & outgoing1),
        ("shared_knn", outgoing0 | outgoing1),
        ("sparse_ot", np.asarray(ot_rows, bool) | np.asarray(ot_columns, bool)),
    ):
        counts[graph, 0] = support.sum(axis=1)
        counts[graph, 1] = support.sum(axis=0)
    frames = []
    for (graph, batch), occurrence in counts.items():
        mean = float(occurrence.mean())
        frames.append(
            pd.DataFrame(
                dict(
                    graph=graph,
                    receiving_batch=batch,
                    point_index=np.arange(len(occurrence)),
                    k_occurrence=occurrence,
                    normalized_k_occurrence=occurrence / mean
                    if mean
                    else np.full(len(occurrence), np.nan),
                )
            )
        )
    return pd.concat(frames, ignore_index=True)


def occurrence_statistics(points):
    """One record per graph/receiving batch; skewness is the population third moment.

    Constant-degree distributions have undefined skewness (NaN), not zero.
    Normalization is per receiving batch and seed, including its zero-degree points.
    """
    records = []
    for (graph, batch), part in points.groupby(["graph", "receiving_batch"], sort=False):
        values = part.k_occurrence.to_numpy(dtype=float)
        mean, sd = values.mean(), values.std(ddof=0)
        records.append(
            dict(
                graph=graph,
                receiving_batch=batch,
                n_points=len(values),
                n_edges=int(values.sum()),
                mean_k_occurrence=mean,
                max_k_occurrence=int(values.max()),
                anti_hub_fraction=float(np.mean(values == 0)),
                receiving_coverage=float(np.mean(values >= 1)),
                skewness=float(np.mean(((values - mean) / sd) ** 3)) if sd else np.nan,
            )
        )
    return pd.DataFrame(records)


def summarize_coverage(raw):
    """Average receiving coverage over corruption seeds, preserving sample SD."""
    return (
        raw.groupby(["mode", "magnitude", "graph", "receiving_batch"], dropna=False)
        .agg(
            receiving_coverage_mean=("receiving_coverage", "mean"),
            receiving_coverage_sd=("receiving_coverage", "std"),
            n_seeds=("corruption_seed", "nunique"),
        )
        .reset_index()
    )


def restore_saved_construction(experiment):
    """Restore saved arrays/directions for diagnostics; do not fit or sample anything."""
    import json
    from types import SimpleNamespace

    from ne_aligned_utils.paper.inputs.synthetic import reconstruct_corruptions

    manifest = json.loads((experiment.run_dir / "config.json").read_text())
    with np.load(experiment.run_dir / "data/construction.npz", allow_pickle=False) as saved:
        for name, key in (
            ("x0_images", "clean0"),
            ("x1_images", "clean1"),
            ("labels0", "labels0"),
            ("labels1", "labels1"),
        ):
            setattr(experiment, name, saved[key])
        if manifest["variant"] == "parallel_pc1":
            experiment.pc1 = saved["pc1"]
            keys = [
                (mode, seed)
                for mode in experiment.config.data.modes
                for seed in experiment.config.corruption.corruption_seeds
            ]
            experiment.class_multipliers = dict(zip(keys, saved["class_multipliers"], strict=True))
        else:
            config = json.loads((experiment.run_dir / "original/config.json").read_text())
            basis = saved["pca_components"][: config["signal_rank"]]
            mean = saved["pca_mean"]
            experiment.clean_by_mode = {
                "original": (saved["clean0"], saved["clean1"]),
                "projected": tuple(
                    (saved[key] - mean) @ basis.T @ basis + mean for key in ("clean0", "clean1")
                ),
            }
            # The canonical reconstruction helper only reads PCA components.
            pca = SimpleNamespace(components_=saved["pca_components"])
            audit = pd.read_csv(experiment.run_dir / "original/diagnostics/corruption_audit.csv")
            experiment.alphas, experiment.betas = {}, {}
            for seed in experiment.config.corruption.corruption_seeds:
                experiment.alphas[seed], experiment.betas[seed] = reconstruct_corruptions(
                    manifest["variant"], audit, pca, seed
                )
