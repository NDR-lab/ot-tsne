from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ne_aligned_utils.benchmark_metrics import (
    iter_aggregate_embeddings,
    load_aggregate_cache,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


METHOD_IDS = {
    "EOT t=0": "eot-t0",
    "OT-tSNE": "ot-tsne",
    "EOT50 + tSNE": "eot-50-tsne",
    "Joint PCA": "joint-pca",
    "Joint tSNE": "joint-tsne",
    "Seurat": "seurat",
    "Harmony 2D": "harmony-2d",
    "Harmony50 + tSNE": "harmony-50-tsne",
    "scVI 2D": "scvi-2d",
    "scVI50 + tSNE": "scvi-50-tsne",
}
METHOD_LABELS = {value: key for key, value in METHOD_IDS.items()}
EPS_METHODS = ("eot-t0", "ot-tsne", "eot-50-tsne")


def _display_cache_path(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_metric_rows(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows = rows.rename(
        columns={
            "neighbourhood_purity": "neighborhood_purity_score",
            "kbet": "kbet_score",
        }
    )
    required = {"dataset", "seed", "method"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Metric table is missing required columns: {sorted(missing)}")
    for column in ("neighborhood_purity_score", "kbet_score"):
        if column not in rows:
            raise ValueError(f"Metric table is missing {column!r}.")

    rows["method_id"] = rows["method"].map(METHOD_IDS).fillna(rows["method"])
    rows["method"] = rows["method_id"].map(METHOD_LABELS).fillna(rows["method"])
    if "eps_times" not in rows:
        rows["eps_times"] = np.nan
    if "effective_k" not in rows:
        rows["effective_k"] = np.nan
    rows["balanced_score"] = rows[
        ["neighborhood_purity_score", "kbet_score"]
    ].mean(axis=1)
    return rows


def summarize_regularization(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = normalize_metric_rows(rows)
    metric_columns = [
        "neighborhood_purity_score",
        "kbet_score",
        "balanced_score",
    ]
    by_dataset = (
        rows.groupby(
            ["dataset", "method_id", "method", "eps_times", "effective_k"],
            dropna=False,
        )[metric_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    by_dataset.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in by_dataset.columns.to_flat_index()
    ]

    overall = (
        rows.groupby(
            ["method_id", "method", "eps_times", "effective_k"], dropna=False
        )[metric_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    overall.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in overall.columns.to_flat_index()
    ]

    candidates = overall[overall["method_id"].isin(EPS_METHODS)].copy()
    selected = candidates.loc[
        candidates.groupby("method_id")["balanced_score_mean"].idxmax()
    ].reset_index(drop=True)
    return by_dataset, overall, selected


def write_overview_tables(
    metric_csv: str | Path, output_dir: str | Path
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = normalize_metric_rows(pd.read_csv(metric_csv))
    by_dataset, overall, selected = summarize_regularization(raw)

    paths = {
        "raw": output_dir / "regularization_raw_metrics.csv",
        "by_dataset": output_dir / "regularization_by_dataset.csv",
        "overall": output_dir / "regularization_overall.csv",
        "selected": output_dir / "selected_regularization.csv",
    }
    raw.to_csv(paths["raw"], index=False)
    by_dataset.to_csv(paths["by_dataset"], index=False)
    overall.to_csv(paths["overall"], index=False)
    selected.to_csv(paths["selected"], index=False)
    return paths


def _save_metric_scatter(rows: pd.DataFrame, output_path: Path) -> None:
    summary = (
        rows.groupby(["dataset", "method"], dropna=False)[
            ["neighborhood_purity_score", "kbet_score"]
        ]
        .mean()
        .reset_index()
    )
    datasets = list(summary["dataset"].drop_duplicates())
    fig, axes = plt.subplots(
        1, max(1, len(datasets)), figsize=(4.2 * max(1, len(datasets)), 4)
    )
    axes = np.atleast_1d(axes)
    for axis, dataset in zip(axes, datasets):
        subset = summary[summary["dataset"] == dataset]
        axis.scatter(
            subset["neighborhood_purity_score"], subset["kbet_score"], s=35
        )
        for row in subset.itertuples():
            axis.annotate(
                str(row.method),
                (row.neighborhood_purity_score, row.kbet_score),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
        axis.set_title(str(dataset))
        axis.set_xlabel("Neighborhood purity")
        axis.set_ylabel("kBET")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_regularization_trajectories(rows: pd.DataFrame, output_path: Path) -> None:
    subset = rows[rows["method_id"].isin(EPS_METHODS)].copy()
    summary = (
        subset.groupby(["dataset", "method", "eps_times"], dropna=False)[
            "balanced_score"
        ]
        .mean()
        .reset_index()
    )
    datasets = list(summary["dataset"].drop_duplicates())
    fig, axes = plt.subplots(
        1, max(1, len(datasets)), figsize=(4.2 * max(1, len(datasets)), 3.8)
    )
    axes = np.atleast_1d(axes)
    for axis, dataset in zip(axes, datasets):
        dataset_rows = summary[summary["dataset"] == dataset]
        for method, method_rows in dataset_rows.groupby("method"):
            method_rows = method_rows.sort_values("eps_times")
            axis.plot(
                method_rows["eps_times"],
                method_rows["balanced_score"],
                marker="o",
                label=method,
            )
        axis.set_xscale("log")
        axis.set_ylim(0, 1)
        axis.set_title(str(dataset))
        axis.set_xlabel("eps_times")
        axis.set_ylabel("Mean purity/kBET")
        axis.grid(alpha=0.2)
    if len(axes):
        axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _selected_eps_by_method(selected: pd.DataFrame) -> dict[str, float]:
    return {
        str(row.method_id): float(row.eps_times)
        for row in selected.itertuples()
        if np.isfinite(row.eps_times)
    }


def _representative_embedding(
    cache: Mapping[str, np.ndarray], method: str, eps_times: float
) -> np.ndarray | None:
    candidates = [
        record
        for record in iter_aggregate_embeddings(cache, methods=[method])
        if int(record["seed_index"]) == 0
        and np.isclose(float(record["eps_times"]), eps_times)
    ]
    if not candidates:
        return None
    return np.asarray(candidates[0]["embedding"], dtype=float)


def _save_embedding_overview(
    cache_paths: Mapping[str, str | Path], selected: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    selected_eps = _selected_eps_by_method(selected)
    methods = [method for method in EPS_METHODS if method in selected_eps]
    inventory = []
    fig, axes = plt.subplots(
        max(1, len(cache_paths)),
        max(1, len(methods)),
        figsize=(4 * max(1, len(methods)), 3.5 * max(1, len(cache_paths))),
        squeeze=False,
    )

    for row_index, (dataset, cache_path) in enumerate(cache_paths.items()):
        path = Path(cache_path)
        display_path = _display_cache_path(path)
        if not path.exists():
            inventory.append(
                {"dataset": dataset, "cache_path": display_path, "status": "missing"}
            )
            for axis in axes[row_index]:
                axis.axis("off")
            continue
        cache = load_aggregate_cache(path)
        batch_ids = np.asarray(cache["batch_ids"], dtype=int)
        inventory.append(
            {"dataset": dataset, "cache_path": display_path, "status": "loaded"}
        )
        for column_index, method in enumerate(methods):
            axis = axes[row_index, column_index]
            embedding = _representative_embedding(
                cache, method, selected_eps[method]
            )
            if embedding is None:
                axis.text(0.5, 0.5, "embedding unavailable", ha="center")
                axis.axis("off")
                continue
            if embedding.shape[0] > 5000:
                index = np.linspace(0, embedding.shape[0] - 1, 5000, dtype=int)
            else:
                index = np.arange(embedding.shape[0])
            axis.scatter(
                embedding[index, 0],
                embedding[index, 1],
                c=batch_ids[index],
                cmap="coolwarm",
                s=2,
                alpha=0.65,
                linewidths=0,
            )
            axis.set_title(
                f"{dataset}: {METHOD_LABELS[method]}\neps={selected_eps[method]:g}"
            )
            axis.set_xticks([])
            axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return pd.DataFrame(inventory)


def generate_overview_artifacts(
    metric_csv: str | Path,
    cache_paths: Mapping[str, str | Path],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths = write_overview_tables(metric_csv, output_dir)
    rows = pd.read_csv(paths["raw"])
    selected = pd.read_csv(paths["selected"])

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths["metric_scatter"] = figure_dir / "metric_scatter.png"
    paths["regularization_trajectories"] = (
        figure_dir / "regularization_trajectories.png"
    )
    paths["embedding_overview"] = figure_dir / "embedding_overview.png"
    _save_metric_scatter(rows, paths["metric_scatter"])
    _save_regularization_trajectories(
        rows, paths["regularization_trajectories"]
    )
    inventory = _save_embedding_overview(
        cache_paths, selected, paths["embedding_overview"]
    )
    paths["cache_inventory"] = output_dir / "cache_inventory.csv"
    inventory.to_csv(paths["cache_inventory"], index=False)
    return paths


def plot_aggregate_cache_overview(
    cache_path: str | Path,
    selected_regularization_csv: str | Path,
    *,
    seed_index: int = 0,
    methods: tuple[str, ...] = (
        "joint-pca",
        "joint-tsne",
        "seurat",
        "eot-t0",
        "ot-tsne",
        "eot-50-tsne",
    ),
    max_points: int = 5000,
):
    """Plot representative embeddings from one aggregate benchmark cache."""
    cache = load_aggregate_cache(cache_path)
    selected = pd.read_csv(selected_regularization_csv)
    selected_eps = _selected_eps_by_method(selected)
    batch_ids = np.asarray(cache["batch_ids"], dtype=int)

    panels: list[tuple[str, np.ndarray]] = []
    for method in methods:
        records = [
            record
            for record in iter_aggregate_embeddings(cache, methods=[method])
            if int(record["seed_index"]) == int(seed_index)
        ]
        if method in EPS_METHODS:
            eps_times = selected_eps.get(method)
            if eps_times is None:
                continue
            records = [
                record
                for record in records
                if np.isclose(float(record["eps_times"]), eps_times)
            ]
        if not records:
            continue
        panels.append((METHOD_LABELS.get(method, method), records[0]["embedding"]))

    n_columns = 3
    n_rows = max(1, int(np.ceil(len(panels) / n_columns)))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4 * n_columns, 3.5 * n_rows),
        squeeze=False,
    )
    for axis, (label, embedding) in zip(axes.flat, panels):
        embedding = np.asarray(embedding, dtype=float)
        if embedding.shape[0] > max_points:
            index = np.linspace(0, embedding.shape[0] - 1, max_points, dtype=int)
        else:
            index = np.arange(embedding.shape[0])
        axis.scatter(
            embedding[index, 0],
            embedding[index, 1],
            c=batch_ids[index],
            cmap="coolwarm",
            s=2,
            alpha=0.65,
            linewidths=0,
        )
        axis.set_title(label)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(panels) :]:
        axis.axis("off")
    dataset = str(np.asarray(cache["dataset"]).item())
    fig.suptitle(f"{dataset}: aggregate cache, seed index {seed_index}")
    fig.tight_layout()
    return fig
