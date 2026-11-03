"""Rank benchmark methods within datasets, then aggregate ranks across datasets.

Raw metric values are never compared across datasets. Hyperparameter values remain separate, and
bio and batch-mixing metric families receive equal weight.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ne_aligned_utils.benchmark.scoring import (
    _embedding_keys,
    load_aggregate_cache,
    method_display_label,
    score_aggregate_cache,
)
from ne_aligned_utils.config import (
    DATASET_LABELS,
    NO_BACKEND,
    SHARED_RESULT_TREES,
    TREE_METHODS,
)

# Metrics

#: Metric direction: +1 means larger is better; -1 means smaller is better.
METRIC_DIRECTION: dict[str, int] = {
    "neighborhood_purity_score": +1,
    "silhouette_score": +1,
    "silhouette_index": +1,
    "ilisi_score": +1,
    "kbet_score": +1,
    "kmeans_rand_index": +1,
    "kmeans_adjusted_rand_index": +1,
    "concordance_score": +1,
    "cross_concordance_score": +1,
    "foscttm_score": -1,
}

#: Metric families receive equal weight in family-weighted aggregates.
METRIC_FAMILY: dict[str, str] = {
    "neighborhood_purity_score": "bio",
    "silhouette_score": "bio",
    "silhouette_index": "bio",
    "kmeans_rand_index": "bio",
    "kmeans_adjusted_rand_index": "bio",
    "ilisi_score": "batch",
    "kbet_score": "batch",
}

#: Default metrics avoid duplicate silhouette measures, reference-dependent scores,
#: and the two-sided mixing score whose ideal value is 0.5.
RANK_METRICS: tuple[str, ...] = (
    "neighborhood_purity_score",
    "silhouette_score",
    "ilisi_score",
    "kbet_score",
)

METRIC_LABELS: dict[str, str] = {
    "neighborhood_purity_score": "Purity",
    "silhouette_score": "Silhouette",
    "ilisi_score": "iLISI",
    "kbet_score": "kBET",
}

# Methods, datasets, trees

#: Which results tree holds each method's aggregate cache; kept here so the library does not
#: import from ``scripts/``.

#: Exclude superseded method definitions even when present in caches.
DEPRECATED_METHODS: tuple[str, ...] = ("cca-2d-uncentered", "cca-50-uncentered")

#: Methods whose score depends on the entropic regularization. Everything else has a single
#: configuration whose value is reused in every epsilon table.
EPS_METHODS: tuple[str, ...] = (
    "eot-t0",
    "eot-50",
    "ot",
    "ot-knndiag",
    "ot-selfotmedian",
    "ot-gramdiag",
)

#: Shared regularization grid for comparing base OT and diagonal variants.
SHARED_REL_REG: tuple[float, ...] = (1.0, 1 / 10, 1 / 100, 1 / 300)


def read_rel_reg_table(path: str | Path, **kwargs: object) -> pd.DataFrame:
    """Read a metric CSV while preserving exact ``rel_reg`` grid values.

    Round-trip float parsing prevents equality filters from dropping values such as
    ``1/300`` after decimal serialization.
    """
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


#: Datasets with a common method pool for directly comparable mean ranks.
MAIN_DATASETS: tuple[str, ...] = ("nCoV", "human_brain", "lupus", "yao_10x_ent_l2l3")

#: Reporting cap tokens per dataset; pin populations when multiple caps exist.
CELL_ARM: dict[str, str] = {
    "nCoV": "n_per_batch_all__",
    "human_brain": "n_per_batch_all__",
    "human_brain_log": "n_per_batch_5000__",
    "lupus": "n_per_batch_all__",
    "yao_10x_ent_l2l3": "n_per_batch_5000_10000__",
    "mnist": "n_per_batch_all__",
    # Tran datasets use uncapped reporting populations.
    "PBMC": "n_per_batch_all__",
    "mouse_atlas": "n_per_batch_all__",
    "hematopoietic": "n_per_batch_all__",
    "hematopoietic_shared": "n_per_batch_all__",
}

#: mnist is ranked separately: it lacks the six raw-count methods (scVI/Seurat/LIGER need
#: counts, and MNIST is pixels), and it is structurally different anyway -- raw pixels with an
#: artificial batch split. Pooling it would mean averaging ranks over unequal competitor pools.
SYNTHETIC_DATASETS: tuple[str, ...] = ("mnist",)


#: Comparison roles for family rollups and colors. Include family entrant counts
#: when interpreting best ranks; larger families have more chances to lead.
METHOD_FAMILY: dict[str, str] = {
    # the methods under study
    "eot-t0": "OT",
    "eot-50": "OT",
    "ot": "OT",
    "ot-knndiag": "OT",
    "ot-selfotmedian": "OT",
    "ot-gramdiag": "OT",
    # everything that models the batch relationship: released tools ...
    "harmony-2d": "integration method",
    "harmony-50": "integration method",
    "scvi-2d": "integration method",
    "scvi-50": "integration method",
    "seurat-2d": "integration method",
    "seurat-50": "integration method",
    "seurat3-2d": "integration method",
    "seurat3-50": "integration method",
    "liger1-2d": "integration method",
    "liger1-50": "integration method",
    "liger-2d": "integration method",
    "liger-50": "integration method",
    "scanorama-2d": "integration method",
    "scanorama-50": "integration method",
    "multimap": "integration method",
    "bc-tsne": "integration method",
    # ... and the two written here. cca uses the cross-covariance x0 @ x1.T; ref fits the
    # reference batch and projects the query into it. Different constructions, same group.
    "cca-2d": "integration method",
    "cca-50": "integration method",
    "cca-2d-uncentered": "integration method",
    "cca-50-uncentered": "integration method",
    "ref": "integration method",
    "ref-our": "integration method",
    # rc-tsne is given the batch id as the label to condition out, which is supervision at the
    # same level as Harmony or scVI get, so it belongs here and not with the no-integration floor.
    "rc-tsne": "integration method",
    # no batch handling at all -- the floor, not a competitor
    "joint-pca": "no integration",
    "joint": "no integration",
}

FAMILY_ORDER: tuple[str, ...] = ("OT", "integration method", "no integration")

#: Okabe-Ito, consistent with the deck's existing "best"/delta colors.
FAMILY_COLORS: dict[str, str] = {
    "OT": "#0072B2",
    "integration method": "#D55E00",
    "no integration": "#999999",
}

#: Direct two-dimensional Harmony and scVI variants excluded by ``fair_subset=True``.
HANDICAPPED_2D_METHODS: tuple[str, ...] = ("harmony-2d", "scvi-2d")


# Step 1 -- scoring


def find_aggregate_cache(
    results_root: Path,
    tree: str,
    dataset: str,
    *,
    config_token: str | Sequence[str] | None = None,
) -> Path | None:
    """Locate a dataset's aggregate embedding cache under a results tree.

    Every ``config_token`` substring must match the token directory, allowing cell
    caps and backend neighborhood settings to be selected together.
    """
    hits = sorted(Path(results_root).glob(f"{tree}/{dataset}/*/{dataset}_benchmark_embeddings.npz"))
    if config_token is not None:
        wanted = [config_token] if isinstance(config_token, str) else list(config_token)
        hits = [h for h in hits if all(w in h.parent.name for w in wanted)]
    if len(hits) > 1 and config_token is None:
        raise ValueError(
            f"{tree}/{dataset} has {len(hits)} config tokens "
            f"({', '.join(h.parent.name for h in hits)}); pass config_token to choose. "
            "Silently taking the first would report one arm's numbers as if it were the only one."
        )
    return hits[0] if hits else None


def resolve_cell_arm(results_root: Path, tree: str, dataset: str) -> str | None:
    """Resolve the dataset's reporting cap token within one results tree.

    Drop an unmatched pin only if all available arms share a single cap. Keep it when
    multiple caps exist so ambiguity cannot silently select a different population.
    """
    pin = CELL_ARM.get(dataset)
    if pin is None:
        return None
    arms = {
        path.parent.name
        for path in Path(results_root).glob(
            f"{tree}/{dataset}/*/{dataset}_benchmark_embeddings.npz"
        )
    }
    if not arms or any(pin in arm for arm in arms):
        return pin
    # The cap is the token's first field (`n_per_batch_all`, `n_per_batch_5000_10000`); the
    # neighborhood is a separate axis, so compare caps rather than whole token names or a
    # MultiMAP tree with two neighborhood arms would look ambiguous on the cap axis.
    caps = {arm.split("__")[0] for arm in arms}
    return None if len(caps) == 1 else pin


def score_trees(
    datasets: Iterable[str],
    *,
    results_root: Path,
    trees: Mapping[str, Sequence[str]] | None = None,
    metrics: Sequence[str] = RANK_METRICS,
    config_token: str | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Score every tree for every dataset into one tidy frame.

    Returns one row per ``(dataset, method, rel_reg, effective_k, seed)`` with the requested
    metric columns. Methods absent from a tree are skipped rather than raising, so a partially
    populated tree still yields everything it does have.
    """
    trees = TREE_METHODS if trees is None else trees
    frames: list[pd.DataFrame] = []
    for dataset in datasets:
        # The cap axis is per dataset and applies to every tree and both backends, so it is
        # resolved once here and combined with whatever token the tree itself needs.
        for tree, tree_methods in trees.items():
            tree_root = Path(results_root)
            backend_tree_token = config_token if tree_root.name == "umap" else None
            if tree in SHARED_RESULT_TREES and tree_root.name in {"tsne", "umap"}:
                tree_root = tree_root.parent
            # Resolved per tree, not once per dataset: which caps exist is a property of the tree.
            cell_arm = resolve_cell_arm(tree_root, tree, dataset)
            if progress and cell_arm != CELL_ARM.get(dataset):
                print(
                    f"  {dataset}/{tree}: no {CELL_ARM.get(dataset)!r} arm; "
                    "scoring the only cap present",
                    flush=True,
                )
            # Direct 2D outputs have no neighborhood and are shared by every arm. MultiMAP is
            # also stored once, but it does have n_neighbors, so select the same arm explicitly.
            if tree == "multimap":
                # An explicit UMAP neighborhood selects the controlled MultiMAP profile.
                # Without it, match the exact native __nn_30 suffix to exclude controlled K=30
                # caches.
                match = re.search(r"nn_(\d+)", config_token or "")
                if match:
                    tree_token = f"nn_{match.group(1)}__min_dist_0p1"
                else:
                    native_hits = sorted(
                        path
                        for path in tree_root.glob(
                            f"{tree}/{dataset}/*/{dataset}_benchmark_embeddings.npz"
                        )
                        if path.parent.name.endswith("__nn_30")
                        and (cell_arm is None or cell_arm in path.parent.name)
                    )
                    if len(native_hits) > 1:
                        raise ValueError(
                            f"expected one native MultiMAP K=30 cache for {dataset}, "
                            f"found {len(native_hits)}: {native_hits}"
                        )
                    # Skip an absent native MultiMAP arm; multiple matching caches remain an error.
                    if not native_hits:
                        continue
                    path = native_hits[0]
                    tree_token = None
            elif tree in SHARED_RESULT_TREES:
                tree_token = None
            else:
                tree_token = backend_tree_token
            if tree != "multimap" or match:
                required = [w for w in (cell_arm, tree_token) if w is not None]
                path = find_aggregate_cache(
                    tree_root,
                    tree,
                    dataset,
                    config_token=required or None,
                )
            if path is None:
                continue
            cache = load_aggregate_cache(path)
            # Derive embedding keys from the backend tree, not stray arrays stored in its caches.
            tree_backend = NO_BACKEND if tree in SHARED_RESULT_TREES else Path(results_root).name
            keys = _embedding_keys(tree_backend)
            available = [
                m for m in tree_methods if keys.get(m) in cache and m not in DEPRECATED_METHODS
            ]
            if not available:
                continue
            if progress:
                print(f"  scoring {dataset}/{tree}: {len(available)} methods", flush=True)
            rows = score_aggregate_cache(
                cache,
                methods=available,
                source_cache=path,
                metric_names=metrics,
                backend=tree_backend,
            )
            # Retain tree and source_cache so each score can be traced to its embedding.
            rows["tree"] = tree
            keep = [
                "dataset",
                "tree",
                "source_cache",
                "method",
                "backend",
                "seed",
                "rel_reg",
                "effective_k",
            ]
            keep += [m for m in metrics if m in rows.columns]
            frames.append(rows[[c for c in keep if c in rows.columns]].copy())
    if not frames:
        return pd.DataFrame(
            columns=[
                "dataset",
                "tree",
                "source_cache",
                "method",
                "backend",
                "seed",
                "rel_reg",
                "effective_k",
                *metrics,
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    out["effective_k"] = pd.to_numeric(out["effective_k"], errors="coerce")
    out["rel_reg"] = pd.to_numeric(out["rel_reg"], errors="coerce")
    return out


# Step 2 -- the method pool


def resolve_method_pool(
    rows: pd.DataFrame,
    datasets: Sequence[str],
    *,
    fair_subset: bool = False,
) -> tuple[str, ...]:
    """Return the method intersection across all requested datasets.

    A common competitor pool makes ranks comparable. Exclude deprecated definitions;
    ``fair_subset`` also excludes methods that integrate directly in two dimensions.
    """
    present = {ds: set(rows.loc[rows["dataset"] == ds, "method"].unique()) for ds in datasets}
    missing = [ds for ds, ms in present.items() if not ms]
    if missing:
        raise ValueError(f"no scored rows for dataset(s) {missing}")
    common = set.intersection(*present.values())
    common -= set(DEPRECATED_METHODS)
    if fair_subset:
        common -= set(HANDICAPPED_2D_METHODS)
    return tuple(sorted(common))


def coverage_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Boolean method x dataset table of which cells have any scored row (for auditing)."""
    return rows.groupby(["method", "dataset"]).size().unstack(fill_value=0).gt(0).sort_index()


# Steps 3-4 -- ranking


#: Suffix used to carry the seed-level standard deviation alongside each metric through the
#: melt/merge in :func:`rank_within`. Not part of any public column name.
_STD_SUFFIX = "__seed_std"

#: Supported cross-dataset and cross-metric rank statistics.
STATISTICS: tuple[str, ...] = ("mean", "median")


def _check_statistic(statistic: str) -> str:
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}, got {statistic!r}")
    return statistic


def _expand_non_rel_reg_methods(
    rows: pd.DataFrame, rel_reg_values: Sequence[float]
) -> pd.DataFrame:
    """Repeat scores without a regularization parameter across the comparison grid.

    Their values stay fixed, but ranks can vary as regularized competitors change.
    """
    rel_reg_free = rows[rows["rel_reg"].isna()].copy()
    swept = rows[rows["rel_reg"].notna()].copy()
    if rel_reg_free.empty:
        return swept
    expanded = [rel_reg_free.assign(rel_reg=float(e)) for e in rel_reg_values]
    return pd.concat([swept, *expanded], ignore_index=True)


def _practical_ranks(adjusted: np.ndarray, tolerance: float) -> np.ndarray:
    """Rank direction-adjusted values, best first, averaging ranks within ties.

    Tie groups stay within ``tolerance`` of their best value; consecutive-gap chaining
    is excluded. Inclusive comparisons use floating-point slack at the boundary.
    Zero tolerance reproduces exact average ranks.
    """
    n = len(adjusted)
    order = np.argsort(-adjusted, kind="stable")
    ranks = np.empty(n, dtype=float)
    limit = tolerance * (1.0 + 1e-9) + 1e-12
    start = 0
    while start < n:
        best = adjusted[order[start]]
        stop = start + 1
        while stop < n and best - adjusted[order[stop]] <= limit:
            stop += 1
        # the group spans positions start+1 .. stop (1-based); every member takes their average
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _resolve_tolerance(tie_tolerance: float | str | None, seed_std: pd.Series) -> float:
    """Resolve an absolute tie tolerance for one ranking cell.

    ``"seed_std"`` uses the median per-method seed standard deviation. Unknown spread
    falls back to zero, allowing only exact ties.
    """
    if tie_tolerance is None:
        return 0.0
    if isinstance(tie_tolerance, (int, float)) and not isinstance(tie_tolerance, bool):
        return float(tie_tolerance)
    if tie_tolerance != "seed_std":
        raise ValueError(
            f"tie_tolerance must be None, a number, or 'seed_std', got {tie_tolerance!r}"
        )
    if seed_std.notna().any():
        value = float(np.nanmedian(seed_std.to_numpy(dtype=float)))
        return value if np.isfinite(value) else 0.0
    return 0.0


def rank_within(
    rows: pd.DataFrame,
    *,
    metrics: Sequence[str] = RANK_METRICS,
    methods: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    rel_reg_values: Sequence[float] = SHARED_REL_REG,
    effective_k: float | None = 15.0,
    tie_tolerance: float | str | None = None,
) -> pd.DataFrame:
    """Average seeds and rank methods within each ``(dataset, rel_reg, metric)`` cell.

    Metric direction determines ordering; best rank is 1 and ties average occupied ranks.
    ``tie_tolerance`` accepts an absolute value or ``"seed_std"``; None uses exact ties.
    Return long-form values, seed spread, ranks, competitor counts, and resolved tolerances.
    """
    frame = rows.copy()
    if datasets is not None:
        frame = frame[frame["dataset"].isin(list(datasets))]
    if methods is not None:
        frame = frame[frame["method"].isin(list(methods))]
    if effective_k is not None and "effective_k" in frame.columns:
        # keep methods that carry no k at all, plus those matching the requested k
        frame = frame[frame["effective_k"].isna() | (frame["effective_k"] == effective_k)]
    if frame.empty:
        raise ValueError("no rows left after filtering; check datasets/methods/effective_k")

    metrics = [m for m in metrics if m in frame.columns]
    if not metrics:
        raise ValueError("none of the requested metrics are present in the input frame")

    # step 1: mean over seeds. The spread over seeds is carried alongside the mean because it is
    # the only available measure of what a difference in step 2 is worth.
    keys = ["dataset", "method", "rel_reg"]
    means = frame.groupby(keys, dropna=False)[list(metrics)].mean().reset_index()
    stds = (
        frame.groupby(keys, dropna=False)[list(metrics)]
        .std(ddof=1)
        .rename(columns={m: f"{m}{_STD_SUFFIX}" for m in metrics})
        .reset_index()
    )
    grouped = means.merge(stds, on=keys, how="left")
    grouped = _expand_non_rel_reg_methods(grouped, rel_reg_values)
    grouped = grouped[grouped["rel_reg"].isin([float(e) for e in rel_reg_values])]

    long = grouped.melt(
        id_vars=keys,
        value_vars=list(metrics),
        var_name="metric",
        value_name="value",
    )
    spread = grouped.melt(
        id_vars=keys,
        value_vars=[f"{m}{_STD_SUFFIX}" for m in metrics],
        var_name="metric",
        value_name="seed_std",
    )
    spread["metric"] = spread["metric"].str.removesuffix(_STD_SUFFIX)
    long = long.merge(spread, on=[*keys, "metric"], how="left").dropna(subset=["value"])

    # step 2: rank inside each (dataset, rel_reg, metric).
    # Multiplying by the direction means rank 1 is always "best", whichever way the metric points.
    pieces = []
    for (_dataset, _eps, metric), group in long.groupby(["dataset", "rel_reg", "metric"]):
        direction = METRIC_DIRECTION.get(metric, +1)
        scored = group.copy()
        tolerance = _resolve_tolerance(tie_tolerance, scored["seed_std"])
        scored["rank"] = _practical_ranks(
            (direction * scored["value"]).to_numpy(dtype=float), tolerance
        )
        scored["n_competitors"] = len(scored)
        scored["tie_tolerance"] = tolerance
        pieces.append(scored)
    ranked = pd.concat(pieces, ignore_index=True)
    ranked["method_label"] = ranked["method"].map(method_display_label)
    ranked["family"] = ranked["method"].map(METHOD_FAMILY).fillna("other")
    ranked["metric_family"] = ranked["metric"].map(METRIC_FAMILY).fillna("bio")
    return ranked


def aggregate_ranks(
    ranked: pd.DataFrame, *, weight: str = "family", statistic: str = "mean"
) -> pd.DataFrame:
    """Collapse ranks across datasets, then metrics, using the chosen statistic throughout.

    ``weight="family"`` gives bio and batch families equal weight regardless of metric
    counts; ``"metric"`` weights metrics equally. Return coverage counts and the statistic
    with aggregate ranks so incomplete coverage remains visible.
    """
    if weight not in {"family", "metric"}:
        raise ValueError(f"weight must be 'family' or 'metric', got {weight!r}")
    _check_statistic(statistic)

    per_metric = ranked.groupby(["method", "rel_reg", "metric"], as_index=False).agg(
        metric_rank=("rank", statistic), n_datasets=("dataset", "nunique")
    )
    per_metric["metric_family"] = per_metric["metric"].map(METRIC_FAMILY).fillna("bio")

    if weight == "family":
        by_family = per_metric.groupby(["method", "rel_reg", "metric_family"], as_index=False).agg(
            family_rank=("metric_rank", statistic)
        )
        agg = by_family.groupby(["method", "rel_reg"], as_index=False).agg(
            aggregate_rank=("family_rank", statistic),
            n_metric_families=("metric_family", "nunique"),
        )
    else:
        agg = per_metric.groupby(["method", "rel_reg"], as_index=False).agg(
            aggregate_rank=("metric_rank", statistic)
        )
        agg["n_metric_families"] = np.nan

    counts = per_metric.groupby(["method", "rel_reg"], as_index=False).agg(
        n_metrics=("metric", "nunique"), n_datasets=("n_datasets", "min")
    )
    agg = agg.merge(counts, on=["method", "rel_reg"], how="left")
    agg["statistic"] = statistic
    agg["method_label"] = agg["method"].map(method_display_label)
    agg["family"] = agg["method"].map(METHOD_FAMILY).fillna("other")
    return agg.sort_values(["rel_reg", "aggregate_rank"]).reset_index(drop=True)


def rank_matrix(agg: pd.DataFrame) -> pd.DataFrame:
    """Method x epsilon matrix of aggregate ranks, rows ordered by best rank achieved."""
    wide = agg.pivot_table(index="method", columns="rel_reg", values="aggregate_rank")
    return wide.loc[wide.min(axis=1).sort_values().index]


def family_rollup(agg: pd.DataFrame) -> pd.DataFrame:
    """Summarize each family's best and mean ranks, retaining entrant counts for comparison."""
    return (
        agg.groupby(["family", "rel_reg"], as_index=False)
        .agg(
            best_rank=("aggregate_rank", "min"),
            mean_rank=("aggregate_rank", "mean"),
            n_methods=("method", "nunique"),
        )
        .sort_values(["rel_reg", "best_rank"])
        .reset_index(drop=True)
    )


# Robustness: dataset bootstrap, top-k frequency, and leave-one-out sensitivity.


def _rank_cube(
    ranked: pd.DataFrame, rel_reg: float
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Return ranks as a ``(methods, datasets, metrics)`` array with axis labels."""
    subset = ranked[ranked["rel_reg"] == float(rel_reg)]
    if subset.empty:
        raise ValueError(f"no ranked rows at rel_reg={rel_reg!r}")
    methods = sorted(subset["method"].unique())
    datasets = [d for d in DATASET_LABELS if d in set(subset["dataset"].unique())]
    datasets += [d for d in sorted(subset["dataset"].unique()) if d not in datasets]
    metric_names = [m for m in RANK_METRICS if m in set(subset["metric"].unique())]

    cube = np.full((len(methods), len(datasets), len(metric_names)), np.nan)
    m_idx = {m: i for i, m in enumerate(methods)}
    d_idx = {d: i for i, d in enumerate(datasets)}
    q_idx = {q: i for i, q in enumerate(metric_names)}
    for method, dataset, metric, rank in subset[["method", "dataset", "metric", "rank"]].itertuples(
        index=False
    ):
        if metric in q_idx:
            cube[m_idx[method], d_idx[dataset], q_idx[metric]] = rank
    return cube, methods, datasets, metric_names


def _aggregate_from_cube(
    cube: np.ndarray,
    metric_names: Sequence[str],
    weight: str = "family",
    statistic: str = "mean",
) -> np.ndarray:
    """Aggregate a rank cube using the same statistic and family weighting as ``aggregate_ranks``.
    """
    reduce = np.nanmean if _check_statistic(statistic) == "mean" else np.nanmedian
    with np.errstate(invalid="ignore"):
        per_metric = reduce(cube, axis=1)  # (methods x metrics)
    if weight != "family":
        with np.errstate(invalid="ignore"):
            return reduce(per_metric, axis=1)
    families = [METRIC_FAMILY.get(q, "bio") for q in metric_names]
    columns = []
    for family in sorted(set(families)):
        idx = [i for i, f in enumerate(families) if f == family]
        with np.errstate(invalid="ignore"):
            columns.append(reduce(per_metric[:, idx], axis=1))
    with np.errstate(invalid="ignore"):
        return reduce(np.column_stack(columns), axis=1)


def bootstrap_aggregate_ranks(
    ranked: pd.DataFrame,
    *,
    rel_reg: float,
    n_boot: int = 2000,
    seed: int = 0,
    weight: str = "family",
    statistic: str = "mean",
    quantiles: tuple[float, float] = (2.5, 97.5),
) -> pd.DataFrame:
    """Estimate percentile intervals for aggregate ranks by resampling datasets.

    Metrics are a fixed evaluation set and are not resampled. With few datasets,
    intervals have limited resolution. Return point estimates, bounds, bootstrap means,
    and resampling metadata per method and regularization.
    """
    cube, methods, datasets, metric_names = _rank_cube(ranked, rel_reg)
    point = _aggregate_from_cube(cube, metric_names, weight, statistic)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(datasets), size=(int(n_boot), len(datasets)))
    replicates = np.stack(
        [_aggregate_from_cube(cube[:, draw, :], metric_names, weight, statistic) for draw in draws]
    )
    lo, hi = np.nanpercentile(replicates, list(quantiles), axis=0)
    return (
        pd.DataFrame(
            {
                "method": methods,
                "rel_reg": float(rel_reg),
                "aggregate_rank": point,
                "lo": lo,
                "hi": hi,
                "boot_mean": np.nanmean(replicates, axis=0),
                "n_boot": int(n_boot),
                "statistic": statistic,
                "method_label": [method_display_label(m) for m in methods],
                "family": [METHOD_FAMILY.get(m, "other") for m in methods],
            }
        )
        .sort_values("aggregate_rank")
        .reset_index(drop=True)
    )


def top_k_frequency(ranked: pd.DataFrame, *, rel_reg: float, k: int = 3) -> pd.DataFrame:
    """Return each method's frequency of top-k finishes across dataset/metric rankings."""
    subset = ranked[ranked["rel_reg"] == float(rel_reg)]
    if subset.empty:
        raise ValueError(f"no ranked rows at rel_reg={rel_reg!r}")
    out = (
        subset.assign(hit=subset["rank"] <= float(k))
        .groupby("method", as_index=False)
        .agg(n_top=("hit", "sum"), n_cells=("hit", "size"))
    )
    out["frequency"] = out["n_top"] / out["n_cells"]
    out["k"] = int(k)
    out["rel_reg"] = float(rel_reg)
    out["method_label"] = out["method"].map(method_display_label)
    out["family"] = out["method"].map(METHOD_FAMILY).fillna("other")
    return out.sort_values(["frequency", "method"], ascending=[False, True]).reset_index(drop=True)


def leave_one_out_ranks(
    ranked: pd.DataFrame,
    *,
    rel_reg: float,
    by: str = "dataset",
    weight: str = "family",
    statistic: str = "mean",
) -> pd.DataFrame:
    """Recompute aggregate ranks with each dataset or metric omitted in turn.

    Return the held-out item and rank change from the full-data estimate.
    """
    if by not in {"dataset", "metric"}:
        raise ValueError(f"by must be 'dataset' or 'metric', got {by!r}")
    cube, methods, datasets, metric_names = _rank_cube(ranked, rel_reg)
    full = _aggregate_from_cube(cube, metric_names, weight, statistic)
    levels = datasets if by == "dataset" else metric_names

    frames = []
    for i, level in enumerate(levels):
        keep = [j for j in range(len(levels)) if j != i]
        if by == "dataset":
            value = _aggregate_from_cube(cube[:, keep, :], metric_names, weight, statistic)
        else:
            value = _aggregate_from_cube(
                cube[:, :, keep], [metric_names[j] for j in keep], weight, statistic
            )
        frames.append(
            pd.DataFrame(
                {
                    "method": methods,
                    "rel_reg": float(rel_reg),
                    "by": by,
                    "held_out": level,
                    "aggregate_rank": value,
                    "delta": value - full,
                    "statistic": statistic,
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out["method_label"] = out["method"].map(method_display_label)
    out["family"] = out["method"].map(METHOD_FAMILY).fillna("other")
    return out


def leave_one_out_matrix(
    ranked: pd.DataFrame, *, rel_reg: float, weight: str = "family", statistic: str = "mean"
) -> pd.DataFrame:
    """Return method-by-fold aggregate ranks, including the full-data estimate in ``all``."""
    cube, methods, _datasets, metric_names = _rank_cube(ranked, rel_reg)
    columns = {"all": _aggregate_from_cube(cube, metric_names, weight, statistic)}
    for by in ("dataset", "metric"):
        loo = leave_one_out_ranks(
            ranked, rel_reg=rel_reg, by=by, weight=weight, statistic=statistic
        )
        for level, group in loo.groupby("held_out", sort=False):
            label = DATASET_LABELS.get(level, METRIC_LABELS.get(level, level))
            columns[f"-{label}"] = (
                group.set_index("method").loc[methods, "aggregate_rank"].to_numpy()
            )
    table = pd.DataFrame(columns, index=pd.Index(methods, name="method"))
    return table.loc[table["all"].sort_values().index]


def per_metric_table(
    ranked: pd.DataFrame, rel_reg: float, *, statistic: str = "mean"
) -> pd.DataFrame:
    """Return method-by-metric ranks at one regularization, ordered by the selected statistic."""
    _check_statistic(statistic)
    subset = ranked[ranked["rel_reg"] == float(rel_reg)]
    wide = subset.pivot_table(index="method", columns="metric", values="rank", aggfunc=statistic)
    wide = wide[[m for m in RANK_METRICS if m in wide.columns]]
    wide.columns = [METRIC_LABELS.get(c, c) for c in wide.columns]
    order = wide.mean(axis=1) if statistic == "mean" else wide.median(axis=1)
    return wide.loc[order.sort_values().index]


__all__ = [
    "DATASET_LABELS",
    "DEPRECATED_METHODS",
    "EPS_METHODS",
    "FAMILY_COLORS",
    "FAMILY_ORDER",
    "HANDICAPPED_2D_METHODS",
    "MAIN_DATASETS",
    "METHOD_FAMILY",
    "METRIC_DIRECTION",
    "METRIC_FAMILY",
    "METRIC_LABELS",
    "RANK_METRICS",
    "SHARED_REL_REG",
    "read_rel_reg_table",
    "STATISTICS",
    "SYNTHETIC_DATASETS",
    "TREE_METHODS",
    "aggregate_ranks",
    "bootstrap_aggregate_ranks",
    "coverage_matrix",
    "family_rollup",
    "find_aggregate_cache",
    "leave_one_out_matrix",
    "leave_one_out_ranks",
    "per_metric_table",
    "rank_matrix",
    "rank_within",
    "resolve_cell_arm",
    "resolve_method_pool",
    "score_trees",
    "top_k_frequency",
]
