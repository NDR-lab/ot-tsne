"""Changes from a reference configuration, for the hyperparameter and initialisation figures.

Two comparisons, both read from benchmark metric tables only:

- Sensitivity grid: one method (default OT-tSNE) over ``rel_reg`` × ``effective_k``. Every point
  is compared with the default point of the same dataset.
- Initialisation: the OT-tSNE variants at the default point, each init compared with the random
  init of the same variant and dataset.

Seeds are averaged within a run, never across settings. A run must hold the same seeds as every
other run, each once (``check_seed_groups``). Every other recorded setting must be the same across
the compared runs; if not, the functions stop and name the setting, rather than average over it.

A change counts as real ("up" or "down") when it is at least ``min_change`` (0.01, one point on
the ×100 scale) and larger than ``noise_factor`` times a noise scale. ``noise="se"`` (default)
uses the standard error of the difference of two seed means, ``sqrt(sd_a^2 / n_a + sd_b^2 /
n_b)``, as a t-test does; ``noise="default_sd"`` uses the reference run's seed SD alone. The
minimum keeps tiny changes of very stable runs from counting.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ne_aligned_utils.evaluation.rank_aggregation import SETTING_COLUMNS, check_seed_groups
from ne_aligned_utils.paper.registry import BIO_METRICS, METRIC_SOURCE_COLUMNS, MIXING_METRICS

#: Composite scores: the plain mean of each half's metrics, computed per seed before averaging.
COMPOSITES: dict[str, tuple[str, ...]] = {"bio": BIO_METRICS, "mixing": MIXING_METRICS}
#: Overall score: the mean of bio and mixing, as in the paper's overall rank.
METRICS: tuple[str, ...] = (*BIO_METRICS, *MIXING_METRICS, *COMPOSITES, "overall")
SWEPT: tuple[str, str] = ("rel_reg", "effective_k")
DEFAULT_POINT: tuple[float, int] = (1.0, 15)
#: The OT-tSNE variants compared across inits, in figure order.
OT_VARIANTS: tuple[str, ...] = ("ot", "ot-selfotmedian", "ot-zerodiag", "ot-gramdiag", "ot-knndiag")
BASELINE_INIT = "random"
NOISE_FACTOR = 2.0
MIN_CHANGE = 0.01
#: Noise scales a change is compared with: the SE of the difference, or the reference's seed SD.
NOISE_RULES: tuple[str, ...] = ("se", "default_sd")
NOISE_RULE = "se"


# Reading


def read_metric_tables(paths: Sequence[Path]) -> pd.DataFrame:
    """Concatenate metric tables: a CSV file, or a results root holding ``metrics.csv``.

    A root's ``task_metadata.csv`` fills ``method_settings`` and ``configuration`` on rows that
    lack them (tables written before those columns existed). ``results_root`` records where each
    row came from, so its aggregate cache can be found again.
    """
    from ne_aligned_utils.benchmark.pipeline import with_run_settings

    frames = []
    for path in map(Path, paths):
        root = path if path.is_dir() else None
        table_path = path / "metrics.csv" if root is not None else path
        if not table_path.is_file():
            raise FileNotFoundError(table_path)
        table = pd.read_csv(table_path)
        metadata = (root or table_path.parent) / "task_metadata.csv"
        if metadata.is_file():
            table = with_run_settings(table, pd.read_csv(metadata))
        table["results_root"] = str((root or table_path.parent).resolve())
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def with_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Short metric names (``purity`` ...) and the per-seed ``bio``, ``mixing`` and ``overall``
    composites."""
    rows = frame.rename(columns=METRIC_SOURCE_COLUMNS)
    missing = [m for m in (*BIO_METRICS, *MIXING_METRICS) if m not in rows.columns]
    if missing:
        raise ValueError(f"metric rows lack {missing}")
    for name, parts in COMPOSITES.items():
        rows[name] = rows[list(parts)].mean(axis=1, skipna=False)
    rows["overall"] = rows[list(COMPOSITES)].mean(axis=1, skipna=False)
    return rows


def _without(settings: object, key: str) -> str:
    """``method_settings`` JSON without one key, for comparing everything else."""
    if not isinstance(settings, str):
        return ""
    return json.dumps({k: v for k, v in json.loads(settings).items() if k != key}, sort_keys=True)


def _require_single(rows: pd.DataFrame, columns: Sequence[str], within: Sequence[str]) -> None:
    """Each column holds one value within each ``within`` group; otherwise name it."""
    for column in columns:
        if column not in rows.columns:
            continue
        counts = rows.groupby(list(within), dropna=False)[column].nunique(dropna=False)
        if (counts > 1).any():
            values = sorted(rows[column].astype(str).unique())[:6]
            raise ValueError(
                f"the compared runs differ in {column!r} ({values}); select one value first, "
                "so different settings are not averaged together"
            )


def _seed_means(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    check_seed_groups(rows, keys)
    grouped = rows.groupby(keys, dropna=False)
    mean = grouped[list(METRICS)].mean()
    sd = grouped[list(METRICS)].std(ddof=1)
    n = grouped["seed"].nunique().rename("n_seeds")
    long = (
        mean.stack()
        .rename("mean")
        .to_frame()
        .join(sd.stack().rename("sd"))
        .reset_index()
        .rename(columns={f"level_{len(keys)}": "metric"})
    )
    return long.merge(n.reset_index(), on=keys)


def _against(
    points: pd.DataFrame,
    reference: pd.DataFrame,
    on: list[str],
    noise_factor: float,
    min_change: float,
    noise: str,
):
    ref = reference[[*on, "metric", "mean", "sd", "n_seeds"]].rename(
        columns={"mean": "ref_mean", "sd": "ref_sd", "n_seeds": "ref_n_seeds"}
    )
    out = points.merge(ref, on=[*on, "metric"], how="left", validate="many_to_one")
    if out["ref_mean"].isna().any():
        missing = out.loc[out["ref_mean"].isna(), on].drop_duplicates().to_dict("records")
        raise ValueError(f"no reference run for {missing[:5]}")
    out["delta"] = out["mean"] - out["ref_mean"]
    out["se"] = np.sqrt(out["sd"] ** 2 / out["n_seeds"] + out["ref_sd"] ** 2 / out["ref_n_seeds"])
    if noise not in NOISE_RULES:
        raise ValueError(f"noise must be one of {NOISE_RULES}, got {noise!r}")
    scale = out["se"] if noise == "se" else out["ref_sd"]
    size = out["delta"].abs()
    beyond = (size > noise_factor * scale) & (size >= min_change - 1e-12)
    out["direction"] = np.where(beyond, np.sign(out["delta"]), 0).astype(int)
    # The rule travels with the table, so figures can say which one they show.
    out["noise_rule"] = noise
    out["noise_factor"] = float(noise_factor)
    out["min_change"] = float(min_change)
    return out


def _pool(changes: pd.DataFrame, keys: list[str], datasets: Sequence[str]) -> pd.DataFrame:
    """Mean change over datasets, with how many datasets move beyond noise either way.

    ``direction`` judges the mean change by the same rule as one dataset's change: at least
    ``min_change``, and beyond ``noise_factor`` times the noise scale of the mean. That scale is
    each dataset's scale combined as for a mean of independent changes, ``sqrt(sum s_d^2) / D``.
    It covers seed noise only, not how much the change varies between datasets.
    """
    counts = changes.groupby([*keys, "metric"])["dataset"].nunique()
    short = counts[counts != len(datasets)]
    if not short.empty:
        raise ValueError(
            f"pooling needs every dataset at every point; these have fewer:\n{short.head(10)}"
        )
    grouped = changes.groupby([*keys, "metric"])
    rule = changes[["noise_rule", "noise_factor", "min_change"]].drop_duplicates()
    if len(rule) != 1:
        raise ValueError("the changes were judged with different noise rules")
    scale_column = "se" if rule.iloc[0]["noise_rule"] == "se" else "ref_sd"
    pooled_scale = grouped[scale_column].apply(lambda s: float(np.sqrt((s**2).sum()) / len(s)))
    pooled = pd.DataFrame(
        {
            "delta": grouped["delta"].mean(),
            "noise_scale": pooled_scale,
            "n_up": grouped["direction"].apply(lambda d: int((d > 0).sum())),
            "n_down": grouped["direction"].apply(lambda d: int((d < 0).sum())),
            "n_datasets": grouped["dataset"].nunique(),
        }
    ).reset_index()
    pooled = pooled.assign(**rule.iloc[0].to_dict())
    size = pooled["delta"].abs()
    beyond = (size > pooled["noise_factor"] * pooled["noise_scale"]) & (
        size >= pooled["min_change"] - 1e-12
    )
    pooled["direction"] = np.where(beyond, np.sign(pooled["delta"]), 0).astype(int)
    return pooled


# Sensitivity grid


def grid_changes(
    frame: pd.DataFrame,
    *,
    datasets: Sequence[str],
    method: str = "ot",
    default: tuple[float, int] = DEFAULT_POINT,
    noise_factor: float = NOISE_FACTOR,
    min_change: float = MIN_CHANGE,
    noise: str = NOISE_RULE,
) -> pd.DataFrame:
    """One row per (dataset, rel_reg, effective_k, metric): seed mean, and change from default."""
    rows = with_scores(frame)
    rows = rows[(rows["method"] == method) & rows["dataset"].isin(list(datasets))]
    if rows.empty:
        raise ValueError(f"no {method!r} rows for {list(datasets)}")
    rows = rows.assign(
        rel_reg=pd.to_numeric(rows["rel_reg"]), effective_k=pd.to_numeric(rows["effective_k"])
    )
    if rows[list(SWEPT)].isna().any().any():
        raise ValueError("some rows record no rel_reg or effective_k")
    others = [c for c in (*SETTING_COLUMNS, "init") if c not in SWEPT]
    _require_single(rows, others, ["dataset"])
    keys = ["dataset", *SWEPT]
    points = _seed_means(rows, keys)
    is_default = np.isclose(points["rel_reg"], default[0]) & (points["effective_k"] == default[1])
    if not is_default.any():
        raise ValueError(f"no default point rel_reg={default[0]}, k={default[1]}")
    changes = _against(points, points[is_default], ["dataset"], noise_factor, min_change, noise)
    return changes.sort_values(["dataset", *SWEPT, "metric"], ignore_index=True)


def pooled_grid(changes: pd.DataFrame, datasets: Sequence[str]) -> pd.DataFrame:
    """Per (rel_reg, effective_k, metric): mean change over datasets and up/down counts."""
    return _pool(changes[changes["dataset"].isin(list(datasets))], list(SWEPT), datasets)


def grid_line(
    changes: pd.DataFrame, sweep: str, default: tuple[float, int] = DEFAULT_POINT
) -> pd.DataFrame:
    """The one-at-a-time line along ``sweep``, the other setting held at its default."""
    if sweep not in SWEPT:
        raise ValueError(f"sweep must be one of {SWEPT}")
    other, value = ("effective_k", default[1]) if sweep == "rel_reg" else ("rel_reg", default[0])
    line = changes[np.isclose(changes[other], value)]
    if line[sweep].nunique() < 2:
        raise ValueError(f"the {sweep} line at {other}={value:g} has fewer than two points")
    return line.sort_values(["dataset", "metric", sweep], ignore_index=True)


def ot_rank_grid(
    paper_frame: pd.DataFrame,
    sweep_frame: pd.DataFrame,
    *,
    datasets: Sequence[str],
    normalization: str,
    default: tuple[float, int] = DEFAULT_POINT,
) -> pd.DataFrame:
    """OT-tSNE's standing among all benchmark methods at each (rel_reg, k) of the sweep.

    ``paper_frame`` is the reported benchmark (``load_canonical_frame``). At each point its
    OT-tSNE rows are replaced by the sweep's rows at that point; every competitor stays at the
    reported configuration. ``source="reported"`` rows rank the paper's own OT-tSNE rows.

    Per block (all, bio, mixing): OT-tSNE's mean rank over datasets (``aggregate_rank``), its
    ``placing`` among the methods ordered by mean rank (1 = best mean rank; ties share the better
    placing), the best other method and its mean rank, and ``lead`` = that rank minus
    OT-tSNE's (positive: OT-tSNE ahead).
    """
    from ne_aligned_utils.paper.summaries.ranks import aggregate_ranks

    sweep = with_scores(sweep_frame)
    sweep = sweep[(sweep["method"] == "ot") & sweep["dataset"].isin(list(datasets))]
    columns = list(paper_frame.columns)
    base = paper_frame[paper_frame["method_key"] != "ot"]

    def rank(frame: pd.DataFrame) -> pd.DataFrame:
        ranks = aggregate_ranks(
            frame,
            datasets=datasets,
            normalization=normalization,
            rel_reg=default[0],
            effective_k=default[1],
        )
        rows = []
        for block, table in ranks.groupby("block"):
            ot = table[table["method_key"] == "ot"].iloc[0]
            others = table[table["method_key"] != "ot"].sort_values("aggregate_rank")
            best = others.iloc[0]
            rows.append(
                dict(
                    block=block,
                    aggregate_rank=float(ot["aggregate_rank"]),
                    n_competitors=int(ot["n_competitors"]),
                    placing=1 + int((others["aggregate_rank"] < ot["aggregate_rank"] - 1e-9).sum()),
                    n_methods=len(table),
                    best_other=str(best["method_key"]),
                    best_other_rank=float(best["aggregate_rank"]),
                    lead=float(best["aggregate_rank"] - ot["aggregate_rank"]),
                )
            )
        return pd.DataFrame(rows)

    pieces = [rank(paper_frame).assign(rel_reg=np.nan, effective_k=np.nan, source="reported")]
    for (rel_reg, k), rows in sweep.groupby(list(SWEPT)):
        swapped = rows.assign(
            method_key="ot",
            # Competitors are filtered to the reported point; the swept values are kept below.
            rel_reg=default[0],
            effective_k=default[1],
        ).reindex(columns=columns)
        frame = pd.concat([base, swapped], ignore_index=True)
        pieces.append(rank(frame).assign(rel_reg=rel_reg, effective_k=k, source="sweep"))
    out = pd.concat(pieces, ignore_index=True)
    out["normalization"] = normalization
    return out


def context_scores(paper_frame: pd.DataFrame, *, datasets: Sequence[str]) -> pd.DataFrame:
    """Seed means of every reported method: one row per (dataset, method_key, metric).

    Raw scores (no min-max), the same composites as the sweep: bio, mixing and overall.
    """
    rows = with_scores(paper_frame[paper_frame["dataset"].isin(list(datasets))])
    return _seed_means(rows, ["dataset", "method_key"])


# Initialisation


def init_changes(
    frame: pd.DataFrame,
    *,
    datasets: Sequence[str],
    methods: Sequence[str] = OT_VARIANTS,
    baseline: str = BASELINE_INIT,
    noise_factor: float = NOISE_FACTOR,
    min_change: float = MIN_CHANGE,
    noise: str = NOISE_RULE,
) -> pd.DataFrame:
    """One row per (dataset, method, init, metric): seed mean and SD, and change from random.

    Runs of one variant may differ only in their init; any other setting that differs (k,
    rel_reg, epochs in ``method_settings`` ...) is an error.
    """
    rows = with_scores(frame)
    rows = rows[rows["method"].isin(list(methods)) & rows["dataset"].isin(list(datasets))]
    if rows.empty:
        raise ValueError(f"no rows for {list(methods)} on {list(datasets)}")
    rows = rows.assign(
        init=rows["init"].fillna(baseline).astype(str),
        settings_besides_init=rows["method_settings"].map(
            lambda s: _without(s, "ot_initialization")
        ),
    )
    if "init_used" in rows.columns:
        used = rows["init_used"].fillna(rows["init"]).astype(str)
        wrong = rows[used != rows["init"]]
        if not wrong.empty:
            raise ValueError(f"{len(wrong)} rows record init_used different from init")
    others = [c for c in SETTING_COLUMNS if c != "method_settings"] + ["settings_besides_init"]
    _require_single(rows, others, ["dataset", "method"])
    keys = ["dataset", "method", "init"]
    points = _seed_means(rows, keys)
    if baseline not in set(points["init"]):
        raise ValueError(f"no {baseline!r} init in the tables; pass the main benchmark root too")
    changes = _against(
        points,
        points[points["init"] == baseline],
        ["dataset", "method"],
        noise_factor,
        min_change,
        noise,
    )
    return changes.sort_values(keys + ["metric"], ignore_index=True)


def pooled_init(changes: pd.DataFrame, datasets: Sequence[str]) -> pd.DataFrame:
    """Per (method, init, metric): mean change from random over datasets, up/down counts, and
    the mean seed SD over datasets."""
    pooled = _pool(changes, ["method", "init"], datasets)
    spread = changes.groupby(["method", "init", "metric"])["sd"].mean().rename("mean_sd")
    return pooled.merge(spread.reset_index(), on=["method", "init", "metric"])


def init_order(inits: Sequence[str], baseline: str = BASELINE_INIT) -> list[str]:
    """Baseline first, then the others in a fixed order."""
    known = [baseline, "spectral", "eot", "pca"]
    present = list(dict.fromkeys(inits))
    return [i for i in known if i in present] + sorted(set(present) - set(known))


# Layouts from aggregate caches


def seed_layouts(
    frame: pd.DataFrame, *, dataset: str, method: str, init: str
) -> tuple[dict[int, np.ndarray], dict]:
    """Every seed's 2D layout for one (dataset, method, init), and the cache they came from.

    Rows come from :func:`read_metric_tables`, already filtered to one configuration; each row's
    ``source_cache`` is read (resolved inside its ``results_root`` if the path is from another
    machine), once per file.
    """
    from ne_aligned_utils.benchmark.scoring import iter_aggregate_embeddings, load_aggregate_cache
    from ne_aligned_utils.paper.inputs.current import _local_cache

    rows = frame[
        (frame["dataset"] == dataset)
        & (frame["method"] == method)
        & (frame["init"].fillna(BASELINE_INIT).astype(str) == init)
    ]
    if rows.empty:
        raise FileNotFoundError(f"no rows for {dataset}/{method}/{init}")
    if rows["seed"].duplicated().any():
        raise ValueError(f"{dataset}/{method}/{init}: a seed appears twice; select one run first")
    layouts: dict[int, np.ndarray] = {}
    caches: dict[str, dict] = {}
    for row in rows.itertuples(index=False):
        source = Path(str(row.source_cache))
        if not source.is_file():
            source = _local_cache(Path(str(row.results_root)), dataset, str(row.source_cache))
        key = str(source)
        if key not in caches:
            caches[key] = load_aggregate_cache(source)
        found = [
            r
            for r in iter_aggregate_embeddings(caches[key], (method,))
            if int(r["seed_index"]) == int(row.seed_index)
            and np.isclose(float(r["rel_reg"]), float(row.rel_reg), equal_nan=True)
            and np.isclose(float(r["effective_k"]), float(row.effective_k), equal_nan=True)
        ]
        if len(found) != 1:
            raise ValueError(f"{dataset}/{method}/{init} seed {row.seed}: {len(found)} layouts")
        layouts[int(row.seed)] = np.asarray(found[0]["embedding"], dtype=float)
    if len({len(v) for v in layouts.values()}) != 1:
        raise ValueError(f"{dataset}/{method}/{init}: seeds embed different numbers of cells")
    return dict(sorted(layouts.items())), next(iter(caches.values()))


def layout_agreement(
    frame: pd.DataFrame,
    *,
    datasets: Sequence[str],
    methods: Sequence[str],
    inits: Sequence[str],
    k: int = 15,
) -> pd.DataFrame:
    """Mean shared-neighbour fraction over all seed pairs, per (dataset, method, init)."""
    from itertools import combinations

    records = []
    for dataset in datasets:
        for method in methods:
            for init in inits:
                layouts, _ = seed_layouts(frame, dataset=dataset, method=method, init=init)
                pairs = list(combinations(sorted(layouts), 2))
                if not pairs:
                    raise ValueError(f"{dataset}/{method}/{init}: fewer than two seeds")
                values = [neighbour_agreement(layouts[a], layouts[b], k) for a, b in pairs]
                records.append(
                    dict(
                        dataset=dataset,
                        method=method,
                        init=init,
                        agreement=float(np.mean(values)),
                        n_pairs=len(pairs),
                    )
                )
    return pd.DataFrame.from_records(records)


# Layout agreement between seeds


def neighbour_agreement(a: np.ndarray, b: np.ndarray, k: int = 15) -> float:
    """Mean fraction of each cell's k nearest neighbours shared by two layouts of the same cells.

    Invariant to rotation, reflection and scale, so it measures how much two seeds' layouts
    agree on local structure, which is what a reader compares between gallery panels.
    """
    from sklearn.neighbors import NearestNeighbors

    if a.shape != b.shape:
        raise ValueError(f"layouts differ in shape: {a.shape} vs {b.shape}")
    k = min(k, len(a) - 1)
    ia = NearestNeighbors(n_neighbors=k + 1).fit(a).kneighbors(a, return_distance=False)[:, 1:]
    ib = NearestNeighbors(n_neighbors=k + 1).fit(b).kneighbors(b, return_distance=False)[:, 1:]
    shared = [len(np.intersect1d(x, y, assume_unique=True)) for x, y in zip(ia, ib, strict=True)]
    return float(np.mean(shared) / k)


def benchmark_metric_summary(
    main_root: Path,
    *,
    datasets: Sequence[str],
    supplementary_root: Path | None = None,
) -> pd.DataFrame:
    """Seed means and SDs of every method, as the benchmark metric grids summarise them.

    The same rows and summary as ``paper.build`` uses for its per-dataset metric figures, so
    axis ranges taken from it match those figures.
    """
    from ne_aligned_utils import config
    from ne_aligned_utils.paper.inputs.current import load_canonical_frame
    from ne_aligned_utils.paper.registry import METRICS
    from ne_aligned_utils.paper.selection.benchmark import CURRENT_BENCHMARK, reported_metric_rows
    from ne_aligned_utils.paper.summaries.metrics import summarise_seeds

    policy = CURRENT_BENCHMARK
    frame = load_canonical_frame(
        Path(main_root),
        datasets=list(datasets),
        supplementary_root=supplementary_root,
        require_source_caches=False,
        policy=policy,
    )
    selected = reported_metric_rows(
        frame,
        backend=policy.backend,
        no_backend=config.NO_BACKEND,
        rel_reg=policy.rel_reg,
        effective_k=policy.effective_k,
    )
    return summarise_seeds(selected, metrics=list(METRICS))


__all__ = [
    "BASELINE_INIT",
    "benchmark_metric_summary",
    "context_scores",
    "COMPOSITES",
    "DEFAULT_POINT",
    "METRICS",
    "MIN_CHANGE",
    "NOISE_FACTOR",
    "NOISE_RULE",
    "NOISE_RULES",
    "OT_VARIANTS",
    "SWEPT",
    "grid_changes",
    "grid_line",
    "init_changes",
    "init_order",
    "layout_agreement",
    "neighbour_agreement",
    "ot_rank_grid",
    "pooled_grid",
    "pooled_init",
    "read_metric_tables",
    "seed_layouts",
    "with_scores",
]
