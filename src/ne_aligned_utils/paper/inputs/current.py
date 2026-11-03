"""Load the canonical benchmark scored rows for paper figures.

The benchmark runner has already computed the metrics.  This module selects the prespecified
paper configuration and translates only the tabular contract expected by the renderers; it never
executes a benchmark method or recomputes an embedding.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ne_aligned_utils import config
from ne_aligned_utils.paper import registry
from ne_aligned_utils.paper.selection.benchmark import canonical_mask

#: Exclude the shared-type subset to avoid counting the same tissue twice in pooled ranks.
CANONICAL_DATASETS: tuple[str, ...] = (
    "nCoV",
    "human_brain",
    "lupus",
    "PBMC",
    "mouse_atlas",
    "hematopoietic",
    "yao_10x_ent_l2l3",
)

#: Rows per dataset that survive the canonical filter. Input tables may also contain sensitivity
#: arms, but those are not required to reproduce the publication figures.
EXPECTED_CANONICAL_ROWS = 69

METRIC_RENAMES = dict(registry.METRIC_SOURCE_COLUMNS)

#: Corrected arms produced beside the frozen benchmark tree.  These are publication replacements,
#: not additional methods: their public method names remain ``multimap``, ``rc-tsne`` and ``ref``.
SUPPLEMENTARY_REPLACEMENTS: dict[str, tuple[str, str]] = {
    "multimap": ("multimap_native", "multimap"),
    "rc-tsne": ("rc_tsne_beta_0p1", "rc-tsne"),
    "ref": ("ref_task_seed", "ref"),
}


def _metric_path(root: Path, dataset: str) -> Path:
    return root / "metrics" / f"{dataset}--all-experiments--all-devices.csv"


def _read_metric_table(root: Path, dataset: str) -> tuple[Path, pd.DataFrame]:
    """Read a dataset export, falling back to the standard combined ``metrics.csv``."""
    dataset_path = _metric_path(root, dataset)
    combined_path = root / "metrics.csv"
    path = dataset_path if dataset_path.is_file() else combined_path
    if not path.is_file():
        raise FileNotFoundError(
            f"missing benchmark metric table: expected {dataset_path} or {combined_path}"
        )
    table = pd.read_csv(path)
    if "dataset" not in table:
        raise ValueError(f"{path} has no 'dataset' column")
    table = table.loc[table["dataset"].astype(str) == dataset].copy()
    if table.empty:
        raise ValueError(f"{path} contains no rows for dataset {dataset!r}")
    return path, table


def _native_multimap_arms(root: Path) -> set[tuple[str, str]]:
    """Read native-profile MultiMAP arms from the combined task metadata when available."""
    path = root / "task_metadata.csv"
    if not path.is_file():
        return set()
    table = pd.read_csv(path)
    required = {"dataset", "method", "arm_id", "method_settings"}
    if not required.issubset(table.columns):
        return set()
    arms: set[tuple[str, str]] = set()
    for row in table.loc[table["method"].astype(str) == "multimap"].itertuples():
        settings = json.loads(str(row.method_settings))
        if bool(settings.get("native_profile", False)):
            arms.add((str(row.dataset), str(row.arm_id)))
    return arms


def _local_cache(root: Path, dataset: str, source: str) -> Path:
    """Resolve a cluster-written aggregate path inside a locally synced result root."""
    source_path = Path(source)
    if source_path.is_file():
        return source_path.resolve()
    if not source_path.is_absolute():
        portable = root / source_path
        if portable.is_file():
            return portable.resolve()
    parts = source_path.parts
    try:
        start = parts.index(dataset)
    except ValueError as exc:
        raise ValueError(
            f"source cache does not contain dataset component {dataset!r}: {source}"
        ) from exc
    candidate = root.joinpath(*parts[start:])
    if not candidate.is_file():
        raise FileNotFoundError(f"metric row points to missing local aggregate: {candidate}")
    return candidate.resolve()


def _multimap_arm_is_native(source_cache: str) -> bool:
    """Whether the MultiMAP arm behind an aggregate cache ran with MultiMAP's native defaults.

    The aggregate does not carry the flag, so it is read from one of the arm's task files. A fresh
    canonical run (``multimap_native = true``) is the native k=30 arm even though it is not loaded
    from the supplementary tree; the frozen main tree holds the matched k=15 arm.
    """
    tasks = sorted(Path(source_cache).parent.glob("tasks/*/*.npz"))
    if not tasks:
        return False
    with np.load(tasks[0], allow_pickle=False) as cache:
        return "multimap_native" in cache.files and bool(cache["multimap_native"])


def multimap_key(native: bool) -> str:
    """Native-default MultiMAP is ``multimap``; the frozen matched-k arm is ``multimap-k15``."""
    return "multimap" if native else "multimap-k15"


def _method_key(
    method: str,
    result_source: str,
    source_cache: str = "",
    *,
    native_profile: bool = False,
) -> str:
    if method == "multimap":
        return multimap_key(
            native_profile
            or result_source == "supplementary:multimap_native"
            or "multimap_native" in Path(source_cache).parts
            or (bool(source_cache) and _multimap_arm_is_native(source_cache))
        )
    return method


def _validate_supplementary_manifest(
    root: Path,
    *,
    benchmark_root: Path,
    arm: str,
    dataset: str,
    base_rows: pd.DataFrame,
) -> Path:
    manifest_path = root / arm / "supplementary_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing supplementary manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    benchmark_hash = str(manifest.get("benchmark_manifest_hash", ""))
    if not benchmark_hash:
        raise ValueError(f"{manifest_path} does not record benchmark_manifest_hash")
    root_hash = benchmark_root.name.rpartition("--")[2]
    if root_hash and not benchmark_hash.startswith(root_hash):
        raise ValueError(
            f"{manifest_path}: benchmark hash {benchmark_hash} does not match {benchmark_root}"
        )
    preprocessing_hash = str(manifest.get("preprocessing_hashes", {}).get(dataset, ""))
    sample_hash = str(manifest.get("sample_hashes", {}).get(dataset, ""))
    expected_preprocessing = set(base_rows["preprocessing_hash"].astype(str))
    expected_sample = set(base_rows["sample_hash"].astype(str))
    if expected_preprocessing != {preprocessing_hash}:
        raise ValueError(
            f"{manifest_path}: {dataset} preprocessing hash does not match the frozen rows"
        )
    if expected_sample != {sample_hash}:
        raise ValueError(f"{manifest_path}: {dataset} sample hash does not match the frozen rows")
    seeds = sorted(int(value) for value in manifest.get("task_seeds", []))
    expected_seeds = sorted(base_rows["seed"].astype(int).tolist())
    if seeds != expected_seeds:
        raise ValueError(
            f"{manifest_path}: task seeds {seeds} do not match frozen seeds {expected_seeds}"
        )
    return manifest_path.resolve()


def _replace_with_supplementary(
    table: pd.DataFrame,
    *,
    benchmark_root: Path,
    dataset: str,
    supplementary_root: Path,
) -> pd.DataFrame:
    """Replace the three frozen method rows while preserving their public benchmark slots."""
    result = table.copy()
    result["result_source"] = "frozen_benchmark"
    result["supplementary_manifest"] = ""
    for method, (arm, directory) in SUPPLEMENTARY_REPLACEMENTS.items():
        mask = result["method"].astype(str) == method
        base_rows = result.loc[mask].copy()
        if len(base_rows) != 3 or base_rows["seed"].astype(int).nunique() != 3:
            raise ValueError(
                f"{dataset}: expected three unique frozen {method!r} seeds, found {len(base_rows)}"
            )
        manifest_path = _validate_supplementary_manifest(
            supplementary_root,
            benchmark_root=benchmark_root,
            arm=arm,
            dataset=dataset,
            base_rows=base_rows,
        )
        method_root = supplementary_root / arm / directory / dataset
        metrics_path = method_root / "metrics.csv"
        aggregate_path = method_root / "aggregate.npz"
        if not metrics_path.is_file() or not aggregate_path.is_file():
            raise FileNotFoundError(
                f"{dataset}/{method}: expected {metrics_path} and {aggregate_path}"
            )
        supplementary = pd.read_csv(metrics_path)
        if set(supplementary["dataset"].astype(str)) != {dataset}:
            raise ValueError(f"{metrics_path} contains rows for another dataset")
        if set(supplementary["method"].astype(str)) != {method}:
            raise ValueError(f"{metrics_path} contains rows for another method")
        if sorted(supplementary["seed"].astype(int)) != sorted(base_rows["seed"].astype(int)):
            raise ValueError(f"{metrics_path} does not contain the three frozen task seeds")

        # Begin with the complete orchestration contract from the frozen slot, then replace every
        # scored/artifact field supplied by the corrected run.  This prevents the smaller
        # supplementary CSV schema from dropping device, pipeline, transport or hash metadata.
        replacement = base_rows.set_index(base_rows["seed"].astype(int), drop=False)
        corrected = supplementary.set_index(supplementary["seed"].astype(int), drop=False)
        for column in corrected.columns:
            replacement[column] = corrected[column]
        replacement["source_cache"] = str(aggregate_path.resolve())
        replacement["result_source"] = f"supplementary:{arm}"
        replacement["supplementary_manifest"] = str(manifest_path)
        result = pd.concat(
            [result.loc[~mask], replacement.reset_index(drop=True)], ignore_index=True
        )
    return result


def load_canonical_frame(
    root: Path,
    *,
    datasets: Sequence[str] = CANONICAL_DATASETS,
    supplementary_root: Path | None = None,
    require_source_caches: bool = True,
) -> pd.DataFrame:
    """Return the canonical paper frame for ``datasets``.

    The returned shape matches the metric/rank/embedding renderers: one row per downstream seed,
    with canonical balanced OT selected before any averaging or ranking.
    """
    root = Path(root).resolve()
    native_multimap_arms = _native_multimap_arms(root)
    pieces: list[pd.DataFrame] = []
    for dataset in datasets:
        path, table = _read_metric_table(root, dataset)
        if supplementary_root is not None:
            table = _replace_with_supplementary(
                table,
                benchmark_root=root,
                dataset=dataset,
                supplementary_root=Path(supplementary_root).resolve(),
            )
        else:
            table["result_source"] = "frozen_benchmark"
            table["supplementary_manifest"] = ""
        canonical = int(canonical_mask(table).sum())
        if canonical != EXPECTED_CANONICAL_ROWS:
            raise ValueError(
                f"{path} has {len(table)} rows of which {canonical} are canonical; "
                f"expected {EXPECTED_CANONICAL_ROWS}"
            )
        pieces.append(table)

    frame = pd.concat(pieces, ignore_index=True)
    frame = frame[canonical_mask(frame)].copy()
    if require_source_caches:
        frame["source_cache"] = [
            str(_local_cache(root, str(dataset), str(source)))
            for dataset, source in zip(frame["dataset"], frame["source_cache"], strict=True)
        ]
    frame = frame.rename(columns=METRIC_RENAMES)
    frame["method_key"] = [
        _method_key(
            str(method),
            str(source),
            str(cache),
            native_profile=(str(dataset), str(arm_id)) in native_multimap_arms,
        )
        for method, source, cache, dataset, arm_id in zip(
            frame["method"],
            frame["result_source"],
            frame["source_cache"],
            frame["dataset"],
            frame["arm_id"],
            strict=True,
        )
    ]
    styles = frame["method_key"].map(registry.METHODS)
    if styles.isna().any():
        missing = sorted(frame.loc[styles.isna(), "method_key"].unique())
        raise KeyError(f"benchmark methods missing from the paper registry: {missing}")
    frame["method_label"] = [style.label for style in styles]
    frame["family"] = [style.family for style in styles]
    frame["subfamily"] = [style.subfamily for style in styles]
    frame["tool"] = [style.tool for style in styles]
    frame["dataset_label"] = frame["dataset"].map(config.DATASET_LABELS).fillna(frame["dataset"])
    frame["tree"] = frame["experiment"]
    frame["n_neighbors"] = frame["method_key"].map(
        lambda value: (
            30.0 if value == "multimap" else 15.0 if value == "multimap-k15" else float("nan")
        )
    )
    frame["init"] = frame["init"].fillna("random")
    frame["backend"] = frame["backend"].replace({"na": config.NO_BACKEND})

    counts = frame.groupby("dataset").size().to_dict()
    wrong = {
        dataset: counts.get(dataset, 0)
        for dataset in datasets
        if counts.get(dataset, 0) != EXPECTED_CANONICAL_ROWS
    }
    if wrong:
        raise ValueError(
            f"canonical selection must have {EXPECTED_CANONICAL_ROWS} rows per "
            f"dataset; found {wrong}"
        )

    for metric in registry.METRICS:
        values = frame[metric]
        if values.isna().any() or (values < 0).any() or (values > 1).any():
            raise ValueError(f"canonical metric {metric!r} is missing or outside [0, 1]")
    return frame.reset_index(drop=True)
