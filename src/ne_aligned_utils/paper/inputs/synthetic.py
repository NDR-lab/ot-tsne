"""Strict cached-run loading for PCA-tail experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

STRUCTURED_FILES = {
    "cost": "diagnostics/cost_ot_diagnostics.csv",
    "affinity": "diagnostics/cross_batch_affinity_metrics.csv",
    "knn": "diagnostics/knn_affinity_metrics.csv",
    "metrics": "metrics/embedding_metrics_raw.csv",
    "summary": "metrics/embedding_metrics_summary.csv",
}

REQUIRED_FILES = (
    "config.json",
    "cost_ot_diagnostics.csv",
    "ot_cross_affinity_diagnostics.csv",
    "knn_diagnostics.csv",
    "metrics.csv",
    "metrics_summary.csv",
)
VARIANT_EXPERIMENT_PREFIX = {
    "one_sided": "tail_pca_hetero",
    "two_sided_correlated_scales": "tail_pca_two_sided_hetero",
    "two_sided_independent_scales": "tail_pca_two_sided_independent_scales",
}
MODE_LABEL = {"original": "Original MNIST", "projected": "Projected control"}


@dataclass(frozen=True)
class SyntheticRuns:
    variant: str
    run_dirs: Mapping[str, Path]
    configs: Mapping[str, dict]
    cost: pd.DataFrame
    affinity: pd.DataFrame
    knn: pd.DataFrame
    metrics: pd.DataFrame
    summary: pd.DataFrame


def _required_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is incompatible; missing columns: {sorted(missing)}")


def load_runs(variant: str, run_dirs: Mapping[str, str | Path]) -> SyntheticRuns:
    """Load one explicit original/projected run pair, refusing discovery or recomputation."""
    if variant not in VARIANT_EXPERIMENT_PREFIX:
        raise ValueError(f"unknown variant {variant!r}; choose {sorted(VARIANT_EXPERIMENT_PREFIX)}")
    if set(run_dirs) != {"original", "projected"}:
        raise ValueError("run_dirs must explicitly contain exactly 'original' and 'projected'")

    resolved = {mode: Path(path).expanduser().resolve() for mode, path in run_dirs.items()}
    configs: dict[str, dict] = {}
    frames: dict[str, list[pd.DataFrame]] = {
        "cost": [],
        "affinity": [],
        "knn": [],
        "metrics": [],
        "summary": [],
    }
    filenames = {
        "cost": "cost_ot_diagnostics.csv",
        "affinity": "ot_cross_affinity_diagnostics.csv",
        "knn": "knn_diagnostics.csv",
        "metrics": "metrics.csv",
        "summary": "metrics_summary.csv",
    }
    prefix = VARIANT_EXPERIMENT_PREFIX[variant]
    for mode, directory in resolved.items():
        config_path = directory / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        manifest = json.loads(config_path.read_text())
        if manifest.get("schema_version") not in (None, 1):
            raise ValueError(f"Unsupported synthetic artifact schema: {directory}")
        structured = manifest.get("schema_version") == 1
        selected_files = STRUCTURED_FILES if structured else filenames
        required = ("config.json", *selected_files.values())
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"load-only run {directory} is incomplete; missing {missing}")
        manifest = json.loads((directory / "config.json").read_text())
        expected_experiment = f"{prefix}_{mode}"
        if manifest.get("experiment") != expected_experiment or manifest.get("mode") != mode:
            raise ValueError(
                f"{directory} is not {expected_experiment!r}: "
                f"experiment={manifest.get('experiment')!r}, mode={manifest.get('mode')!r}"
            )
        if manifest.get("batch_sampling") != "disjoint_stratified_exact_balance_v1":
            raise ValueError(
                f"{directory} is a legacy paired-image run; final plates require "
                "disjoint_stratified_exact_balance_v1 sampling; select compatible completed runs"
            )
        configs[mode] = manifest
        for key, filename in selected_files.items():
            frame = pd.read_csv(directory / filename)
            if any("pair" in column.lower() for column in frame.columns):
                raise ValueError(f"Legacy exact-pair column in {directory / filename}")
            if frame.empty:
                raise ValueError(f"{directory / filename} is empty")
            if "mode" not in frame or set(frame["mode"].astype(str)) != {mode}:
                raise ValueError(f"{directory / filename} does not contain only mode={mode!r}")
            frames[key].append(frame)

    compatibility_keys = (
        "magnitudes",
        "corruption_seeds",
        "embedding_seeds",
        "epsilon_policies",
        "rel_reg",
        "class_corruption_scales",
        "signal_rank",
        "batch_sampling",
        "n_per_digit_per_batch",
        "sampling_seed",
        "alpha_tail_pca_indices",
        "beta_tail_pca_indices",
        "construction_sha256",
        "multimap_profile",
    )
    for key in compatibility_keys:
        if configs["original"].get(key) != configs["projected"].get(key):
            raise ValueError(f"selected runs disagree on {key!r}")

    combined = {key: pd.concat(parts, ignore_index=True) for key, parts in frames.items()}
    _required_columns(
        combined["cost"],
        {
            "mode",
            "corruption_seed",
            "magnitude",
            "rel_reg",
            "epsilon_policy",
            "epsilon_over_reference",
            "relative_ot_plan_change",
        },
        Path("cost_ot_diagnostics.csv"),
    )
    affinity_columns = {
        "mode",
        "corruption_seed",
        "magnitude",
        "rel_reg",
        "epsilon_policy",
        "raw_same_digit_mass_fraction",
        "relative_ot_plan_change",
    }
    if variant == "two_sided_independent_scales":
        affinity_columns.update(
            {
                "top_k",
                "eot_row_digit_purity",
                "eot_row_nonempty_row_fraction",
                "eot_column_digit_purity",
                "eot_column_nonempty_row_fraction",
                "eot_shared_or_digit_purity",
                "eot_shared_or_nonempty_row_fraction",
                "eot_mutual_and_digit_purity",
                "eot_mutual_and_nonempty_row_fraction",
            }
        )
    else:
        affinity_columns.add("sparse_cross_digit_purity")
    _required_columns(
        combined["affinity"], affinity_columns, Path("ot_cross_affinity_diagnostics.csv")
    )
    _required_columns(
        combined["knn"],
        {
            "mode",
            "corruption_seed",
            "magnitude",
            "shared_correct_class_neighbor_fraction",
            "mutual_correct_class_neighbor_fraction",
            "within_batch_knn_edge_preservation",
        },
        Path("knn_diagnostics.csv"),
    )
    _required_columns(
        combined["summary"],
        {
            "mode",
            "magnitude",
            "display_method",
            "epsilon_policy",
            "rel_reg",
            "purity",
            "purity_std",
            "kbet",
            "kbet_std",
        },
        Path("metrics_summary.csv"),
    )
    return SyntheticRuns(variant, resolved, configs, **combined)


DEFAULT_RUNS = {
    "one_sided": {
        "original": "results/tail_pca_hetero_original/runs/config_4c3fccd2d7fa",
        # Required by the shared run loader for compatibility checks; never plotted.
        "projected": "results/tail_pca_hetero_projected/runs/config_2977e7844045",
    },
    "two_sided_correlated_scales": {
        "original": "results/tail_pca_two_sided_hetero_original/runs/config_4f3990b00557",
        "projected": "results/tail_pca_two_sided_hetero_projected/runs/config_d010ccdebce4",
    },
    "two_sided_independent_scales": {
        "original": (
            "results/tail_pca_two_sided_independent_scales_original/runs/config_ec638fd6784e"
        ),
        "projected": (
            "results/tail_pca_two_sided_independent_scales_projected/runs/config_671572660fdb"
        ),
    },
}

SAMPLING = "disjoint_stratified_exact_balance_v1"
GALLERY_FILES = (
    "mag_10__ot__tsne__corruption_seed_0__embedding_seed_0"
    "__epsilon_policy_adaptive_per_magnitude__eps_1__k_15.npz",
    "mag_10__multimap__default-guarded-scaling__corruption_seed_0__embedding_seed_0__k_30.npz",
    "mag_10__scanorama-50-tsne__tsne__corruption_seed_0__embedding_seed_0.npz",
)


def cache_directory(source_root: Path) -> Path:
    return source_root / "results/paper_figures/synthetic_experiment/final"


def mnist_directory(source_root: Path) -> Path:
    for directory in (source_root / "data/torchvision", source_root / "data"):
        if (directory / "MNIST/raw").is_dir():
            return directory
    raise FileNotFoundError(f"saved MNIST raw data are missing under {source_root / 'data'}")


def load_directed(runs: SyntheticRuns, cache_root: Path) -> pd.DataFrame:
    """Read the existing matched-support cache; never solve a transport problem."""
    if runs.configs["original"].get("schema_version") == 1:
        saved = runs.affinity.query(
            "mode == 'original' and epsilon_policy == 'adaptive_per_magnitude' and rel_reg == 1"
        ).copy()
        saved["batch_sampling"] = SAMPLING
        path = runs.run_dirs["original"] / STRUCTURED_FILES["affinity"]
    else:
        path = cache_root / f"{runs.variant}_directed_ot_top15.csv"
        saved = pd.read_csv(path)
    required = {
        "corruption_seed",
        "magnitude",
        "rel_reg",
        "top_k",
        "batch_sampling",
        "eot_row_digit_purity",
        "eot_row_nonempty_row_fraction",
        "eot_column_digit_purity",
        "eot_column_nonempty_row_fraction",
        "eot_shared_or_digit_purity",
        "eot_shared_or_nonempty_row_fraction",
        "eot_mutual_and_digit_purity",
        "eot_mutual_and_nonempty_row_fraction",
    }
    _required_columns(saved, required, path)
    expected = {
        (int(seed), float(magnitude))
        for seed in runs.configs["original"]["corruption_seeds"]
        for magnitude in (0.0, *runs.configs["original"]["magnitudes"])
    }
    observed = set(zip(saved.corruption_seed.astype(int), saved.magnitude.astype(float)))
    if not (
        observed == expected
        and len(saved) == len(expected)
        and set(saved.rel_reg) == {1.0}
        and set(saved.top_k) == {15}
        and set(saved.batch_sampling) == {SAMPLING}
        and not any("pair" in column for column in saved.columns)
    ):
        raise ValueError(f"incompatible saved directed diagnostics: {path}; no recomputation")
    return saved


def load_coordinates(path: Path, n_points: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as cache:
        embedding = np.asarray(cache["embedding"])
    if embedding.shape != (n_points, 2) or not np.isfinite(embedding).all():
        raise ValueError(f"invalid saved coordinates: {path}")
    return embedding


def load_joint_embedding(runs: SyntheticRuns, cache_root: Path) -> np.ndarray:
    path = (
        runs.run_dirs["original"]
        / "embeddings/joint_tsne/lambda_10/corruption_seed_0/embedding_seed_0.npz"
        if runs.configs["original"].get("schema_version") == 1
        else cache_root / f"{runs.variant}_unintegrated_tsne_disjoint_mag10.npz"
    )
    return load_coordinates(path, 20 * int(runs.configs["original"]["n_per_digit_per_batch"]))


def preflight(
    variant: str, *, source_root: Path, output_dir: Path, run_root: Path | None = None
) -> SyntheticRuns:
    """Check all saved stages before any plotting or output creation."""
    source_root, output_dir = source_root.resolve(), output_dir.resolve()
    if output_dir.is_relative_to(source_root) or source_root.is_relative_to(output_dir):
        raise ValueError("synthetic output must not overlap the source checkout")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"output must be new or empty: {output_dir}")
    runs = (
        load_experiment(run_root)
        if run_root is not None
        else load_runs(
            variant,
            {mode: source_root / relative for mode, relative in DEFAULT_RUNS[variant].items()},
        )
    )
    if runs.variant != variant:
        raise ValueError("Requested variant differs from selected run")
    mnist_directory(source_root)
    for directory in runs.run_dirs.values():
        if not audit_path(directory).is_file():
            raise FileNotFoundError(audit_path(directory))
    n_points = 20 * int(runs.configs["original"]["n_per_digit_per_batch"])
    for filename in GALLERY_FILES:
        load_coordinates(gallery_path(runs.run_dirs["original"], filename), n_points)
    load_directed(runs, cache_directory(source_root))
    load_joint_embedding(runs, cache_directory(source_root))
    return runs


def audit_path(directory: Path) -> Path:
    manifest = json.loads((directory / "config.json").read_text())
    return directory / (
        "diagnostics/corruption_audit.csv"
        if manifest.get("schema_version") == 1
        else "class_corruption_audit.csv"
    )


def gallery_path(directory: Path, filename: str) -> Path:
    """Resolve canonical gallery inputs across explicit legacy and structured schemas."""
    manifest = json.loads((directory / "config.json").read_text())
    if manifest.get("schema_version") != 1:
        return directory / filename
    from ne_aligned_utils.experiments.synthetic import embedding_method_id

    specs = (
        ("ot", "tsne", "adaptive_per_magnitude", 1.0, 15),
        ("multimap", "default-guarded-scaling", None, None, 30),
        ("scanorama-50-tsne", "tsne", None, None, None),
    )
    spec = specs[GALLERY_FILES.index(filename)]
    if spec[0] == "multimap" and "multimap_profile" in manifest:
        profile = manifest["multimap_profile"]
        if profile not in {"default", "k15"}:
            raise ValueError(f"Unsupported MultiMAP profile: {profile}")
        spec = ("multimap", "native-defaults", None, None, 15 if profile == "k15" else None)
    method_id = embedding_method_id(*spec)
    return directory / "embeddings" / method_id / "lambda_10/corruption_seed_0/embedding_seed_0.npz"


def load_experiment(run_root: Path, *, expected_config: dict | None = None) -> SyntheticRuns:
    """Load one explicitly named structured run, without discovery or construction."""
    run_root = Path(run_root)
    path = run_root / "config.json"
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("variant") != "two_sided_independent_scales"
    ):
        raise ValueError(f"Incompatible structured experiment manifest: {path}")
    if expected_config is not None and manifest.get("config") != json.loads(
        json.dumps(expected_config)
    ):
        raise ValueError(f"Incompatible requested experiment configuration: {path}")
    runs = load_runs(
        manifest["variant"],
        {"original": run_root / "original", "projected": run_root / "signal_projected"},
    )
    if not manifest.get("construction_sha256") or any(
        config.get("construction_sha256") != manifest["construction_sha256"]
        for config in runs.configs.values()
    ):
        raise ValueError("Run representations have different construction fingerprints")
    return runs


def reconstruct_corruptions(
    variant: str, audit: pd.DataFrame, pca: PCA, corruption_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the per-class corruption vectors (alpha, beta) one recorded seed applied."""
    rows = audit.query("corruption_seed == @corruption_seed")

    def vectors(part: pd.DataFrame) -> np.ndarray:
        part = part.sort_values("class")
        indices = np.asarray(json.loads(part.tail_pca_indices.iloc[0]), dtype=int)
        basis = pca.components_[indices]
        coefficients = np.stack(part.tail_pc_coefficients.map(json.loads).map(np.asarray))
        directions = coefficients @ basis
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        scale_column = "configured_scale" if "configured_scale" in part else "precenter_scale"
        scales = part[scale_column].to_numpy(float)
        result = scales[:, None] * directions
        return result - result.mean(axis=0, keepdims=True)

    if variant == "one_sided":
        beta = vectors(rows)
        return np.zeros_like(beta), beta
    return vectors(rows.query("batch == 0")), vectors(rows.query("batch == 1"))
