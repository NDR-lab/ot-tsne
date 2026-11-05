"""Configuration and shared artifacts for the inline parallel-PC1 MNIST experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ne_aligned_utils.experiments.synthetic import (
    CompetitorsConfig,
    EmbeddingConfig,
    EvaluationConfig,
    ExperimentPaths,
    OtConfig,
    SyntheticExperiment,
    canonical_json,
    config_record,
    write_compatible_manifest,
)
from ne_aligned_utils.methods import compute_ot_plan
from ne_aligned_utils.paper.inputs.synthetic import STRUCTURED_FILES

VARIANT = "parallel_pc1"
EXPERIMENT = "mnist_parallel_batch_effect"


@dataclass(frozen=True)
class ParallelDataConfig:
    random_seed: int = 0
    n_per_digit: int = 100
    modes: tuple = ("constant", "heterogeneous")


@dataclass(frozen=True)
class ParallelCorruptionConfig:
    magnitudes: tuple = (1.0, 4.0, 10.0, 20.0)
    corruption_seeds: tuple = (0, 1, 2)
    multiplier_range: tuple = (-2.0, 2.0)
    batch_multipliers: tuple = (-1.0, 1.0)


@dataclass(frozen=True)
class ParallelCompetitorsConfig(CompetitorsConfig):
    multimap_preprocessing: tuple = ("centered", "uncentered")


@dataclass(frozen=True)
class ParallelConfig:
    data: ParallelDataConfig = field(default_factory=ParallelDataConfig)
    corruption: ParallelCorruptionConfig = field(default_factory=ParallelCorruptionConfig)
    ot: OtConfig = field(default_factory=OtConfig)
    competitors: ParallelCompetitorsConfig = field(default_factory=ParallelCompetitorsConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


class ParallelExperiment(SyntheticExperiment):
    """Reuse graph, epsilon, scoring-cache and CSV utilities without orthogonal controls."""

    def __init__(self, config: ParallelConfig, paths: ExperimentPaths):
        validate_parallel_config(config)
        self.config, self.paths = config, paths
        digest = hashlib.sha256(canonical_json(config_record(config)).encode()).hexdigest()[:8]
        name = paths.run_name or f"{EXPERIMENT}__seed{config.data.random_seed}__{digest}"
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("run_name must be a single directory name")
        self.run_dir = paths.results_root / EXPERIMENT / name

    def make_batches(self, mode, magnitude, corruption_seed):
        weights = self.class_multipliers[mode, corruption_seed]
        return tuple(
            clean + magnitude * batch_scale * weights[labels, None] * self.pc1
            for clean, labels, batch_scale in zip(
                (self.x0_images, self.x1_images),
                (self.labels0, self.labels1),
                self.config.corruption.batch_multipliers,
            )
        )

    def initialize_run(self):
        """Save construction and exact configuration; initialize lambda-zero epsilon."""
        construction = dict(
            batch0_indices=self.batch0_indices,
            batch1_indices=self.batch1_indices,
            labels0=self.labels0,
            labels1=self.labels1,
            clean0=self.x0_images,
            clean1=self.x1_images,
            pca_components=self.pca.components_,
            pca_mean=self.pca.mean_,
            pca_variance=self.pca.explained_variance_,
            pc1=self.pc1,
            class_multipliers=np.stack(
                [
                    self.class_multipliers[mode, seed]
                    for mode in self.config.data.modes
                    for seed in self.config.corruption.corruption_seeds
                ]
            ),
        )
        self.construction_sha256 = hashlib.sha256(
            b"".join(np.ascontiguousarray(x).tobytes() for x in construction.values())
        ).hexdigest()
        manifest = dict(
            schema_version=1,
            variant=VARIANT,
            config=config_record(self.config),
            construction_sha256=self.construction_sha256,
        )
        write_compatible_manifest(self.run_dir / "config.json", manifest)
        (self.run_dir / "data").mkdir(exist_ok=True)
        np.savez_compressed(self.run_dir / "data/construction.npz", **construction)
        self.mode_dirs = {}
        self.reference_epsilons = {}
        self.clean_by_mode = {
            mode: (self.x0_images, self.x1_images) for mode in self.config.data.modes
        }
        for mode in self.config.data.modes:
            directory = self.run_dir / mode
            self.mode_dirs[mode] = directory
            write_compatible_manifest(directory / "config.json", {**manifest, "mode": mode})
            for subdirectory in ("diagnostics", "metrics", "embeddings", "figures"):
                (directory / subdirectory).mkdir(exist_ok=True)
            self.audit.query("mode == @mode").to_csv(
                directory / "diagnostics/corruption_audit.csv", index=False
            )
            self.reference_epsilons[mode] = {}
            for rel_reg in self.config.ot.rel_reg:
                _, artifacts = compute_ot_plan(
                    self.x0_images,
                    self.x1_images,
                    rel_reg=rel_reg,
                    tol=self.config.ot.ot_tol,
                    max_iter=self.config.ot.ot_max_iter,
                    return_artifacts=True,
                )
                self.reference_epsilons[mode][float(rel_reg)] = float(artifacts["epsilon"])

    def load_saved(self):
        return load_parallel_run(self.run_dir, expected_config=config_record(self.config))


def load_parallel_run(run_dir, *, expected_config=None):
    """Read only one explicit completed run, refusing incompatible configurations."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "config.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("variant") != VARIANT:
        raise ValueError("Not a compatible parallel-PC1 run")
    if expected_config is not None and manifest["config"] != json.loads(
        json.dumps(expected_config)
    ):
        raise ValueError("Requested configuration differs from saved parallel run")
    validate_parallel_config(
        ParallelConfig(
            data=ParallelDataConfig(**manifest["config"]["data"]),
            corruption=ParallelCorruptionConfig(**manifest["config"]["corruption"]),
            ot=OtConfig(**manifest["config"]["ot"]),
            competitors=ParallelCompetitorsConfig(**manifest["config"]["competitors"]),
            embedding=EmbeddingConfig(**manifest["config"]["embedding"]),
            evaluation=EvaluationConfig(**manifest["config"]["evaluation"]),
        )
    )
    with np.load(run_dir / "data/construction.npz", allow_pickle=False) as saved_data:
        fingerprint = hashlib.sha256(
            b"".join(np.ascontiguousarray(saved_data[key]).tobytes() for key in saved_data.files)
        ).hexdigest()
    if fingerprint != manifest.get("construction_sha256"):
        raise ValueError("Saved parallel construction fingerprint differs from manifest")
    required_columns = {
        "cost": {
            "mode",
            "magnitude",
            "corruption_seed",
            "rel_reg",
            "epsilon_policy",
            "epsilon_over_reference",
            "relative_ot_plan_change",
            "decomposition_identity_error_max_abs",
            "nonseparable_cost_rms",
        },
        "affinity": {
            "mode",
            "magnitude",
            "corruption_seed",
            "rel_reg",
            "epsilon_policy",
            "eot_shared_or_digit_purity",
            "eot_mutual_and_nonempty_row_fraction",
        },
        "knn": {
            "mode",
            "magnitude",
            "corruption_seed",
            "shared_correct_class_neighbor_fraction",
            "mutual_correct_class_neighbor_fraction",
            "mutual_nonempty_row_fraction",
        },
        "metrics": {
            "mode",
            "magnitude",
            "method_id",
            "corruption_seed",
            "embedding_seed",
            "purity",
            "kbet",
        },
        "summary": {
            "mode",
            "magnitude",
            "method_id",
            "purity",
            "kbet",
            "purity_std",
            "kbet_std",
            "preprocessing",
        },
    }
    frames = {key: [] for key in STRUCTURED_FILES}
    modes = manifest["config"]["data"]["modes"]
    for mode in modes:
        directory = run_dir / mode
        saved = json.loads((directory / "config.json").read_text())
        if saved != {**manifest, "mode": mode}:
            raise ValueError("Parallel mode manifest differs from run manifest")
        for key, filename in STRUCTURED_FILES.items():
            frame = pd.read_csv(directory / filename)
            if not required_columns[key].issubset(frame.columns):
                raise ValueError(f"Incomplete parallel result columns: {directory / filename}")
            if frame.empty or set(frame["mode"]) != {mode}:
                raise ValueError(f"Missing or incorrect mode in {directory / filename}")
            frames[key].append(frame)
    return SimpleNamespace(
        variant=VARIANT,
        run_dir=run_dir,
        manifest=manifest,
        run_dirs={mode: run_dir / mode for mode in modes},
        **{key: pd.concat(parts, ignore_index=True) for key, parts in frames.items()},
    )


def validate_parallel_config(config):
    if tuple(config.data.modes) != ("constant", "heterogeneous"):
        raise ValueError("Retain constant and heterogeneous modes, in that order")
    if config.data.n_per_digit < 2:
        raise ValueError("At least two observations per digit and batch are required")
    lo, hi = config.corruption.multiplier_range
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("Multiplier range must be finite and increasing")
    shifts = np.asarray(config.corruption.batch_multipliers)
    if shifts.shape != (2,) or not np.isfinite(shifts).all() or shifts[0] == shifts[1]:
        raise ValueError("Two distinct finite batch multipliers are required")
    if config.competitors.multimap_profile not in {"default", "k15"}:
        raise ValueError("multimap_profile must be default or k15")
    preprocessing = config.competitors.multimap_preprocessing
    if (
        not preprocessing
        or len(set(preprocessing)) != len(preprocessing)
        or not set(preprocessing).issubset({"centered", "uncentered"})
    ):
        raise ValueError("Choose unique centered/uncentered MultiMAP variants")
    if set(config.ot.epsilon_policies) != {"fixed_at_lambda_0", "adaptive_per_magnitude"}:
        raise ValueError("Retain both epsilon policies")
    for values in (
        config.corruption.magnitudes,
        config.corruption.corruption_seeds,
        config.embedding.embedding_seeds,
        config.ot.rel_reg,
    ):
        if not values or len(set(values)) != len(values):
            raise ValueError("Nonempty unique grids are required")
    if any(x < 0 or not np.isfinite(x) for x in config.corruption.magnitudes):
        raise ValueError("Magnitudes must be finite and nonnegative")
    if any(x <= 0 or not np.isfinite(x) for x in config.ot.rel_reg):
        raise ValueError("rel_reg must be finite and positive")
