"""Shared configuration, graph utilities and artifacts for the inline MNIST walkthrough.

Scientific construction and experiment loops live in
notebooks/mnist_orthogonal_batch_effect_walkthrough.ipynb. No computation occurs on import.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from torchvision.datasets import MNIST

from ne_aligned_utils.config import DEFAULT_TSNE_EARLY_EXAGGERATION_ITER, DEFAULT_TSNE_N_ITER
from ne_aligned_utils.methods import (
    compute_ot_plan,
    compute_tsne_embedding_from_arrays,
)
from ne_aligned_utils.methods.ot_methods import within_batch_knn_indices

VARIANT = "two_sided_independent_scales"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DataConfig:
    random_seed: int = 0
    n_per_digit: int = 100
    signal_variance_target: float = 0.9
    signal_rank: int | None = None
    tail_pool_size: int = 30
    tail_variance_rel_tol: float = 1e-10
    tail_pca_indices: tuple | None = None
    modes: tuple = ("original", "projected")


@dataclass(frozen=True)
class CorruptionConfig:
    magnitudes: tuple = (1.0, 4.0, 10.0, 20.0)
    corruption_seeds: tuple = (0, 1, 2)
    batch1_seed_offset: int = 10000
    batch0_scale_seed_offset: int = 20000
    batch1_scale_seed_offset: int = 30000
    class_corruption_scales: tuple = field(
        default_factory=lambda: tuple(np.linspace(0.35, 2.60, 10))
    )


@dataclass(frozen=True)
class OtConfig:
    epsilon_policies: tuple = ("fixed_at_lambda_0", "adaptive_per_magnitude")
    rel_reg: tuple = (1.0, 10.0)
    ot_effective_k: int = 15
    ot_tol: float = 1e-12
    ot_max_iter: int = 10000


@dataclass(frozen=True)
class CompetitorsConfig:
    cross_batch_k: int = 15
    multimap_profile: str = "default"  # "default" or "k15"; other parameters stay native
    run_multimap: bool = True
    run_scanorama_50_tsne: bool = True


@dataclass(frozen=True)
class EmbeddingConfig:
    """t-SNE schedule: ``early_exaggeration_iter`` exaggerated, then ``n_epochs`` iterations.

    The defaults are openTSNE's (250 + 500), as in the benchmark. Earlier saved runs used
    ``n_epochs=200, early_exaggeration_iter=None``: OT-tSNE then exaggerated the first third of
    200, and the Scanorama/joint t-SNE tails used openTSNE's 250 before their 200.
    """

    embedding_seeds: tuple = (0, 1, 2)
    n_epochs: int = DEFAULT_TSNE_N_ITER
    early_exaggeration_iter: int | None = DEFAULT_TSNE_EARLY_EXAGGERATION_ITER


@dataclass(frozen=True)
class EvaluationConfig:
    kbet_k: int = 15


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    ot: OtConfig = field(default_factory=OtConfig)
    competitors: CompetitorsConfig = field(default_factory=CompetitorsConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def sample_mnist(data_root: Path, n_per_digit: int, sampling_seed: int, *, download=False):
    """Return disjoint balanced batches and their source indices, in reference RNG order."""
    dataset = MNIST(data_root, train=True, download=download)
    all_x = dataset.data.numpy().reshape(-1, 28 * 28)
    all_y = dataset.targets.numpy().astype(int)
    rng = np.random.default_rng(sampling_seed)
    batch0_indices, batch1_indices = ([], [])
    for digit in range(10):
        candidates = np.flatnonzero(all_y == digit)
        selected = rng.choice(candidates, 2 * n_per_digit, replace=False)
        rng.shuffle(selected)
        batch0_indices.extend(selected[:n_per_digit])
        batch1_indices.extend(selected[n_per_digit:])
    batch0_indices = np.asarray(batch0_indices)
    batch1_indices = np.asarray(batch1_indices)
    rng.shuffle(batch0_indices)
    rng.shuffle(batch1_indices)
    if np.intersect1d(batch0_indices, batch1_indices).size:
        raise AssertionError("MNIST batches must be disjoint")
    x0_images = all_x[batch0_indices].astype(np.float64) / 255.0
    x1_images = all_x[batch1_indices].astype(np.float64) / 255.0
    labels0 = all_y[batch0_indices]
    labels1 = all_y[batch1_indices]
    expected = np.full(10, n_per_digit)
    if not (
        np.array_equal(np.bincount(labels0, minlength=10), expected)
        and np.array_equal(np.bincount(labels1, minlength=10), expected)
    ):
        raise AssertionError("Each batch must contain exactly n_per_digit images per class")
    return x0_images, labels0, x1_images, labels1, batch0_indices, batch1_indices


@dataclass(frozen=True)
class ExperimentPaths:
    results_root: Path
    data_root: Path
    run_name: str | None = None
    download_mnist: bool = False


def canonical_json(value):
    return json.dumps(value, sort_keys=True, indent=2) + "\n"


def config_record(config) -> dict:
    """``asdict(config)``, omitting ``embedding.early_exaggeration_iter`` when it is None.

    None is the schedule of runs saved before the setting existed, so those runs keep their
    directory hash and manifest and still load.
    """
    record = asdict(config)
    embedding = record.get("embedding", {})
    if "early_exaggeration_iter" in embedding and embedding["early_exaggeration_iter"] is None:
        del embedding["early_exaggeration_iter"]
    return record


def run_directory(config: ExperimentConfig, paths: ExperimentPaths) -> Path:
    digest = hashlib.sha256(canonical_json(config_record(config)).encode()).hexdigest()[:8]
    name = (
        paths.run_name or f"mnist_orthogonal_batch_effect__seed{config.data.random_seed}__{digest}"
    )
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("run_name must be a single directory name")
    return paths.results_root / "mnist_orthogonal_batch_effect" / name


class SyntheticExperiment:
    """Notebook state plus shared graph, cache and artifact operations."""

    def __init__(self, config: ExperimentConfig, paths: ExperimentPaths):
        validate_config(config)
        self.config, self.paths = config, paths
        self.run_dir = run_directory(config, paths)
        self.base_config = {
            "sampling_seed": self.config.data.random_seed,
            "n_per_digit_per_batch": self.config.data.n_per_digit,
            "batch_sampling": "disjoint_stratified_exact_balance_v1",
            "magnitudes": self.config.corruption.magnitudes,
            "modes": self.config.data.modes,
            "corruption_seeds": self.config.corruption.corruption_seeds,
            "embedding_seeds": self.config.embedding.embedding_seeds,
            "epsilon_policies": self.config.ot.epsilon_policies,
            "pca_n_components": "full_spectrum",
            "pca_solver": "full",
            "signal_variance_target": self.config.data.signal_variance_target,
            "signal_rank_requested": self.config.data.signal_rank,
            "tail_pool_size_per_batch": self.config.data.tail_pool_size,
            "batch1_seed_offset": self.config.corruption.batch1_seed_offset,
            "batch0_scale_seed_offset": self.config.corruption.batch0_scale_seed_offset,
            "batch1_scale_seed_offset": self.config.corruption.batch1_scale_seed_offset,
            "class_scale_assignment": "independent_permutation_per_batch_and_corruption_seed",
            "tail_variance_relative_tolerance": self.config.data.tail_variance_rel_tol,
            "tail_pca_indices_requested": self.config.data.tail_pca_indices,
            "class_corruption_scales": self.config.corruption.class_corruption_scales,
            "rel_reg": self.config.ot.rel_reg,
            "cross_batch_k": self.config.competitors.cross_batch_k,
            "ot_effective_k": self.config.ot.ot_effective_k,
            "multimap_profile": self.config.competitors.multimap_profile,
            "n_epochs": self.config.embedding.n_epochs,
            "ot_tol": self.config.ot.ot_tol,
            "ot_max_iter": self.config.ot.ot_max_iter,
            "kbet_k": self.config.evaluation.kbet_k,
            "purity": {
                "average": "macro_by_label",
                "min_label_size": 5,
                "neighbors": "min(label_size - 1, 15)",
                "exclude_self": True,
            },
            "analysis_clipping": False,
            "original_coordinate_system": "raw_mnist_pixels",
            "projected_coordinate_system": "pca_mean_plus_leading_subspace_reconstruction",
            "scanorama_approx": False,
        }
        # Recorded only when set, so runs saved before the setting existed stay compatible.
        if self.config.embedding.early_exaggeration_iter is not None:
            self.base_config["early_exaggeration_iter"] = (
                self.config.embedding.early_exaggeration_iter
            )

    def prepare_controls(self):
        """Reconstruct signal-projected pixels; assert zero nuisance projection."""
        x0_centered, x1_centered = (
            self.x0_images - self.pca.mean_,
            self.x1_images - self.pca.mean_,
        )
        x0_projected = x0_centered @ self.lead_basis.T @ self.lead_basis + self.pca.mean_
        x1_projected = x1_centered @ self.lead_basis.T @ self.lead_basis + self.pca.mean_
        x_centered = np.vstack([x0_centered, x1_centered])
        x_projected = np.vstack([x0_projected - self.pca.mean_, x1_projected - self.pca.mean_])
        tail_indices_basis = self.pca.components_[self.tail_indices]
        self.clean_by_mode = {
            "original": (self.x0_images, self.x1_images),
            "projected": (x0_projected, x1_projected),
        }
        tail_projection_original = np.linalg.norm(x_centered @ tail_indices_basis.T, axis=1)
        tail_projection_projected = np.linalg.norm(x_projected @ tail_indices_basis.T, axis=1)
        print(
            "original tail-projection max/RMS:",
            tail_projection_original.max(),
            np.sqrt(np.mean(tail_projection_original**2)),
        )
        print(
            "projected tail-projection max/RMS:",
            tail_projection_projected.max(),
            np.sqrt(np.mean(tail_projection_projected**2)),
        )
        assert tail_projection_projected.max() < 1e-10

    def make_batches(
        self, mode: str, magnitude: float, corruption_seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        base0, base1 = self.clean_by_mode[mode]
        x0 = base0 + float(magnitude) * self.alphas[corruption_seed][self.labels0]
        x1 = base1 + float(magnitude) * self.betas[corruption_seed][self.labels1]
        return (x0, x1)

    def compute_policy_plan(self, x0, x1, mode, rel_reg, epsilon_policy, return_artifacts=False):
        common = dict(
            tol=self.config.ot.ot_tol,
            max_iter=self.config.ot.ot_max_iter,
            return_artifacts=return_artifacts,
        )
        if epsilon_policy == "fixed_at_lambda_0":
            return compute_ot_plan(
                x0, x1, epsilon=self.reference_epsilons[mode][float(rel_reg)], **common
            )
        if epsilon_policy == "adaptive_per_magnitude":
            return compute_ot_plan(x0, x1, rel_reg=rel_reg, **common)
        raise ValueError(f"Unknown epsilon policy: {epsilon_policy}")

    def squared_cost(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.maximum(
            np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2 * a @ b.T, 0.0
        )

    @staticmethod
    def directed_knn(query, candidates, k):
        return (
            NearestNeighbors(n_neighbors=k).fit(candidates).kneighbors(query, return_distance=False)
        )

    def edge_purity(self, neighbors, query_labels, candidate_labels):
        return float(np.mean(candidate_labels[neighbors] == query_labels[:, None]))

    def within_preservation(self, base, perturbed, k):
        before = within_batch_knn_indices(base, k)
        after = within_batch_knn_indices(perturbed, k)
        return float(
            np.mean([len(set(before[i]).intersection(after[i])) / k for i in range(len(base))])
        )

    def support_stats(self, support, query_labels, candidate_labels):
        same_digit = query_labels[:, None] == candidate_labels[None, :]
        degree = support.sum(axis=1)
        covered = degree > 0
        purity = np.divide(
            (support & same_digit).sum(axis=1),
            degree,
            out=np.zeros(len(query_labels), float),
            where=covered,
        )
        mean_purity = float(purity[covered].mean()) if np.any(covered) else np.nan
        return (mean_purity, float(covered.mean()), float(degree.mean()))

    @staticmethod
    def rectangular_topk_support(plan, k, axis):
        support = np.zeros_like(plan, dtype=bool)
        if axis == 1:
            indices = np.argpartition(plan, -k, axis=1)[:, -k:]
            support[np.arange(plan.shape[0])[:, None], indices] = True
        elif axis == 0:
            indices = np.argpartition(plan, -k, axis=0)[-k:, :]
            support[indices, np.arange(plan.shape[1])[None, :]] = True
        else:
            raise ValueError("axis must be 0 or 1")
        return support

    def knn_diagnostics(self, base0, base1, x0, x1, k):
        forward = self.directed_knn(x0, x1, k)
        backward = self.directed_knn(x1, x0, k)
        rows0, rows1 = (np.arange(len(x0)), np.arange(len(x1)))
        forward_support = np.zeros((len(x0), len(x1)), dtype=bool)
        backward_support = np.zeros_like(forward_support)
        forward_support[rows0[:, None], forward] = True
        backward_support[backward, rows1[:, None]] = True
        shared_purity, shared_coverage, _ = self.support_stats(
            forward_support | backward_support, self.labels0, self.labels1
        )
        mutual_purity, mutual_coverage, _ = self.support_stats(
            forward_support & backward_support, self.labels0, self.labels1
        )
        preserve0 = self.within_preservation(base0, x0, k)
        preserve1 = self.within_preservation(base1, x1, k)
        return {
            "batch0_to_batch1_correct_class_neighbor_fraction": self.edge_purity(
                forward, self.labels0, self.labels1
            ),
            "batch1_to_batch0_correct_class_neighbor_fraction": self.edge_purity(
                backward, self.labels1, self.labels0
            ),
            "shared_correct_class_neighbor_fraction": shared_purity,
            "shared_nonempty_row_fraction": shared_coverage,
            "mutual_correct_class_neighbor_fraction": mutual_purity,
            "mutual_nonempty_row_fraction": mutual_coverage,
            "batch0_within_knn_edge_preservation": preserve0,
            "batch1_within_knn_edge_preservation": preserve1,
            "within_batch_knn_edge_preservation": 0.5 * (preserve0 + preserve1),
        }

    def prepare_paper_gallery(self):
        """Cache the manuscript's joint PCA-50/t-SNE control, outside the renderer.

        This reproduces the former figure script's baseline: lambda=10, corruption
        seed=0, embedding seed=0, perplexity=30, and the public t-SNE default epochs.
        Its settings deliberately differ from the integration grid's epoch setting.
        """
        if 0 not in self.config.corruption.corruption_seeds:
            raise ValueError("The canonical paper gallery requires corruption seed 0")
        x0, x1 = self.make_batches("original", 10.0, 0)
        path = self.cache_path("original", 10.0, "joint-tsne", "tsne", 0, 0)
        metadata = dict(
            method="joint-tsne",
            magnitude=10.0,
            corruption_seed=0,
            embedding_seed=0,
            perplexity=30.0,
            pca_components=50,
            n_iter=None,
            backend="tsne",
        )
        self.load_or_compute(
            path,
            lambda: compute_tsne_embedding_from_arrays(
                x0, x1, random_state=0, perplexity=30.0, pca_components=50
            ),
            metadata,
        )

    def cache_path(
        self,
        mode,
        magnitude,
        method,
        backend,
        corruption_seed,
        embedding_seed,
        epsilon_policy=None,
        rel_reg=None,
        k=None,
    ):
        method_id = embedding_method_id(method, backend, epsilon_policy, rel_reg, k)
        return (
            self.mode_dirs[mode]
            / "embeddings"
            / method_id
            / f"lambda_{magnitude:g}"
            / f"corruption_seed_{corruption_seed}"
            / f"embedding_seed_{embedding_seed}.npz"
        )

    def load_or_compute(self, path: Path, compute, metadata: dict):
        metadata = {
            **metadata,
            "experiment_config": config_record(self.config),
            "construction_sha256": self.construction_sha256,
        }
        sidecar = path.with_suffix(".json")
        if path.exists():
            if not sidecar.is_file() or sidecar.read_text() != canonical_json(metadata):
                raise ValueError(f"Incompatible embedding cache metadata: {path}")
            with np.load(path, allow_pickle=False) as saved:
                embedding = saved["embedding"]
                elapsed = float(saved["runtime_seconds"])
        else:
            started = time.perf_counter()
            embedding = np.asarray(compute(), dtype=float)
            elapsed = time.perf_counter() - started
            validate_embedding(embedding, len(self.labels))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, embedding=embedding, runtime_seconds=elapsed)
            sidecar.write_text(canonical_json(metadata))
        validate_embedding(embedding, len(self.labels))
        return embedding, elapsed

    def initialize_run(self):
        """Record construction and exact manifests; refuse incompatible cache reuse."""
        self.prepare_controls()
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
        )
        self.construction_sha256 = hashlib.sha256(
            b"".join(np.ascontiguousarray(array).tobytes() for array in construction.values())
        ).hexdigest()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "variant": VARIANT,
            "config": config_record(self.config),
            "construction_sha256": self.construction_sha256,
        }
        write_compatible_manifest(self.run_dir / "config.json", manifest)
        data_dir = self.run_dir / "data"
        data_dir.mkdir(exist_ok=True)
        np.savez_compressed(data_dir / "construction.npz", **construction)
        self.mode_dirs, self.reference_epsilons = {}, {}
        for mode in self.config.data.modes:
            directory = self.run_dir / ("signal_projected" if mode == "projected" else mode)
            resolved = {
                **self.base_config,
                "schema_version": SCHEMA_VERSION,
                "experiment": f"tail_pca_two_sided_independent_scales_{mode}",
                "mode": mode,
                "signal_rank": self.signal_rank,
                "alpha_tail_pca_indices": self.alpha_tail_indices.tolist(),
                "beta_tail_pca_indices": self.beta_tail_indices.tolist(),
                "construction_sha256": self.construction_sha256,
            }
            write_compatible_manifest(directory / "config.json", resolved)
            self.mode_dirs[mode] = directory
            for subdir in ("diagnostics", "metrics", "embeddings", "figures"):
                (directory / subdir).mkdir(exist_ok=True)
            self.audit.assign(mode=mode).to_csv(
                directory / "diagnostics/corruption_audit.csv", index=False
            )
            self.reference_epsilons[mode] = {}
            clean0, clean1 = self.clean_by_mode[mode]
            for rel_reg in self.config.ot.rel_reg:
                _, artifacts = compute_ot_plan(
                    clean0,
                    clean1,
                    rel_reg=rel_reg,
                    tol=self.config.ot.ot_tol,
                    max_iter=self.config.ot.ot_max_iter,
                    return_artifacts=True,
                )
                self.reference_epsilons[mode][float(rel_reg)] = float(artifacts["epsilon"])

    def save_stage(self, stage):
        """Save complete tables after each stage, including stable method IDs."""
        specifications = {
            "cost": [
                (
                    self.cost_diagnostics if hasattr(self, "cost_diagnostics") else None,
                    "diagnostics/cost_ot_diagnostics.csv",
                )
            ],
            "affinity": [
                (
                    getattr(self, "affinity_diagnostics", None),
                    "diagnostics/cross_batch_affinity_metrics.csv",
                ),
                (
                    getattr(self, "knn_diagnostics_table", None),
                    "diagnostics/knn_affinity_metrics.csv",
                ),
            ],
            "embedding": [(getattr(self, "results", None), "metrics/embedding_metrics_raw.csv")],
            "summary": [
                (getattr(self, "summary", None), "metrics/embedding_metrics_summary.csv"),
                (
                    getattr(self, "per_corruption_summary", None),
                    "metrics/embedding_metrics_per_corruption.csv",
                ),
            ],
        }
        for frame, relative in specifications[stage]:
            if frame is None:
                raise RuntimeError(f"Run the {stage} stage before saving it")
            for mode, directory in self.mode_dirs.items():
                frame.loc[frame["mode"] == mode].to_csv(directory / relative, index=False)

    def verify_costs(self):
        """Protect the analytic identity and projected fixed-epsilon invariance."""
        assert self.cost_diagnostics.decomposition_identity_error_max_abs.max() < 1e-8
        control = self.cost_diagnostics.query(
            "mode == 'projected' and epsilon_policy == 'fixed_at_lambda_0'"
        )
        assert len(control), "Fixed-epsilon projected control is required"
        assert control.relative_ot_plan_change.max() < 1e-7

    def load_saved(self):
        """Load exactly this named run, checking all global settings before tables."""
        from ne_aligned_utils.paper.inputs.synthetic import load_experiment

        return load_experiment(self.run_dir, expected_config=config_record(self.config))


def write_compatible_manifest(path, manifest):
    serialized = canonical_json(manifest)
    if path.exists() and path.read_text() != serialized:
        raise ValueError(f"Incompatible run manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized)


def validate_embedding(embedding, n_points):
    if embedding.shape != (n_points, 2) or not np.isfinite(embedding).all():
        raise ValueError("Embedding must contain finite coordinates for every observation")


def embedding_method_id(method, backend, epsilon_policy=None, rel_reg=None, k=None):
    if method == "ot":
        policy = {
            "fixed_at_lambda_0": "fixed_epsilon",
            "adaptive_per_magnitude": "adaptive_epsilon",
        }[epsilon_policy]
        return f"ot_tsne__{policy}__rel_reg_{rel_reg:g}__k_{k:g}"
    if method == "multimap":
        suffix = "default" if k is None or pd.isna(k) else f"k_{k:g}"
        return f"multimap__{backend}__{suffix}"
    if method in {"scanorama-50-tsne", "joint-tsne"}:
        return method.replace("-", "_")
    raise ValueError(f"Unknown embedding method: {method}")


def label_methods(frame):
    """Keep configuration columns explicit; labels are presentation only."""
    frame = frame.copy()

    def identity(row):
        return embedding_method_id(
            row.method,
            row.backend,
            row.get("epsilon_policy"),
            row.get("rel_reg"),
            row.get("effective_k"),
        )

    frame["method_id"] = frame.apply(identity, axis=1)

    def label(row):
        if row.method == "ot":
            policy = "fixed ε" if row.epsilon_policy == "fixed_at_lambda_0" else "adaptive ε"
            return f"OT-tSNE · {policy} · rel_reg={row.rel_reg:g}"
        if row.method == "multimap":
            if row.backend in {"native-defaults", "uncentered-svd-defaults"}:
                name = "MultiMAP" if pd.isna(row.effective_k) else "MultiMAP k=15"
                if row.get("preprocessing") in {"centered", "uncentered"}:
                    return f"{name} · {row.preprocessing}"
                return name
            return f"MultiMAP · k={row.effective_k:g} · {row.preprocessing}"
        return "Scanorama → t-SNE"

    frame["display_method"] = frame.apply(label, axis=1)
    return frame


def validate_config(config):
    if config.competitors.multimap_profile not in {"default", "k15"}:
        raise ValueError("multimap_profile must be default or k15")
    if set(config.data.modes) != {"original", "projected"} or len(config.data.modes) != 2:
        raise ValueError("Both original and signal-projected controls are required")
    if not 0 < config.data.signal_variance_target <= 1 or config.data.tail_pool_size < 1:
        raise ValueError("Invalid signal variance target or tail pool size")
    if config.corruption.batch0_scale_seed_offset == config.corruption.batch1_scale_seed_offset:
        raise ValueError("Class-scale RNG streams must be independent")
    scales = np.asarray(config.corruption.class_corruption_scales)
    if scales.shape != (10,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Supply ten finite positive class scales")
    if set(config.ot.epsilon_policies) != {"fixed_at_lambda_0", "adaptive_per_magnitude"}:
        raise ValueError("Retain fixed and adaptive epsilon policies for the mechanism control")
    for values in (
        config.corruption.corruption_seeds,
        config.embedding.embedding_seeds,
        config.corruption.magnitudes,
        config.ot.rel_reg,
    ):
        if not values or len(set(values)) != len(values):
            raise ValueError("Experiment grids must be nonempty and contain unique values")
    if any(value < 0 for value in config.corruption.magnitudes):
        raise ValueError("Corruption magnitudes must be nonnegative")
    if any(value <= 0 for value in config.ot.rel_reg):
        raise ValueError("OT rel_reg must be positive")
