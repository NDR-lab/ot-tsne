#!/usr/bin/env python3
"""Manifest-driven implementation of the canonical real-data benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ne_aligned_utils.benchmark.data import load_benchmark_data
from ne_aligned_utils.benchmark.sweep.config import (
    COUNT_NORMALIZED_METHODS,
    LEGACY_COUNT_LIBRARY_SIZE,
    config_token,
    format_rel_reg,
    format_token,
    resolve_task_output_path,
)
from ne_aligned_utils.benchmark.sweep.registry import (
    DIAGONAL_CONSTRUCTIONS as DIAGONALS,
    EPS_K_METHODS,
    INTERMEDIATE_DIMENSION_KEYS,
    KNNDIAG_SYMMETRIZATION_METHODS,
    OT_PLAN_METHODS,
    SYMMETRIZATION_METHODS,
    UOT_METHODS,
    VISUALIZED_METHODS,
    input_kind_for_method,
    representation_family,
)
from ne_aligned_utils.config import DEFAULT_INIT_JITTER, ROOT, thread_env

DEFAULT_MANIFEST = ROOT / "configs" / "benchmark.toml"


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def data_root(manifest: dict[str, Any]) -> Path:
    override = os.environ.get("NE_ALIGNED_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return resolve_repo_path(manifest["data_root"])


def configured_result_root(manifest: dict[str, Any]) -> Path:
    override = os.environ.get("NE_ALIGNED_RESULT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return resolve_repo_path(manifest["result_root"])


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    with path.open("rb") as handle:
        manifest = tomllib.load(handle)
    return manifest, canonical_hash(manifest)


def scientific_method_settings(manifest: dict[str, Any], method: str) -> dict[str, Any]:
    settings = manifest["settings"]
    result: dict[str, Any] = {"final_dimension": int(settings["embedding_dim"])}
    if method in VISUALIZED_METHODS:
        result.update(
            visualization="tsne",
            visualization_epochs=int(settings["visualization_epochs"]),
            perplexity=float(settings["perplexity"]),
        )
        # Absent from older manifests, whose run ids must not change.
        if "tsne_early_exaggeration_iter" in settings:
            result["tsne_early_exaggeration_iter"] = int(settings["tsne_early_exaggeration_iter"])
    if method in COUNT_NORMALIZED_METHODS and "count_library_size" in settings:
        # Absent from older manifests, whose run ids must not change.
        result["count_library_size"] = str(settings["count_library_size"])
    if method in INTERMEDIATE_DIMENSION_KEYS:
        result[INTERMEDIATE_DIMENSION_KEYS[method]] = 50
    if method in OT_PLAN_METHODS:
        result.update(
            # Only OT-tSNE starts from this init. EOT's layouts always start from random
            # coordinates, so their run ids must not move with the setting.
            ot_initialization=settings["ot_init"] if method in EPS_K_METHODS else "random",
            ot_tolerance=float(settings["ot_tolerance"]),
            ot_max_iterations=int(settings["ot_max_iterations"]),
        )
        # Absent from older manifests, whose run ids must not change.
        if "ot_init_jitter" in settings:
            result["ot_init_jitter"] = float(settings["ot_init_jitter"])
        # Only OT-tSNE's spectral blocks use the plan SVD. Absent from older manifests,
        # whose run ids must not change; those ran the full SVD (see ot_svd_task_args).
        if method == "ot" and "ot_svd_solver" in settings:
            result["ot_svd_solver"] = str(settings["ot_svd_solver"])
            result["ot_svd_directions"] = int(settings["ot_svd_directions"])
    # Absent from older manifests, whose run ids must not change; those ran rliger's default.
    if method in LIGER_METHODS and "liger_n_cores" in settings:
        result["liger_n_cores"] = int(settings["liger_n_cores"])
    if method == "ref":
        result.update(
            reference_pca_dimension=50,
            preselected_features=True,
            initialization_seed=settings["ref_initialization_seed"],
        )
    elif method == "multimap":
        native = bool(settings["multimap_native"])
        result.update(reduction="per_batch", native_profile=native)
        if not native:
            result["n_neighbors"] = int(settings["n_neighbors"])
    elif method == "bc-tsne":
        result["reduction"] = "batch_aware_internal"
    elif method == "rc-tsne":
        result.update(
            pca_components=int(settings["pca_components"]),
            beta=float(settings["rc_tsne_beta"]),
        )
    return result


#: SVD of manifests written before the solver was a setting: every direction.
LEGACY_OT_SVD_SOLVER = "full"


def ot_svd_task_args(settings: dict[str, Any]) -> list[str]:
    """Command-line flags for OT-tSNE's plan SVD. Manifests without the setting ran the full
    SVD, so they get it explicitly and still reproduce under the newer command-line default."""
    solver = str(settings.get("ot_svd_solver", LEGACY_OT_SVD_SOLVER))
    args = ["--svd-solver", solver]
    if "ot_svd_directions" in settings:
        args.extend(["--svd-directions", str(int(settings["ot_svd_directions"]))])
    return args


#: Benchmark methods that run the current LIGER and take its core setting.
LIGER_METHODS = ("liger-2d", "liger-50")


def liger_cores_task_args(method: str, settings: dict[str, Any]) -> list[str]:
    """Command-line flag for LIGER's iNMF cores. Manifests without the setting get none, so
    they run at rliger's default of 2 cores, as they did."""
    if method in LIGER_METHODS and "liger_n_cores" in settings:
        return ["--liger-n-cores", str(int(settings["liger_n_cores"]))]
    return []


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return str(value) != ""


def _number(value: Any) -> str:
    return f"{float(value):g}"


def configuration_label(
    method: str,
    *,
    rel_reg: Any = None,
    effective_k: Any = None,
    graph_symmetrization: Any = None,
    knndiag_symmetrization: Any = None,
    rel_reg_m: Any = None,
    method_settings: dict[str, Any] | None = None,
) -> str:
    """A short readable name for the settings that define a run.

    It names only what tells runs of the same method apart: the swept OT settings, and the method
    settings that define an arm (MultiMAP native or matched-k, the rc-tsne beta, a non-random
    OT-tSNE initialization). The device is
    left out, since it changes timings rather than results; readers still group by it. The full
    settings are in the ``method_settings`` column.
    """
    parts = [method]
    if _present(rel_reg):
        parts.append(f"rel_reg={_number(rel_reg)}")
    if _present(effective_k):
        parts.append(f"k={int(float(effective_k))}")
    if method in SYMMETRIZATION_METHODS and _present(graph_symmetrization):
        parts.append(f"sym={graph_symmetrization}")
    if method in KNNDIAG_SYMMETRIZATION_METHODS and _present(knndiag_symmetrization):
        parts.append(f"knndiag_sym={knndiag_symmetrization}")
    if method in UOT_METHODS and _present(rel_reg_m):
        parts.append(f"rel_reg_m={_number(rel_reg_m)}")
    settings = method_settings or {}
    if method == "multimap" and "native_profile" in settings:
        native = bool(settings["native_profile"])
        parts.append("native" if native else f"k={int(settings['n_neighbors'])}")
    if method == "rc-tsne" and "beta" in settings:
        parts.append(f"beta={_number(settings['beta'])}")
    # Only a non-random init is named, so the benchmark's own labels stay as they were.
    if method in EPS_K_METHODS and settings.get("ot_initialization", "random") != "random":
        parts.append(f"init={settings['ot_initialization']}")
        jitter = float(settings.get("ot_init_jitter", DEFAULT_INIT_JITTER))
        if jitter != DEFAULT_INIT_JITTER:
            parts.append(f"jitter={_number(jitter)}")
    return " ".join(parts)


def task_configuration_label(task: Task) -> str:
    return configuration_label(
        task.method,
        rel_reg=task.rel_reg,
        effective_k=task.effective_k,
        graph_symmetrization=task.graph_symmetrization,
        knndiag_symmetrization=task.knndiag_symmetrization,
        rel_reg_m=task.rel_reg_m,
        method_settings=task.method_settings,
    )


def with_run_settings(metrics: Any, task_metadata: Any) -> Any:
    """``metrics`` with ``method_settings`` and ``configuration`` on every row it can resolve.

    Tables written by ``finalize`` already carry both. Older ones get them from
    ``task_metadata.csv`` through ``arm_id``; a row whose arm is not listed there (for example one
    swapped in from a supplementary tree) keeps empty values. Returns a copy.
    """
    import pandas as pd

    frame = metrics.copy()
    if "method_settings" not in frame.columns:
        frame["method_settings"] = pd.Series(np.nan, index=frame.index, dtype=object)
    frame["method_settings"] = frame["method_settings"].astype(object)
    missing = frame["method_settings"].isna()
    if missing.any() and {"arm_id", "method_settings"} <= set(task_metadata.columns):
        by_arm = task_metadata.drop_duplicates("arm_id").set_index("arm_id")["method_settings"]
        frame.loc[missing, "method_settings"] = frame.loc[missing, "arm_id"].map(by_arm)
    if "configuration" not in frame.columns:
        frame["configuration"] = pd.Series(np.nan, index=frame.index, dtype=object)
    frame["configuration"] = frame["configuration"].astype(object)
    unlabelled = frame["configuration"].isna() & frame["method_settings"].notna()

    def label(row: pd.Series) -> str:
        return configuration_label(
            str(row["method"]),
            rel_reg=row.get("rel_reg"),
            effective_k=row.get("effective_k"),
            graph_symmetrization=row.get("graph_symmetrization"),
            knndiag_symmetrization=row.get("knndiag_symmetrization"),
            rel_reg_m=row.get("rel_reg_m"),
            method_settings=json.loads(str(row["method_settings"])),
        )

    if unlabelled.any():
        frame.loc[unlabelled, "configuration"] = frame.loc[unlabelled].apply(label, axis=1)
    return frame


@dataclass(frozen=True)
class Task:
    experiment: str
    dataset: str
    method: str
    pipeline: str
    method_seed: int
    visualization_seed: int
    rel_reg: float | None
    effective_k: int | None
    graph_symmetrization: str
    knndiag_symmetrization: str
    rel_reg_m: float | None
    device: str
    preprocessing_hash: str
    sample_hash: str
    manifest_hash: str
    method_settings: dict[str, Any]

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "preprocessing_manifest_hash": self.preprocessing_hash,
            "representation_family": representation_family(self.method),
            "fixed_sample_hash": self.sample_hash,
            "method": self.method,
            "pipeline": self.pipeline,
            "method_seed": self.method_seed,
            "visualization_seed": self.visualization_seed,
            "rel_reg": self.rel_reg,
            "effective_k": self.effective_k,
            "diagonal_construction": DIAGONALS.get(self.method, "not_applicable"),
            "graph_symmetrization": self.graph_symmetrization,
            "knndiag_symmetrization": self.knndiag_symmetrization,
            "transport": (
                {"type": "unbalanced", "rel_reg_m": self.rel_reg_m}
                if self.method.startswith("ot-uot-")
                else {"type": "balanced"}
            ),
            "backend": "tsne",
            "device": self.device,
            "method_settings": self.method_settings,
        }

    @property
    def run_id(self) -> str:
        return canonical_hash(self.identity)[:20]

    @property
    def arm_identity(self) -> dict[str, Any]:
        identity = dict(self.identity)
        for key in ("method_seed", "visualization_seed", "rel_reg", "effective_k"):
            identity.pop(key)
        return identity

    @property
    def arm_id(self) -> str:
        return f"{self.method}--{canonical_hash(self.arm_identity)[:16]}"


def preprocessing_state(
    manifest: dict[str, Any], dataset: str, *, allow_pending: bool, persist_sample: bool
):
    root = data_root(manifest)
    path = root / dataset / "preprocessing_manifest.json"
    if not path.exists():
        if allow_pending:
            return "PENDING_PREPROCESSING", "PENDING_FIXED_SAMPLE"
        raise FileNotFoundError(
            f"{path} is missing. Prepare the dataset first: "
            f"uv run python scripts/prepare_data.py {dataset}"
        )
    prep = json.loads(path.read_text())
    prep_hash = str(prep["preprocessing_hash"])
    dataset_spec = manifest["datasets"][dataset]
    if dataset_spec.get("sample_policy") != "all_cells":
        raise ValueError(
            f"{dataset}: benchmark requires sample_policy='all_cells', got "
            f"{dataset_spec.get('sample_policy')!r}."
        )
    if dataset_spec.get("max_cells_batch0") or dataset_spec.get("max_cells_batch1"):
        raise ValueError(f"{dataset}: all_cells cannot be combined with cell caps.")
    caps = [
        dataset_spec.get("max_cells_batch0") or None,
        dataset_spec.get("max_cells_batch1") or None,
    ]
    sample_seed = int(manifest["seeds"]["sample"])
    data = load_benchmark_data(
        dataset,
        data_root=root,
        seed=sample_seed,
        max_cells_per_batch=None,
        max_cells_batch0=caps[0],
        max_cells_batch1=caps[1],
    )
    cell_ids0 = np.load(root / dataset / "cell_ids1.npy").astype(str)
    cell_ids1 = np.load(root / dataset / "cell_ids2.npy").astype(str)
    # The sample hash identifies the ordered cell set, not an inert seed. sample_seed remains
    # provenance metadata so a future stochastic policy can use it without changing this contract.
    identity = {
        "dataset": dataset,
        "preprocessing_hash": prep_hash,
        "sample_policy": "all_cells",
        "indices0": data.sample_indices0.tolist(),
        "indices1": data.sample_indices1.tolist(),
        "cell_ids0": cell_ids0[data.sample_indices0].tolist(),
        "cell_ids1": cell_ids1[data.sample_indices1].tolist(),
    }
    sample_hash = canonical_hash(identity)
    if persist_sample:
        sample_dir = root / dataset / "fixed_samples" / sample_hash
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_manifest = sample_dir / "sample_manifest.json"
        encoded = (
            json.dumps(
                {
                    "sample_hash": sample_hash,
                    "identity": identity,
                    "metadata": {"sample_seed": sample_seed},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if sample_manifest.exists() and sample_manifest.read_text() != encoded:
            raise ValueError(f"Refusing to replace inconsistent fixed sample at {sample_manifest}.")
        with tempfile.NamedTemporaryFile(
            mode="w", dir=sample_dir, prefix=".sample_manifest.", suffix=".tmp", delete=False
        ) as handle:
            handle.write(encoded)
            manifest_tmp = Path(handle.name)
        manifest_tmp.replace(sample_manifest)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=sample_dir, prefix=".sample_indices.", suffix=".tmp", delete=False
        ) as handle:
            np.savez_compressed(
                handle,
                sample_indices0=data.sample_indices0,
                sample_indices1=data.sample_indices1,
            )
            indices_tmp = Path(handle.name)
        indices_tmp.replace(sample_dir / "sample_indices.npz")
    return prep_hash, sample_hash


def method_devices(
    declared: list[str], method: str, device_filter: str | None = None
) -> list[str]:
    """Devices one method runs on in one experiment.

    Without a filter, the experiment's declared devices. With ``--device cuda``, every method that
    solves an OT plan (OT-tSNE and its variants, and EOT; the only ones that can solve on a GPU)
    runs on the GPU in every experiment, even where the experiment declares only the CPU; other
    methods have no GPU task. With ``--device cpu``, only declared CPU tasks.
    """
    if method not in OT_PLAN_METHODS:
        declared = [device for device in declared if device == "cpu"]
    if device_filter is None:
        return list(declared)
    if device_filter in declared:
        return [device_filter]
    if device_filter == "cuda" and method in OT_PLAN_METHODS:
        return ["cuda"]
    return []


def expand_tasks(
    manifest: dict[str, Any],
    manifest_hash: str,
    *,
    dataset_filter: str | None = None,
    experiment_filter: str | None = None,
    method_filter: str | None = None,
    rel_reg_filter: float | None = None,
    effective_k_filter: int | None = None,
    profile_name: str | None = None,
    device_filter: str | None = None,
    include_disabled: bool = False,
    allow_pending: bool = False,
    persist_sample: bool = False,
) -> list[Task]:
    profile = manifest.get("profiles", {}).get(profile_name, {}) if profile_name else {}
    downstream = [
        int(value) for value in profile.get("downstream_seeds", manifest["seeds"]["downstream"])
    ]
    if downstream != list(range(len(downstream))):
        raise ValueError("The reused aggregate core currently requires downstream seeds 0..N-1.")
    tasks: list[Task] = []
    for experiment in manifest["experiments"]:
        if (not include_disabled and not experiment.get("enabled", True)) or (
            experiment_filter and experiment["name"] != experiment_filter
        ):
            continue
        for dataset in experiment["datasets"]:
            if dataset_filter and dataset != dataset_filter:
                continue
            if not include_disabled and not manifest["datasets"][dataset].get("enabled", False):
                continue
            prep_hash, sample_hash = preprocessing_state(
                manifest,
                dataset,
                allow_pending=allow_pending,
                persist_sample=persist_sample,
            )
            declared_devices = experiment.get("devices", [experiment.get("device", "cpu")])
            symmetrizations = experiment.get("graph_symmetrizations", ["or"])
            methods = [
                method
                for method in experiment["methods"]
                if method_filter is None or method == method_filter
            ]
            if profile.get("canonical_only", False):
                symmetrizations = [experiment.get("canonical_graph_symmetrization", "or")]
            for method in methods:
                rel_regs = (
                    experiment.get("rel_reg", manifest["grid"]["rel_reg"])
                    if method in OT_PLAN_METHODS
                    else [None]
                )
                if profile.get("canonical_only", False) and method in OT_PLAN_METHODS:
                    rel_regs = [manifest["grid"]["canonical_rel_reg"]]
                effective_ks = (
                    manifest["grid"]["effective_k"] if method in EPS_K_METHODS else [None]
                )
                method_symms = symmetrizations if method in DIAGONALS else ["or"]
                rel_reg_ms = experiment.get("rel_reg_m", [None]) or []
                if not method.startswith("ot-uot-"):
                    rel_reg_ms = [None]
                elif profile.get("canonical_only", False):
                    rel_reg_ms = [experiment["canonical_rel_reg_m"]]
                devices = method_devices(declared_devices, method, device_filter)
                for device in devices:
                    for graph_symm in method_symms:
                        for rel_reg_m in rel_reg_ms:
                            for rel_reg in rel_regs:
                                for effective_k in effective_ks:
                                    for seed in downstream:
                                        tasks.append(
                                            Task(
                                                experiment=experiment["name"],
                                                dataset=dataset,
                                                method=method,
                                                pipeline=str(experiment["pipeline"]),
                                                method_seed=seed,
                                                visualization_seed=seed,
                                                rel_reg=rel_reg,
                                                effective_k=effective_k,
                                                graph_symmetrization=graph_symm,
                                                knndiag_symmetrization=experiment.get(
                                                    "knndiag_symmetrization", "or"
                                                ),
                                                rel_reg_m=rel_reg_m,
                                                device=device,
                                                preprocessing_hash=prep_hash,
                                                sample_hash=sample_hash,
                                                manifest_hash=manifest_hash,
                                                method_settings={
                                                    **scientific_method_settings(manifest, method),
                                                    **(
                                                        {"ot_dtype": experiment["cuda_dtype"]}
                                                        if "cuda_dtype" in experiment
                                                        else {}
                                                    ),
                                                },
                                            )
                                        )
    return _filter_grid_point(
        tasks, rel_reg_filter=rel_reg_filter, effective_k_filter=effective_k_filter
    )


def _filter_grid_point(
    tasks: list[Task], *, rel_reg_filter: float | None, effective_k_filter: int | None
) -> list[Task]:
    """Keep the tasks at one rel_reg and/or one k: one line of the grid, or one point.

    Tasks without that setting are dropped. A value no task has is an error, so a typo cannot
    select nothing. The kept tasks are the very tasks of the full grid (same run_id and file).
    """
    for name, value in (("rel_reg", rel_reg_filter), ("effective_k", effective_k_filter)):
        if value is None:
            continue
        available = sorted({getattr(task, name) for task in tasks} - {None})
        if value not in available:
            raise ValueError(f"no selected task has {name}={value!r}; the grid has {available}.")
        tasks = [task for task in tasks if getattr(task, name) == value]
    return tasks


def result_base(manifest: dict[str, Any], manifest_hash: str) -> Path:
    artifact_root = os.environ.get("NE_ALIGNED_ARTIFACT_ROOT")
    if artifact_root:
        return Path(artifact_root).expanduser().resolve()
    return configured_result_root(manifest) / f"{manifest['name']}--{manifest_hash[:12]}"


def arm_dir(manifest: dict[str, Any], task: Task) -> Path:
    return result_base(manifest, task.manifest_hash) / task.dataset / task.experiment / task.arm_id


def _experiment(manifest: dict[str, Any], task: Task) -> dict[str, Any]:
    return next(item for item in manifest["experiments"] if item["name"] == task.experiment)


def point_suffix(manifest: dict[str, Any], task: Task) -> str:
    """Suffix of the aggregate and metrics files of the task's arm.

    Empty unless the experiment sets ``aggregate_per_point``. Then every (rel_reg, k) point gets
    its own ``aggregate--rel_reg_<r>__k_<k>.npz`` and ``metrics--...csv``, so any subset of the grid
    can be aggregated and scored, and later runs reuse the points already done.
    """
    if not _experiment(manifest, task).get("aggregate_per_point", False):
        return ""
    rel_reg = "na" if task.rel_reg is None else format_token(format_rel_reg(task.rel_reg))
    effective_k = "na" if task.effective_k is None else format_token(task.effective_k)
    return f"--rel_reg_{rel_reg}__k_{effective_k}"


def aggregate_path(manifest: dict[str, Any], task: Task) -> Path:
    return arm_dir(manifest, task) / f"aggregate{point_suffix(manifest, task)}.npz"


def arm_metrics_path(manifest: dict[str, Any], task: Task) -> Path:
    return arm_dir(manifest, task) / f"metrics{point_suffix(manifest, task)}.csv"


def common_cli(manifest: dict[str, Any], task: Task) -> list[str]:
    spec = manifest["datasets"][task.dataset]
    settings = manifest["settings"]
    args = [
        "--dataset",
        task.dataset,
        "--data-root",
        str(data_root(manifest)),
        "--preprocessing-seed",
        str(manifest["seeds"]["preprocessing"]),
        "--sample-seed",
        str(manifest["seeds"]["sample"]),
        "--preprocessing-hash",
        task.preprocessing_hash,
        "--sample-hash",
        task.sample_hash,
        "--representation-family",
        representation_family(task.method),
        "--method-pipeline",
        task.pipeline,
        "--benchmark-manifest-hash",
        task.manifest_hash,
        "--embedding-dim",
        str(settings["embedding_dim"]),
        "--n-epochs",
        str(settings["visualization_epochs"]),
        "--pca-components",
        str(settings["pca_components"]),
        "--perplexity",
        str(settings["perplexity"]),
        "--n-neighbors",
        str(settings["n_neighbors"]),
        "--tol",
        str(settings["ot_tolerance"]),
        "--max-iter",
        str(settings["ot_max_iterations"]),
        "--threads",
        str(manifest["resources"]["cpu"]["threads"]),
        "--n-jobs",
        str(manifest["resources"]["cpu"]["threads"]),
        "--ot-device",
        task.device,
        "--graph-symmetrization",
        task.graph_symmetrization,
        "--knndiag-symmetrization",
        task.knndiag_symmetrization,
    ]
    if task.method in EPS_K_METHODS:
        # Only OT-tSNE takes the init setting; every other method refuses a non-random one.
        args.extend(["--init", str(settings["ot_init"])])
    if "tsne_early_exaggeration_iter" in settings:
        args.extend(
            ["--tsne-early-exaggeration-iter", str(settings["tsne_early_exaggeration_iter"])]
        )
    if "ot_init_jitter" in settings:
        args.extend(["--init-jitter", str(settings["ot_init_jitter"])])
    if task.method == "ot":
        args.extend(ot_svd_task_args(settings))
    args.extend(liger_cores_task_args(task.method, settings))
    # Older manifests lack the setting and ran with the selected-gene divisor; pass it
    # explicitly so they still reproduce under the newer command-line default.
    count_library_size = settings.get("count_library_size", LEGACY_COUNT_LIBRARY_SIZE)
    args.extend(["--count-library-size", str(count_library_size)])
    if task.device == "cpu":
        args.append("--require-cpu")
    if spec.get("max_cells_batch0"):
        args.extend(["--max-cells-batch0", str(spec["max_cells_batch0"])])
    if spec.get("max_cells_batch1"):
        args.extend(["--max-cells-batch1", str(spec["max_cells_batch1"])])
    if task.rel_reg_m is not None:
        args.extend(["--rel-reg-m", str(task.rel_reg_m)])
    return args


def task_output(manifest: dict[str, Any], task: Task) -> Path:
    output_dir = arm_dir(manifest, task) / "tasks"
    settings = manifest["settings"]
    ns = argparse.Namespace(
        output_dir=output_dir,
        dataset=task.dataset,
        max_cells_per_batch=None,
        max_cells_batch0=manifest["datasets"][task.dataset].get("max_cells_batch0") or None,
        max_cells_batch1=manifest["datasets"][task.dataset].get("max_cells_batch1") or None,
        embedding_dim=settings["embedding_dim"],
        n_epochs=settings["visualization_epochs"],
        pca_components=settings["pca_components"],
        backend="tsne",
        n_neighbors=None,
    )
    return resolve_task_output_path(
        output_dir / config_token(ns),
        task_kind=task.method,
        seed=task.method_seed,
        rel_reg=task.rel_reg,
        effective_k=task.effective_k,
        backend="tsne",
    )


def task_command(manifest: dict[str, Any], task: Task) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ne_aligned_utils.benchmark.sweep",
        "run-task",
        *common_cli(manifest, task),
        "--output-dir",
        str(arm_dir(manifest, task) / "tasks"),
        "--task-kind",
        task.method,
        "--seed",
        str(task.method_seed),
        "--method-seed",
        str(task.method_seed),
        "--visualization-seed",
        str(task.visualization_seed),
        "--run-id",
        task.run_id,
        "--task-output",
        str(task_output(manifest, task)),
    ]
    if task.rel_reg is not None:
        command.extend(["--rel-reg", repr(task.rel_reg)])
    if task.effective_k is not None:
        command.extend(["--effective-k", str(task.effective_k)])
    if _experiment(manifest, task).get("reuse_ot_cache", False) and task.method == "ot":
        # Tasks that differ only in k or seed share the plan and its SVD (methods/ot_cache.py).
        cache_dir = result_base(manifest, task.manifest_hash) / "ot_cache" / task.dataset
        command.extend(["--ot-cache-dir", str(cache_dir)])
    if task.method == "multimap":
        command.append(
            "--multimap-native"
            if bool(task.method_settings["native_profile"])
            else "--no-multimap-native"
        )
    if task.method == "rc-tsne":
        command.extend(["--rc-beta", repr(float(task.method_settings["beta"]))])
    return command


def format_plan(manifest: dict[str, Any], tasks: list[Task]) -> str:
    lines = [
        "experiment\tdataset\trepresentation_family\tpipeline\tmethod\tarm\tseed\t"
        "rel_reg\trel_reg_m\trel_reg_role\tconfiguration_role\tk\tsymm\t"
        "knndiag_symm\ttransport\tdevice\trun_id\toutput"
    ]
    canonical_rel_reg = float(manifest["grid"]["canonical_rel_reg"])
    for task in tasks:
        rel_reg_role = (
            "not_applicable"
            if task.rel_reg is None
            else "canonical"
            if task.rel_reg == canonical_rel_reg
            else "sensitivity"
        )
        configuration_role = (
            "canonical"
            if task.method in DIAGONALS
            and task.rel_reg == canonical_rel_reg
            and task.graph_symmetrization == "or"
            else "sensitivity"
            if task.method in DIAGONALS
            else "not_applicable"
        )
        lines.append(
            "\t".join(
                str(value)
                for value in (
                    task.experiment,
                    task.dataset,
                    representation_family(task.method),
                    task.pipeline,
                    task.method,
                    task.arm_id,
                    task.method_seed,
                    task.rel_reg,
                    task.rel_reg_m,
                    rel_reg_role,
                    configuration_role,
                    task.effective_k,
                    task.graph_symmetrization,
                    task.knndiag_symmetrization,
                    "unbalanced" if task.method.startswith("ot-uot-") else "balanced",
                    task.device,
                    task.run_id,
                    task_output(manifest, task),
                )
            )
        )
    if any(task.preprocessing_hash.startswith("PENDING_") for task in tasks):
        suffix = "; PENDING hashes/run_ids are provisional until preprocessing exists."
    else:
        suffix = "; preprocessing/sample hashes and run_ids are final."
    lines.append(f"# {len(tasks)} tasks{suffix}")
    return "\n".join(lines) + "\n"

def completed_task_is_valid(manifest: dict[str, Any], task: Task) -> bool:
    """Return True when an existing task artifact is safe to reuse."""
    path = task_output(manifest, task)
    if not path.is_file():
        return False

    expected = {
        "task_kind": task.method,
        "seed": task.method_seed,
        "method_seed": task.method_seed,
        "visualization_seed": task.visualization_seed,
        "input_source": input_kind_for_method(task.method),
        "run_id": task.run_id,
        "preprocessing_hash": task.preprocessing_hash,
        "sample_hash": task.sample_hash,
        "representation_family": representation_family(task.method),
        "method_pipeline": task.pipeline,
    }

    try:
        with np.load(path, allow_pickle=False) as payload:
            for field, value in expected.items():
                if field not in payload:
                    return False
                if np.asarray(payload[field]).item() != value:
                    return False

            if "embedding" not in payload:
                return False

            embedding = np.asarray(payload["embedding"])
            if (
                embedding.ndim != 2
                or embedding.shape[1] != int(manifest["settings"]["embedding_dim"])
                or not np.isfinite(embedding).all()
            ):
                return False

    except (OSError, KeyError, ValueError):
        return False

    return True

def run_tasks(manifest: dict[str, Any], tasks: list[Task]) -> None:
    threads = int(manifest["resources"]["cpu"]["threads"])
    env = dict(os.environ)
    env.update(thread_env(threads))
    env["CUDA_VISIBLE_DEVICES"] = (
        "" if all(task.device == "cpu" for task in tasks) else env.get("CUDA_VISIBLE_DEVICES", "")
    )

    for task in tasks:
        output = task_output(manifest, task)
        output.parent.mkdir(parents=True, exist_ok=True)

        if completed_task_is_valid(manifest, task):
            print(
                f"Skipping completed task: {task.method}, "
                f"seed={task.method_seed}, output={output}"
            )
            continue

        subprocess.run(task_command(manifest, task), cwd=ROOT, env=env, check=True)


def selection_name(
    dataset_filter: str | None,
    experiment_filter: str | None,
    device_filter: str | None,
    rel_reg_filter: float | None = None,
    effective_k_filter: int | None = None,
) -> str | None:
    """File stem of a filtered selection, or ``None`` for the whole benchmark.

    The rel_reg and k filters are appended only when given, so other names are unchanged.
    """
    if (
        dataset_filter is None
        and experiment_filter is None
        and device_filter is None
        and rel_reg_filter is None
        and effective_k_filter is None
    ):
        return None
    parts = [
        dataset_filter or "all-datasets",
        experiment_filter or "all-experiments",
        device_filter or "all-devices",
    ]
    if rel_reg_filter is not None:
        parts.append(f"rel_reg_{format_token(format_rel_reg(rel_reg_filter))}")
    if effective_k_filter is not None:
        parts.append(f"k_{format_token(effective_k_filter)}")
    return "--".join(parts)


def write_task_metadata(
    manifest: dict[str, Any],
    tasks: list[Task],
    *,
    dataset_filter: str | None = None,
    experiment_filter: str | None = None,
    device_filter: str | None = None,
    rel_reg_filter: float | None = None,
    effective_k_filter: int | None = None,
) -> Path:
    """Write a readable view of canonical task metadata without creating a second authority.

    The whole benchmark writes ``task_metadata.csv``. A filtered selection (for example the GPU
    jobs) writes ``task_metadata/<selection>.csv`` instead, so it never replaces the full table.
    """
    if not tasks:
        raise ValueError("Cannot write task metadata for an empty task selection.")
    rows: list[dict[str, Any]] = []
    for task in tasks:
        output = task_output(manifest, task)
        if not output.is_file():
            continue
        rows.append(
            {
                "dataset": task.dataset,
                "experiment": task.experiment,
                "method": task.method,
                "pipeline": task.pipeline,
                "input_representation": input_kind_for_method(task.method),
                "representation_family": representation_family(task.method),
                "seed": task.method_seed,
                "visualization_seed": task.visualization_seed,
                "rel_reg": "" if task.rel_reg is None else task.rel_reg,
                "effective_k": "" if task.effective_k is None else task.effective_k,
                "diagonal_construction": DIAGONALS.get(task.method, "not_applicable"),
                "graph_symmetrization": task.graph_symmetrization,
                "knndiag_symmetrization": task.knndiag_symmetrization,
                "transport": "unbalanced" if task.method.startswith("ot-uot-") else "balanced",
                "rel_reg_m": "" if task.rel_reg_m is None else task.rel_reg_m,
                "backend": "tsne",
                "device": task.device,
                "method_settings": json.dumps(task.method_settings, sort_keys=True),
                "preprocessing_hash": task.preprocessing_hash,
                "sample_hash": task.sample_hash,
                "run_id": task.run_id,
                "arm_id": task.arm_id,
                "manifest_hash": task.manifest_hash,
                "output_embedding_path": str(
                    output.relative_to(result_base(manifest, task.manifest_hash))
                ),
            }
        )
    base = result_base(manifest, tasks[0].manifest_hash)
    selection = selection_name(
        dataset_filter, experiment_filter, device_filter, rel_reg_filter, effective_k_filter
    )
    output = (
        base / "task_metadata.csv"
        if selection is None
        else base / "task_metadata" / f"{selection}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _union_fieldnames(rows) or ["dataset", "experiment", "method", "run_id"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved readable task metadata to {output}")
    return output


def arm_groups(tasks: list[Task], manifest: dict[str, Any] | None = None) -> list[list[Task]]:
    """Tasks aggregated together: one arm, or one point of an arm with per-point aggregation."""
    groups: dict[tuple[str, str, str, str], list[Task]] = {}
    for task in tasks:
        suffix = "" if manifest is None else point_suffix(manifest, task)
        groups.setdefault((task.dataset, task.experiment, task.arm_id, suffix), []).append(task)
    return [groups[key] for key in sorted(groups)]


def aggregate_tasks(manifest: dict[str, Any], tasks: list[Task]) -> None:
    threads = int(manifest["resources"]["cpu"]["threads"])
    env = dict(os.environ)
    env.update(thread_env(threads))
    for group in arm_groups(tasks, manifest):
        task = group[0]
        rel_regs = sorted(
            {item.rel_reg for item in group if item.rel_reg is not None}, reverse=True
        )
        effective_ks = sorted({item.effective_k for item in group if item.effective_k is not None})
        command = [
            sys.executable,
            "-m",
            "ne_aligned_utils.benchmark.sweep",
            "aggregate",
            *common_cli(manifest, task),
            "--output-dir",
            str(arm_dir(manifest, task) / "tasks"),
            "--methods",
            task.method,
            "--n-seeds",
            str(len({item.method_seed for item in group})),
            "--output-cache",
            str(aggregate_path(manifest, task)),
            "--replace",
        ]
        if rel_regs:
            command.extend(["--rel-reg", *(repr(value) for value in rel_regs)])
        if effective_ks:
            command.extend(["--effective-ks", *(str(value) for value in effective_ks)])
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def _union_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    """Every column of every row, in first-seen order; methods may add their own columns."""
    return list(dict.fromkeys(key for row in rows for key in row))


def score_tasks(
    manifest: dict[str, Any],
    tasks: list[Task],
    *,
    dataset_filter: str | None = None,
    experiment_filter: str | None = None,
    device_filter: str | None = None,
    rel_reg_filter: float | None = None,
    effective_k_filter: int | None = None,
) -> None:
    combined: list[dict[str, str]] = []
    threads = int(manifest["resources"]["cpu"]["threads"])
    env = dict(os.environ)
    env.update(thread_env(threads))
    env["LOKY_MAX_CPU_COUNT"] = str(threads)
    canonical_rel_reg = float(manifest["grid"]["canonical_rel_reg"])
    for group in arm_groups(tasks, manifest):
        task = group[0]
        metrics_path = arm_metrics_path(manifest, task)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ne_aligned_utils.benchmark.sweep",
                "score-aggregate",
                "--aggregate-cache",
                str(aggregate_path(manifest, task)),
                "--output-csv",
                str(metrics_path),
                "--methods",
                task.method,
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        with metrics_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                # Label each row by its own rel_reg and k: an arm may hold several grid points.
                row_rel_reg = float(row["rel_reg"]) if _present(row.get("rel_reg")) else None
                row_effective_k = (
                    int(float(row["effective_k"])) if _present(row.get("effective_k")) else None
                )
                row.update(
                    {
                        "experiment": task.experiment,
                        "arm_id": task.arm_id,
                        "preprocessing_hash": task.preprocessing_hash,
                        "sample_hash": task.sample_hash,
                        "representation_family": representation_family(task.method),
                        "pipeline": task.pipeline,
                        "rel_reg_role": (
                            "canonical"
                            if row_rel_reg == canonical_rel_reg
                            else "sensitivity"
                            if row_rel_reg is not None
                            else "not_applicable"
                        ),
                        "graph_symmetrization": task.graph_symmetrization,
                        "knndiag_symmetrization": task.knndiag_symmetrization,
                        "transport": "unbalanced"
                        if task.method.startswith("ot-uot-")
                        else "balanced",
                        "rel_reg_m": "" if task.rel_reg_m is None else str(task.rel_reg_m),
                        "device": task.device,
                        # Every setting of the run, so rows that differ anywhere stay apart.
                        "method_settings": json.dumps(task.method_settings, sort_keys=True),
                        "configuration": configuration_label(
                            task.method,
                            rel_reg=row_rel_reg,
                            effective_k=row_effective_k,
                            graph_symmetrization=task.graph_symmetrization,
                            knndiag_symmetrization=task.knndiag_symmetrization,
                            rel_reg_m=task.rel_reg_m,
                            method_settings=task.method_settings,
                        ),
                    }
                )
                combined.append(row)
    if not combined:
        return
    base = result_base(manifest, tasks[0].manifest_hash)
    selection = selection_name(
        dataset_filter, experiment_filter, device_filter, rel_reg_filter, effective_k_filter
    )
    output = base / "metrics.csv" if selection is None else base / "metrics" / f"{selection}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _union_fieldnames(combined)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)
    print(f"Saved combined metrics to {output}")
    if selection is None:
        for dataset in sorted({row["dataset"] for row in combined}):
            dataset_rows = [row for row in combined if row["dataset"] == dataset]
            dataset_output = base / "metrics" / f"{dataset}--all-experiments--all-devices.csv"
            dataset_output.parent.mkdir(parents=True, exist_ok=True)
            with dataset_output.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(dataset_rows)
            print(f"Saved dataset metrics to {dataset_output}")


def validate_tasks(manifest: dict[str, Any], tasks: list[Task]) -> None:
    """Check that every selected task, aggregate, and scored arm exists and is readable."""
    missing_tasks: list[Path] = []
    invalid_tasks: list[Path] = []
    for task in tasks:
        path = task_output(manifest, task)
        if not path.is_file():
            missing_tasks.append(path)
            continue
        try:
            with np.load(path, allow_pickle=False) as payload:
                expected = {
                    "run_id": task.run_id,
                    "preprocessing_hash": task.preprocessing_hash,
                    "sample_hash": task.sample_hash,
                    "representation_family": representation_family(task.method),
                    "method_pipeline": task.pipeline,
                    "task_kind": task.method,
                    "method_seed": task.method_seed,
                    "visualization_seed": task.visualization_seed,
                }
                mismatch = any(
                    field not in payload or np.asarray(payload[field]).item() != value
                    for field, value in expected.items()
                )
                embedding = np.asarray(payload["embedding"])
                mismatch = mismatch or embedding.ndim != 2 or embedding.shape[1] != 2
                mismatch = mismatch or not np.isfinite(embedding).all()
            if mismatch:
                invalid_tasks.append(path)
        except (OSError, ValueError, KeyError):
            invalid_tasks.append(path)

    groups = arm_groups(tasks, manifest)
    missing_aggregates = [
        aggregate_path(manifest, group[0])
        for group in groups
        if not aggregate_path(manifest, group[0]).is_file()
    ]
    missing_metrics = [
        arm_metrics_path(manifest, group[0])
        for group in groups
        if not arm_metrics_path(manifest, group[0]).is_file()
    ]
    print(
        f"Selected {len(tasks)} tasks in {len(groups)} arms: "
        f"missing tasks={len(missing_tasks)}, invalid tasks={len(invalid_tasks)}, "
        f"missing aggregates={len(missing_aggregates)}, missing metrics={len(missing_metrics)}"
    )
    problems = missing_tasks + invalid_tasks + missing_aggregates + missing_metrics
    if problems:
        for path in problems[:25]:
            print(f"MISSING_OR_INVALID\t{path}")
        if len(problems) > 25:
            print(f"... and {len(problems) - 25} more")
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "plan",
            "run",
            "run-local",
            "run-aggregate",
            "aggregate",
            "score",
            "validate",
            "finalize",
            "reproduce",
        ),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--method", default=None)
    # One line or point of the rel_reg x k grid; the tasks are those of the full grid.
    parser.add_argument("--rel-reg", type=float, default=None)
    parser.add_argument("--effective-k", type=int, default=None)
    parser.add_argument("--profile", choices=("smoke",), default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--allow-unprepared", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-plan", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest, manifest_hash = load_manifest(manifest_path)
    if args.include_disabled and args.command != "plan" and not args.experiment:
        raise SystemExit(
            "Executing disabled groups requires both --include-disabled and an explicit "
            "--experiment, so special experiments cannot enter the main benchmark accidentally."
        )
    tasks = expand_tasks(
        manifest,
        manifest_hash,
        dataset_filter=args.dataset,
        experiment_filter=args.experiment,
        method_filter=args.method,
        rel_reg_filter=args.rel_reg,
        effective_k_filter=args.effective_k,
        profile_name=args.profile,
        device_filter=args.device,
        include_disabled=args.include_disabled,
        allow_pending=args.allow_unprepared or args.command == "plan",
        persist_sample=args.command in {"run", "run-local", "run-aggregate", "reproduce"},
    )
    selection = dict(
        dataset_filter=args.dataset,
        experiment_filter=args.experiment,
        device_filter=args.device,
        rel_reg_filter=args.rel_reg,
        effective_k_filter=args.effective_k,
    )
    if args.command == "plan":
        plan = format_plan(manifest, tasks)
        print(plan, end="")
        if args.output_plan is not None:
            args.output_plan.parent.mkdir(parents=True, exist_ok=True)
            args.output_plan.write_text(plan)
    elif args.command in {"run", "run-local"}:
        run_tasks(manifest, tasks)
        write_task_metadata(
            manifest,
            tasks,
            **selection,
        )
    elif args.command == "run-aggregate":
        run_tasks(manifest, tasks)
        aggregate_tasks(manifest, tasks)
    elif args.command == "aggregate":
        aggregate_tasks(manifest, tasks)
    elif args.command == "score":
        score_tasks(
            manifest,
            tasks,
            **selection,
        )
    elif args.command == "validate":
        validate_tasks(manifest, tasks)
    elif args.command == "finalize":
        score_tasks(
            manifest,
            tasks,
            **selection,
        )
        validate_tasks(manifest, tasks)
        write_task_metadata(
            manifest,
            tasks,
            **selection,
        )
    elif args.command == "reproduce":
        run_tasks(manifest, tasks)
        aggregate_tasks(manifest, tasks)
        score_tasks(
            manifest,
            tasks,
            **selection,
        )
        validate_tasks(manifest, tasks)
        write_task_metadata(
            manifest,
            tasks,
            **selection,
        )


if __name__ == "__main__":
    main()
