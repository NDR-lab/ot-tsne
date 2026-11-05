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
    resolve_task_output_path,
)
from ne_aligned_utils.benchmark.sweep.registry import (
    DIAGONAL_CONSTRUCTIONS as DIAGONALS,
    EPS_K_METHODS,
    INTERMEDIATE_DIMENSION_KEYS,
    OT_PLAN_METHODS,
    VISUALIZED_METHODS,
    input_kind_for_method,
    representation_family,
)
from ne_aligned_utils.config import ROOT, thread_env

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
            ot_initialization=settings["ot_init"],
            ot_tolerance=float(settings["ot_tolerance"]),
            ot_max_iterations=int(settings["ot_max_iterations"]),
        )
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
            f"{path} is missing. Build it with scripts/prepare/build_dataset.py first."
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
    return tasks


def result_base(manifest: dict[str, Any], manifest_hash: str) -> Path:
    artifact_root = os.environ.get("NE_ALIGNED_ARTIFACT_ROOT")
    if artifact_root:
        return Path(artifact_root).expanduser().resolve()
    return configured_result_root(manifest) / f"{manifest['name']}--{manifest_hash[:12]}"


def arm_dir(manifest: dict[str, Any], task: Task) -> Path:
    return result_base(manifest, task.manifest_hash) / task.dataset / task.experiment / task.arm_id


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
        "--init",
        str(settings["ot_init"]),
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
    if "tsne_early_exaggeration_iter" in settings:
        args.extend(
            ["--tsne-early-exaggeration-iter", str(settings["tsne_early_exaggeration_iter"])]
        )
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
    dataset_filter: str | None, experiment_filter: str | None, device_filter: str | None
) -> str | None:
    """File stem of a filtered selection, or ``None`` for the whole benchmark."""
    if dataset_filter is None and experiment_filter is None and device_filter is None:
        return None
    return "--".join(
        (
            dataset_filter or "all-datasets",
            experiment_filter or "all-experiments",
            device_filter or "all-devices",
        )
    )


def write_task_metadata(
    manifest: dict[str, Any],
    tasks: list[Task],
    *,
    dataset_filter: str | None = None,
    experiment_filter: str | None = None,
    device_filter: str | None = None,
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
    selection = selection_name(dataset_filter, experiment_filter, device_filter)
    output = (
        base / "task_metadata.csv"
        if selection is None
        else base / "task_metadata" / f"{selection}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["dataset", "experiment", "method", "run_id"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved readable task metadata to {output}")
    return output


def arm_groups(tasks: list[Task]) -> list[list[Task]]:
    groups: dict[tuple[str, str, str], list[Task]] = {}
    for task in tasks:
        groups.setdefault((task.dataset, task.experiment, task.arm_id), []).append(task)
    return [groups[key] for key in sorted(groups)]


def aggregate_tasks(manifest: dict[str, Any], tasks: list[Task]) -> None:
    threads = int(manifest["resources"]["cpu"]["threads"])
    env = dict(os.environ)
    env.update(thread_env(threads))
    for group in arm_groups(tasks):
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
            str(arm_dir(manifest, task) / "aggregate.npz"),
            "--replace",
        ]
        if rel_regs:
            command.extend(["--rel-reg", *(repr(value) for value in rel_regs)])
        if effective_ks:
            command.extend(["--effective-ks", *(str(value) for value in effective_ks)])
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def score_tasks(
    manifest: dict[str, Any],
    tasks: list[Task],
    *,
    dataset_filter: str | None = None,
    experiment_filter: str | None = None,
    device_filter: str | None = None,
) -> None:
    combined: list[dict[str, str]] = []
    threads = int(manifest["resources"]["cpu"]["threads"])
    env = dict(os.environ)
    env.update(thread_env(threads))
    env["LOKY_MAX_CPU_COUNT"] = str(threads)
    for group in arm_groups(tasks):
        task = group[0]
        directory = arm_dir(manifest, task)
        metrics_path = directory / "metrics.csv"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ne_aligned_utils.benchmark.sweep",
                "score-aggregate",
                "--aggregate-cache",
                str(directory / "aggregate.npz"),
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
                            if task.rel_reg == float(manifest["grid"]["canonical_rel_reg"])
                            else "sensitivity"
                            if task.rel_reg is not None
                            else "not_applicable"
                        ),
                        "graph_symmetrization": task.graph_symmetrization,
                        "knndiag_symmetrization": task.knndiag_symmetrization,
                        "transport": "unbalanced"
                        if task.method.startswith("ot-uot-")
                        else "balanced",
                        "rel_reg_m": "" if task.rel_reg_m is None else str(task.rel_reg_m),
                        "device": task.device,
                    }
                )
                combined.append(row)
    if not combined:
        return
    base = result_base(manifest, tasks[0].manifest_hash)
    selection = selection_name(dataset_filter, experiment_filter, device_filter)
    output = base / "metrics.csv" if selection is None else base / "metrics" / f"{selection}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(combined[0])
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

    groups = arm_groups(tasks)
    missing_aggregates = [
        arm_dir(manifest, group[0]) / "aggregate.npz"
        for group in groups
        if not (arm_dir(manifest, group[0]) / "aggregate.npz").is_file()
    ]
    missing_metrics = [
        arm_dir(manifest, group[0]) / "metrics.csv"
        for group in groups
        if not (arm_dir(manifest, group[0]) / "metrics.csv").is_file()
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
        profile_name=args.profile,
        device_filter=args.device,
        include_disabled=args.include_disabled,
        allow_pending=args.allow_unprepared or args.command == "plan",
        persist_sample=args.command in {"run", "run-local", "run-aggregate", "reproduce"},
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
            dataset_filter=args.dataset,
            experiment_filter=args.experiment,
            device_filter=args.device,
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
            dataset_filter=args.dataset,
            experiment_filter=args.experiment,
            device_filter=args.device,
        )
    elif args.command == "validate":
        validate_tasks(manifest, tasks)
    elif args.command == "finalize":
        score_tasks(
            manifest,
            tasks,
            dataset_filter=args.dataset,
            experiment_filter=args.experiment,
            device_filter=args.device,
        )
        validate_tasks(manifest, tasks)
        write_task_metadata(
            manifest,
            tasks,
            dataset_filter=args.dataset,
            experiment_filter=args.experiment,
            device_filter=args.device,
        )
    elif args.command == "reproduce":
        run_tasks(manifest, tasks)
        aggregate_tasks(manifest, tasks)
        score_tasks(
            manifest,
            tasks,
            dataset_filter=args.dataset,
            experiment_filter=args.experiment,
            device_filter=args.device,
        )
        validate_tasks(manifest, tasks)
        write_task_metadata(
            manifest,
            tasks,
            dataset_filter=args.dataset,
            experiment_filter=args.experiment,
            device_filter=args.device,
        )


if __name__ == "__main__":
    main()
