from __future__ import annotations

import argparse
import csv
import pickle
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ne_aligned_utils.data import make_fig1_data, make_joint_gmm  # noqa: E402


SUPPORTED_DATASETS = (
    "nCoV",
    "atac",
    "lupus",
    "human_brain",
    "Boris_sc",
    "Boris sc",
    "boris_sc",
    "yao_10x_ent_l2l3",
    "fig1",
    "gmm",
    "mnist",
)
DATASET_ALIASES = {
    "Boris sc": "Boris_sc",
    "boris_sc": "Boris_sc",
}
TASK_KINDS = (
    "eot-t0",
    "ot-tsne",
    "eot-50-tsne",
    "joint-pca",
    "joint-tsne",
    "seurat",
    "harmony-2d",
    "harmony-50-tsne",
    "scvi-2d",
    "scvi-50-tsne",
)
SCVI_METHODS = ("scvi-2d", "scvi-50-tsne")
DEFAULT_METHODS = tuple(method for method in TASK_KINDS if method not in SCVI_METHODS)
OT_METHODS = ("eot-t0", "ot-tsne", "eot-50-tsne")
DEFAULT_EPS_TIMES = (1.0,)
DEFAULT_EFFECTIVE_KS = (15,)
DEFAULT_EMBEDDING_DIM = 2
DEFAULT_N_EPOCHS = 200
DEFAULT_TSNE_PCA_COMPONENTS = 50
DEFAULT_PERPLEXITY = 30.0
DEFAULT_TOL = 1e-12
DEFAULT_MAX_ITER = int(1e4)
DEFAULT_GMM_THETA = 3.0


SLURM_MANIFEST_FIELDNAMES = [
    "task_kind",
    "seed",
    "eps_times",
    "effective_k",
    "job_id",
    "job_name",
    "output_path",
    "log_path",
    "command",
]


@dataclass(frozen=True)
class BenchmarkData:
    dataset: str
    x0: np.ndarray
    x1: np.ndarray
    labels0: np.ndarray | None
    labels1: np.ndarray | None
    label_names: tuple[str, ...]
    dataset_names: tuple[str, str]
    sample_indices0: np.ndarray
    sample_indices1: np.ndarray
    input_source: str = "processed"
    source_path: str | None = None
    feature_names: tuple[str, ...] = ()

    @property
    def joint_data(self) -> np.ndarray:
        return np.concatenate([self.x0, self.x1], axis=0)

    @property
    def batch_ids(self) -> np.ndarray:
        return np.concatenate(
            [
                np.zeros(self.x0.shape[0], dtype=int),
                np.ones(self.x1.shape[0], dtype=int),
            ]
        )

    @property
    def labels_available(self) -> bool:
        return self.labels0 is not None and self.labels1 is not None


def format_token(value: float | int | str | None) -> str:
    text = "none" if value is None else str(value)
    text = text.replace("-", "m").replace(".", "p")
    return "".join(char if char.isalnum() or char == "_" else "_" for char in text)


def canonical_dataset_name(dataset: str) -> str:
    return DATASET_ALIASES.get(str(dataset), str(dataset))


def resolve_data_root(data_root: Path | None) -> Path:
    if data_root is None:
        return ROOT / "data"
    path = Path(data_root).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def resolve_output_base(output_dir: Path | None, dataset: str) -> Path:
    if output_dir is None:
        return ROOT / "data" / "benchmark_embeddings" / dataset
    path = Path(output_dir).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def config_token(args: argparse.Namespace) -> str:
    max_cells = getattr(args, "max_cells_per_batch", None)
    max_cells0 = getattr(args, "max_cells_batch0", None)
    max_cells1 = getattr(args, "max_cells_batch1", None)
    if max_cells0 is not None or max_cells1 is not None:
        cap0 = max_cells if max_cells0 is None else max_cells0
        cap1 = max_cells if max_cells1 is None else max_cells1
        n_token = f"{'all' if cap0 is None else int(cap0)}_{'all' if cap1 is None else int(cap1)}"
    else:
        n_token = "all" if max_cells is None else str(int(max_cells))

    return "__".join(
        [
            f"n_per_batch_{format_token(n_token)}",
            f"q_{format_token(getattr(args, 'embedding_dim', DEFAULT_EMBEDDING_DIM))}",
            f"epochs_{format_token(getattr(args, 'n_epochs', DEFAULT_N_EPOCHS))}",
            f"tsne_pca_{format_token(getattr(args, 'tsne_pca_components', DEFAULT_TSNE_PCA_COMPONENTS))}",
        ]
    )


def resolve_task_root(args: argparse.Namespace) -> Path:
    return resolve_output_base(getattr(args, "output_dir", None), args.dataset) / config_token(args)


def resolve_task_output_path(
    task_root: Path,
    *,
    task_kind: str,
    seed: int,
    eps_times: float | None = None,
    effective_k: int | None = None,
) -> Path:
    seed_token = format_token(seed)
    if task_kind == "eot-t0":
        return task_root / f"eot_t0_seed_{seed_token}_eps_times_{format_token(f'{float(eps_times):g}')}.npz"
    if task_kind == "ot-tsne":
        return (
            task_root
            / f"ot_tsne_seed_{seed_token}_eps_times_{format_token(f'{float(eps_times):g}')}"
            f"_k_{format_token(effective_k)}.npz"
        )
    if task_kind == "eot-50-tsne":
        return task_root / f"eot_50_tsne_seed_{seed_token}_eps_times_{format_token(f'{float(eps_times):g}')}.npz"
    if task_kind == "joint-pca":
        return task_root / f"joint_pca_seed_{seed_token}.npz"
    if task_kind == "joint-tsne":
        return task_root / f"joint_tsne_seed_{seed_token}.npz"
    if task_kind == "seurat":
        return task_root / f"seurat_seed_{seed_token}.npz"
    if task_kind == "harmony-2d":
        return task_root / f"harmony_2d_seed_{seed_token}.npz"
    if task_kind == "harmony-50-tsne":
        return task_root / f"harmony_50_tsne_seed_{seed_token}.npz"
    if task_kind == "scvi-2d":
        return task_root / f"scvi_2d_seed_{seed_token}.npz"
    if task_kind == "scvi-50-tsne":
        return task_root / f"scvi_50_tsne_seed_{seed_token}.npz"
    raise ValueError(f"Unsupported task_kind={task_kind!r}.")


def resolve_aggregate_output_path(task_root: Path, dataset: str) -> Path:
    return task_root / f"{format_token(dataset)}_benchmark_embeddings.npz"


def _encode_labels(labels0: np.ndarray, labels1: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    values = np.concatenate([labels0.astype(str), labels1.astype(str)], axis=0)
    names = tuple(sorted(pd.unique(values).tolist()))
    mapping = {name: idx for idx, name in enumerate(names)}
    encoded0 = np.asarray([mapping[value] for value in labels0.astype(str)], dtype=int)
    encoded1 = np.asarray([mapping[value] for value in labels1.astype(str)], dtype=int)
    return encoded0, encoded1, names


def _labels_are_valid(labels: np.ndarray | None, n: int) -> bool:
    if labels is None:
        return False
    labels = np.asarray(labels)
    return labels.ndim == 1 and labels.shape[0] == int(n)


def _load_method_labels(
    frame: pd.DataFrame,
    *,
    x0_rows: int,
    x1_rows: int,
    label_col: str,
    dataset_name0: str,
    dataset_name1: str,
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[str, ...], tuple[str, str]]:
    if "method" not in frame.columns or label_col not in frame.columns:
        return None, None, (), (dataset_name0, dataset_name1)

    method = frame["method"].astype(str).to_numpy()
    label_values = frame[label_col].astype(str).to_numpy()
    labels0_raw = label_values[method == "data1"]
    labels1_raw = label_values[method == "data2"]
    if labels0_raw.shape[0] != x0_rows or labels1_raw.shape[0] != x1_rows:
        return None, None, (), (dataset_name0, dataset_name1)

    labels0, labels1, label_names = _encode_labels(labels0_raw, labels1_raw)
    names = [dataset_name0, dataset_name1]
    for idx, method_name in enumerate(("data1", "data2")):
        subset = frame.loc[frame["method"].astype(str) == method_name]
        for name_col in ("batch", "batchname", "replicate_id"):
            if name_col in subset.columns and len(subset) > 0:
                names[idx] = str(subset[name_col].iloc[0])
                break
    return labels0, labels1, label_names, (names[0], names[1])


def _load_two_batch_directory_dataset(
    *,
    dataset: str,
    dataset_dir: Path,
    default_dataset_names: tuple[str, str],
) -> BenchmarkData:
    x0_path = dataset_dir / "X1.npy"
    x1_path = dataset_dir / "X2.npy"
    if not x0_path.exists() or not x1_path.exists():
        x0_path = dataset_dir / "data1.npy"
        x1_path = dataset_dir / "data2.npy"
    if not x0_path.exists() or not x1_path.exists():
        raise FileNotFoundError(
            f"{dataset_dir} must contain X1.npy/X2.npy or data1.npy/data2.npy."
        )

    x0 = np.load(x0_path)
    x1 = np.load(x1_path)
    labels0 = np.load(dataset_dir / "labels1.npy") if (dataset_dir / "labels1.npy").exists() else None
    labels1 = np.load(dataset_dir / "labels2.npy") if (dataset_dir / "labels2.npy").exists() else None
    label_names: tuple[str, ...] = ()
    dataset_names = default_dataset_names

    if _labels_are_valid(labels0, x0.shape[0]) and _labels_are_valid(labels1, x1.shape[0]):
        labels0 = np.asarray(labels0, dtype=int)
        labels1 = np.asarray(labels1, dtype=int)
        names_path = dataset_dir / "cell_type_names.npy"
        label_names = (
            tuple(str(value) for value in np.load(names_path, allow_pickle=True).tolist())
            if names_path.exists()
            else tuple(str(value) for value in sorted(np.unique(np.concatenate([labels0, labels1]))))
        )
    else:
        labels0 = labels1 = None

    dataset_names_path = dataset_dir / "dataset_names.npy"
    if dataset_names_path.exists():
        loaded_names = tuple(str(value) for value in np.load(dataset_names_path, allow_pickle=True).tolist())
        if len(loaded_names) >= 2:
            dataset_names = (loaded_names[0], loaded_names[1])

    meta_path = dataset_dir / "cells_metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        for label_col in ("cell_type", "cell_types", "label", "labels"):
            if label_col in meta.columns:
                meta_labels0, meta_labels1, meta_label_names, meta_dataset_names = _load_method_labels(
                    meta,
                    x0_rows=x0.shape[0],
                    x1_rows=x1.shape[0],
                    label_col=label_col,
                    dataset_name0=dataset_names[0],
                    dataset_name1=dataset_names[1],
                )
                if meta_labels0 is not None and meta_labels1 is not None:
                    labels0, labels1, label_names = meta_labels0, meta_labels1, meta_label_names
                dataset_names = meta_dataset_names
                break

    return BenchmarkData(
        dataset=dataset,
        x0=np.asarray(x0, dtype=float),
        x1=np.asarray(x1, dtype=float),
        labels0=labels0,
        labels1=labels1,
        label_names=label_names,
        dataset_names=dataset_names,
        sample_indices0=np.arange(x0.shape[0], dtype=int),
        sample_indices1=np.arange(x1.shape[0], dtype=int),
    )


def _dense_raw_counts(matrix: Any, *, name: str) -> np.ndarray:
    if sp.issparse(matrix):
        data = matrix.data
        if np.any(~np.isfinite(data)):
            raise ValueError(f"{name} contains non-finite raw count values.")
        if np.any(data < 0.0):
            raise ValueError(f"{name} contains negative raw count values.")
        if not np.allclose(data, np.round(data)):
            raise ValueError(f"{name} contains non-integer raw count values.")
        matrix = matrix.toarray()
    else:
        matrix = np.asarray(matrix)
        if np.any(~np.isfinite(matrix)):
            raise ValueError(f"{name} contains non-finite raw count values.")
        if np.any(matrix < 0.0):
            raise ValueError(f"{name} contains negative raw count values.")
        if not np.allclose(matrix, np.round(matrix)):
            raise ValueError(f"{name} contains non-integer raw count values.")
    return np.asarray(matrix, dtype=float)


def _read_feature_list(path: Path) -> tuple[str, ...]:
    with path.open() as handle:
        return tuple(line.strip() for line in handle if line.strip())


def _load_h5ad_counts_subset(
    path: Path,
    *,
    obs_names: pd.Index | list[str] | np.ndarray,
    var_names: tuple[str, ...] | list[str] | np.ndarray | None = None,
    layer: str | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if not path.exists():
        raise FileNotFoundError(path)
    adata = ad.read_h5ad(path, backed="r")
    try:
        obs_index = pd.Index([str(value) for value in obs_names])
        missing_obs = obs_index.difference(pd.Index(adata.obs_names.astype(str)))
        if len(missing_obs) > 0:
            raise ValueError(
                f"{path} is missing {len(missing_obs)} requested cells, "
                f"including {missing_obs[:5].tolist()}."
            )

        if var_names is None:
            var_index = pd.Index(adata.var_names.astype(str))
        else:
            var_index = pd.Index([str(value) for value in var_names])
            missing_var = var_index.difference(pd.Index(adata.var_names.astype(str)))
            if len(missing_var) > 0:
                raise ValueError(
                    f"{path} is missing {len(missing_var)} requested genes, "
                    f"including {missing_var[:5].tolist()}."
                )

        subset = adata[obs_index.tolist(), var_index.tolist()]
        matrix = subset.X if layer is None else subset.layers[layer]
        return _dense_raw_counts(matrix, name=str(path)), tuple(str(value) for value in var_index)
    finally:
        adata.file.close()


def _split_count_matrix_by_method(
    counts: np.ndarray,
    frame: pd.DataFrame,
    *,
    dataset: str,
    label_col: str,
    dataset_names: tuple[str, str],
    source_path: Path,
    feature_names: tuple[str, ...],
) -> BenchmarkData:
    methods = frame["method"].astype(str).to_numpy()
    mask0 = methods == "data1"
    mask1 = methods == "data2"
    if not mask0.any() or not mask1.any():
        raise ValueError(f"{dataset} raw-count metadata must contain both data1 and data2 methods.")
    x0 = counts[mask0]
    x1 = counts[mask1]
    labels0, labels1, label_names, dataset_names = _load_method_labels(
        frame,
        x0_rows=x0.shape[0],
        x1_rows=x1.shape[0],
        label_col=label_col,
        dataset_name0=dataset_names[0],
        dataset_name1=dataset_names[1],
    )
    return BenchmarkData(
        dataset=dataset,
        x0=np.asarray(x0, dtype=float),
        x1=np.asarray(x1, dtype=float),
        labels0=labels0,
        labels1=labels1,
        label_names=label_names,
        dataset_names=dataset_names,
        sample_indices0=np.arange(x0.shape[0], dtype=int),
        sample_indices1=np.arange(x1.shape[0], dtype=int),
        input_source="raw_counts",
        source_path=str(source_path),
        feature_names=feature_names,
    )


def _load_ncov_raw_counts(data_root: Path) -> BenchmarkData:
    dataset_dir = data_root / "nCoV"
    source_path = dataset_dir / "combined_1000genes.h5ad"
    meta = pd.read_csv(dataset_dir / "cells_metadata.csv", index_col=0)
    counts, feature_names = _load_h5ad_counts_subset(
        source_path,
        obs_names=meta.index,
        layer="counts",
    )
    return _split_count_matrix_by_method(
        counts,
        meta,
        dataset="nCoV",
        label_col="cell_type",
        dataset_names=("nCoV-data1", "nCoV-data2"),
        source_path=source_path,
        feature_names=feature_names,
    )


def _load_boris_raw_counts(data_root: Path) -> BenchmarkData:
    dataset_dir = data_root / "human_brain"
    source_path = dataset_dir / "replicates.h5ad"
    selected_path = dataset_dir / "combined_1000genes.h5ad"
    meta = pd.read_csv(dataset_dir / "cells_metadata.csv", index_col=0)
    selected = ad.read_h5ad(selected_path, backed="r")
    try:
        feature_names = tuple(str(value) for value in selected.var_names)
    finally:
        selected.file.close()

    counts, feature_names = _load_h5ad_counts_subset(
        source_path,
        obs_names=meta.index,
        var_names=feature_names,
    )
    cell_metadata_path = dataset_dir / "cell_metadata.csv"
    if cell_metadata_path.exists():
        cell_metadata = pd.read_csv(cell_metadata_path, index_col=0)
        meta = meta.join(cell_metadata[["cell_types"]], how="left")

    return _split_count_matrix_by_method(
        counts,
        meta,
        dataset="Boris_sc",
        label_col="cell_types",
        dataset_names=("Boris_sc-data1", "Boris_sc-data2"),
        source_path=source_path,
        feature_names=feature_names,
    )


def _load_lupus_raw_counts(data_root: Path) -> BenchmarkData:
    dataset_dir = data_root / "lupus"
    source_path = dataset_dir / "raw_counts.h5ad"
    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path} does not exist. Run "
            "`python scripts/prepare_lupus_raw_counts.py --output data/lupus/raw_counts.h5ad` "
            "before running lupus scVI benchmarks."
        )

    meta = pd.read_csv(dataset_dir / "cells_metadata.csv", index_col=0)
    feature_names = _read_feature_list(dataset_dir / "features.txt")
    counts, feature_names = _load_h5ad_counts_subset(
        source_path,
        obs_names=meta.index,
        var_names=feature_names,
    )
    dataset_names = ("lupus-data1", "lupus-data2")
    dataset_names_path = dataset_dir / "dataset_names.npy"
    if dataset_names_path.exists():
        loaded_names = tuple(str(value) for value in np.load(dataset_names_path, allow_pickle=True).tolist())
        if len(loaded_names) >= 2:
            dataset_names = (loaded_names[0], loaded_names[1])

    return _split_count_matrix_by_method(
        counts,
        meta,
        dataset="lupus",
        label_col="cell_type",
        dataset_names=dataset_names,
        source_path=source_path,
        feature_names=feature_names,
    )


def _load_yao_raw_counts(data_root: Path) -> BenchmarkData:
    source_path = data_root / "yao_new" / "ent_L23.pkl"
    ent_meta_path = data_root / "yao_new" / "ent_l2l3_meta.csv"
    full_meta_path = data_root / "yao_10x" / "10x_metadata.csv"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not ent_meta_path.exists():
        raise FileNotFoundError(ent_meta_path)
    if not full_meta_path.exists():
        raise FileNotFoundError(full_meta_path)

    with source_path.open("rb") as handle:
        raw = pickle.load(handle)
    counts = raw["counts"]
    feature_names = tuple(str(value) for value in raw["genes"])

    ent_meta = pd.read_csv(ent_meta_path)
    full_meta = pd.read_csv(full_meta_path, usecols=["sample_name"])
    positions = pd.Series(np.arange(len(full_meta), dtype=np.int64), index=full_meta["sample_name"].astype(str))
    subset_positions = positions.reindex(ent_meta["sample_name"].astype(str))
    if subset_positions.isna().any():
        missing = ent_meta.loc[subset_positions.isna(), "sample_name"].astype(str).head().tolist()
        raise ValueError(f"Could not map {subset_positions.isna().sum()} Yao ENT cells to raw counts, including {missing}.")
    counts = counts[subset_positions.to_numpy(dtype=np.int64)]
    counts = _dense_raw_counts(counts, name=str(source_path))

    batch_values = ent_meta["donor_sex_label"].astype(str)
    batch_order = [value for value in ("F", "M") if value in set(batch_values)]
    if len(batch_order) != 2:
        batch_order = sorted(batch_values.dropna().unique().tolist())[:2]
    if len(batch_order) != 2:
        raise ValueError("Yao raw ENT subset needs exactly two donor_sex_label batches.")

    label_values = ent_meta["cluster_label"].astype(str)
    label_names = tuple(sorted(label_values.unique()))
    label_to_index = {name: idx for idx, name in enumerate(label_names)}
    labels = label_values.map(label_to_index).to_numpy(dtype=int)
    mask0 = batch_values.to_numpy() == batch_order[0]
    mask1 = batch_values.to_numpy() == batch_order[1]

    return BenchmarkData(
        dataset="yao_10x_ent_l2l3",
        x0=np.asarray(counts[mask0], dtype=float),
        x1=np.asarray(counts[mask1], dtype=float),
        labels0=np.asarray(labels[mask0], dtype=int),
        labels1=np.asarray(labels[mask1], dtype=int),
        label_names=label_names,
        dataset_names=(f"ent_l2l3-{batch_order[0]}", f"ent_l2l3-{batch_order[1]}"),
        sample_indices0=np.arange(int(mask0.sum()), dtype=int),
        sample_indices1=np.arange(int(mask1.sum()), dtype=int),
        input_source="raw_counts",
        source_path=str(source_path),
        feature_names=feature_names,
    )


def load_scvi_raw_dataset(dataset: str, data_root: Path) -> BenchmarkData:
    dataset = canonical_dataset_name(dataset)
    if dataset == "yao_10x_ent_l2l3":
        return _load_yao_raw_counts(data_root)
    if dataset == "nCoV":
        return _load_ncov_raw_counts(data_root)
    if dataset == "Boris_sc":
        return _load_boris_raw_counts(data_root)
    if dataset == "lupus":
        return _load_lupus_raw_counts(data_root)
    raise ValueError(f"Raw-count scVI input is not configured for dataset={dataset!r}.")


def load_named_dataset(
    dataset: str,
    data_root: Path,
    seed: int,
    *,
    input_kind: str = "processed",
) -> BenchmarkData:
    data_root = resolve_data_root(data_root)
    dataset = canonical_dataset_name(dataset)
    if input_kind == "raw_counts":
        return load_scvi_raw_dataset(dataset, data_root)
    if input_kind != "processed":
        raise ValueError(f"Unsupported input_kind={input_kind!r}.")

    if dataset == "fig1":
        x0, x1 = make_fig1_data(seed=int(seed))
        return BenchmarkData(
            dataset=dataset,
            x0=np.asarray(x0, dtype=float),
            x1=np.asarray(x1, dtype=float),
            labels0=None,
            labels1=None,
            label_names=(),
            dataset_names=("fig1-data1", "fig1-data2"),
            sample_indices0=np.arange(x0.shape[0], dtype=int),
            sample_indices1=np.arange(x1.shape[0], dtype=int),
        )

    if dataset == "gmm":
        x0, x1 = make_joint_gmm(seed=int(seed), theta=DEFAULT_GMM_THETA)
        _, _, labels0, labels1 = make_joint_gmm(seed=int(seed), theta=DEFAULT_GMM_THETA, clean=True)
        return BenchmarkData(
            dataset=dataset,
            x0=np.asarray(x0, dtype=float),
            x1=np.asarray(x1, dtype=float),
            labels0=np.asarray(labels0, dtype=int),
            labels1=np.asarray(labels1, dtype=int),
            label_names=tuple(str(value) for value in sorted(np.unique(np.concatenate([labels0, labels1])))),
            dataset_names=("gmm-data1", "gmm-data2"),
            sample_indices0=np.arange(x0.shape[0], dtype=int),
            sample_indices1=np.arange(x1.shape[0], dtype=int),
        )

    if dataset == "mnist":
        return _load_two_batch_directory_dataset(
            dataset=dataset,
            dataset_dir=data_root / "mnist",
            default_dataset_names=("mnist-batch0", "mnist-batch1"),
        )

    if dataset == "nCoV":
        dataset_dir = data_root / "nCoV"
        x0 = np.load(dataset_dir / "X1.npy")
        x1 = np.load(dataset_dir / "X2.npy")
        labels0 = np.load(dataset_dir / "labels1.npy") if (dataset_dir / "labels1.npy").exists() else None
        labels1 = np.load(dataset_dir / "labels2.npy") if (dataset_dir / "labels2.npy").exists() else None
        if not _labels_are_valid(labels0, x0.shape[0]) or not _labels_are_valid(labels1, x1.shape[0]):
            labels0 = labels1 = None
            label_names: tuple[str, ...] = ()
        else:
            labels0 = np.asarray(labels0, dtype=int)
            labels1 = np.asarray(labels1, dtype=int)
            names_path = dataset_dir / "cell_type_names.npy"
            label_names = (
                tuple(str(value) for value in np.load(names_path, allow_pickle=True).tolist())
                if names_path.exists()
                else tuple(str(value) for value in sorted(np.unique(np.concatenate([labels0, labels1]))))
            )

        dataset_names = ("nCoV-data1", "nCoV-data2")
        meta_path = dataset_dir / "cells_metadata.csv"
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            _, _, _, dataset_names = _load_method_labels(
                meta,
                x0_rows=x0.shape[0],
                x1_rows=x1.shape[0],
                label_col="cell_type",
                dataset_name0=dataset_names[0],
                dataset_name1=dataset_names[1],
            )
        return BenchmarkData(
            dataset=dataset,
            x0=np.asarray(x0, dtype=float),
            x1=np.asarray(x1, dtype=float),
            labels0=labels0,
            labels1=labels1,
            label_names=label_names,
            dataset_names=dataset_names,
            sample_indices0=np.arange(x0.shape[0], dtype=int),
            sample_indices1=np.arange(x1.shape[0], dtype=int),
        )

    if dataset == "lupus":
        return _load_two_batch_directory_dataset(
            dataset=dataset,
            dataset_dir=data_root / "lupus",
            default_dataset_names=("lupus-data1", "lupus-data2"),
        )

    if dataset == "atac":
        dataset_dir = data_root / "atac"
        x0 = np.load(dataset_dir / "data1.npy")
        x1 = np.load(dataset_dir / "data2.npy")
        labels0 = labels1 = None
        label_names = ()
        dataset_names = ("atac-data1", "atac-data2")
        meta_path = dataset_dir / "cells_metadata.csv"
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            labels0, labels1, label_names, dataset_names = _load_method_labels(
                meta,
                x0_rows=x0.shape[0],
                x1_rows=x1.shape[0],
                label_col="cell_type",
                dataset_name0=dataset_names[0],
                dataset_name1=dataset_names[1],
            )
        return BenchmarkData(
            dataset=dataset,
            x0=np.asarray(x0, dtype=float),
            x1=np.asarray(x1, dtype=float),
            labels0=labels0,
            labels1=labels1,
            label_names=label_names,
            dataset_names=dataset_names,
            sample_indices0=np.arange(x0.shape[0], dtype=int),
            sample_indices1=np.arange(x1.shape[0], dtype=int),
        )

    if dataset in {"human_brain", "Boris_sc"}:
        dataset_dir = data_root / "human_brain"
        x0 = np.load(dataset_dir / "X1.npy")
        x1 = np.load(dataset_dir / "X2.npy")
        labels0 = labels1 = None
        label_names = ()
        dataset_names = (
            ("Boris_sc-data1", "Boris_sc-data2")
            if dataset == "Boris_sc"
            else ("human_brain-data1", "human_brain-data2")
        )
        cell_metadata_path = dataset_dir / "cell_metadata.csv"
        cells_metadata_path = dataset_dir / "cells_metadata.csv"
        if cell_metadata_path.exists() and cells_metadata_path.exists():
            cell_metadata = pd.read_csv(cell_metadata_path, index_col=0)
            cells_metadata = pd.read_csv(cells_metadata_path, index_col=0)
            joined = cells_metadata.join(cell_metadata[["cell_types"]], how="left")
            labels0, labels1, label_names, dataset_names = _load_method_labels(
                joined,
                x0_rows=x0.shape[0],
                x1_rows=x1.shape[0],
                label_col="cell_types",
                dataset_name0=dataset_names[0],
                dataset_name1=dataset_names[1],
            )
        return BenchmarkData(
            dataset=dataset,
            x0=np.asarray(x0, dtype=float),
            x1=np.asarray(x1, dtype=float),
            labels0=labels0,
            labels1=labels1,
            label_names=label_names,
            dataset_names=dataset_names,
            sample_indices0=np.arange(x0.shape[0], dtype=int),
            sample_indices1=np.arange(x1.shape[0], dtype=int),
        )

    if dataset == "yao_10x_ent_l2l3":
        dataset_dir = data_root / "yao_10x_ent_l2l3"
        x = np.load(dataset_dir / "ent_l2l3_pca50.npy")
        meta = pd.read_csv(dataset_dir / "ent_l2l3_meta.csv")
        batch_values = meta["donor_sex_label"].astype(str)
        batch_order = [value for value in ("F", "M") if value in set(batch_values)]
        if len(batch_order) != 2:
            batch_order = sorted(batch_values.dropna().unique().tolist())[:2]
        if len(batch_order) != 2:
            raise ValueError("yao_10x_ent_l2l3 needs exactly two donor_sex_label batches.")

        label_values = meta["cluster_label"].astype(str)
        label_names = tuple(sorted(label_values.unique()))
        label_to_index = {name: idx for idx, name in enumerate(label_names)}
        labels = label_values.map(label_to_index).to_numpy(dtype=int)

        mask0 = batch_values.to_numpy() == batch_order[0]
        mask1 = batch_values.to_numpy() == batch_order[1]
        return BenchmarkData(
            dataset=dataset,
            x0=np.asarray(x[mask0], dtype=float),
            x1=np.asarray(x[mask1], dtype=float),
            labels0=np.asarray(labels[mask0], dtype=int),
            labels1=np.asarray(labels[mask1], dtype=int),
            label_names=label_names,
            dataset_names=(f"ent_l2l3-{batch_order[0]}", f"ent_l2l3-{batch_order[1]}"),
            sample_indices0=np.arange(int(mask0.sum()), dtype=int),
            sample_indices1=np.arange(int(mask1.sum()), dtype=int),
        )

    raise ValueError(f"Unsupported dataset={dataset!r}.")


def _allocate_group_sample_sizes(labels: np.ndarray, n_target: int) -> dict[str, int]:
    values, counts = np.unique(labels.astype(str), return_counts=True)
    n_target = min(int(n_target), int(counts.sum()))
    if n_target <= 0:
        return {str(value): 0 for value in values}

    ideal = counts / counts.sum() * n_target
    sizes = np.floor(ideal).astype(int)
    min_per_group = 1 if n_target >= len(values) else 0
    if min_per_group:
        sizes = np.maximum(sizes, min_per_group)
    sizes = np.minimum(sizes, counts)

    remaining = int(n_target - sizes.sum())
    if remaining > 0:
        order = np.argsort(ideal - sizes)[::-1]
        while remaining > 0:
            changed = False
            for idx in order:
                if sizes[idx] < counts[idx]:
                    sizes[idx] += 1
                    remaining -= 1
                    changed = True
                    if remaining == 0:
                        break
            if not changed:
                break
    elif remaining < 0:
        order = np.argsort(ideal - sizes)
        while remaining < 0:
            changed = False
            for idx in order:
                if sizes[idx] > min_per_group:
                    sizes[idx] -= 1
                    remaining += 1
                    changed = True
                    if remaining == 0:
                        break
            if not changed:
                break

    return {str(value): int(size) for value, size in zip(values, sizes)}


def downsample_indices(
    n_rows: int,
    *,
    n_target: int | None,
    labels: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    n_rows = int(n_rows)
    rng = np.random.default_rng(int(seed))
    if n_target is None or n_rows <= int(n_target):
        return np.arange(n_rows, dtype=int)

    n_target = int(n_target)
    if _labels_are_valid(labels, n_rows):
        labels = np.asarray(labels)
        allocation = _allocate_group_sample_sizes(labels, n_target)
        parts: list[np.ndarray] = []
        for offset, (label_value, n_group) in enumerate(allocation.items()):
            if n_group <= 0:
                continue
            group_indices = np.where(labels.astype(str) == label_value)[0]
            group_rng = np.random.default_rng(int(seed) + 1009 * (offset + 1))
            chosen = group_rng.choice(group_indices, size=int(n_group), replace=False)
            parts.append(np.asarray(chosen, dtype=int))
        if parts:
            indices = np.concatenate(parts)
            return rng.permutation(indices).astype(int)

    return rng.choice(n_rows, size=n_target, replace=False).astype(int)


def load_benchmark_data(
    dataset: str,
    *,
    data_root: Path,
    seed: int,
    max_cells_per_batch: int | None,
    max_cells_batch0: int | None = None,
    max_cells_batch1: int | None = None,
    input_kind: str = "processed",
) -> BenchmarkData:
    data = load_named_dataset(dataset, data_root=data_root, seed=seed, input_kind=input_kind)
    cap0 = max_cells_per_batch if max_cells_batch0 is None else max_cells_batch0
    cap1 = max_cells_per_batch if max_cells_batch1 is None else max_cells_batch1
    idx0 = downsample_indices(
        data.x0.shape[0],
        n_target=cap0,
        labels=data.labels0,
        seed=int(seed),
    )
    idx1 = downsample_indices(
        data.x1.shape[0],
        n_target=cap1,
        labels=data.labels1,
        seed=int(seed) + 10_000,
    )
    labels0 = None if data.labels0 is None else np.asarray(data.labels0, dtype=int)[idx0]
    labels1 = None if data.labels1 is None else np.asarray(data.labels1, dtype=int)[idx1]
    return BenchmarkData(
        dataset=data.dataset,
        x0=np.asarray(data.x0[idx0], dtype=float),
        x1=np.asarray(data.x1[idx1], dtype=float),
        labels0=labels0,
        labels1=labels1,
        label_names=data.label_names,
        dataset_names=data.dataset_names,
        sample_indices0=np.asarray(data.sample_indices0, dtype=int)[idx0],
        sample_indices1=np.asarray(data.sample_indices1, dtype=int)[idx1],
        input_source=data.input_source,
        source_path=data.source_path,
        feature_names=data.feature_names,
    )


def common_metadata(data: BenchmarkData, args: argparse.Namespace, *, task_kind: str) -> dict[str, Any]:
    labels_available = bool(data.labels_available)
    metadata: dict[str, Any] = {
        "task_kind": task_kind,
        "dataset": data.dataset,
        "seed": int(args.seed),
        "embedding_dim": int(args.embedding_dim),
        "n_epochs": int(args.n_epochs),
        "tsne_pca_components": int(args.tsne_pca_components),
        "perplexity": float(args.perplexity),
        "max_cells_per_batch": -1 if args.max_cells_per_batch is None else int(args.max_cells_per_batch),
        "max_cells_batch0": -1 if args.max_cells_batch0 is None else int(args.max_cells_batch0),
        "max_cells_batch1": -1 if args.max_cells_batch1 is None else int(args.max_cells_batch1),
        "batch_sizes": np.asarray([data.x0.shape[0], data.x1.shape[0]], dtype=int),
        "batch_ids": data.batch_ids,
        "sample_indices0": np.asarray(data.sample_indices0, dtype=int),
        "sample_indices1": np.asarray(data.sample_indices1, dtype=int),
        "dataset_names": np.asarray(data.dataset_names),
        "labels_available": labels_available,
        "labels0": np.asarray(data.labels0 if labels_available else [], dtype=int),
        "labels1": np.asarray(data.labels1 if labels_available else [], dtype=int),
        "label_names": np.asarray(data.label_names),
        "input_source": str(data.input_source),
        "source_path": "" if data.source_path is None else str(data.source_path),
        "n_genes": int(data.x0.shape[1]),
    }
    if data.feature_names:
        metadata["feature_names"] = np.asarray(data.feature_names)
    return metadata


def _npz_value(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray(-1)
    if isinstance(value, Path):
        return np.asarray(str(value))
    if isinstance(value, dict):
        return np.asarray(str(value))
    array = np.asarray(value)
    if array.dtype == object:
        return array.astype(str)
    return array


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: _npz_value(value) for key, value in arrays.items()})


def _compute_task(data: BenchmarkData, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    from ne_aligned_utils.methods import (
        compute_eot_embedding_from_arrays,
        compute_eot_then_tsne_embedding_from_arrays,
        compute_ot_tsne_embedding_from_arrays,
        compute_harmony_embedding_from_arrays,
        compute_harmony_then_tsne_embedding_from_arrays,
        compute_pca_embedding_from_arrays,
        compute_scvi_embedding_from_arrays,
        compute_scvi_then_tsne_embedding_from_arrays,
        compute_seurat_embedding_from_arrays,
        compute_tsne_embedding_from_arrays,
    )

    if args.task_kind == "eot-t0":
        embedding, artifacts = compute_eot_embedding_from_arrays(
            data.x0,
            data.x1,
            t=0.0,
            eps_times=float(args.eps_times),
            tol=float(args.tol),
            max_iter=int(args.max_iter),
            embd_dim=int(args.embedding_dim),
            return_artifacts=True,
        )
        return embedding, artifacts, {"eps_times": float(args.eps_times), "t": 0.0}

    if args.task_kind == "ot-tsne":
        embedding, artifacts = compute_ot_tsne_embedding_from_arrays(
            data.x0,
            data.x1,
            eps_times=float(args.eps_times),
            effective_k=int(args.effective_k),
            random_state=int(args.seed),
            n_epochs=int(args.n_epochs),
            tol=float(args.tol),
            max_iter=int(args.max_iter),
            return_artifacts=True,
        )
        return embedding, artifacts, {
            "eps_times": float(args.eps_times),
            "effective_k": int(args.effective_k),
        }

    if args.task_kind == "eot-50-tsne":
        embedding, artifacts = compute_eot_then_tsne_embedding_from_arrays(
            data.x0,
            data.x1,
            t=0.0,
            eps_times=float(args.eps_times),
            eot_dim=50,
            random_state=int(args.seed),
            perplexity=float(args.perplexity),
            n_jobs=int(args.n_jobs),
            tol=float(args.tol),
            max_iter=int(args.max_iter),
            n_iter=int(args.n_epochs),
            return_artifacts=True,
        )
        return embedding, artifacts, {"eps_times": float(args.eps_times), "t": 0.0, "eot_dim": 50}

    if args.task_kind == "joint-pca":
        embedding, artifacts = compute_pca_embedding_from_arrays(
            data.x0,
            data.x1,
            embd_dim=int(args.embedding_dim),
            random_state=int(args.seed),
            return_artifacts=True,
        )
        return embedding, artifacts, {}

    if args.task_kind == "joint-tsne":
        embedding, artifacts = compute_tsne_embedding_from_arrays(
            data.x0,
            data.x1,
            random_state=int(args.seed),
            perplexity=float(args.perplexity),
            n_jobs=int(args.n_jobs),
            pca_components=int(args.tsne_pca_components),
            n_iter=int(args.n_epochs),
            return_artifacts=True,
        )
        return embedding, artifacts, {}

    if args.task_kind == "seurat":
        embedding, artifacts = compute_seurat_embedding_from_arrays(
            data.x0,
            data.x1,
            embd_dim=int(args.embedding_dim),
            return_artifacts=True,
        )
        return embedding, artifacts, {}

    if args.task_kind == "harmony-2d":
        embedding, artifacts = compute_harmony_embedding_from_arrays(
            data.x0,
            data.x1,
            embd_dim=2,
            random_state=int(args.seed),
            return_artifacts=True,
        )
        return embedding, artifacts, {"harmony_dim": 2}

    if args.task_kind == "harmony-50-tsne":
        embedding, artifacts = compute_harmony_then_tsne_embedding_from_arrays(
            data.x0,
            data.x1,
            harmony_dim=50,
            random_state=int(args.seed),
            perplexity=float(args.perplexity),
            n_jobs=int(args.n_jobs),
            n_iter=int(args.n_epochs),
            return_artifacts=True,
        )
        return embedding, artifacts, {"harmony_dim": 50}

    if args.task_kind == "scvi-2d":
        embedding, artifacts = compute_scvi_embedding_from_arrays(
            data.x0,
            data.x1,
            latent_dim=2,
            random_state=int(args.seed),
            max_epochs=400,
            train_kwargs={"enable_progress_bar": False, "batch_size": 2048},
            return_artifacts=True,
        )
        return embedding, artifacts, {"scvi_latent_dim": 2}

    if args.task_kind == "scvi-50-tsne":
        embedding, artifacts = compute_scvi_then_tsne_embedding_from_arrays(
            data.x0,
            data.x1,
            latent_dim=50,
            random_state=int(args.seed),
            max_epochs=400,
            perplexity=float(args.perplexity),
            n_jobs=int(args.n_jobs),
            n_iter=int(args.n_epochs),
            train_kwargs={"enable_progress_bar": False, "batch_size": 2048},
            return_artifacts=True,
        )
        return embedding, artifacts, {"scvi_latent_dim": 50}

    raise ValueError(f"Unsupported task_kind={args.task_kind!r}.")


def run_task(args: argparse.Namespace) -> None:
    output_path = Path(args.task_output).expanduser()
    if output_path.exists() and not args.force:
        print(f"Task output already exists at {output_path}; use --force to recompute.")
        return

    data = load_benchmark_data(
        args.dataset,
        data_root=resolve_data_root(args.data_root),
        seed=int(args.seed),
        max_cells_per_batch=args.max_cells_per_batch,
        max_cells_batch0=args.max_cells_batch0,
        max_cells_batch1=args.max_cells_batch1,
        input_kind="raw_counts" if args.task_kind in SCVI_METHODS else "processed",
    )
    started = time.perf_counter()
    embedding, artifacts, method_params = _compute_task(data, args)
    payload = common_metadata(data, args, task_kind=args.task_kind)
    payload.update(artifacts)
    payload.update(method_params)
    payload["embedding"] = np.asarray(embedding, dtype=float)
    payload["total_seconds"] = float(time.perf_counter() - started)
    save_npz(output_path, **payload)
    print(f"Saved {args.task_kind} embedding to {output_path}")


def _load_embedding(path: Path, n_total: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as cache:
        embedding = np.asarray(cache["embedding"], dtype=float)
    if embedding.ndim != 2 or embedding.shape[0] != int(n_total):
        raise ValueError(f"{path} has embedding shape {embedding.shape}, expected {n_total} rows.")
    return embedding


def _simple_paths(task_root: Path, task_kind: str, seeds: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            str(resolve_task_output_path(task_root, task_kind=task_kind, seed=int(seed)))
            for seed in seeds.tolist()
        ]
    )


def _eps_paths(task_root: Path, task_kind: str, seeds: np.ndarray, eps_values: np.ndarray) -> np.ndarray:
    paths = np.empty((len(seeds), len(eps_values)), dtype=object)
    for seed_idx, seed in enumerate(seeds.tolist()):
        for eps_idx, eps_times in enumerate(eps_values.tolist()):
            paths[seed_idx, eps_idx] = str(
                resolve_task_output_path(
                    task_root,
                    task_kind=task_kind,
                    seed=int(seed),
                    eps_times=float(eps_times),
                )
            )
    return paths


def _eps_k_paths(
    task_root: Path,
    task_kind: str,
    seeds: np.ndarray,
    eps_values: np.ndarray,
    effective_ks: np.ndarray,
) -> np.ndarray:
    paths = np.empty((len(seeds), len(eps_values), len(effective_ks)), dtype=object)
    for seed_idx, seed in enumerate(seeds.tolist()):
        for eps_idx, eps_times in enumerate(eps_values.tolist()):
            for k_idx, effective_k in enumerate(effective_ks.tolist()):
                paths[seed_idx, eps_idx, k_idx] = str(
                    resolve_task_output_path(
                        task_root,
                        task_kind=task_kind,
                        seed=int(seed),
                        eps_times=float(eps_times),
                        effective_k=int(effective_k),
                    )
                )
    return paths


def _stack_path_array(paths: np.ndarray, n_total: int) -> np.ndarray:
    flat_embeddings = [_load_embedding(Path(path), n_total=n_total) for path in paths.reshape(-1)]
    first_shape = flat_embeddings[0].shape
    for path, embedding in zip(paths.reshape(-1), flat_embeddings):
        if embedding.shape != first_shape:
            raise ValueError(f"{path} has embedding shape {embedding.shape}, expected {first_shape}.")
    return np.stack(flat_embeddings, axis=0).reshape(paths.shape + first_shape)


def aggregate(args: argparse.Namespace) -> None:
    task_root = resolve_task_root(args)
    output_path = (
        resolve_aggregate_output_path(task_root, args.dataset)
        if args.output_cache is None
        else Path(args.output_cache).expanduser()
    )

    seeds = np.arange(int(args.n_seeds), dtype=int)
    eps_values = np.asarray(args.eps_times, dtype=float)
    effective_ks = np.asarray(args.effective_ks, dtype=int)
    methods = tuple(args.methods)

    data0 = load_benchmark_data(
        args.dataset,
        data_root=resolve_data_root(args.data_root),
        seed=int(seeds[0]),
        max_cells_per_batch=args.max_cells_per_batch,
        max_cells_batch0=args.max_cells_batch0,
        max_cells_batch1=args.max_cells_batch1,
    )
    n0, n1 = data0.x0.shape[0], data0.x1.shape[0]
    n_total = n0 + n1

    sample_indices0 = np.empty((len(seeds), n0), dtype=int)
    sample_indices1 = np.empty((len(seeds), n1), dtype=int)
    labels_available = bool(data0.labels_available)
    labels0 = np.empty((len(seeds), n0), dtype=int) if labels_available else np.empty((0,), dtype=int)
    labels1 = np.empty((len(seeds), n1), dtype=int) if labels_available else np.empty((0,), dtype=int)

    for seed_idx, seed in enumerate(seeds.tolist()):
        data = load_benchmark_data(
            args.dataset,
            data_root=resolve_data_root(args.data_root),
            seed=int(seed),
            max_cells_per_batch=args.max_cells_per_batch,
            max_cells_batch0=args.max_cells_batch0,
            max_cells_batch1=args.max_cells_batch1,
        )
        if data.x0.shape[0] != n0 or data.x1.shape[0] != n1:
            raise ValueError("Aggregate requires identical sampled batch sizes across seeds.")
        sample_indices0[seed_idx] = data.sample_indices0
        sample_indices1[seed_idx] = data.sample_indices1
        if labels_available:
            if not data.labels_available:
                raise ValueError("Labels were available for seed 0 but missing for a later seed.")
            labels0[seed_idx] = data.labels0
            labels1[seed_idx] = data.labels1

    payload: dict[str, Any] = {
        "dataset": args.dataset,
        "methods": np.asarray(methods),
        "seeds": seeds,
        "eps_times": eps_values,
        "effective_ks": effective_ks,
        "embedding_dim": int(args.embedding_dim),
        "n_epochs": int(args.n_epochs),
        "tsne_pca_components": int(args.tsne_pca_components),
        "perplexity": float(args.perplexity),
        "max_cells_per_batch": -1 if args.max_cells_per_batch is None else int(args.max_cells_per_batch),
        "max_cells_batch0": -1 if args.max_cells_batch0 is None else int(args.max_cells_batch0),
        "max_cells_batch1": -1 if args.max_cells_batch1 is None else int(args.max_cells_batch1),
        "batch_sizes": np.asarray([n0, n1], dtype=int),
        "batch_ids": np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)]),
        "dataset_names": np.asarray(data0.dataset_names),
        "labels_available": labels_available,
        "labels0": labels0,
        "labels1": labels1,
        "label_names": np.asarray(data0.label_names),
        "sample_indices0": sample_indices0,
        "sample_indices1": sample_indices1,
    }

    simple_method_outputs = {
        "joint-pca": ("joint_pca_embeddings", "joint_pca_paths"),
        "joint-tsne": ("joint_tsne_embeddings", "joint_tsne_paths"),
        "seurat": ("seurat_embeddings", "seurat_paths"),
        "harmony-2d": ("harmony_2d_embeddings", "harmony_2d_paths"),
        "harmony-50-tsne": ("harmony_50_tsne_embeddings", "harmony_50_tsne_paths"),
        "scvi-2d": ("scvi_2d_embeddings", "scvi_2d_paths"),
        "scvi-50-tsne": ("scvi_50_tsne_embeddings", "scvi_50_tsne_paths"),
    }
    for method, (embedding_key, path_key) in simple_method_outputs.items():
        if method not in methods:
            continue
        paths = _simple_paths(task_root, method, seeds)
        payload[path_key] = paths
        payload[embedding_key] = _stack_path_array(paths, n_total=n_total)

    eps_method_outputs = {
        "eot-t0": ("eot_t0_embeddings", "eot_t0_paths"),
        "eot-50-tsne": ("eot_50_tsne_embeddings", "eot_50_tsne_paths"),
    }
    for method, (embedding_key, path_key) in eps_method_outputs.items():
        if method not in methods:
            continue
        paths = _eps_paths(task_root, method, seeds, eps_values)
        payload[path_key] = paths
        payload[embedding_key] = _stack_path_array(paths, n_total=n_total)

    if "ot-tsne" in methods:
        paths = _eps_k_paths(task_root, "ot-tsne", seeds, eps_values, effective_ks)
        payload["ot_tsne_paths"] = paths
        payload["ot_tsne_embeddings"] = _stack_path_array(paths, n_total=n_total)

    save_npz(output_path, **payload)
    print(f"Saved aggregated embedding cache to {output_path}")


def _aggregate_cache_methods(cache: dict[str, np.ndarray], requested_methods: list[str] | None) -> tuple[str, ...]:
    if requested_methods is not None:
        return tuple(requested_methods)
    if "methods" in cache:
        return tuple(str(value) for value in np.asarray(cache["methods"]).tolist())
    return DEFAULT_METHODS


def _method_display_label(
    method: str,
    *,
    eps_times: float | None = None,
    effective_k: int | None = None,
) -> str:
    labels = {
        "eot-t0": "EOT t=0",
        "ot-tsne": "OT-tSNE",
        "eot-50-tsne": "EOT50 + tSNE",
        "joint-pca": "Joint PCA",
        "joint-tsne": "Joint tSNE",
        "seurat": "Seurat",
        "harmony-2d": "Harmony 2D",
        "harmony-50-tsne": "Harmony50 + tSNE",
        "scvi-2d": "scVI 2D",
        "scvi-50-tsne": "scVI50 + tSNE",
    }
    label = labels.get(method, method)
    details = []
    if eps_times is not None:
        details.append(f"eps={float(eps_times):g}")
    if effective_k is not None:
        details.append(f"k={int(effective_k)}")
    return label if not details else f"{label} ({', '.join(details)})"


def _require_cache_key(cache: dict[str, np.ndarray], key: str, method: str) -> np.ndarray:
    if key not in cache:
        raise KeyError(f"Aggregate cache is missing {key!r}; was method {method!r} aggregated?")
    return cache[key]


def _iter_aggregate_metric_embeddings(
    cache: dict[str, np.ndarray],
    methods: tuple[str, ...],
):
    seeds = np.asarray(cache["seeds"], dtype=int)
    eps_values = np.asarray(cache.get("eps_times", []), dtype=float)
    effective_ks = np.asarray(cache.get("effective_ks", []), dtype=int)

    simple_method_outputs = {
        "joint-pca": "joint_pca_embeddings",
        "joint-tsne": "joint_tsne_embeddings",
        "seurat": "seurat_embeddings",
        "harmony-2d": "harmony_2d_embeddings",
        "harmony-50-tsne": "harmony_50_tsne_embeddings",
        "scvi-2d": "scvi_2d_embeddings",
        "scvi-50-tsne": "scvi_50_tsne_embeddings",
    }
    eps_method_outputs = {
        "eot-t0": "eot_t0_embeddings",
        "eot-50-tsne": "eot_50_tsne_embeddings",
    }

    for method in methods:
        if method in simple_method_outputs:
            embedding_key = simple_method_outputs[method]
            embeddings = _require_cache_key(cache, embedding_key, method)
            for seed_index, seed in enumerate(seeds.tolist()):
                yield {
                    "method": method,
                    "method_label": _method_display_label(method),
                    "seed_index": int(seed_index),
                    "seed": int(seed),
                    "eps_times": np.nan,
                    "effective_k": np.nan,
                    "embedding_key": embedding_key,
                    "embedding": np.asarray(embeddings[seed_index], dtype=float),
                }
            continue

        if method in eps_method_outputs:
            embedding_key = eps_method_outputs[method]
            embeddings = _require_cache_key(cache, embedding_key, method)
            for seed_index, seed in enumerate(seeds.tolist()):
                for eps_index, eps_times in enumerate(eps_values.tolist()):
                    yield {
                        "method": method,
                        "method_label": _method_display_label(method, eps_times=float(eps_times)),
                        "seed_index": int(seed_index),
                        "seed": int(seed),
                        "eps_times": float(eps_times),
                        "effective_k": np.nan,
                        "embedding_key": embedding_key,
                        "embedding": np.asarray(embeddings[seed_index, eps_index], dtype=float),
                    }
            continue

        if method == "ot-tsne":
            embedding_key = "ot_tsne_embeddings"
            embeddings = _require_cache_key(cache, embedding_key, method)
            for seed_index, seed in enumerate(seeds.tolist()):
                for eps_index, eps_times in enumerate(eps_values.tolist()):
                    for k_index, effective_k in enumerate(effective_ks.tolist()):
                        yield {
                            "method": method,
                            "method_label": _method_display_label(
                                method,
                                eps_times=float(eps_times),
                                effective_k=int(effective_k),
                            ),
                            "seed_index": int(seed_index),
                            "seed": int(seed),
                            "eps_times": float(eps_times),
                            "effective_k": int(effective_k),
                            "embedding_key": embedding_key,
                            "embedding": np.asarray(
                                embeddings[seed_index, eps_index, k_index],
                                dtype=float,
                            ),
                        }
            continue

        raise ValueError(f"Unsupported aggregate metric method={method!r}.")


def score_aggregate(args: argparse.Namespace) -> None:
    from ne_aligned_utils.benchmark_metrics import (
        load_aggregate_cache,
        score_aggregate_cache,
    )

    aggregate_path = Path(args.aggregate_cache).expanduser()
    rows = score_aggregate_cache(
        load_aggregate_cache(aggregate_path),
        methods=args.methods,
        source_cache=aggregate_path,
    )

    output_csv = (
        aggregate_path.with_name(f"{aggregate_path.stem}_metrics.csv")
        if args.output_csv is None
        else Path(args.output_csv).expanduser()
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_csv, index=False)
    print(f"Saved aggregate metrics to {output_csv}")


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_cpu_wrapped_command(
    command: list[str],
    cpus_per_task: int | None,
    *,
    allow_cuda: bool = False,
) -> list[str]:
    env = ["env"]
    if not allow_cuda:
        env.append("CUDA_VISIBLE_DEVICES=")
    if cpus_per_task is not None:
        threads = str(int(cpus_per_task))
        env.extend(
            [
                f"OMP_NUM_THREADS={threads}",
                f"OPENBLAS_NUM_THREADS={threads}",
                f"MKL_NUM_THREADS={threads}",
                f"NUMEXPR_NUM_THREADS={threads}",
            ]
        )
    return [*env, *command]


def build_sbatch_command(
    *,
    job_name: str,
    log_path: Path,
    wrapped_command: list[str],
    chdir: Path,
    time_limit: str | None,
    partition: str | None,
    account: str | None,
    cpus_per_task: int | None,
    mem: str | None,
    extra_args: list[str],
) -> list[str]:
    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        job_name,
        "--output",
        str(log_path),
        "--chdir",
        str(chdir),
    ]
    if time_limit is not None:
        command.extend(["--time", time_limit])
    if partition is not None:
        command.extend(["--partition", partition])
    if account is not None:
        command.extend(["--account", account])
    if cpus_per_task is not None:
        command.extend(["--cpus-per-task", str(int(cpus_per_task))])
    if mem is not None:
        command.extend(["--mem", mem])
    command.extend(extra_args)
    command.extend(["--wrap", shlex.join(wrapped_command)])
    return command


def build_common_run_args(args: argparse.Namespace) -> list[str]:
    common = [
        "--dataset",
        args.dataset,
        "--seed",
        "{seed}",
        "--embedding-dim",
        str(int(args.embedding_dim)),
        "--n-epochs",
        str(int(args.n_epochs)),
        "--tsne-pca-components",
        str(int(args.tsne_pca_components)),
        "--perplexity",
        f"{float(args.perplexity):g}",
        "--n-jobs",
        str(int(args.n_jobs)),
        "--tol",
        f"{float(args.tol):g}",
        "--max-iter",
        str(int(args.max_iter)),
        "--data-root",
        str(resolve_data_root(args.data_root)),
        "--output-dir",
        str(resolve_output_base(args.output_dir, args.dataset)),
    ]
    if args.max_cells_per_batch is not None:
        common.extend(["--max-cells-per-batch", str(int(args.max_cells_per_batch))])
    if args.max_cells_batch0 is not None:
        common.extend(["--max-cells-batch0", str(int(args.max_cells_batch0))])
    if args.max_cells_batch1 is not None:
        common.extend(["--max-cells-batch1", str(int(args.max_cells_batch1))])
    return common


def submit_slurm(args: argparse.Namespace) -> None:
    task_root = resolve_task_root(args)
    logs_dir = task_root / "slurm_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = task_root / "slurm_jobs.csv"
    if args.overwrite_manifest and manifest_path.exists():
        manifest_path.unlink()

    script_path = Path(__file__).resolve()
    python_executable = args.python_executable or sys.executable
    common_template = build_common_run_args(args)
    methods = tuple(args.methods)
    job_counter = 0

    def submit_one(
        *,
        task_kind: str,
        seed: int,
        output_path: Path,
        eps_times: float | None = None,
        effective_k: int | None = None,
    ) -> str:
        nonlocal job_counter
        job_counter += 1
        job_name = f"{args.slurm_job_prefix}-{job_counter:04d}-{task_kind.replace('-', '')}"
        log_path = logs_dir / f"{job_name}.log"
        run_args = [value.format(seed=int(seed)) for value in common_template]
        task_args = [
            "run-task",
            "--task-kind",
            task_kind,
            "--task-output",
            str(output_path),
            *run_args,
        ]
        if eps_times is not None:
            task_args.extend(["--eps-times", f"{float(eps_times):g}"])
        if effective_k is not None:
            task_args.extend(["--effective-k", str(int(effective_k))])
        if args.force:
            task_args.append("--force")

        python_command = [python_executable, str(script_path), *task_args]
        wrapped_command = build_cpu_wrapped_command(
            python_command,
            args.slurm_cpus_per_task,
            allow_cuda=bool(args.slurm_use_gpu),
        )
        sbatch_command = build_sbatch_command(
            job_name=job_name,
            log_path=log_path,
            wrapped_command=wrapped_command,
            chdir=ROOT,
            time_limit=args.slurm_time,
            partition=args.slurm_partition,
            account=args.slurm_account,
            cpus_per_task=args.slurm_cpus_per_task,
            mem=args.slurm_mem,
            extra_args=args.slurm_extra_arg,
        )

        if args.dry_run:
            job_id = f"dryrun{job_counter:04d}"
            print(shlex.join(sbatch_command))
        else:
            result = subprocess.run(sbatch_command, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip()
            print(f"Submitted {job_name} as job {job_id}")

        append_csv_row(
            manifest_path,
            SLURM_MANIFEST_FIELDNAMES,
            {
                "task_kind": task_kind,
                "seed": str(int(seed)),
                "eps_times": "" if eps_times is None else f"{float(eps_times):g}",
                "effective_k": "" if effective_k is None else str(int(effective_k)),
                "job_id": job_id,
                "job_name": job_name,
                "output_path": str(output_path),
                "log_path": str(log_path),
                "command": shlex.join(wrapped_command),
            },
        )
        return job_id

    print(f"Writing task outputs under {task_root}")
    for seed in range(int(args.n_seeds)):
        for method in methods:
            if method == "ot-tsne":
                for eps_times in args.eps_times:
                    for effective_k in args.effective_ks:
                        submit_one(
                            task_kind=method,
                            seed=seed,
                            eps_times=float(eps_times),
                            effective_k=int(effective_k),
                            output_path=resolve_task_output_path(
                                task_root,
                                task_kind=method,
                                seed=seed,
                                eps_times=float(eps_times),
                                effective_k=int(effective_k),
                            ),
                        )
                continue

            if method in OT_METHODS:
                for eps_times in args.eps_times:
                    submit_one(
                        task_kind=method,
                        seed=seed,
                        eps_times=float(eps_times),
                        output_path=resolve_task_output_path(
                            task_root,
                            task_kind=method,
                            seed=seed,
                            eps_times=float(eps_times),
                        ),
                    )
                continue

            submit_one(
                task_kind=method,
                seed=seed,
                output_path=resolve_task_output_path(task_root, task_kind=method, seed=seed),
            )

    aggregate_command = [
        python_executable,
        str(script_path),
        "aggregate",
        "--dataset",
        args.dataset,
        "--n-seeds",
        str(int(args.n_seeds)),
        "--methods",
        *methods,
        "--embedding-dim",
        str(int(args.embedding_dim)),
        "--n-epochs",
        str(int(args.n_epochs)),
        "--tsne-pca-components",
        str(int(args.tsne_pca_components)),
        "--perplexity",
        f"{float(args.perplexity):g}",
        "--data-root",
        str(resolve_data_root(args.data_root)),
        "--output-dir",
        str(resolve_output_base(args.output_dir, args.dataset)),
    ]
    if set(methods).intersection(OT_METHODS):
        aggregate_command.extend(["--eps-times", *(f"{float(value):g}" for value in args.eps_times)])
    if "ot-tsne" in methods:
        aggregate_command.extend(["--effective-k", *(str(int(value)) for value in args.effective_ks)])
    if args.max_cells_per_batch is not None:
        aggregate_command.extend(["--max-cells-per-batch", str(int(args.max_cells_per_batch))])
    if args.max_cells_batch0 is not None:
        aggregate_command.extend(["--max-cells-batch0", str(int(args.max_cells_batch0))])
    if args.max_cells_batch1 is not None:
        aggregate_command.extend(["--max-cells-batch1", str(int(args.max_cells_batch1))])
    print(f"Wrote Slurm manifest to {manifest_path}")
    print("After all jobs finish, run:")
    print(shlex.join(aggregate_command))


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-cells-per-batch", type=int, default=None)
    parser.add_argument("--max-cells-batch0", type=int, default=None)
    parser.add_argument("--max-cells-batch1", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument("--tsne-pca-components", type=int, default=DEFAULT_TSNE_PCA_COMPONENTS)
    parser.add_argument("--perplexity", type=float, default=DEFAULT_PERPLEXITY)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)


def add_sweep_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--eps-times", nargs="+", type=float, default=list(DEFAULT_EPS_TIMES))
    parser.add_argument(
        "--effective-k",
        "--effective-ks",
        dest="effective_ks",
        nargs="+",
        type=int,
        default=list(DEFAULT_EFFECTIVE_KS),
    )
    parser.add_argument("--methods", nargs="+", choices=TASK_KINDS, default=list(DEFAULT_METHODS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute benchmark embeddings from named repo datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit-slurm")
    add_common_options(submit_parser)
    add_sweep_options(submit_parser)
    submit_parser.add_argument("--n-seeds", type=int, required=True)
    submit_parser.add_argument("--python-executable", type=str, default=None)
    submit_parser.add_argument("--overwrite-manifest", action="store_true")
    submit_parser.add_argument("--force", action="store_true")
    submit_parser.add_argument("--dry-run", action="store_true")
    submit_parser.add_argument("--slurm-job-prefix", type=str, default="embed-bench")
    submit_parser.add_argument("--slurm-time", type=str, default=None)
    submit_parser.add_argument("--slurm-partition", type=str, default=None)
    submit_parser.add_argument("--slurm-account", type=str, default=None)
    submit_parser.add_argument("--slurm-cpus-per-task", type=int, default=None)
    submit_parser.add_argument("--slurm-mem", type=str, default=None)
    submit_parser.add_argument("--slurm-use-gpu", action="store_true")
    submit_parser.add_argument("--slurm-extra-arg", action="append", default=[])

    task_parser = subparsers.add_parser("run-task")
    add_common_options(task_parser)
    task_parser.add_argument("--task-kind", choices=TASK_KINDS, required=True)
    task_parser.add_argument("--seed", type=int, required=True)
    task_parser.add_argument("--task-output", type=Path, required=True)
    task_parser.add_argument("--eps-times", type=float, default=None)
    task_parser.add_argument("--effective-k", type=int, default=None)
    task_parser.add_argument("--force", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate")
    add_common_options(aggregate_parser)
    add_sweep_options(aggregate_parser)
    aggregate_parser.add_argument("--n-seeds", type=int, required=True)
    aggregate_parser.add_argument("--output-cache", type=Path, default=None)

    score_parser = subparsers.add_parser("score-aggregate")
    score_parser.add_argument("--aggregate-cache", type=Path, required=True)
    score_parser.add_argument("--output-csv", type=Path, default=None)
    score_parser.add_argument("--methods", nargs="+", choices=TASK_KINDS, default=None)

    args = parser.parse_args()
    if hasattr(args, "dataset"):
        args.dataset = canonical_dataset_name(args.dataset)
    validate_args(parser, args)
    return args


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if hasattr(args, "embedding_dim") and args.embedding_dim <= 0:
        parser.error("--embedding-dim must be positive.")
    if hasattr(args, "n_epochs") and args.n_epochs <= 0:
        parser.error("--n-epochs must be positive.")
    if hasattr(args, "tsne_pca_components") and args.tsne_pca_components <= 0:
        parser.error("--tsne-pca-components must be positive.")
    if hasattr(args, "perplexity") and args.perplexity <= 0:
        parser.error("--perplexity must be positive.")
    if hasattr(args, "n_jobs") and args.n_jobs <= 0:
        parser.error("--n-jobs must be positive.")
    if hasattr(args, "tol") and args.tol <= 0:
        parser.error("--tol must be positive.")
    if hasattr(args, "max_iter") and args.max_iter <= 0:
        parser.error("--max-iter must be positive.")
    if hasattr(args, "max_cells_per_batch") and args.max_cells_per_batch is not None and args.max_cells_per_batch <= 0:
        parser.error("--max-cells-per-batch must be positive when provided.")
    if hasattr(args, "max_cells_batch0") and args.max_cells_batch0 is not None and args.max_cells_batch0 <= 0:
        parser.error("--max-cells-batch0 must be positive when provided.")
    if hasattr(args, "max_cells_batch1") and args.max_cells_batch1 is not None and args.max_cells_batch1 <= 0:
        parser.error("--max-cells-batch1 must be positive when provided.")
    if hasattr(args, "n_seeds") and args.n_seeds <= 0:
        parser.error("--n-seeds must be positive.")
    if hasattr(args, "slurm_cpus_per_task") and args.slurm_cpus_per_task is not None:
        if args.slurm_cpus_per_task <= 0:
            parser.error("--slurm-cpus-per-task must be positive.")
    if hasattr(args, "eps_times") and args.eps_times is not None:
        eps_values = args.eps_times if isinstance(args.eps_times, list) else [args.eps_times]
        if any(float(value) <= 0 for value in eps_values):
            parser.error("--eps-times values must be positive.")
    if hasattr(args, "effective_ks"):
        if any(int(value) <= 0 for value in args.effective_ks):
            parser.error("--effective-k values must be positive.")
    if getattr(args, "task_kind", None) in OT_METHODS and args.eps_times is None:
        parser.error("--eps-times is required for OT-related run-task commands.")
    if getattr(args, "task_kind", None) == "ot-tsne" and args.effective_k is None:
        parser.error("--effective-k is required for --task-kind ot-tsne.")
    if hasattr(args, "methods") and hasattr(args, "eps_times"):
        methods = set(args.methods)
        if "ot-tsne" in methods and not getattr(args, "effective_ks", None):
            parser.error("--effective-k is required when ot-tsne is requested.")
        if methods.intersection(OT_METHODS) and not getattr(args, "eps_times", None):
            parser.error("--eps-times is required when OT-related methods are requested.")


def main() -> None:
    args = parse_args()
    if args.command == "run-task":
        run_task(args)
        return
    if args.command == "submit-slurm":
        submit_slurm(args)
        return
    if args.command == "aggregate":
        aggregate(args)
        return
    if args.command == "score-aggregate":
        score_aggregate(args)
        return
    raise ValueError(f"Unsupported command {args.command!r}.")


if __name__ == "__main__":
    main()
