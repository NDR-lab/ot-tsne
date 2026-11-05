"""Load, validate, and reproducibly sample benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ne_aligned_utils.config import (
    DATASET_ALIASES,
    DATASET_DIRECTORIES,
    ROOT,
    TWO_BATCH_DATASETS,
)


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


def canonical_dataset_name(dataset: str) -> str:
    return DATASET_ALIASES.get(str(dataset), str(dataset))


def resolve_data_root(data_root: Path | None) -> Path:
    if data_root is None:
        return ROOT / "data"
    path = Path(data_root).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _encode_labels(
    labels0: np.ndarray, labels1: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
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

    if not frame.index.is_unique:
        raise ValueError("cells_metadata.csv contains duplicate cell IDs.")
    if frame[label_col].isna().any():
        raise ValueError(f"cells_metadata.csv contains missing values in {label_col!r}.")
    method = frame["method"].astype(str).to_numpy()
    observed_methods = set(method)
    if observed_methods != {"data1", "data2"}:
        raise ValueError(
            "cells_metadata.csv must contain exactly data1 and data2 methods; "
            f"found {sorted(observed_methods)}."
        )
    label_values = frame[label_col].astype(str).to_numpy()
    labels0_raw = label_values[method == "data1"]
    labels1_raw = label_values[method == "data2"]
    if labels0_raw.shape[0] != x0_rows or labels1_raw.shape[0] != x1_rows:
        raise ValueError(
            "cells_metadata.csv batch sizes do not match X1.npy and X2.npy: "
            f"metadata has {labels0_raw.shape[0]}/{labels1_raw.shape[0]}, "
            f"arrays have {x0_rows}/{x1_rows}."
        )

    labels0, labels1, label_names = _encode_labels(labels0_raw, labels1_raw)
    names = [dataset_name0, dataset_name1]
    for idx, method_name in enumerate(("data1", "data2")):
        subset = frame.loc[frame["method"].astype(str) == method_name]
        for name_col in ("batch", "batchname", "replicate_id"):
            if name_col in subset.columns and len(subset) > 0:
                values = subset[name_col].dropna().astype(str).unique()
                if len(values) != 1:
                    raise ValueError(
                        f"Method {method_name} has {len(values)} values in {name_col!r}; "
                        "a batch must have one name."
                    )
                names[idx] = str(values[0])
                break
    return labels0, labels1, label_names, (names[0], names[1])


def _first_existing_dir(root: Path, *names: str) -> Path:
    """Return the first existing directory, or the canonical first candidate."""
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return root / names[0]


def _load_two_batch_directory_dataset(
    *,
    dataset: str,
    dataset_dir: Path,
    default_dataset_names: tuple[str, str],
) -> BenchmarkData:
    x0_path = dataset_dir / "X1.npy"
    x1_path = dataset_dir / "X2.npy"
    if not x0_path.exists() or not x1_path.exists():
        raise FileNotFoundError(f"{dataset_dir} must contain X1.npy and X2.npy.")

    x0 = np.load(x0_path)
    x1 = np.load(x1_path)
    labels0 = (
        np.load(dataset_dir / "labels1.npy") if (dataset_dir / "labels1.npy").exists() else None
    )
    labels1 = (
        np.load(dataset_dir / "labels2.npy") if (dataset_dir / "labels2.npy").exists() else None
    )
    label_names: tuple[str, ...] = ()
    dataset_names = default_dataset_names

    if _labels_are_valid(labels0, x0.shape[0]) and _labels_are_valid(labels1, x1.shape[0]):
        labels0 = np.asarray(labels0, dtype=int)
        labels1 = np.asarray(labels1, dtype=int)
        names_path = dataset_dir / "cell_type_names.npy"
        label_names = (
            tuple(str(value) for value in np.load(names_path, allow_pickle=True).tolist())
            if names_path.exists()
            else tuple(
                str(value) for value in sorted(np.unique(np.concatenate([labels0, labels1])))
            )
        )
    else:
        labels0 = labels1 = None

    dataset_names_path = dataset_dir / "dataset_names.npy"
    if dataset_names_path.exists():
        loaded_names = tuple(
            str(value) for value in np.load(dataset_names_path, allow_pickle=True).tolist()
        )
        if len(loaded_names) >= 2:
            dataset_names = (loaded_names[0], loaded_names[1])

    meta_path = dataset_dir / "cells_metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path, index_col=0)
        for label_col in ("cell_type", "cell_types", "label", "labels"):
            if label_col in meta.columns:
                meta_labels0, meta_labels1, meta_label_names, meta_dataset_names = (
                    _load_method_labels(
                        meta,
                        x0_rows=x0.shape[0],
                        x1_rows=x1.shape[0],
                        label_col=label_col,
                        dataset_name0=dataset_names[0],
                        dataset_name1=dataset_names[1],
                    )
                )
                if meta_labels0 is not None and meta_labels1 is not None:
                    if labels0 is not None and labels1 is not None and label_names:
                        decoded0 = np.asarray(label_names, dtype=str)[labels0]
                        decoded1 = np.asarray(label_names, dtype=str)[labels1]
                        meta_values = np.asarray(meta_label_names, dtype=str)
                        labels0_match = np.array_equal(decoded0, meta_values[meta_labels0])
                        labels1_match = np.array_equal(decoded1, meta_values[meta_labels1])
                        if not labels0_match or not labels1_match:
                            raise ValueError(
                                f"{dataset_dir} saved labels disagree with cells_metadata.csv."
                            )
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
        if not np.array_equal(data, np.round(data)):
            raise ValueError(f"{name} contains non-integer raw count values.")
        matrix = matrix.toarray()
    else:
        matrix = np.asarray(matrix)
        if np.any(~np.isfinite(matrix)):
            raise ValueError(f"{name} contains non-finite raw count values.")
        if np.any(matrix < 0.0):
            raise ValueError(f"{name} contains negative raw count values.")
        if not np.array_equal(matrix, np.round(matrix)):
            raise ValueError(f"{name} contains non-integer raw count values.")
    return np.asarray(matrix, dtype=float)


def _read_feature_list(path: Path) -> tuple[str, ...]:
    with path.open() as handle:
        features = [line.strip() for line in handle if line.strip()]
    if features and features[0].lower() in {"gene", "feature"}:
        features.pop(0)
    return tuple(features)


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
        if not obs_index.is_unique:
            raise ValueError("Requested cell IDs contain duplicates.")
        if not adata.obs_names.is_unique:
            raise ValueError(f"{path} contains duplicate cell IDs.")
        if not adata.var_names.is_unique:
            raise ValueError(f"{path} contains duplicate gene names.")
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
            if not var_index.is_unique:
                raise ValueError("Requested gene names contain duplicates.")
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
    if len(frame) != counts.shape[0]:
        raise ValueError(
            f"{dataset} metadata has {len(frame)} rows but the count matrix has {counts.shape[0]}."
        )
    methods = frame["method"].astype(str).to_numpy()
    observed_methods = set(methods)
    if observed_methods != {"data1", "data2"}:
        raise ValueError(
            f"{dataset} raw-count metadata must contain exactly data1 and data2; "
            f"found {sorted(observed_methods)}."
        )
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


def _load_combined_counts_h5ad(
    data_root: Path,
    dataset: str,
    *,
    dataset_names: tuple[str, str],
) -> BenchmarkData:
    """Load raw counts and metadata from a combined h5ad dataset directory."""
    dataset_dir = data_root / dataset
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
        dataset=dataset,
        label_col="cell_type",
        dataset_names=dataset_names,
        source_path=source_path,
        feature_names=feature_names,
    )


def _load_ncov_raw_counts(data_root: Path) -> BenchmarkData:
    return _load_combined_counts_h5ad(data_root, "nCoV", dataset_names=("nCoV-data1", "nCoV-data2"))


def _load_human_brain_raw_counts(
    data_root: Path, dataset: str = "human_brain"
) -> BenchmarkData:
    dataset_dir = data_root / DATASET_DIRECTORIES[dataset]
    combined_path = dataset_dir / "combined_1000genes.h5ad"
    if combined_path.exists():
        combined = ad.read_h5ad(combined_path, backed="r")
        try:
            has_counts = "counts" in combined.layers
        finally:
            combined.file.close()
        if has_counts:
            return _load_combined_counts_h5ad(
                data_root,
                DATASET_DIRECTORIES[dataset],
                dataset_names=("replicate 4", "replicate 1"),
            )
    source_path = dataset_dir / "raw" / "replicates.h5ad"
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
    cell_metadata_path = dataset_dir / "raw" / "cell_metadata.csv"
    if cell_metadata_path.exists():
        cell_metadata = pd.read_csv(cell_metadata_path, index_col=0)
        meta = meta.join(cell_metadata[["cell_types"]], how="left")

    return _split_count_matrix_by_method(
        counts,
        meta,
        dataset=dataset,
        label_col="cell_types",
        dataset_names=(f"{dataset}-data1", f"{dataset}-data2"),
        source_path=source_path,
        feature_names=feature_names,
    )


def _load_lupus_raw_counts(data_root: Path) -> BenchmarkData:
    dataset_dir = data_root / "lupus"
    combined_path = dataset_dir / "combined_1000genes.h5ad"
    if combined_path.exists():
        return _load_combined_counts_h5ad(data_root, "lupus", dataset_names=("CTRL", "STIM"))
    source_path = dataset_dir / "raw" / "raw_counts.h5ad"
    legacy_source_path = dataset_dir / "raw_counts.h5ad"
    if not source_path.exists() and legacy_source_path.exists():
        source_path = legacy_source_path
    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path} does not exist. Run "
            "`python scripts/prepare/prepare_lupus_raw_counts.py` "
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
        loaded_names = tuple(
            str(value) for value in np.load(dataset_names_path, allow_pickle=True).tolist()
        )
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


#: Prepared ENT L2/3 raw-count subset.
YAO_RAW_COUNTS_CACHE = "ent_L23_raw_counts.npz"


def _load_yao_raw_counts(data_root: Path) -> BenchmarkData:
    """Load Yao ENT L2/3 raw counts from the prepared subset cache."""
    prepared_dir = data_root / "yao_10x_ent_l2l3"
    if (prepared_dir / "combined_1000genes.h5ad").exists():
        return _load_combined_counts_h5ad(
            data_root,
            "yao_10x_ent_l2l3",
            dataset_names=("ent_l2l3-F", "ent_l2l3-M"),
        )
    yao_dir = data_root / "yao"
    source_path = yao_dir / YAO_RAW_COUNTS_CACHE
    ent_meta_path = yao_dir / "ent_l2l3_meta.csv"
    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path} is missing. Prepare the canonical dataset with:\n"
            "    uv run python scripts/prepare_data.py yao_10x_ent_l2l3"
        )
    if not ent_meta_path.exists():
        raise FileNotFoundError(ent_meta_path)

    with np.load(source_path, allow_pickle=False) as cached:
        counts = sp.csr_matrix(
            (cached["data"], cached["indices"], cached["indptr"]),
            shape=tuple(int(value) for value in cached["shape"]),
        )
        feature_names = tuple(str(value) for value in cached["genes"])
    counts = _dense_raw_counts(counts, name=str(source_path))

    ent_meta = pd.read_csv(ent_meta_path)
    if len(ent_meta) != counts.shape[0]:
        raise ValueError(
            f"{source_path} has {counts.shape[0]} cells but {ent_meta_path} has "
            f"{len(ent_meta)} rows; rebuild the cache."
        )

    batch_values = ent_meta["donor_sex_label"].astype(str)
    batch_order = [value for value in ("F", "M") if value in set(batch_values)]
    if set(batch_values) != {"F", "M"}:
        raise ValueError(
            "Yao raw ENT subset requires donor_sex_label values F and M only; "
            f"found {sorted(set(batch_values))}."
        )

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
    if dataset in {"human_brain", "human_brain_log"}:
        return _load_human_brain_raw_counts(data_root, dataset)
    if dataset == "lupus":
        return _load_lupus_raw_counts(data_root)
    if dataset in TWO_BATCH_DATASETS:
        # Keep raw and processed views on the same batch labels.
        return replace(
            _load_combined_counts_h5ad(
                data_root, dataset, dataset_names=TWO_BATCH_DATASETS[dataset]
            ),
            dataset_names=TWO_BATCH_DATASETS[dataset],
        )
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
    if input_kind == "joint_pca":
        processed = load_named_dataset(dataset, data_root, seed, input_kind="processed")
        dataset_dir = (
            _first_existing_dir(data_root, "MNIST", "mnist")
            if dataset == "mnist"
            else data_root / DATASET_DIRECTORIES[dataset]
        )
        x0_path = dataset_dir / "X1_pca50.npy"
        x1_path = dataset_dir / "X2_pca50.npy"
        if not x0_path.exists() or not x1_path.exists():
            raise FileNotFoundError(
                f"{dataset!r} requires the fixed preprocessing files {x0_path.name} and "
                f"{x1_path.name}. Rebuild the dataset before running common-PCA methods."
            )
        x0 = np.load(x0_path)
        x1 = np.load(x1_path)
        if x0.shape[0] != processed.x0.shape[0] or x1.shape[0] != processed.x1.shape[0]:
            raise ValueError(
                f"Joint PCA rows {x0.shape[0]}/{x1.shape[0]} do not match processed rows "
                f"{processed.x0.shape[0]}/{processed.x1.shape[0]}."
            )
        if x0.ndim != 2 or x1.ndim != 2 or x0.shape[1] != x1.shape[1]:
            raise ValueError("Joint PCA arrays must be 2D and have equal widths.")
        return replace(
            processed,
            x0=np.asarray(x0, dtype=float),
            x1=np.asarray(x1, dtype=float),
            input_source="joint_pca",
            source_path=f"{x0_path}+{x1_path}",
            feature_names=tuple(f"PC{i + 1}" for i in range(x0.shape[1])),
        )
    if input_kind != "processed":
        raise ValueError(f"Unsupported input_kind={input_kind!r}.")

    if dataset == "mnist":
        return _load_two_batch_directory_dataset(
            dataset=dataset,
            dataset_dir=_first_existing_dir(data_root, "MNIST", "mnist"),
            default_dataset_names=("mnist-batch0", "mnist-batch1"),
        )

    if dataset == "nCoV":
        dataset_dir = data_root / "nCoV"
        x0 = np.load(dataset_dir / "X1.npy")
        x1 = np.load(dataset_dir / "X2.npy")
        labels0 = (
            np.load(dataset_dir / "labels1.npy") if (dataset_dir / "labels1.npy").exists() else None
        )
        labels1 = (
            np.load(dataset_dir / "labels2.npy") if (dataset_dir / "labels2.npy").exists() else None
        )
        if not _labels_are_valid(labels0, x0.shape[0]) or not _labels_are_valid(
            labels1, x1.shape[0]
        ):
            labels0 = labels1 = None
            label_names: tuple[str, ...] = ()
        else:
            labels0 = np.asarray(labels0, dtype=int)
            labels1 = np.asarray(labels1, dtype=int)
            names_path = dataset_dir / "cell_type_names.npy"
            label_names = (
                tuple(str(value) for value in np.load(names_path, allow_pickle=True).tolist())
                if names_path.exists()
                else tuple(
                    str(value) for value in sorted(np.unique(np.concatenate([labels0, labels1])))
                )
            )

        dataset_names = ("nCoV-data1", "nCoV-data2")
        meta_path = dataset_dir / "cells_metadata.csv"
        if meta_path.exists():
            meta = pd.read_csv(meta_path, index_col=0)
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

    if dataset in {"human_brain", "human_brain_log"}:
        dataset_dir = data_root / DATASET_DIRECTORIES[dataset]
        if (dataset_dir / "labels1.npy").exists() and (dataset_dir / "labels2.npy").exists():
            return _load_two_batch_directory_dataset(
                dataset=dataset,
                dataset_dir=dataset_dir,
                default_dataset_names=(f"{dataset}-data1", f"{dataset}-data2"),
            )
        x0 = np.load(dataset_dir / "X1.npy")
        x1 = np.load(dataset_dir / "X2.npy")
        labels0 = labels1 = None
        label_names = ()
        # Preserve the requested human-brain panel in batch labels.
        dataset_names = (f"{dataset}-data1", f"{dataset}-data2")
        cell_metadata_path = dataset_dir / "raw" / "cell_metadata.csv"
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
        prepared_dir = data_root / "yao_10x_ent_l2l3"
        if (prepared_dir / "X1.npy").exists() and (prepared_dir / "X2.npy").exists():
            return _load_two_batch_directory_dataset(
                dataset=dataset,
                dataset_dir=prepared_dir,
                default_dataset_names=("ent_l2l3-F", "ent_l2l3-M"),
            )
        candidates = [
            (data_root / "yao_10x_ent_l2l3", "ent_l2l3_pca50.npy", "ent_l2l3_meta.csv"),
            (data_root / "yao", "ent_L23_PCA50.npy", "ent_l2l3_meta.csv"),
        ]
        for dataset_dir, array_name, meta_name in candidates:
            if (dataset_dir / array_name).exists() and (dataset_dir / meta_name).exists():
                x = np.load(dataset_dir / array_name)
                meta = pd.read_csv(dataset_dir / meta_name)
                break
        else:
            raise FileNotFoundError(
                "yao_10x_ent_l2l3 needs ent_l2l3_pca50.npy and ent_l2l3_meta.csv under "
                f"{data_root / 'yao_10x_ent_l2l3'}, or ent_L23_PCA50.npy and "
                f"ent_l2l3_meta.csv under {data_root / 'yao'}."
            )
        batch_values = meta["donor_sex_label"].astype(str)
        batch_order = [value for value in ("F", "M") if value in set(batch_values)]
        if set(batch_values) != {"F", "M"}:
            raise ValueError(
                "yao_10x_ent_l2l3 requires donor_sex_label values F and M only; "
                f"found {sorted(set(batch_values))}."
            )

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

    if dataset in TWO_BATCH_DATASETS:
        data = _load_two_batch_directory_dataset(
            dataset=dataset,
            dataset_dir=data_root / dataset,
            default_dataset_names=TWO_BATCH_DATASETS[dataset],
        )
        # Replace generic mouse-atlas batch labels with protocol names.
        return replace(data, dataset_names=TWO_BATCH_DATASETS[dataset])

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


# Preprocessing divides each cell by its total over all genes, times 1e4, then applies log1p,
# before keeping the selected genes. Count-based methods receive only the selected genes'
# counts, so their own normalisation would divide by the selected-gene total instead.
PROCESSED_NORMALIZATION_TARGET = 1e4
_LIBRARY_SIZE_RTOL = 1e-3


def all_gene_library_sizes(
    processed: BenchmarkData, counts: BenchmarkData
) -> tuple[np.ndarray, np.ndarray]:
    """Recover each cell's all-gene count total from the processed and raw-count views.

    ``processed = log1p(counts / total * 1e4)`` holds entrywise, so every nonzero count gives
    ``total = counts * 1e4 / expm1(processed)``. All of a cell's nonzero entries must agree;
    otherwise the two views are not the same cells, genes and normalisation, and this raises.
    Cells without counts in the selected genes get a total of 1; their normalised values are
    zero whatever the divisor.
    """
    for name in ("sample_indices0", "sample_indices1"):
        if not np.array_equal(getattr(processed, name), getattr(counts, name)):
            raise ValueError(f"processed and raw-count views differ in {name}.")
    if processed.feature_names and counts.feature_names and (
        tuple(processed.feature_names) != tuple(counts.feature_names)
    ):
        raise ValueError("processed and raw-count views have different genes.")
    totals = []
    for x, c in ((processed.x0, counts.x0), (processed.x1, counts.x1)):
        x = np.asarray(x, dtype=float)
        c = np.asarray(c, dtype=float)
        if x.shape != c.shape:
            raise ValueError(f"processed {x.shape} and raw-count {c.shape} shapes differ.")
        nonzero = c > 0
        if not np.array_equal(nonzero, x > 0):
            raise ValueError("processed and raw-count views disagree on which entries are zero.")
        safe = np.where(nonzero, np.expm1(x), 1.0)
        estimates = np.where(nonzero, c * PROCESSED_NORMALIZATION_TARGET / safe, np.nan)
        has_counts = nonzero.any(axis=1)
        total = np.ones(x.shape[0])
        if has_counts.any():
            rows = estimates[has_counts]
            low, high = np.nanmin(rows, axis=1), np.nanmax(rows, axis=1)
            if np.any(high > low * (1.0 + _LIBRARY_SIZE_RTOL)):
                raise ValueError(
                    "processed values are not log1p(counts / total * 1e4) with one total per "
                    "cell; the all-gene library size cannot be recovered."
                )
            total[has_counts] = np.nanmedian(rows, axis=1)
        totals.append(total)
    return totals[0], totals[1]
