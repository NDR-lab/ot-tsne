"""Build the common two-batch files used by benchmark dataset notebooks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.decomposition import PCA

from ne_aligned_utils.config import DATA_DIR

PREPROCESSING_RANDOM_SEED = 0
PREPROCESSING_SCHEMA_VERSION = "unified-real-data-v2"


def _update_hash(hasher, value):
    """Hash an array without relying on ``repr`` or object-array memory addresses."""
    array = np.asarray(value)
    hasher.update(str(array.dtype).encode())
    hasher.update(json.dumps(array.shape).encode())
    if array.dtype.kind in "OUS":
        for item in array.astype(str).ravel(order="C"):
            encoded = item.encode("utf-8")
            hasher.update(len(encoded).to_bytes(8, "big"))
            hasher.update(encoded)
    else:
        hasher.update(np.ascontiguousarray(array).tobytes())


def array_sha256(value) -> str:
    hasher = hashlib.sha256()
    _update_hash(hasher, value)
    return hasher.hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def canonical_json_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_preprocessing_manifest(
    out,
    *,
    source_paths,
    loader,
    preprocessing_seed,
    n_top_genes,
    n_pca,
):
    """Write the stable scientific identity of a unified preprocessing result.

    Paths are provenance only. Source *content* hashes and hashes of every standardized
    representation enter the scientific identity, so moving a source file does not make a new
    dataset while changing any source, selected feature, cell, or numeric output does.
    """
    source_paths = tuple(Path(path).resolve() for path in source_paths)
    cells = np.asarray(out["adata_hv"].obs_names.astype(str))
    n0 = out["X1"].shape[0]
    identity = {
        "schema_version": PREPROCESSING_SCHEMA_VERSION,
        "dataset": str(out["dataset"]),
        "loader": str(loader),
        "preprocessing_seed": int(preprocessing_seed),
        "policy": {
            "min_genes_per_cell": 10,
            "min_cells_per_gene": 3,
            "feature_selection": "scanpy.seurat_v3_batch_aware_raw_counts",
            "n_top_genes": int(n_top_genes),
            "normalization": "normalize_total_1e4_then_log1p",
            "joint_pca_components": None if n_pca is None else int(n_pca),
        },
        "batch_names": list(out["batch_names"]),
        "source_content_sha256": {
            path.name: file_sha256(path) for path in sorted(source_paths, key=lambda p: p.name)
        },
        "selected_features_sha256": array_sha256(out["features"]),
        "cell_ids_sha256": {
            "data1": array_sha256(cells[:n0]),
            "data2": array_sha256(cells[n0:]),
        },
        "representations_sha256": {
            key: array_sha256(out[key])
            for key in ("X1", "X2", "X1_counts", "X2_counts", "X1_pca", "X2_pca")
            if out[key] is not None
        },
    }
    manifest = {
        "preprocessing_hash": canonical_json_sha256(identity),
        "identity": identity,
        "provenance": {
            "source_files": [path.name for path in source_paths],
            "output_directory": Path(out["res_path"]).name,
            "path_policy": "portable logical names; local absolute paths are not recorded",
        },
    }
    path = Path(out["res_path"]) / "preprocessing_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def dense(X):
    """Densify `X` if it is sparse, so callers can ignore the distinction."""
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def validate_output_path(res_path, *, allow_canonical_write=False):
    """Reject accidental writes below the canonical input directory."""
    path = Path(res_path)
    if path.resolve().is_relative_to(DATA_DIR.resolve()) and not allow_canonical_write:
        raise ValueError(
            f"Refusing to overwrite canonical data under {DATA_DIR}. "
            "Write to results/preprocessing, or pass allow_canonical_write=True deliberately."
        )
    return path


def joint_pca(X1, X2, n_components, *, random_seed=PREPROCESSING_RANDOM_SEED):
    """Fit one reproducible PCA to both batches and return the two score matrices."""
    n_comp = min(n_components, X1.shape[1], X1.shape[0] + X2.shape[0])
    scores = PCA(n_components=n_comp, random_state=random_seed).fit_transform(
        np.concatenate((X1, X2))
    )
    return scores[: X1.shape[0]], scores[X1.shape[0] :]


def encode_labels(values, n_batch0):
    """Encode labels once, then split them at the batch boundary."""
    names, labels = np.unique(np.asarray(values, dtype=str), return_inverse=True)
    return names, labels[:n_batch0], labels[n_batch0:]


def write_dataset_files(
    output_dir,
    *,
    X1,
    X2,
    batch_names,
    features,
    metadata,
    label_col="cell_type",
    X1_counts=None,
    X2_counts=None,
    X1_pca=None,
    X2_pca=None,
    combined_adata=None,
    combined_filename=None,
    allow_canonical_write=False,
):
    """Write the repository's standard two-batch dataset files."""
    output_dir = validate_output_path(output_dir, allow_canonical_write=allow_canonical_write)
    if len(metadata) != X1.shape[0] + X2.shape[0]:
        raise ValueError("metadata rows must equal the rows in X1 and X2.")
    if len(features) != X1.shape[1] or X1.shape[1] != X2.shape[1]:
        raise ValueError("features must match the shared columns of X1 and X2.")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X1.npy", X1)
    np.save(output_dir / "X2.npy", X2)
    np.save(output_dir / "dataset_names.npy", np.asarray(batch_names, dtype=object))
    # Persist plain Unicode rather than an object array so identity checks never require pickle.
    cell_ids = np.asarray(metadata.index.astype(str), dtype=str)
    np.save(output_dir / "cell_ids1.npy", cell_ids[: X1.shape[0]])
    np.save(output_dir / "cell_ids2.npy", cell_ids[X1.shape[0] :])
    pd.Series(features, name="gene").to_csv(output_dir / "features.txt", index=False)
    metadata.to_csv(output_dir / "cells_metadata.csv")

    if (X1_counts is None) != (X2_counts is None):
        raise ValueError("Provide both raw-count matrices, or neither.")
    if X1_counts is not None:
        if X1_counts.shape != X1.shape or X2_counts.shape != X2.shape:
            raise ValueError("Raw-count matrices must match X1 and X2.")
        np.save(output_dir / "X1_counts.npy", X1_counts)
        np.save(output_dir / "X2_counts.npy", X2_counts)

    if (X1_pca is None) != (X2_pca is None):
        raise ValueError("Provide both PCA matrices, or neither.")
    if X1_pca is not None:
        if X1_pca.shape[0] != X1.shape[0] or X2_pca.shape[0] != X2.shape[0]:
            raise ValueError("PCA matrices must match the rows of X1 and X2.")
        if X1_pca.shape[1] != X2_pca.shape[1]:
            raise ValueError("PCA matrices must have the same number of columns.")
        n_components = X1_pca.shape[1]
        np.save(output_dir / f"X1_pca{n_components}.npy", X1_pca)
        np.save(output_dir / f"X2_pca{n_components}.npy", X2_pca)

    if label_col is not None and label_col in metadata:
        names, y1, y2 = encode_labels(metadata[label_col], X1.shape[0])
        np.save(output_dir / "labels1.npy", y1)
        np.save(output_dir / "labels2.npy", y2)
        np.save(output_dir / "cell_type_names.npy", names)

    if combined_adata is not None:
        filename = combined_filename or f"combined_{len(features)}genes.h5ad"
        combined_adata.write(output_dir / filename)

    return output_dir


def prepare(
    adata,
    res_path,
    dataset,
    keep_types=None,
    n_top_genes=1000,
    n_pca=50,
    random_seed=PREPROCESSING_RANDOM_SEED,
    batch_names=("b1", "b2"),
    extra_label_cols=(),
    save=True,
    allow_canonical_write=False,
):
    """Filter, select shared genes, log-normalise, split batches and optionally save.

    ``adata.layers["counts"]`` must contain raw counts. Canonical files under ``data/`` are
    protected unless ``allow_canonical_write=True`` is explicitly supplied. The unified
    ``seurat_v3`` model always ranks raw counts before normalization. PCA randomness is isolated
    behind ``random_seed`` and defaults to the fixed preprocessing seed rather than any downstream
    benchmark seed.
    """
    res_path = Path(res_path)
    if save:
        res_path = validate_output_path(res_path, allow_canonical_write=allow_canonical_write)
    methods = set(adata.obs["method"].astype(str).unique())
    assert methods == {"data1", "data2"}, f"obs['method'] must be data1/data2, got {methods}"

    assert "counts" in adata.layers, "layers['counts'] is missing, see the docstring"
    values = adata.layers["counts"]
    values = values.data if issparse(values) else np.asarray(values)  # nonzeros suffice
    assert np.array_equal(values, np.round(values)) and values.min(initial=0) >= 0, (
        "layers['counts'] is not raw counts, see the docstring"
    )

    if keep_types is not None:
        adata = adata[adata.obs["cell_type"].isin(keep_types)]
    adata = adata.copy()

    if not isinstance(adata.obs["cell_type"].dtype, pd.CategoricalDtype):
        adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    adata.obs["cell_type"] = adata.obs["cell_type"].cat.remove_unused_categories()
    n_obs_before, n_vars_before = adata.shape

    # Match the cell/gene thresholds used by the original Seurat workflow.
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=3)
    print(
        f"[{dataset}] filters dropped "
        f"{n_obs_before - adata.n_obs} cells and {n_vars_before - adata.n_vars} genes"
    )

    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
        batch_key="method",
        layer="counts",
    )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata_hv = adata[:, adata.var["highly_variable"]].copy()
    features = adata_hv.var_names.to_list()

    # Split after gene selection so both batches have identical columns.
    adata_hv1 = adata_hv[adata_hv.obs["method"] == "data1", features].copy()
    adata_hv2 = adata_hv[adata_hv.obs["method"] == "data2", features].copy()

    X1 = dense(adata_hv1.X)
    X2 = dense(adata_hv2.X)

    # Keep raw counts for methods that require them.
    X1_counts = dense(adata_hv1.layers["counts"])
    X2_counts = dense(adata_hv2.layers["counts"])

    # Encode labels jointly so the same integer has the same meaning in both batches.
    ordered_obs = pd.concat([adata_hv1.obs.copy(), adata_hv2.obs.copy()], axis=0)
    unique_cell_types, y = np.unique(ordered_obs["cell_type"].astype(str), return_inverse=True)
    y1, y2 = y[: adata_hv1.n_obs], y[adata_hv1.n_obs :]

    extra = {}
    for col in extra_label_cols:
        names, codes = np.unique(ordered_obs[col].astype(str), return_inverse=True)
        extra[col] = dict(
            names=names, codes1=codes[: adata_hv1.n_obs], codes2=codes[adata_hv1.n_obs :]
        )

    # Fit one reproducible PCA so both batches share a coordinate system.
    X1_pca = X2_pca = None
    pca_names = ()
    if n_pca is not None:
        X1_pca, X2_pca = joint_pca(X1, X2, n_pca, random_seed=random_seed)
        n_comp = X1_pca.shape[1]
        pca_names = (f"X1_pca{n_comp}.npy", f"X2_pca{n_comp}.npy")

    if save:
        write_dataset_files(
            res_path,
            X1=X1,
            X2=X2,
            batch_names=batch_names,
            features=features,
            metadata=ordered_obs,
            X1_counts=X1_counts,
            X2_counts=X2_counts,
            X1_pca=X1_pca,
            X2_pca=X2_pca,
            combined_adata=adata_hv[list(ordered_obs.index)].copy(),
            combined_filename=f"combined_{n_top_genes}genes.h5ad",
            allow_canonical_write=allow_canonical_write,
        )
        if X1_pca is not None:
            print(
                f"[{dataset}] joint PCA {X1_pca.shape} + {X2_pca.shape} -> {', '.join(pca_names)}"
            )
        print(f"[{dataset}] saved to {res_path}")

    return dict(
        dataset=dataset,
        res_path=res_path,
        batch_names=tuple(batch_names),
        X1=X1,
        X2=X2,
        X1_counts=X1_counts,
        X2_counts=X2_counts,
        y1=y1,
        y2=y2,
        X1_pca=X1_pca,
        X2_pca=X2_pca,
        unique_cell_types=unique_cell_types,
        features=features,
        extra=extra,
        adata_hv=adata_hv[list(ordered_obs.index)].copy(),
    )


def describe_batches(adata, batch_col="batch"):
    """Print counts, detected genes and sparsity per batch. True if all counts are integer."""
    all_integer = True
    for b in adata.obs[batch_col].astype(str).unique():
        sub = adata[adata.obs[batch_col].astype(str) == b]
        X = sub.layers["counts"] if "counts" in sub.layers else sub.X
        # sparse input is summarised without densifying, so this stays usable on big matrices
        if issparse(X):
            n_genes_per_cell = X.getnnz(axis=1)
            counts_per_cell = np.asarray(X.sum(axis=1)).ravel()
            zero_fraction = 1.0 - X.nnz / (X.shape[0] * X.shape[1])
            values = X.data
        else:
            X = dense(X)
            n_genes_per_cell = (X > 0).sum(1)
            counts_per_cell = X.sum(1)
            zero_fraction = float((X == 0).mean())
            values = X
        is_integer = bool(np.array_equal(values, np.round(values)))
        all_integer &= is_integer
        print(f"--- {b} ({sub.n_obs} cells x {sub.n_vars} genes) ---")
        print("integer counts:", is_integer, "| min:", X.min())
        print(f"zero fraction: {zero_fraction:.3f}")
        print(
            f"genes/cell   min {n_genes_per_cell.min():.0f}, "
            f"median {np.median(n_genes_per_cell):.0f}, max {n_genes_per_cell.max():.0f}"
        )
        print(
            f"counts/cell  min {counts_per_cell.min():.0f}, "
            f"median {np.median(counts_per_cell):.0f}, max {counts_per_cell.max():.0f}"
        )
        print()
    return all_integer


def check_saved(out, counts_are_integer=True):
    """Assert the saved arrays are consistent, and print what was checked."""
    print(f"--- {out['dataset']} ---")
    print("X1:", out["X1"].shape, "  X2:", out["X2"].shape)
    print(f"{len(out['unique_cell_types'])} cell types:", list(out["unique_cell_types"]))

    b1, b2 = out["batch_names"]
    counts = (
        pd.DataFrame(
            {
                str(b1): pd.Series(out["y1"]).value_counts(),
                str(b2): pd.Series(out["y2"]).value_counts(),
            }
        )
        .fillna(0)
        .astype(int)
    )
    counts.index = out["unique_cell_types"][counts.index]
    print(counts.sort_index().to_string())

    lo = min(out["X1"].min(), out["X2"].min())
    hi = max(out["X1"].max(), out["X2"].max())
    print(f"log-normalised range: {lo:.3f} .. {hi:.3f}")

    # The saved counts must cover the same cells and genes as X1/X2. normalize_total ran
    # before the HVG subset, so the library sizes are not recoverable from the selected
    # genes alone; what must hold is that the sparsity patterns agree exactly.
    for k, kc in [("X1", "X1_counts"), ("X2", "X2_counts")]:
        C, X = out[kc], out[k]
        assert C.shape == X.shape, f"{kc} shape mismatch"
        assert np.array_equal(C > 0, X > 0), f"{kc} and {k} disagree on which entries are zero"
        if counts_are_integer:
            assert np.array_equal(C, np.round(C)), f"{kc} is not integer-valued"
    print(
        "counts files: shapes match, same sparsity pattern as X1/X2, integer:",
        counts_are_integer,
    )
    c_lo = min(out["X1_counts"].min(), out["X2_counts"].min())
    c_hi = max(out["X1_counts"].max(), out["X2_counts"].max())
    print(f"raw count range: {c_lo:.0f} .. {c_hi:.0f}")

    pca1, pca2 = out.get("X1_pca"), out.get("X2_pca")
    if pca1 is not None:
        assert pca1.shape[0] == out["X1"].shape[0], "X1_pca does not cover the X1 cells"
        assert pca2.shape[0] == out["X2"].shape[0], "X2_pca does not cover the X2 cells"
        assert pca1.shape[1] == pca2.shape[1], "the two PCA blocks differ in dimension"
        print(
            f"PCA: {pca1.shape} + {pca2.shape} "
            "(one PCA fitted over both batches, then split by batch)"
        )
