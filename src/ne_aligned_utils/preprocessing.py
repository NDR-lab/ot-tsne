import os

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.decomposition import PCA

# preprocessing shared by the load_*.ipynb notebooks


def dense(X):
    """Densify `X` if it is sparse, so callers can ignore the distinction."""
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def prepare(
    adata,
    res_path,
    dataset,
    keep_types=None,
    n_top_genes=1000,
    n_pca=50,
    batch_names=("b1", "b2"),
    extra_label_cols=(),
    save=True,
):
    """Preprocess a two-batch AnnData the way real_data060925.R does and save the arrays.

    `adata` holds both batches, with `obs["method"]` marking each cell as `data1` or
    `data2` and `obs["cell_type"]` holding the labels. The steps mirror the R pipeline:
    `CreateSeuratObject(min.cells=3, min.features=10)`,
    `SelectIntegrationFeatures(nfeatures=n_top_genes)` and
    `NormalizeData(LogNormalize, scale.factor=1e4)`.

    `layers["counts"]` must hold **raw counts**: `seurat_v3` gene selection is run on it and
    it is saved as `X1_counts.npy` / `X2_counts.npy`. Already-normalised input needs more
    than a different HVG flavor, so it is rejected rather than silently mishandled.

    The `n_top_genes` genes are shared by both batches, not selected per batch:
    `batch_key="method"` makes the selection batch-*aware*, i.e. scanpy ranks genes within
    each batch and then takes the top `n_top_genes` by median rank across batches into a
    single `var["highly_variable"]` mask. Both batches are re-indexed to the resulting
    feature list, so `X1` and `X2` have identical columns in identical order.

    Parameters
    ----------
    res_path, dataset
        Where to write the arrays, and the name used in the printouts.
    keep_types
        Restrict to these cell types *before* filtering and gene selection, so that a
        subset gets its own HVGs. `None` keeps every cell type.
    n_pca
        Components of the PCA, which is fitted on `X1` stacked on `X2` (one PCA over both
        batches, not one per batch) and then split by batch into `X1_pca<n_pca>.npy` and
        `X2_pca<n_pca>.npy`. `None` skips it.
    batch_names
        Names of the two batches, saved as `dataset_names.npy` and used as plot titles.
    extra_label_cols
        Further `obs` columns to encode as integer codes, for colouring embeddings. They
        are not saved as `.npy` files, only returned (and kept in `cells_metadata.csv`).

    Returns
    -------
    dict with `X1`, `X2`, `X1_counts`, `X2_counts`, `X1_pca`, `X2_pca`, `y1`, `y2`,
    `unique_cell_types`, `features`, `extra`, `adata_hv`, and the `dataset` / `res_path` /
    `batch_names` used.
    """
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

    # CreateSeuratObject(min.cells=3, min.features=10)
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=3)
    print(
        f"[{dataset}] filters dropped "
        f"{n_obs_before - adata.n_obs} cells and {n_vars_before - adata.n_vars} genes"
    )

    # SelectIntegrationFeatures(nfeatures=n_top_genes), ranking genes on the raw counts
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
        batch_key="method",
        layer="counts",
    )

    # NormalizeData(LogNormalize, scale.factor=1e4)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata_hv = adata[:, adata.var["highly_variable"]].copy()
    features = adata_hv.var_names.to_list()

    # split into the two batches, same gene order in both
    adata_hv1 = adata_hv[adata_hv.obs["method"] == "data1", features].copy()
    adata_hv2 = adata_hv[adata_hv.obs["method"] == "data2", features].copy()

    X1 = dense(adata_hv1.X)
    X2 = dense(adata_hv2.X)

    # raw counts for exactly the same cells and genes, in the same column order
    X1_counts = dense(adata_hv1.layers["counts"])
    X2_counts = dense(adata_hv2.layers["counts"])

    # label encoding shared by both batches, so an integer means the same type in each
    unique_cell_types, y = np.unique(adata_hv.obs["cell_type"].astype(str), return_inverse=True)
    is_data1 = (adata_hv.obs["method"] == "data1").to_numpy()
    y1, y2 = y[is_data1], y[~is_data1]

    extra = {}
    for col in extra_label_cols:
        names, codes = np.unique(adata_hv.obs[col].astype(str), return_inverse=True)
        extra[col] = dict(names=names, codes1=codes[is_data1], codes2=codes[~is_data1])

    # ONE PCA over both batches, then split by batch, so the two blocks share a coordinate
    # system. Saved per batch, mirroring X1/X2 and X1_counts/X2_counts.
    X1_pca = X2_pca = None
    pca_names = ()
    if n_pca is not None:
        n_comp = min(n_pca, X1.shape[1], X1.shape[0] + X2.shape[0])
        pca = PCA(n_components=n_comp).fit_transform(np.concatenate((X1, X2)))
        X1_pca, X2_pca = pca[: X1.shape[0]], pca[X1.shape[0]:]
        pca_names = (f"X1_pca{n_comp}.npy", f"X2_pca{n_comp}.npy")

    if save:
        os.makedirs(res_path, exist_ok=True)
        np.save(os.path.join(res_path, "X1.npy"), X1)
        np.save(os.path.join(res_path, "X2.npy"), X2)
        np.save(os.path.join(res_path, "X1_counts.npy"), X1_counts)
        np.save(os.path.join(res_path, "X2_counts.npy"), X2_counts)
        np.save(os.path.join(res_path, "labels1.npy"), y1)
        np.save(os.path.join(res_path, "labels2.npy"), y2)
        np.save(os.path.join(res_path, "cell_type_names.npy"), unique_cell_types)
        np.save(os.path.join(res_path, "dataset_names.npy"), np.array(list(batch_names)))
        if X1_pca is not None:
            for name, arr in zip(pca_names, (X1_pca, X2_pca)):
                np.save(os.path.join(res_path, name), arr)
            print(
                f"[{dataset}] joint PCA {X1_pca.shape} + {X2_pca.shape} "
                f"-> {', '.join(pca_names)}"
            )
        pd.Series(features, name="gene").to_csv(
            os.path.join(res_path, "features.txt"), index=False
        )
        adata_hv.obs.to_csv(os.path.join(res_path, "cells_metadata.csv"))
        adata_hv.write(os.path.join(res_path, f"combined_{n_top_genes}genes.h5ad"))
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
        adata_hv=adata_hv,
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
    counts = pd.DataFrame(
        {
            str(b1): pd.Series(out["y1"]).value_counts(),
            str(b2): pd.Series(out["y2"]).value_counts(),
        }
    ).fillna(0).astype(int)
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
