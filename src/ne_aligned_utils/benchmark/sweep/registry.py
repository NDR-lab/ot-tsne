"""Method families and representation routing for benchmark sweeps."""

TASK_KINDS = (
    "eot-t0",
    "ot",
    "eot-50",
    "joint-pca",
    "joint",
    "cca-2d",
    "cca-50",
    "cca-2d-uncentered",
    "cca-50-uncentered",
    "harmony-2d",
    "harmony-50",
    "scvi-2d",
    "scvi-50",
    "ot-knndiag",
    "ot-selfotmedian",
    "ot-gramdiag",
    "ot-uot-topk",
    "ot-uot-mass",
    "ot-zerodiag",
    "ot-zerodiag-split",
    "ref",
    "ref-our",
    "scanorama-2d",
    "scanorama-50",
    "seurat-2d",
    "seurat-50",
    "seurat3-2d",
    "seurat3-50",
    "liger1-2d",
    "liger1-50",
    "liger-2d",
    "liger-50",
    "bc-tsne",
    "multimap",
    "rc-tsne",
)

SCVI_METHODS = ("scvi-2d", "scvi-50")
DIAG_VARIANT_METHODS = ("ot-knndiag", "ot-selfotmedian", "ot-gramdiag")
UOT_METHODS = ("ot-uot-topk", "ot-uot-mass")
ZERODIAG_METHODS = ("ot-zerodiag", "ot-zerodiag-split")
EXTRA_INTEGRATION_METHODS = ("ref", "ref-our", "scanorama-2d", "scanorama-50")
R_METHODS = (
    "seurat-2d",
    "seurat-50",
    "seurat3-2d",
    "seurat3-50",
    "liger1-2d",
    "liger1-50",
    "liger-2d",
    "liger-50",
    "bc-tsne",
)
R_RAW_COUNT_METHODS = (
    "seurat-2d",
    "seurat-50",
    "seurat3-2d",
    "seurat3-50",
    "liger1-2d",
    "liger1-50",
    "liger-2d",
    "liger-50",
)
RAW_COUNT_METHODS = SCVI_METHODS + R_RAW_COUNT_METHODS + ("ref",)
PY_SUBPROCESS_METHODS = ("multimap",)
CPP_SUBPROCESS_METHODS = ("rc-tsne",)
UNCENTERED_CCA_METHODS = ("cca-2d-uncentered", "cca-50-uncentered")

DEFAULT_METHODS = tuple(
    method
    for method in TASK_KINDS
    if method not in SCVI_METHODS
    and method not in DIAG_VARIANT_METHODS
    and method not in UOT_METHODS
    and method not in ZERODIAG_METHODS
    and method not in EXTRA_INTEGRATION_METHODS
    and method not in R_METHODS
    and method not in PY_SUBPROCESS_METHODS
    and method not in CPP_SUBPROCESS_METHODS
    and method not in UNCENTERED_CCA_METHODS
)

#: OT-tSNE and its variants.
OT_TSNE_METHODS = ("ot", *DIAG_VARIANT_METHODS, *UOT_METHODS, *ZERODIAG_METHODS)
#: EOT: a separate method, not an OT-tSNE variant.
EOT_METHODS = ("eot-t0", "eot-50")
#: Every method that solves an entropic OT plan between the batches. Only the solver settings
#: (rel_reg, initialisation, tolerance, device) are shared; EOT is still its own method.
OT_PLAN_METHODS = (*OT_TSNE_METHODS, *EOT_METHODS)
COMMON_PCA_METHODS = (*OT_PLAN_METHODS, "joint", "harmony-2d", "harmony-50", "rc-tsne")
EPS_K_METHODS = OT_TSNE_METHODS
SYMMETRIZATION_METHODS = ("ot", *DIAG_VARIANT_METHODS, *ZERODIAG_METHODS)
KNNDIAG_SYMMETRIZATION_METHODS = ("ot-knndiag",)
DIAGONAL_CONSTRUCTIONS = {
    "ot": "historical_default",
    "ot-gramdiag": "gram",
    "ot-knndiag": "euclidean_knn_or",
    "ot-selfotmedian": "self_ot_median",
    "ot-zerodiag": "zero_cross_batch_only",
    "ot-zerodiag-split": "zero_cross_batch_split_budget",
    "ot-uot-topk": "historical_default",
    "ot-uot-mass": "historical_default",
}
VISUALIZED_METHODS = (
    "eot-50",
    "ot",
    "ot-gramdiag",
    "ot-knndiag",
    "ot-selfotmedian",
    "ot-zerodiag",
    "ot-zerodiag-split",
    "ot-uot-topk",
    "ot-uot-mass",
    "joint",
    "cca-50",
    "harmony-50",
    "scanorama-50",
    "scvi-50",
    "seurat-50",
    "liger-50",
)
INTERMEDIATE_DIMENSION_KEYS = {
    "eot-50": "eot_dimension",
    "cca-50": "cca_dimension",
    "harmony-50": "harmony_dimension",
    "scanorama-50": "scanorama_dimension",
    "scvi-50": "scvi_latent_dimension",
    "seurat-50": "seurat_dimension",
    "liger-50": "liger_dimension",
}


def input_kind_for_method(task_kind: str) -> str:
    """Return the persisted representation consumed by a benchmark method."""
    if task_kind in RAW_COUNT_METHODS:
        return "raw_counts"
    if task_kind in COMMON_PCA_METHODS:
        return "joint_pca"
    return "processed"


def representation_family(method: str) -> str:
    """Return the stable v2 identity label for a method's input representation."""
    if method == "ref":
        return "raw_counts_reference_pca50"
    return {
        "joint_pca": "common_joint_pca50",
        "processed": "selected_gene_processed_1000d",
        "raw_counts": "selected_gene_raw_counts",
    }[input_kind_for_method(method)]

__all__ = [name for name in globals() if name.isupper()] + [
    "input_kind_for_method",
    "representation_family",
]
