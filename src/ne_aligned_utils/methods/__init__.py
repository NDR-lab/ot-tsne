"""OT-tSNE and benchmark embedding methods.

Public methods accept two batches in the same feature space and return a joint embedding with
batch 0 first. Method-specific preprocessing is documented in ``PREPROCESSING_STEPS.md``.

Names containing ``tsne`` identify the method family. The ``backend`` argument selects the actual
2D embedder; stored result names are resolved through ``config.canonical_name``.
"""

import time
import warnings

import numpy as np
import scipy.sparse as sp
import umap
from openTSNE import TSNE, TSNEEmbedding, affinity, initialization
from openTSNE.affinity import Affinities
from openTSNE.initialization import jitter
from sklearn.decomposition import PCA

from ne_aligned_utils.config import (
    DEFAULT_BACKEND,
    DEFAULT_UMAP_N_NEIGHBORS,
)
from ne_aligned_utils.methods.common import (
    joint_data_and_batches as _joint_data_and_batches,
    split_joint_embedding as _split_joint_embedding,
    validate_batch_sizes as _validate_batch_sizes,
    validate_embedding_dim as _validate_embedding_dim,
    validate_matrix as _validate_matrix,
    validate_pair_arrays as _validate_pair_arrays,
    validate_positive_integer as _validate_positive_integer,
    stage as _stage,
)
from ne_aligned_utils.methods.embedders import (
    _safe_perplexity,
    compute_pca_embedding,
    compute_tsne_embedding,
    compute_umap_embedding,
    embed_2d,
    embedder_artifacts,
)
from ne_aligned_utils.methods.baselines import (
    compute_pca_embedding_from_arrays,
    compute_tsne_embedding_from_arrays,
)
from ne_aligned_utils.methods.direct_integration_methods import (
    compute_policar_ref_tsne_embedding_from_arrays,
    compute_ref_tsne_embedding_from_arrays,
)
from ne_aligned_utils.methods.general_integration_methods import (
    _run_harmony,
    compute_cca_embedding_from_arrays,
    compute_cca_then_tsne_embedding_from_arrays,
    compute_harmony_embedding_from_arrays,
    compute_harmony_then_tsne_embedding_from_arrays,
    compute_scanorama_embedding_from_arrays,
    compute_scanorama_then_tsne_embedding_from_arrays,
    compute_scvi_embedding_from_arrays,
    compute_scvi_then_tsne_embedding_from_arrays,
)
from ne_aligned_utils.methods.ot_transport import (
    _median_squared_distance,
    BALANCED_TRANSPORT,
    CPU_DEVICE,
    CUDA_DEVICE,
    OT_DEVICE_CHOICES,
    TORCH_CPU_DEVICE,
    TRANSPORT_CHOICES,
    UNBALANCED_TRANSPORT,
    compute_eot_embedding_from_arrays,
    compute_eot_embedding_from_plan,
    compute_eot_then_tsne_embedding_from_arrays,
    compute_ot_plan,
)
from ne_aligned_utils.methods.ot_graph import (
    OT_TSNE_INIT_CHOICES,
    EffectiveKSplit,
    resolve_effective_k_split,
    build_ot_tsne_four_block_matrix,
    build_ot_tsne_zerodiag_four_block_matrix,
    _wants_eot_basis,
    _build_four_block_parts,
    _topk_scores,
    _row_topk_indices,
    _row_or_col_topk_mask,
    GRAPH_SYMMETRIZATIONS,
    DEFAULT_GRAPH_SYMMETRIZATION,
    _validate_symmetrization,
    _validate_symmetrizations_for_backend,
    _mutual_topk_mask,
    _symmetrised,
    four_block_directed_knn,
    cross_batch_directed_knn,
    sparsify_four_block_top_k,
    resolve_cross_batch_k,
    sparsify_four_block_cross_batch_only,
    sparsify_four_block_cross_batch_split,
    _variable_topk_mask_along,
    resolve_mass_budgets,
    sparsify_four_block_top_k_mass_aware,
    sparsify_four_block_top_percentage,
    normalize_ot_tsne_graph,
    SPECTRAL_SOLVER_SEED,
    _flip_init_signs,
    _finalize_tsne_init,
    _is_symmetric,
    _spectral_init,
    _resolve_plan_init,
    run_umap_from_graph,
    run_tsne_from_graph,
    _embed_ot_graph,
)
from ne_aligned_utils.methods.ot_methods import (
    compute_ot_tsne_embedding_from_plan,
    compute_ot_tsne_embedding_from_arrays,
    _compute_uot_tsne_embedding_from_arrays,
    compute_ot_tsne_uot_topk_embedding_from_arrays,
    compute_ot_tsne_uot_mass_embedding_from_arrays,
    within_batch_knn_indices,
    _adjacency_from_indices,
    _within_batch_knn_block,
    compute_ot_tsne_knndiag_embedding_from_plan,
    compute_ot_tsne_knndiag_embedding_from_arrays,
    compute_ot_tsne_diagonal_embedding_from_plan,
    compute_ot_tsne_selfotmedian_embedding_from_arrays,
    compute_ot_tsne_gramdiag_embedding_from_arrays,
    compute_ot_tsne_zerodiag_embedding_from_plan,
    compute_ot_tsne_zerodiag_split_embedding_from_plan,
    compute_ot_tsne_zerodiag_split_embedding_from_arrays,
    compute_ot_tsne_zerodiag_embedding_from_arrays,
)

__all__ = [
    "BALANCED_TRANSPORT",
    "OT_TSNE_INIT_CHOICES",
    "SPECTRAL_SOLVER_SEED",
    "TRANSPORT_CHOICES",
    "UNBALANCED_TRANSPORT",
    "EffectiveKSplit",
    "build_ot_tsne_four_block_matrix",
    "build_ot_tsne_zerodiag_four_block_matrix",
    "compute_eot_embedding_from_arrays",
    "compute_eot_embedding_from_plan",
    "compute_eot_then_tsne_embedding_from_arrays",
    "compute_harmony_embedding_from_arrays",
    "compute_harmony_then_tsne_embedding_from_arrays",
    "compute_ot_plan",
    "compute_ot_tsne_embedding_from_plan",
    "compute_ot_tsne_embedding_from_arrays",
    "compute_ot_tsne_knndiag_embedding_from_plan",
    "compute_ot_tsne_knndiag_embedding_from_arrays",
    "compute_ot_tsne_diagonal_embedding_from_plan",
    "compute_ot_tsne_selfotmedian_embedding_from_arrays",
    "compute_ot_tsne_gramdiag_embedding_from_arrays",
    "compute_ot_tsne_zerodiag_embedding_from_plan",
    "compute_ot_tsne_zerodiag_embedding_from_arrays",
    "compute_ot_tsne_zerodiag_split_embedding_from_plan",
    "compute_ot_tsne_zerodiag_split_embedding_from_arrays",
    "compute_ot_tsne_uot_topk_embedding_from_arrays",
    "compute_ot_tsne_uot_mass_embedding_from_arrays",
    "compute_policar_ref_tsne_embedding_from_arrays",
    "compute_ref_tsne_embedding_from_arrays",
    "compute_scanorama_embedding_from_arrays",
    "compute_scanorama_then_tsne_embedding_from_arrays",
    "compute_pca_embedding",
    "compute_pca_embedding_from_arrays",
    "compute_scvi_embedding_from_arrays",
    "compute_scvi_then_tsne_embedding_from_arrays",
    "compute_cca_embedding_from_arrays",
    "compute_cca_then_tsne_embedding_from_arrays",
    "compute_tsne_embedding",
    "compute_tsne_embedding_from_arrays",
    "normalize_ot_tsne_graph",
    "resolve_cross_batch_k",
    "resolve_effective_k_split",
    "resolve_mass_budgets",
    "run_tsne_from_graph",
    "sparsify_four_block_cross_batch_only",
    "sparsify_four_block_cross_batch_split",
    "sparsify_four_block_top_k",
    "sparsify_four_block_top_k_mass_aware",
    "sparsify_four_block_top_percentage",
]
