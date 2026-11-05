"""Public OT-tSNE pipelines and graph variants."""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ne_aligned_utils.config import (
    DEFAULT_BACKEND,
    DEFAULT_TSNE_EARLY_EXAGGERATION_ITER,
    DEFAULT_TSNE_N_ITER,
)
from ne_aligned_utils.methods.common import (
    stage as _stage,
    validate_matrix as _validate_matrix,
    validate_pair_arrays as _validate_pair_arrays,
    validate_positive_integer as _validate_positive_integer,
)
from ne_aligned_utils.methods.ot_affinity import (
    BlockScores,
    DiagonalSource,
    PlanDiagonals,
    PlanSpectrum,
    SuppliedDiagonals,
    WithinBatchKnn,
    graph_from_pairs,
    select_four_block,
)
from ne_aligned_utils.methods.ot_graph import (
    DEFAULT_GRAPH_SYMMETRIZATION,
    _build_four_block_parts,
    _embed_ot_graph,
    _resolve_plan_init,
    _validate_symmetrization,
    _validate_symmetrizations_for_backend,
    _wants_eot_basis,
    normalize_ot_tsne_graph,
    resolve_effective_k_split,
    resolve_mass_budgets,
    sparsify_four_block_top_percentage,
)
from ne_aligned_utils.methods.ot_transport import (
    CPU_DEVICE,
    UNBALANCED_TRANSPORT,
    compute_ot_plan,
)


def _block_graph(
    plan: np.ndarray,
    *,
    init: str | np.ndarray,
    effective_k: int,
    rule: str,
    within_batch: DiagonalSource,
    graph_symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
    mass_budgets: tuple[np.ndarray, np.ndarray] | None = None,
    binarize: bool = True,
    directed_knn: bool = False,
):
    """Build a normalized sparse graph from a plan and a within-batch source.

    Return ``(graph, selection, singular_values, resolved_init)``. Compute the SVD
    only for plan-derived diagonals or EOT initialization; otherwise singular values
    are ``None``. ``resolved_init`` replaces ``"eot"`` with its coordinates.
    """
    plan = np.asarray(plan, dtype=float)
    if plan.ndim != 2:
        raise ValueError(f"plan must be a 2D array, got shape {plan.shape!r}.")
    n0, n1 = plan.shape
    needs_spectrum = isinstance(within_batch, PlanDiagonals) or _wants_eot_basis(init)
    spectrum = PlanSpectrum.from_plan(plan) if needs_spectrum else None
    resolved_init = _resolve_plan_init(
        init, plan, spectrum.eot_coordinates() if _wants_eot_basis(init) else None
    )
    selection = select_four_block(
        BlockScores.from_plan(plan, within_batch, spectrum),
        effective_k=effective_k,
        rule=rule,
        within_batch_knn=within_batch if isinstance(within_batch, WithinBatchKnn) else None,
        graph_symmetrization=graph_symmetrization,
        mass_budgets=mass_budgets,
        binarize=binarize,
        directed_knn=directed_knn,
    )
    graph = graph_from_pairs(selection.rows, selection.cols, n0 + n1)
    singular_values = None if spectrum is None else spectrum.singular_values
    return graph, selection, singular_values, resolved_init


def _spectrum_artifacts(singular_values: np.ndarray | None) -> dict[str, object]:
    """Record singular values when an SVD was needed."""
    return {} if singular_values is None else {"singular_values": singular_values}


def compute_ot_tsne_embedding_from_plan(
    plan: np.ndarray,
    effective_k: int | None = 15,
    percentage: float | None = None,
    binarize: bool = True,
    ensure_nonempty: bool = True,
    random_state: int = 0,
    n_epochs: int = DEFAULT_TSNE_N_ITER,
    n_jobs: int = 1,
    early_exaggeration_iter: int | None = DEFAULT_TSNE_EARLY_EXAGGERATION_ITER,
    init_scale: float = 10000.0,
    init: str | np.ndarray = "random",
    backend: str = DEFAULT_BACKEND,
    x_joint: np.ndarray | None = None,
    mass_budgets: tuple[np.ndarray, np.ndarray] | None = None,
    graph_symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Embed a transport plan after four-block affinity selection and normalization.

    Use ``effective_k`` for per-block quotas or ``percentage`` for dense threshold
    selection. Optional mass budgets vary cross-batch quotas. Mutual (``"and"``)
    symmetrization is supported only for fixed-budget t-SNE graphs.
    With ``return_artifacts=True``, also return graph and initialization diagnostics.
    """
    plan = np.asarray(plan, dtype=float)
    if plan.ndim != 2:
        raise ValueError(f"plan must be a 2D array, got shape {plan.shape!r}.")
    n0, n1 = plan.shape
    if mass_budgets is not None and backend != DEFAULT_BACKEND:
        # UMAP requires a fixed-width neighbor list; mass budgets give variable widths.
        raise ValueError(
            "mass-aware budgets are only supported by the t-SNE backend; "
            f"got backend={backend!r}, whose neighbour list has one fixed width per point."
        )
    _validate_symmetrizations_for_backend(backend, graph_symmetrization=graph_symmetrization)
    if graph_symmetrization != DEFAULT_GRAPH_SYMMETRIZATION:
        # Reject modes these sparsifiers cannot represent.
        if mass_budgets is not None:
            raise ValueError(
                "graph_symmetrization='and' is not supported together with mass-aware budgets: "
                "sparsify_four_block_top_k_mass_aware has no mutual mode."
            )
        if percentage is not None:
            raise ValueError(
                "graph_symmetrization='and' is not supported together with percentage "
                "sparsification: sparsify_four_block_top_percentage has no mutual mode."
            )

    init_info: dict[str, object] = {}
    # Postprocessing also contributes to the accumulated pre-embedding time.
    four_block_matrix = None
    knn_indices = None
    with _stage(timer, "time_pre_embedding"), _stage(timer, "time_ot_postprocess"):
        if percentage is not None:
            if effective_k is not None:
                raise ValueError("Specify either effective_k or percentage, not both.")
            four_block_matrix, singular_values, eot_coordinates = _build_four_block_parts(
                plan, return_eot=_wants_eot_basis(init)
            )
            resolved_init = _resolve_plan_init(init, plan, eot_coordinates)
            sparse_four_block_matrix = sparsify_four_block_top_percentage(
                four_block_matrix,
                n0=plan.shape[0],
                n1=plan.shape[1],
                percentage=percentage,
                ensure_nonempty=ensure_nonempty,
                binarize=binarize,
            )
            split = None
            graph = normalize_ot_tsne_graph(sparse_four_block_matrix)
        else:
            if effective_k is None:
                raise ValueError("effective_k is required when percentage is not provided.")
            graph, selection, singular_values, resolved_init = _block_graph(
                plan,
                init=init,
                effective_k=effective_k,
                rule="split" if mass_budgets is None else "mass",
                within_batch=PlanDiagonals(),
                graph_symmetrization=graph_symmetrization,
                mass_budgets=mass_budgets,
                binarize=binarize,
                directed_knn=backend == "umap",
            )
            split = selection.split
            knn_indices = selection.knn_indices
    with _stage(timer, "time_embedding"):
        embedding, backend_artifacts = _embed_ot_graph(
            backend=backend,
            graph=graph,
            four_block_matrix=four_block_matrix,
            knn_indices=knn_indices,
            n0=n0,
            n1=n1,
            effective_k=effective_k,
            within_batch_indices=None,
            x_joint=x_joint,
            random_state=random_state,
            n_epochs=n_epochs,
            n_jobs=n_jobs,
            early_exaggeration_iter=early_exaggeration_iter,
            init_scale=init_scale,
            init=resolved_init,
            init_info=init_info,
            timer=timer,
        )
    if not return_artifacts:
        return embedding

    artifacts: dict[str, object] = {
        **backend_artifacts,
        "graph": graph,
        **_spectrum_artifacts(singular_values),
        "percentage": None if percentage is None else float(percentage),
        "graph_symmetrization": str(graph_symmetrization),
        **init_info,
    }
    if split is not None:
        artifacts.update(
            effective_k=int(split.effective_k),
            batch0_k=int(split.batch0_k),
            batch1_k=int(split.batch1_k),
        )
    if mass_budgets is not None:
        artifacts.update(
            batch0_mass_budgets=np.asarray(mass_budgets[0], dtype=int),
            batch1_mass_budgets=np.asarray(mass_budgets[1], dtype=int),
        )
    return embedding, artifacts
def compute_ot_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    rel_reg: float = 1.0,
    effective_k: int | None = 15,
    percentage: float | None = None,
    binarize: bool = True,
    ensure_nonempty: bool = True,
    random_state: int = 0,
    n_epochs: int = DEFAULT_TSNE_N_ITER,
    n_jobs: int = 1,
    early_exaggeration_iter: int | None = DEFAULT_TSNE_EARLY_EXAGGERATION_ITER,
    tol: float = 1e-12,
    max_iter: int = int(1e4),
    init_scale: float = 10000.0,
    init: str | np.ndarray = "random",
    backend: str = DEFAULT_BACKEND,
    device: str = CPU_DEVICE,
    graph_symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Compute a balanced entropic transport plan and embed its four-block affinities."""
    x0, x1 = _validate_pair_arrays(x0, x1)
    with _stage(timer, "time_pre_embedding"):
        plan, ot_artifacts = compute_ot_plan(
            x0,
            x1,
            rel_reg=rel_reg,
            tol=tol,
            max_iter=max_iter,
            device=device,
            return_artifacts=True,
            timer=timer,
        )
    embedding, method_artifacts = compute_ot_tsne_embedding_from_plan(
        plan,
        effective_k=effective_k,
        percentage=percentage,
        binarize=binarize,
        ensure_nonempty=ensure_nonempty,
        random_state=random_state,
        n_epochs=n_epochs,
        n_jobs=n_jobs,
        early_exaggeration_iter=early_exaggeration_iter,
        init_scale=init_scale,
        init=init,
        backend=backend,
        x_joint=np.vstack([x0, x1]),
        graph_symmetrization=graph_symmetrization,
        return_artifacts=True,
        timer=timer,
    )

    if not return_artifacts:
        return embedding

    artifacts: dict[str, object] = {
        **method_artifacts,
        "ot_plan": plan,
        "epsilon": ot_artifacts["epsilon"],
        "median_squared_distance": ot_artifacts["median_squared_distance"],
        # The task runner reads torch_device to keep runtime comparisons device-specific.
        "ot_device": ot_artifacts["ot_device"],
        "torch_device": ot_artifacts["ot_device"],
    }
    return embedding, artifacts

def _compute_uot_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    mass_aware: bool,
    rel_reg: float = 1.0,
    rel_reg_m: float = 1.0,
    reg_m: float | None = None,
    solver_method: str = "sinkhorn",
    k_min: int = 1,
    effective_k: int | None = 15,
    binarize: bool = True,
    random_state: int = 0,
    n_epochs: int = DEFAULT_TSNE_N_ITER,
    n_jobs: int = 1,
    early_exaggeration_iter: int | None = DEFAULT_TSNE_EARLY_EXAGGERATION_ITER,
    tol: float = 1e-12,
    max_iter: int = int(1e4),
    init_scale: float = 10000.0,
    init: str | np.ndarray = "random",
    backend: str = DEFAULT_BACKEND,
    device: str = CPU_DEVICE,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Shared body of the two unbalanced-OT variants; ``mass_aware`` is the only difference."""
    x0, x1 = _validate_pair_arrays(x0, x1)
    with _stage(timer, "time_pre_embedding"):
        plan, ot_artifacts = compute_ot_plan(
            x0,
            x1,
            rel_reg=rel_reg,
            tol=tol,
            max_iter=max_iter,
            transport=UNBALANCED_TRANSPORT,
            rel_reg_m=rel_reg_m,
            reg_m=reg_m,
            solver_method=solver_method,
            device=device,
            return_artifacts=True,
            timer=timer,
        )

    mass_budgets = None
    if mass_aware:
        if effective_k is None:
            raise ValueError("effective_k is required for the mass-aware unbalanced variant.")
        split = resolve_effective_k_split(
            effective_k=effective_k, n0=plan.shape[0], n1=plan.shape[1]
        )
        # Cross-batch budgets are capped by the opposite batch's quota.
        mass_budgets = (
            resolve_mass_budgets(ot_artifacts["plan_row_mass"], split.batch1_k, k_min=k_min),
            resolve_mass_budgets(ot_artifacts["plan_col_mass"], split.batch0_k, k_min=k_min),
        )

    embedding, method_artifacts = compute_ot_tsne_embedding_from_plan(
        plan,
        effective_k=effective_k,
        binarize=binarize,
        random_state=random_state,
        n_epochs=n_epochs,
        n_jobs=n_jobs,
        early_exaggeration_iter=early_exaggeration_iter,
        init_scale=init_scale,
        init=init,
        backend=backend,
        x_joint=np.vstack([x0, x1]),
        mass_budgets=mass_budgets,
        return_artifacts=True,
        timer=timer,
    )

    if not return_artifacts:
        return embedding

    artifacts: dict[str, object] = {
        **method_artifacts,
        "ot_plan": plan,
        "epsilon": ot_artifacts["epsilon"],
        "median_squared_distance": ot_artifacts["median_squared_distance"],
        # The task runner reads torch_device to keep runtime comparisons device-specific.
        "ot_device": ot_artifacts["ot_device"],
        "torch_device": ot_artifacts["ot_device"],
        "transport": ot_artifacts["transport"],
        "reg_m": ot_artifacts["reg_m"],
        "rel_reg_m": ot_artifacts["rel_reg_m"],
        "plan_total_mass": ot_artifacts["plan_total_mass"],
        "plan_nnz_fraction": ot_artifacts["plan_nnz_fraction"],
        "plan_row_mass": ot_artifacts["plan_row_mass"],
        "plan_col_mass": ot_artifacts["plan_col_mass"],
    }
    return embedding, artifacts


def compute_ot_tsne_uot_topk_embedding_from_arrays(
    x0: np.ndarray, x1: np.ndarray, **kwargs: object
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Embed an unbalanced plan with fixed per-block neighbor quotas.

    Marginal relaxation affects neighbor rankings; binarization discards edge magnitudes.
    """
    return _compute_uot_tsne_embedding_from_arrays(x0, x1, mass_aware=False, **kwargs)


def compute_ot_tsne_uot_mass_embedding_from_arrays(
    x0: np.ndarray, x1: np.ndarray, **kwargs: object
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Embed an unbalanced plan with cross-batch budgets scaled by transported mass."""
    return _compute_uot_tsne_embedding_from_arrays(x0, x1, mass_aware=True, **kwargs)


def within_batch_knn_indices(x: np.ndarray, k: int, metric: str = "euclidean") -> np.ndarray:
    """Block-local indices of each point's ``k`` nearest neighbors within one batch.

    Shape ``(n, k)``, self excluded, indices in ``0..n-1``. Shared by the graph the t-SNE arm
    embeds and the neighbor list the UMAP arm embeds, so the two cannot disagree about which
    neighbors the knndiag variant chose.
    """
    from sklearn.neighbors import NearestNeighbors

    x = _validate_matrix(x, "x")
    n = x.shape[0]
    k = _validate_positive_integer(k, "k")
    if k >= n:
        raise ValueError(f"k={k} must be smaller than the number of points {n}.")
    neighbors = NearestNeighbors(n_neighbors=k, metric=metric)
    neighbors.fit(x)
    # Without a query, scikit-learn excludes each point by index, so identical points tied at
    # zero distance cannot push a point out of its own neighbour list.
    return neighbors.kneighbors(return_distance=False)


def _adjacency_from_indices(
    neighbor_idx: np.ndarray,
    n: int,
    symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
) -> np.ndarray:
    """Build a loop-free binary adjacency from block-local neighbor indices.

    ``"or"`` keeps either endpoint's choice; ``"and"`` requires both. This within-batch
    mode is independent of cross-batch OT symmetrization in the kNN-diagonal variant.
    """
    symmetrization = _validate_symmetrization(symmetrization, "symmetrization")
    adjacency = np.zeros((n, n), dtype=float)
    adjacency[np.arange(n)[:, None], neighbor_idx] = 1.0
    combine = np.maximum if symmetrization == "or" else np.minimum
    adjacency = combine(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def _within_batch_knn_block(
    x: np.ndarray,
    k: int,
    metric: str = "euclidean",
    symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
) -> np.ndarray:
    """Binary within-batch kNN adjacency with self-loops excluded.

    Each point keeps an edge to its ``k`` nearest neighbors; the graph is then symmetrized by
    ``symmetrization``. ``"or"`` -- the default -- keeps an edge if *either* endpoint selected the
    other, matching how the four-block diagonal is symmetrized downstream; ``"and"`` keeps it only
    if both did. The result is already sparse and binary, so it is injected into the graph *after*
    the top-k sparsifier rather than passed through it.
    """
    symmetrization = _validate_symmetrization(symmetrization, "symmetrization")
    x = _validate_matrix(x, "x")
    n = x.shape[0]
    k = _validate_positive_integer(k, "k")
    if k >= n:
        raise ValueError(f"k={k} must be smaller than the number of points {n}.")

    neighbor_idx = within_batch_knn_indices(x, k, metric=metric)
    row_ids = np.arange(n)[:, None]

    adjacency = np.zeros((n, n), dtype=float)
    adjacency[row_ids, neighbor_idx] = 1.0
    combine = np.maximum if symmetrization == "or" else np.minimum
    adjacency = combine(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def compute_ot_tsne_knndiag_embedding_from_plan(
    plan: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    effective_k: int = 15,
    binarize: bool = True,
    random_state: int = 0,
    n_epochs: int = DEFAULT_TSNE_N_ITER,
    n_jobs: int = 1,
    early_exaggeration_iter: int | None = DEFAULT_TSNE_EARLY_EXAGGERATION_ITER,
    init_scale: float = 10000.0,
    init: str | np.ndarray = "random",
    backend: str = DEFAULT_BACKEND,
    x_joint: np.ndarray | None = None,
    graph_symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
    knndiag_symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Embed cross-plan affinities with fixed within-batch kNN edges.

    Within-batch neighbors use the proportional per-batch quotas and are inserted
    without score selection. ``graph_symmetrization`` controls cross-batch OT edges;
    ``knndiag_symmetrization`` independently controls within-batch kNN edges.
    """
    plan = np.asarray(plan, dtype=float)
    if plan.ndim != 2:
        raise ValueError(f"plan must be a 2D array, got shape {plan.shape!r}.")
    x0 = _validate_matrix(x0, "x0")
    x1 = _validate_matrix(x1, "x1")
    n0, n1 = plan.shape
    if x0.shape[0] != n0 or x1.shape[0] != n1:
        raise ValueError(
            f"x0/x1 row counts ({x0.shape[0]}, {x1.shape[0]}) must match the plan "
            f"shape {plan.shape}."
        )
    if effective_k is None:
        raise ValueError("effective_k is required for the kNN-diagonal variant.")
    _validate_symmetrizations_for_backend(
        backend,
        graph_symmetrization=graph_symmetrization,
        knndiag_symmetrization=knndiag_symmetrization,
    )

    init_info: dict[str, object] = {}
    # Within-batch kNN construction is part of OT graph postprocessing.
    with _stage(timer, "time_pre_embedding"), _stage(timer, "time_ot_postprocess"):
        split = resolve_effective_k_split(effective_k=effective_k, n0=n0, n1=n1)
        within_batch = WithinBatchKnn(
            within_batch_knn_indices(x0, split.batch0_k),
            within_batch_knn_indices(x1, split.batch1_k),
            symmetrization=knndiag_symmetrization,
        )
        graph, selection, singular_values, resolved_init = _block_graph(
            plan,
            init=init,
            effective_k=effective_k,
            rule="split",
            within_batch=within_batch,
            graph_symmetrization=graph_symmetrization,
            binarize=binarize,
            directed_knn=backend == "umap",
        )
    with _stage(timer, "time_embedding"):
        embedding, backend_artifacts = _embed_ot_graph(
            backend=backend,
            graph=graph,
            four_block_matrix=None,
            knn_indices=selection.knn_indices,
            n0=n0,
            n1=n1,
            effective_k=effective_k,
            within_batch_indices=None,
            x_joint=x_joint if x_joint is not None else np.vstack([x0, x1]),
            random_state=random_state,
            n_epochs=n_epochs,
            n_jobs=n_jobs,
            early_exaggeration_iter=early_exaggeration_iter,
            init_scale=init_scale,
            init=resolved_init,
            init_info=init_info,
            timer=timer,
        )
    if not return_artifacts:
        return embedding

    artifacts: dict[str, object] = {
        **backend_artifacts,
        **_spectrum_artifacts(singular_values),
        "effective_k": int(split.effective_k),
        "batch0_k": int(split.batch0_k),
        "batch1_k": int(split.batch1_k),
        "diag0_edges": selection.diag_edges[0],
        "diag1_edges": selection.diag_edges[1],
        "graph_symmetrization": str(graph_symmetrization),
        "knndiag_symmetrization": str(knndiag_symmetrization),
        **init_info,
    }
    return embedding, artifacts


def compute_ot_tsne_knndiag_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    rel_reg: float = 1.0,
    effective_k: int = 15,
    binarize: bool = True,
    random_state: int = 0,
    n_epochs: int = DEFAULT_TSNE_N_ITER,
    n_jobs: int = 1,
    early_exaggeration_iter: int | None = DEFAULT_TSNE_EARLY_EXAGGERATION_ITER,
    tol: float = 1e-12,
    max_iter: int = int(1e4),
    init_scale: float = 10000.0,
    init: str | np.ndarray = "random",
    backend: str = DEFAULT_BACKEND,
    device: str = CPU_DEVICE,
    graph_symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
    knndiag_symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """End-to-end kNN-diagonal OT-tSNE from raw arrays (see the plan-level docstring)."""
    x0, x1 = _validate_pair_arrays(x0, x1)
    with _stage(timer, "time_pre_embedding"):
        plan, ot_artifacts = compute_ot_plan(
            x0,
            x1,
            rel_reg=rel_reg,
            tol=tol,
            max_iter=max_iter,
            device=device,
            return_artifacts=True,
            timer=timer,
        )
    result = compute_ot_tsne_knndiag_embedding_from_plan(
        plan,
        x0,
        x1,
        effective_k=effective_k,
        binarize=binarize,
        random_state=random_state,
        n_epochs=n_epochs,
        n_jobs=n_jobs,
        early_exaggeration_iter=early_exaggeration_iter,
        init_scale=init_scale,
        init=init,
        backend=backend,
        graph_symmetrization=graph_symmetrization,
        knndiag_symmetrization=knndiag_symmetrization,
        x_joint=np.vstack([x0, x1]),
        return_artifacts=return_artifacts,
        timer=timer,
    )
    if not return_artifacts:
        return result
    embedding, method_artifacts = result
    artifacts: dict[str, object] = {
        **method_artifacts,
        "epsilon": ot_artifacts["epsilon"],
        "median_squared_distance": ot_artifacts["median_squared_distance"],
        # The task runner reads torch_device to keep runtime comparisons device-specific.
        "ot_device": ot_artifacts["ot_device"],
        "torch_device": ot_artifacts["ot_device"],
    }
    return embedding, artifacts


def compute_ot_tsne_diagonal_embedding_from_plan(
    plan: np.ndarray,
    self_plan0: np.ndarray,
    self_plan1: np.ndarray,
    effective_k: int = 15,
    binarize: bool = True,
    random_state: int = 0,
    n_epochs: int = DEFAULT_TSNE_N_ITER,
    n_jobs: int = 1,
    early_exaggeration_iter: int | None = DEFAULT_TSNE_EARLY_EXAGGERATION_ITER,
    init_scale: float = 10000.0,
    init: str | np.ndarray = "random",
    backend: str = DEFAULT_BACKEND,
    x_joint: np.ndarray | None = None,
    graph_symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Embed a plan with caller-supplied within-batch score matrices.

    ``self_plan0`` and ``self_plan1`` may be self-OT or Gram matrices. Their shapes
    must match their batches. All blocks use the same top-k and symmetrization rules.
    """
    plan = np.asarray(plan, dtype=float)
    if plan.ndim != 2:
        raise ValueError(f"plan must be a 2D array, got shape {plan.shape!r}.")
    n0, n1 = plan.shape
    self_plan0 = _validate_matrix(self_plan0, "self_plan0")
    self_plan1 = _validate_matrix(self_plan1, "self_plan1")
    if self_plan0.shape != (n0, n0):
        raise ValueError(f"self_plan0 must have shape {(n0, n0)}, got {self_plan0.shape}.")
    if self_plan1.shape != (n1, n1):
        raise ValueError(f"self_plan1 must have shape {(n1, n1)}, got {self_plan1.shape}.")
    if effective_k is None:
        raise ValueError("effective_k is required for the self-OT-diagonal variant.")
    _validate_symmetrizations_for_backend(backend, graph_symmetrization=graph_symmetrization)

    init_info: dict[str, object] = {}
    with _stage(timer, "time_pre_embedding"), _stage(timer, "time_ot_postprocess"):
        graph, selection, singular_values, resolved_init = _block_graph(
            plan,
            init=init,
            effective_k=effective_k,
            rule="split",
            within_batch=SuppliedDiagonals(self_plan0, self_plan1),
            graph_symmetrization=graph_symmetrization,
            binarize=binarize,
            directed_knn=backend == "umap",
        )
        split = selection.split
    with _stage(timer, "time_embedding"):
        embedding, backend_artifacts = _embed_ot_graph(
            backend=backend,
            graph=graph,
            four_block_matrix=None,
            knn_indices=selection.knn_indices,
            n0=n0,
            n1=n1,
            effective_k=effective_k,
            within_batch_indices=None,
            x_joint=x_joint,
            random_state=random_state,
            n_epochs=n_epochs,
            n_jobs=n_jobs,
            early_exaggeration_iter=early_exaggeration_iter,
            init_scale=init_scale,
            init=resolved_init,
            init_info=init_info,
            timer=timer,
        )
    if not return_artifacts:
        return embedding

    artifacts: dict[str, object] = {
        **backend_artifacts,
        **_spectrum_artifacts(singular_values),
        "effective_k": int(split.effective_k),
        "batch0_k": int(split.batch0_k),
        "batch1_k": int(split.batch1_k),
        "diag0_edges": selection.diag_edges[0],
        "diag1_edges": selection.diag_edges[1],
        "graph_symmetrization": str(graph_symmetrization),
        **init_info,
    }
    return embedding, artifacts


def compute_ot_tsne_selfotmedian_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    rel_reg: float = 1.0,
    effective_k: int = 15,
    binarize: bool = True,
    random_state: int = 0,
    n_epochs: int = DEFAULT_TSNE_N_ITER,
    n_jobs: int = 1,
    early_exaggeration_iter: int | None = DEFAULT_TSNE_EARLY_EXAGGERATION_ITER,
    tol: float = 1e-12,
    max_iter: int = int(1e4),
    init_scale: float = 10000.0,
    init: str | np.ndarray = "random",
    backend: str = DEFAULT_BACKEND,
    device: str = CPU_DEVICE,
    graph_symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Embed cross-batch OT with self-OT within-batch affinities.

    Each plan uses ``epsilon = rel_reg * median_squared_distance`` from its own
    pair of inputs, giving the same relative regularization in all three plans.
    """
    x0, x1 = _validate_pair_arrays(x0, x1)
    # Cost and solve timings accumulate across all three plans; self solves are also timed
    # separately.
    with _stage(timer, "time_pre_embedding"):
        plan, ot_artifacts = compute_ot_plan(
            x0,
            x1,
            rel_reg=rel_reg,
            tol=tol,
            max_iter=max_iter,
            device=device,
            return_artifacts=True,
            timer=timer,
        )
        # Each self-plan uses its own median to resolve epsilon.
        with _stage(timer, "time_ot_solve_self"):
            self_plan0, self0_art = compute_ot_plan(
                x0,
                x0,
                rel_reg=rel_reg,
                tol=tol,
                max_iter=max_iter,
                device=device,
                return_artifacts=True,
                timer=timer,
            )
            self_plan1, self1_art = compute_ot_plan(
                x1,
                x1,
                rel_reg=rel_reg,
                tol=tol,
                max_iter=max_iter,
                device=device,
                return_artifacts=True,
                timer=timer,
            )
    result = compute_ot_tsne_diagonal_embedding_from_plan(
        plan,
        self_plan0,
        self_plan1,
        effective_k=effective_k,
        binarize=binarize,
        random_state=random_state,
        n_epochs=n_epochs,
        n_jobs=n_jobs,
        early_exaggeration_iter=early_exaggeration_iter,
        init_scale=init_scale,
        init=init,
        backend=backend,
        x_joint=np.vstack([x0, x1]),
        graph_symmetrization=graph_symmetrization,
        return_artifacts=return_artifacts,
        timer=timer,
    )
    if not return_artifacts:
        return result
    embedding, method_artifacts = result
    artifacts: dict[str, object] = {
        **method_artifacts,
        "epsilon": float(ot_artifacts["epsilon"]),
        "epsilon_self0": float(self0_art["epsilon"]),
        "epsilon_self1": float(self1_art["epsilon"]),
        "diagonal_epsilon": "own_median",
        "median_squared_distance": ot_artifacts["median_squared_distance"],
        # The task runner reads torch_device to keep runtime comparisons device-specific.
        "ot_device": ot_artifacts["ot_device"],
        "torch_device": ot_artifacts["ot_device"],
    }
    return embedding, artifacts


def compute_ot_tsne_gramdiag_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    rel_reg: float = 1.0,
    effective_k: int = 15,
    binarize: bool = True,
    random_state: int = 0,
    n_epochs: int = DEFAULT_TSNE_N_ITER,
    n_jobs: int = 1,
    early_exaggeration_iter: int | None = DEFAULT_TSNE_EARLY_EXAGGERATION_ITER,
    tol: float = 1e-12,
    max_iter: int = int(1e4),
    init_scale: float = 10000.0,
    init: str | np.ndarray = "random",
    backend: str = DEFAULT_BACKEND,
    device: str = CPU_DEVICE,
    graph_symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Embed OT with within-batch Gram affinities ``P P.T`` and ``P.T P``.

    For ``P = U S V.T``, these use ``S**2`` rather than the default diagonals'
    linearly weighted spectrum. They also omit the default batch-size factors.
    All blocks use the same selection rules; no additional OT solve is required.
    """
    x0, x1 = _validate_pair_arrays(x0, x1)
    with _stage(timer, "time_pre_embedding"):
        plan, ot_artifacts = compute_ot_plan(
            x0,
            x1,
            rel_reg=rel_reg,
            tol=tol,
            max_iter=max_iter,
            device=device,
            return_artifacts=True,
            timer=timer,
        )
        # Gram products count as graph postprocessing.
        with _stage(timer, "time_ot_postprocess"):
            gram0 = plan @ plan.T
            gram1 = plan.T @ plan
    result = compute_ot_tsne_diagonal_embedding_from_plan(
        plan,
        gram0,
        gram1,
        effective_k=effective_k,
        binarize=binarize,
        random_state=random_state,
        n_epochs=n_epochs,
        n_jobs=n_jobs,
        early_exaggeration_iter=early_exaggeration_iter,
        init_scale=init_scale,
        init=init,
        backend=backend,
        x_joint=np.vstack([x0, x1]),
        graph_symmetrization=graph_symmetrization,
        return_artifacts=return_artifacts,
        timer=timer,
    )
    if not return_artifacts:
        return result
    embedding, method_artifacts = result
    artifacts: dict[str, object] = {
        **method_artifacts,
        "epsilon": float(ot_artifacts["epsilon"]),
        "median_squared_distance": ot_artifacts["median_squared_distance"],
        # The task runner reads torch_device to keep runtime comparisons device-specific.
        "ot_device": ot_artifacts["ot_device"],
        "torch_device": ot_artifacts["ot_device"],
    }
    return embedding, artifacts


def compute_ot_tsne_zerodiag_embedding_from_plan(
    plan: np.ndarray,
    effective_k: int = 15,
    binarize: bool = True,
    random_state: int = 0,
    n_epochs: int = DEFAULT_TSNE_N_ITER,
    n_jobs: int = 1,
    early_exaggeration_iter: int | None = DEFAULT_TSNE_EARLY_EXAGGERATION_ITER,
    init_scale: float = 10000.0,
    init: str | np.ndarray = "random",
    backend: str = DEFAULT_BACKEND,
    x_joint: np.ndarray | None = None,
    graph_symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Embed ``[[0, P], [P.T, 0]]`` using the full budget for cross-batch neighbors.

    Zero blocks are excluded from selection to avoid creating edges from tied zeros. Reallocating
    the full budget keeps graph density comparable with the base method.
    """
    plan = np.asarray(plan, dtype=float)
    if plan.ndim != 2:
        raise ValueError(f"plan must be a 2D array, got shape {plan.shape!r}.")
    n0, n1 = plan.shape
    if effective_k is None:
        raise ValueError("effective_k is required for the zero-diagonal variant.")
    _validate_symmetrizations_for_backend(backend, graph_symmetrization=graph_symmetrization)

    init_info: dict[str, object] = {}
    with _stage(timer, "time_pre_embedding"), _stage(timer, "time_ot_postprocess"):
        graph, selection, singular_values, resolved_init = _block_graph(
            plan,
            init=init,
            effective_k=effective_k,
            rule="cross_only",
            within_batch=None,
            graph_symmetrization=graph_symmetrization,
            binarize=binarize,
            directed_knn=backend == "umap",
        )
        diag0_edges, diag1_edges = selection.diag_edges
    with _stage(timer, "time_embedding"):
        embedding, backend_artifacts = _embed_ot_graph(
            backend=backend,
            graph=graph,
            four_block_matrix=None,
            knn_indices=selection.knn_indices,
            n0=n0,
            n1=n1,
            effective_k=effective_k,
            within_batch_indices=None,
            x_joint=x_joint,
            random_state=random_state,
            n_epochs=n_epochs,
            n_jobs=n_jobs,
            early_exaggeration_iter=early_exaggeration_iter,
            init_scale=init_scale,
            init=resolved_init,
            init_info=init_info,
            cross_batch_only=True,
            timer=timer,
        )
    if not return_artifacts:
        return embedding

    artifacts: dict[str, object] = {
        **backend_artifacts,
        **_spectrum_artifacts(singular_values),
        "effective_k": int(effective_k),
        "cross_batch_k": int(effective_k),
        "within_batch_k": 0,
        "diag0_edges": diag0_edges,
        "diag1_edges": diag1_edges,
        "graph_symmetrization": str(graph_symmetrization),
        **init_info,
    }
    return embedding, artifacts


def compute_ot_tsne_zerodiag_split_embedding_from_plan(
    plan: np.ndarray,
    effective_k: int = 15,
    binarize: bool = True,
    random_state: int = 0,
    n_epochs: int = DEFAULT_TSNE_N_ITER,
    n_jobs: int = 1,
    early_exaggeration_iter: int | None = DEFAULT_TSNE_EARLY_EXAGGERATION_ITER,
    init_scale: float = 10000.0,
    init: str | np.ndarray = "random",
    backend: str = DEFAULT_BACKEND,
    x_joint: np.ndarray | None = None,
    graph_symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Remove within-batch edges while preserving the base method's cross-batch edges.

    The proportional split can give the two batches different degrees, so this variant supports
    t-SNE only; UMAP's precomputed-neighbor interface requires one fixed width.
    """
    plan = np.asarray(plan, dtype=float)
    if plan.ndim != 2:
        raise ValueError(f"plan must be a 2D array, got shape {plan.shape!r}.")
    n0, n1 = plan.shape
    if effective_k is None:
        raise ValueError("effective_k is required for the zero-diagonal variant.")
    if backend != DEFAULT_BACKEND:
        raise ValueError(
            "the split-budget zero-diagonal variant is only supported by the t-SNE backend; got "
            f"backend={backend!r}, whose neighbour list has one fixed width per point. Under the "
            "proportional split a batch-0 point keeps batch1_k neighbours and a batch-1 point "
            "batch0_k, which that list cannot express. Use ot-zerodiag for a UMAP arm."
        )

    init_info: dict[str, object] = {}
    with _stage(timer, "time_pre_embedding"), _stage(timer, "time_ot_postprocess"):
        graph, selection, singular_values, resolved_init = _block_graph(
            plan,
            init=init,
            effective_k=effective_k,
            rule="cross_split",
            within_batch=None,
            graph_symmetrization=graph_symmetrization,
            binarize=binarize,
        )
        split = selection.split
        diag0_edges, diag1_edges = selection.diag_edges
    with _stage(timer, "time_embedding"):
        embedding, backend_artifacts = _embed_ot_graph(
            backend=backend,
            graph=graph,
            four_block_matrix=None,
            n0=n0,
            n1=n1,
            effective_k=effective_k,
            within_batch_indices=None,
            x_joint=x_joint,
            random_state=random_state,
            n_epochs=n_epochs,
            n_jobs=n_jobs,
            early_exaggeration_iter=early_exaggeration_iter,
            init_scale=init_scale,
            init=resolved_init,
            init_info=init_info,
            timer=timer,
        )
    if not return_artifacts:
        return embedding

    artifacts: dict[str, object] = {
        **backend_artifacts,
        **_spectrum_artifacts(singular_values),
        "effective_k": int(split.effective_k),
        "batch0_k": int(split.batch0_k),
        "batch1_k": int(split.batch1_k),
        "within_batch_k": 0,
        "graph_symmetrization": graph_symmetrization,
        "diag0_edges": diag0_edges,
        "diag1_edges": diag1_edges,
        **init_info,
    }
    return embedding, artifacts


def compute_ot_tsne_zerodiag_split_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    rel_reg: float = 1.0,
    effective_k: int = 15,
    binarize: bool = True,
    random_state: int = 0,
    n_epochs: int = DEFAULT_TSNE_N_ITER,
    n_jobs: int = 1,
    early_exaggeration_iter: int | None = DEFAULT_TSNE_EARLY_EXAGGERATION_ITER,
    tol: float = 1e-12,
    max_iter: int = int(1e4),
    init_scale: float = 10000.0,
    init: str | np.ndarray = "random",
    backend: str = DEFAULT_BACKEND,
    device: str = CPU_DEVICE,
    graph_symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """End-to-end split-budget zero-diagonal OT-tSNE (see the plan-level docstring)."""
    x0, x1 = _validate_pair_arrays(x0, x1)
    with _stage(timer, "time_pre_embedding"):
        plan, ot_artifacts = compute_ot_plan(
            x0,
            x1,
            rel_reg=rel_reg,
            tol=tol,
            max_iter=max_iter,
            device=device,
            return_artifacts=True,
            timer=timer,
        )
    result = compute_ot_tsne_zerodiag_split_embedding_from_plan(
        plan,
        effective_k=effective_k,
        binarize=binarize,
        random_state=random_state,
        n_epochs=n_epochs,
        n_jobs=n_jobs,
        early_exaggeration_iter=early_exaggeration_iter,
        init_scale=init_scale,
        init=init,
        backend=backend,
        x_joint=np.vstack([x0, x1]),
        graph_symmetrization=graph_symmetrization,
        return_artifacts=return_artifacts,
        timer=timer,
    )
    if not return_artifacts:
        return result
    embedding, method_artifacts = result
    artifacts: dict[str, object] = {
        **method_artifacts,
        "epsilon": ot_artifacts["epsilon"],
        "median_squared_distance": ot_artifacts["median_squared_distance"],
        "ot_device": ot_artifacts["ot_device"],
        "torch_device": ot_artifacts["ot_device"],
    }
    return embedding, artifacts


def compute_ot_tsne_zerodiag_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    rel_reg: float = 1.0,
    effective_k: int = 15,
    binarize: bool = True,
    random_state: int = 0,
    n_epochs: int = DEFAULT_TSNE_N_ITER,
    n_jobs: int = 1,
    early_exaggeration_iter: int | None = DEFAULT_TSNE_EARLY_EXAGGERATION_ITER,
    tol: float = 1e-12,
    max_iter: int = int(1e4),
    init_scale: float = 10000.0,
    init: str | np.ndarray = "random",
    backend: str = DEFAULT_BACKEND,
    device: str = CPU_DEVICE,
    graph_symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """End-to-end zero-diagonal OT-tSNE from raw arrays (see the plan-level docstring)."""
    x0, x1 = _validate_pair_arrays(x0, x1)
    with _stage(timer, "time_pre_embedding"):
        plan, ot_artifacts = compute_ot_plan(
            x0,
            x1,
            rel_reg=rel_reg,
            tol=tol,
            max_iter=max_iter,
            device=device,
            return_artifacts=True,
            timer=timer,
        )
    result = compute_ot_tsne_zerodiag_embedding_from_plan(
        plan,
        effective_k=effective_k,
        binarize=binarize,
        random_state=random_state,
        n_epochs=n_epochs,
        n_jobs=n_jobs,
        early_exaggeration_iter=early_exaggeration_iter,
        init_scale=init_scale,
        init=init,
        backend=backend,
        x_joint=np.vstack([x0, x1]),
        graph_symmetrization=graph_symmetrization,
        return_artifacts=return_artifacts,
        timer=timer,
    )
    if not return_artifacts:
        return result
    embedding, method_artifacts = result
    artifacts: dict[str, object] = {
        **method_artifacts,
        "epsilon": float(ot_artifacts["epsilon"]),
        "median_squared_distance": ot_artifacts["median_squared_distance"],
        # The task runner reads torch_device to keep runtime comparisons device-specific.
        "ot_device": ot_artifacts["ot_device"],
        "torch_device": ot_artifacts["ot_device"],
    }
    return embedding, artifacts
