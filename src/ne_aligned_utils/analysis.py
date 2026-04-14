from dataclasses import dataclass
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from openTSNE import TSNE
from openTSNE.affinity import Affinities
from sklearn.decomposition import PCA

from ne_aligned_utils.data import load_nCoV
from ne_aligned_utils.utils import compute_ot_plan


@dataclass(frozen=True)
class NCovAnalysisData:
    x1: np.ndarray
    x2: np.ndarray
    labels1: np.ndarray
    labels2: np.ndarray
    cell_type_names: tuple[str, ...]
    dataset_names: tuple[str, str]

    @property
    def joint_data(self) -> np.ndarray:
        return np.concatenate([self.x1, self.x2], axis=0)

    @property
    def joint_labels(self) -> np.ndarray:
        return np.concatenate([self.labels1, self.labels2], axis=0)

    @property
    def batch_ids(self) -> np.ndarray:
        return np.concatenate(
            [
                np.zeros(self.x1.shape[0], dtype=int),
                np.ones(self.x2.shape[0], dtype=int),
            ]
        )

    @property
    def short_dataset_names(self) -> tuple[str, str]:
        return tuple(_short_dataset_name(name) for name in self.dataset_names)


@dataclass(frozen=True)
class EmbeddingViews:
    name: str
    first: np.ndarray
    second: np.ndarray
    joint: np.ndarray


def _copy_embedding_views(embeddings: EmbeddingViews, first: np.ndarray, second: np.ndarray, joint: np.ndarray) -> EmbeddingViews:
    return EmbeddingViews(name=embeddings.name, first=first, second=second, joint=joint)


def _resolve_ncov_path(data_path: str | Path) -> Path:
    path = Path(data_path)
    repo_root = Path(__file__).resolve().parents[2]
    base_candidates = [path]
    if not path.is_absolute():
        base_candidates.append(repo_root / path)

    candidates: list[Path] = []
    for base in base_candidates:
        candidates.append(base)
        if base.name != "nCoV":
            candidates.append(base / "nCoV")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find nCoV data. Checked: {checked}")


def _short_dataset_name(name: str) -> str:
    return name.split("-")[-1]


def _safe_perplexity(n_samples: int, perplexity: float) -> float:
    max_perplexity = max(5, (n_samples - 1) // 3)
    return float(min(perplexity, max_perplexity))


def _cell_type_palette(cell_type_names: tuple[str, ...], cmap_name: str = "tab20") -> np.ndarray:
    cmap = plt.get_cmap(cmap_name, len(cell_type_names))
    return np.array([cmap(i) for i in range(len(cell_type_names))])


def load_ncov_analysis_data(data_path: str | Path = "data/nCoV") -> NCovAnalysisData:
    resolved_path = _resolve_ncov_path(data_path)
    x1, x2, cell_type_names, labels1, labels2, dataset_names = load_nCoV(
        str(resolved_path),
        return_meta=True,
    )
    return NCovAnalysisData(
        x1=np.asarray(x1),
        x2=np.asarray(x2),
        labels1=np.asarray(labels1, dtype=int),
        labels2=np.asarray(labels2, dtype=int),
        cell_type_names=tuple(str(name) for name in cell_type_names),
        dataset_names=tuple(str(name) for name in dataset_names),
    )


def compute_pca_embeddings(
    data: NCovAnalysisData,
    n_components: int = 50,
    random_state: int = 0,
) -> EmbeddingViews:
    first = PCA(n_components=n_components, random_state=random_state).fit_transform(data.x1)
    second = PCA(n_components=n_components, random_state=random_state).fit_transform(data.x2)
    joint = PCA(n_components=n_components, random_state=random_state).fit_transform(data.joint_data)
    return EmbeddingViews(name="PCA", first=first, second=second, joint=joint)


def _run_tsne(
    x: np.ndarray,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_jobs: int = 1,
) -> np.ndarray:
    tsne = TSNE(
        random_state=random_state,
        perplexity=_safe_perplexity(x.shape[0], perplexity),
        initialization="pca",
        n_jobs=n_jobs,
        verbose=False,
    )
    return np.asarray(tsne.fit(x))


def compute_tsne_embeddings(
    pca_embeddings: EmbeddingViews,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_jobs: int = 1,
) -> EmbeddingViews:
    return EmbeddingViews(
        name="t-SNE",
        first=_run_tsne(
            pca_embeddings.first,
            random_state=random_state,
            perplexity=perplexity,
            n_jobs=n_jobs,
        ),
        second=_run_tsne(
            pca_embeddings.second,
            random_state=random_state,
            perplexity=perplexity,
            n_jobs=n_jobs,
        ),
        joint=_run_tsne(
            pca_embeddings.joint,
            random_state=random_state,
            perplexity=perplexity,
            n_jobs=n_jobs,
        ),
    )


def compute_eot_embeddings(
    base_embeddings: EmbeddingViews,
    t: int = 3,
    eps_times: float = 1.0,
    tol: float = 1e-12,
    max_iter: int = int(1e4),
    embd_dim: int = 2,
) -> EmbeddingViews:
    n0 = base_embeddings.first.shape[0]
    n1 = base_embeddings.second.shape[0]
    plan = compute_ot_plan(
        base_embeddings.first,
        base_embeddings.second,
        eps_times=eps_times,
        tol=tol,
        maxIter=max_iter,
    )
    u, singular_values, vt = np.linalg.svd(np.asarray(plan, dtype=float), full_matrices=False)
    start = 1
    stop = start + embd_dim
    left = u[:, start:stop] * (singular_values[start:stop] ** t) * np.sqrt(n0)
    right = vt.T[:, start:stop] * (singular_values[start:stop] ** t) * np.sqrt(n1)
    joint = np.concatenate([left, right], axis=0)
    return EmbeddingViews(name=f"EOT t={t}", first=left, second=right, joint=joint)


def compute_ot_plan_from_views(
    base_embeddings: EmbeddingViews,
    eps_times: float = 10.0,
    tol: float = 1e-12,
    max_iter: int = int(1e4),
) -> np.ndarray:
    return np.asarray(
        compute_ot_plan(
            base_embeddings.first,
            base_embeddings.second,
            eps_times=eps_times,
            tol=tol,
            maxIter=max_iter,
        ),
        dtype=float,
    )


def compute_ot_plan_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    eps_times: float = 10.0,
    tol: float = 1e-12,
    max_iter: int = int(1e4),
) -> np.ndarray:
    return np.asarray(
        compute_ot_plan(
            np.asarray(x0, dtype=float),
            np.asarray(x1, dtype=float),
            eps_times=eps_times,
            tol=tol,
            maxIter=max_iter,
        ),
        dtype=float,
    )


def build_ot_tsne_four_block_matrix(plan: np.ndarray) -> np.ndarray:
    n0, n1 = plan.shape
    u, singular_values, vt = np.linalg.svd(np.asarray(plan, dtype=float), full_matrices=False)
    eot = np.concatenate([u * np.sqrt(n0), vt.T * np.sqrt(n1)], axis=0)
    matrix = eot @ np.diag(singular_values) @ eot.T / np.sqrt(n0 * n1)
    matrix = np.asarray(matrix, dtype=float)
    matrix[matrix < 0] = 0.0
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _row_or_col_topk_mask(block: np.ndarray, row_k: int, col_k: int) -> np.ndarray:
    n_rows, n_cols = block.shape
    row_k = min(max(1, int(row_k)), n_cols)
    col_k = min(max(1, int(col_k)), n_rows)
    mask = np.zeros_like(block, dtype=bool)

    if row_k >= n_cols:
        mask[:, :] = True
    else:
        row_idx = np.argpartition(block, n_cols - row_k, axis=1)[:, -row_k:]
        rows = np.arange(n_rows, dtype=int)[:, None]
        mask[rows, row_idx] = True

    if col_k >= n_rows:
        mask[:, :] = True
    else:
        col_idx = np.argpartition(block, n_rows - col_k, axis=0)[-col_k:, :]
        cols = np.arange(n_cols, dtype=int)[None, :]
        mask[col_idx, cols] = True

    return mask


def _top_percentage_neighbors_block_mask(block: np.ndarray, percentage: float) -> np.ndarray:
    n_rows, n_cols = block.shape
    row_k = max(1, int(np.ceil(float(percentage) * n_cols)))
    col_k = max(1, int(np.ceil(float(percentage) * n_rows)))
    return _row_or_col_topk_mask(block, row_k=row_k, col_k=col_k)


def _ensure_nonempty_mask(mask: np.ndarray, block: np.ndarray) -> np.ndarray:
    mask = np.array(mask, dtype=bool, copy=True)

    empty_rows = np.where(~mask.any(axis=1))[0]
    if empty_rows.size > 0:
        row_argmax = np.argmax(block[empty_rows], axis=1)
        mask[empty_rows, row_argmax] = True

    empty_cols = np.where(~mask.any(axis=0))[0]
    if empty_cols.size > 0:
        col_argmax = np.argmax(block[:, empty_cols], axis=0)
        mask[col_argmax, empty_cols] = True

    return mask


def sparsify_four_block_top_neighbors(
    matrix: np.ndarray,
    n0: int,
    n1: int,
    percentage: float = 0.15,
    ensure_nonempty: bool = True,
    binarize: bool = False,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (int(n0) + int(n1), int(n0) + int(n1)):
        raise ValueError(
            f"Expected a square 4-block matrix with shape {(int(n0) + int(n1), int(n0) + int(n1))}, got {matrix.shape}."
        )
    if not (0.0 < float(percentage) <= 1.0):
        raise ValueError(f"percentage must be in (0, 1], got {percentage}.")

    mask = np.zeros_like(matrix, dtype=bool)
    for row_slice, col_slice in (
        (slice(0, n0), slice(0, n0)),
        (slice(0, n0), slice(n0, n0 + n1)),
        (slice(n0, n0 + n1), slice(0, n0)),
        (slice(n0, n0 + n1), slice(n0, n0 + n1)),
    ):
        block = matrix[row_slice, col_slice]
        block_mask = _top_percentage_neighbors_block_mask(block, percentage)
        if ensure_nonempty:
            block_mask = _ensure_nonempty_mask(block_mask, block)
        mask[row_slice, col_slice] = block_mask

    np.fill_diagonal(mask, False)
    sparse_matrix = np.where(mask, matrix, 0.0)
    sparse_matrix = np.nan_to_num(sparse_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    sparse_matrix[sparse_matrix < 0] = 0.0
    if binarize:
        sparse_matrix = (sparse_matrix > 0.0).astype(float)
    return sparse_matrix


def normalize_ot_tsne_graph(matrix: np.ndarray) -> np.ndarray:
    graph = np.asarray(matrix, dtype=float)
    if graph.ndim != 2:
        raise ValueError(f"Expected a 2D 4-block matrix, got shape {graph.shape}.")

    graph = np.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0)
    graph[graph < 0] = 0.0
    graph = (graph > 0.0).astype(float)
    graph = np.maximum(graph, graph.T)
    np.fill_diagonal(graph, 0.0)

    total_mass = float(graph.sum())
    if total_mass <= 0.0:
        raise ValueError("Graph has zero total mass after normalization.")
    return graph / total_mass


def run_tsne_from_graph(
    graph: np.ndarray,
    random_state: int = 0,
    n_epochs: int = 200,
    init_scale: float = 10000.0,
) -> np.ndarray:
    graph = np.asarray(graph, dtype=float)
    graph = np.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0)
    graph[graph < 0] = 0.0
    np.fill_diagonal(graph, 0.0)

    total_mass = float(graph.sum())
    if total_mass <= 0.0:
        raise ValueError("Graph has zero total mass; cannot run t-SNE.")
    graph = graph / total_mass

    n_points = graph.shape[0]
    init = np.random.RandomState(int(random_state)).randn(n_points, 2)
    init = init / (init[:, 0].std() * float(init_scale))
    n_early = int(n_epochs) // 3
    n_late = int(n_epochs) - n_early

    affinities = Affinities()
    affinities.P = sp.csr_matrix(graph)
    embedder = TSNE(
        n_components=2,
        random_state=int(random_state),
        verbose=False,
        early_exaggeration_iter=n_early,
        n_iter=n_late,
    )
    return np.asarray(embedder.fit(affinities=affinities, initialization=init))


def standardize_embedding_views(embeddings: EmbeddingViews, eps: float = 1e-12) -> EmbeddingViews:
    mean = embeddings.joint.mean(axis=0, keepdims=True)
    std = embeddings.joint.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return _copy_embedding_views(
        embeddings,
        first=(embeddings.first - mean) / std,
        second=(embeddings.second - mean) / std,
        joint=(embeddings.joint - mean) / std,
    )


def plot_embedding_panels(
    embeddings: EmbeddingViews,
    data: NCovAnalysisData,
    color_by: str = "batch",
    point_size: float = 3.0,
    alpha: float = 0.85,
    joint_shuffle_seed: int = 0,
    figsize: tuple[float, float] = (15, 5),
    standardize: bool = False,
    aspect: str | float = "equal",
):
    if color_by not in {"batch", "cell_type"}:
        raise ValueError("color_by must be either 'batch' or 'cell_type'")

    plotted_embeddings = (
        standardize_embedding_views(embeddings) if standardize else embeddings
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    batch_palette = np.array(["tab:red", "tab:blue"], dtype=object)

    if color_by == "batch":
        first_colors = batch_palette[0]
        second_colors = batch_palette[1]
        joint_colors = batch_palette[data.batch_ids]
        legend_handles = [
            mlines.Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=6,
                markerfacecolor=batch_palette[i],
                markeredgecolor="none",
                label=data.short_dataset_names[i],
            )
            for i in range(2)
        ]
        legend_title = "Batch"
    else:
        palette = _cell_type_palette(data.cell_type_names)
        first_colors = palette[data.labels1]
        second_colors = palette[data.labels2]
        joint_colors = palette[data.joint_labels]
        legend_handles = [
            mlines.Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=6,
                markerfacecolor=palette[i],
                markeredgecolor="none",
                label=label,
            )
            for i, label in enumerate(data.cell_type_names)
        ]
        legend_title = "Cell type"

    panel_specs = [
        (axes[0], plotted_embeddings.first, first_colors, data.short_dataset_names[0]),
        (axes[1], plotted_embeddings.second, second_colors, data.short_dataset_names[1]),
    ]

    for ax, values, colors, title_suffix in panel_specs:
        ax.scatter(values[:, 0], values[:, 1], s=point_size, c=colors, alpha=alpha)
        ax.set_title(f"{embeddings.name} on {title_suffix}")
        ax.set_aspect(aspect)
        ax.axis("off")

    order = np.random.default_rng(joint_shuffle_seed).permutation(plotted_embeddings.joint.shape[0])
    axes[2].scatter(
        plotted_embeddings.joint[order, 0],
        plotted_embeddings.joint[order, 1],
        s=point_size,
        c=joint_colors[order],
        alpha=alpha,
    )
    axes[2].set_title(f"Joint {embeddings.name}")
    axes[2].set_aspect(aspect)
    axes[2].axis("off")
    axes[2].legend(
        handles=legend_handles,
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
    )

    fig.suptitle(f"{embeddings.name} embeddings colored by {color_by.replace('_', ' ')}")
    return fig, axes


def plot_joint_embedding(
    embedding: np.ndarray,
    data: NCovAnalysisData,
    color_by: str = "batch",
    point_size: float = 3.0,
    alpha: float = 0.85,
    shuffle_seed: int = 0,
    figsize: tuple[float, float] = (6, 5),
    aspect: str | float = "equal",
):
    if color_by not in {"batch", "cell_type"}:
        raise ValueError("color_by must be either 'batch' or 'cell_type'")

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    order = np.random.default_rng(shuffle_seed).permutation(embedding.shape[0])

    if color_by == "batch":
        palette = np.array(["tab:red", "tab:blue"], dtype=object)
        colors = palette[data.batch_ids]
        legend_handles = [
            mlines.Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=6,
                markerfacecolor=palette[i],
                markeredgecolor="none",
                label=data.short_dataset_names[i],
            )
            for i in range(2)
        ]
        legend_title = "Batch"
        title = "Joint OT-tSNE colored by batch"
    else:
        palette = _cell_type_palette(data.cell_type_names)
        colors = palette[data.joint_labels]
        legend_handles = [
            mlines.Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=6,
                markerfacecolor=palette[i],
                markeredgecolor="none",
                label=label,
            )
            for i, label in enumerate(data.cell_type_names)
        ]
        legend_title = "Cell type"
        title = "Joint OT-tSNE colored by cell type"

    ax.scatter(
        embedding[order, 0],
        embedding[order, 1],
        s=point_size,
        c=colors[order],
        alpha=alpha,
    )
    ax.set_title(title)
    ax.set_aspect(aspect)
    ax.axis("off")
    ax.legend(
        handles=legend_handles,
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
    )
    return fig, ax


def plot_joint_embedding_pair(
    embedding: np.ndarray,
    data: NCovAnalysisData,
    point_size: float = 3.0,
    alpha: float = 0.85,
    shuffle_seed: int = 0,
    figsize: tuple[float, float] = (12, 5),
    aspect: str | float = "equal",
):
    order = np.random.default_rng(shuffle_seed).permutation(embedding.shape[0])
    batch_palette = np.array(["tab:red", "tab:blue"], dtype=object)
    cell_type_palette = _cell_type_palette(data.cell_type_names)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    axes[0].scatter(
        embedding[order, 0],
        embedding[order, 1],
        s=point_size,
        c=cell_type_palette[data.joint_labels][order],
        alpha=alpha,
    )
    axes[0].set_title("Cell type")
    axes[0].set_aspect(aspect)
    axes[0].axis("off")
    axes[0].legend(
        handles=[
            mlines.Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=6,
                markerfacecolor=cell_type_palette[i],
                markeredgecolor="none",
                label=label,
            )
            for i, label in enumerate(data.cell_type_names)
        ],
        title="Cell type",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
    )

    axes[1].scatter(
        embedding[order, 0],
        embedding[order, 1],
        s=point_size,
        c=batch_palette[data.batch_ids][order],
        alpha=alpha,
    )
    axes[1].set_title("Batch")
    axes[1].set_aspect(aspect)
    axes[1].axis("off")
    axes[1].legend(
        handles=[
            mlines.Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=6,
                markerfacecolor=batch_palette[i],
                markeredgecolor="none",
                label=data.short_dataset_names[i],
            )
            for i in range(2)
        ],
        title="Batch",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
    )

    fig.suptitle("Joint OT-tSNE")
    return fig, axes


def plot_matrix(
    matrix: np.ndarray,
    title: str,
    figsize: tuple[float, float] = (6, 5),
    cmap: str = "magma",
    aspect: str | float = "auto",
):
    matrix = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    image = ax.imshow(matrix, cmap=cmap, interpolation="nearest", aspect=aspect)
    ax.set_title(title)
    ax.set_xlabel("index")
    ax.set_ylabel("index")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax


def compute_ot_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    eps_times: float = 1.0,
    percentage: float = 0.005,
    tol: float = 1e-12,
    max_iter: int = int(1e4),
    ensure_nonempty: bool = True,
    binarize: bool = True,
    random_state: int = 0,
    n_epochs: int = 200,
    init_scale: float = 10000.0,
) -> np.ndarray:
    ot_plan = compute_ot_plan_from_arrays(
        x0,
        x1,
        eps_times=eps_times,
        tol=tol,
        max_iter=max_iter,
    )
    four_block = build_ot_tsne_four_block_matrix(ot_plan)
    sparse_four_block = sparsify_four_block_top_neighbors(
        four_block,
        n0=np.asarray(x0).shape[0],
        n1=np.asarray(x1).shape[0],
        percentage=percentage,
        ensure_nonempty=ensure_nonempty,
        binarize=binarize,
    )
    graph = normalize_ot_tsne_graph(sparse_four_block)
    return run_tsne_from_graph(
        graph,
        random_state=random_state,
        n_epochs=n_epochs,
        init_scale=init_scale,
    )


def plot_ot_tsne_sweep_grid(
    data: NCovAnalysisData,
    percentages: tuple[float, ...] = (0.005, 0.01, 0.05, 0.1, 0.2),
    eps_values: tuple[float, ...] = (1.0, 10.0, 100.0, 300.0),
    point_size: float = 1.5,
    alpha: float = 0.7,
    binarize: bool = True,
    shuffle_seed: int = 0,
    aspect: str | float = "auto",
    random_state: int = 0,
    n_epochs: int = 200,
    init_scale: float = 10000.0,
):
    percentages = tuple(float(value) for value in percentages)
    eps_values = tuple(float(value) for value in eps_values)
    embeddings: dict[tuple[float, float], np.ndarray] = {}

    batch_palette = np.array(["tab:red", "tab:blue"], dtype=object)
    cell_type_palette = _cell_type_palette(data.cell_type_names)
    order = np.random.default_rng(shuffle_seed).permutation(data.joint_data.shape[0])

    n_rows = len(eps_values)
    n_cols = 2 * len(percentages)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.8 * n_cols, 2.8 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for row_idx, eps_times in enumerate(eps_values):
        for col_idx, percentage in enumerate(percentages):
            embedding = compute_ot_tsne_embedding_from_arrays(
                data.x1,
                data.x2,
                eps_times=eps_times,
                percentage=percentage,
                binarize=binarize,
                random_state=random_state,
                n_epochs=n_epochs,
                init_scale=init_scale,
            )
            embeddings[(eps_times, percentage)] = embedding

            ax_cell = axes[row_idx, 2 * col_idx]
            ax_batch = axes[row_idx, 2 * col_idx + 1]

            ax_cell.scatter(
                embedding[order, 0],
                embedding[order, 1],
                s=point_size,
                c=cell_type_palette[data.joint_labels][order],
                alpha=alpha,
            )
            ax_batch.scatter(
                embedding[order, 0],
                embedding[order, 1],
                s=point_size,
                c=batch_palette[data.batch_ids][order],
                alpha=alpha,
            )

            if row_idx == 0:
                ax_cell.set_title(f"p={percentage:g}\ncell type")
                ax_batch.set_title(f"p={percentage:g}\nbatch")

            if col_idx == 0:
                ax_cell.set_ylabel(f"eps={eps_times:g}")

            for ax in (ax_cell, ax_batch):
                ax.set_aspect(aspect)
                ax.axis("off")

    fig.suptitle("Joint OT-tSNE sweep")
    return embeddings, fig, axes


__all__ = [
    "EmbeddingViews",
    "NCovAnalysisData",
    "compute_eot_embeddings",
    "compute_ot_plan_from_arrays",
    "compute_ot_plan_from_views",
    "compute_ot_tsne_embedding_from_arrays",
    "compute_pca_embeddings",
    "compute_tsne_embeddings",
    "load_ncov_analysis_data",
    "build_ot_tsne_four_block_matrix",
    "plot_ot_tsne_sweep_grid",
    "plot_embedding_panels",
    "plot_joint_embedding_pair",
    "plot_joint_embedding",
    "plot_matrix",
    "normalize_ot_tsne_graph",
    "run_tsne_from_graph",
    "sparsify_four_block_top_neighbors",
    "standardize_embedding_views",
]
