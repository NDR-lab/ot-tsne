from dataclasses import dataclass
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from ne_aligned_utils.data import load_nCoV
from ne_aligned_utils.methods import compute_ot_tsne_embedding_from_arrays


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


def standardize_embedding(embedding: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=float)
    mean = embedding.mean(axis=0, keepdims=True)
    std = embedding.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (embedding - mean) / std


def plot_embedding_panels(
    embedding: np.ndarray,
    data: NCovAnalysisData,
    name: str = "embedding",
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

    embedding = np.asarray(embedding, dtype=float)
    if embedding.shape[0] != data.joint_data.shape[0]:
        raise ValueError(
            "embedding and data must contain the same number of samples, "
            f"got {embedding.shape[0]} and {data.joint_data.shape[0]}."
        )
    plotted_embedding = standardize_embedding(embedding) if standardize else embedding
    first_embedding = plotted_embedding[: data.x1.shape[0]]
    second_embedding = plotted_embedding[data.x1.shape[0] :]

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
        (axes[0], first_embedding, first_colors, data.short_dataset_names[0]),
        (axes[1], second_embedding, second_colors, data.short_dataset_names[1]),
    ]

    for ax, values, colors, title_suffix in panel_specs:
        ax.scatter(values[:, 0], values[:, 1], s=point_size, c=colors, alpha=alpha)
        ax.set_title(f"{name} on {title_suffix}")
        ax.set_aspect(aspect)
        ax.axis("off")

    order = np.random.default_rng(joint_shuffle_seed).permutation(plotted_embedding.shape[0])
    axes[2].scatter(
        plotted_embedding[order, 0],
        plotted_embedding[order, 1],
        s=point_size,
        c=joint_colors[order],
        alpha=alpha,
    )
    axes[2].set_title(f"Joint {name}")
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

    fig.suptitle(f"{name} embeddings colored by {color_by.replace('_', ' ')}")
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
                effective_k=None,
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
    "NCovAnalysisData",
    "load_ncov_analysis_data",
    "plot_ot_tsne_sweep_grid",
    "plot_embedding_panels",
    "plot_joint_embedding_pair",
    "plot_joint_embedding",
    "plot_matrix",
    "standardize_embedding",
]
