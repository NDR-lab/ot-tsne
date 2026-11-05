"""Embed the supplied nCoV PCA inputs with OT-tSNE and save the README figure."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ne_aligned_utils.methods import compute_ot_tsne_embedding_from_arrays


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data/nCoV/preprocessed")
    parser.add_argument("--output", type=Path, default=root / "figures/ncov_embedding.png")
    args = parser.parse_args()

    # These scores come from one PCA fitted jointly to both samples.
    x0 = np.load(args.data_dir / "X1_pca50.npy", allow_pickle=False)
    x1 = np.load(args.data_dir / "X2_pca50.npy", allow_pickle=False)
    embedding = compute_ot_tsne_embedding_from_arrays(
        x0,
        x1,
        rel_reg=1.0,
        effective_k=15,
        random_state=0,
        n_jobs=4,
    )
    if embedding.shape != (len(x0) + len(x1), 2) or not np.isfinite(embedding).all():
        raise ValueError("Expected a finite joint two-dimensional embedding.")

    # Annotations are used only for coloring, never for fitting the embedding.
    labels = np.concatenate(
        [
            np.load(args.data_dir / "labels1.npy", allow_pickle=False),
            np.load(args.data_dir / "labels2.npy", allow_pickle=False),
        ]
    )
    cell_types = np.load(args.data_dir / "cell_type_names.npy", allow_pickle=False)
    batch = np.repeat([0, 1], [len(x0), len(x1)])

    # Shuffle drawing order so one batch does not systematically cover the other.
    order = np.random.default_rng(0).permutation(len(embedding))
    coordinates = embedding[order]
    batch, labels = batch[order], labels[order]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    batch_colors = ["#1f77b4", "#ff7f0e"]
    axes[0].scatter(*coordinates.T, c=np.array(batch_colors)[batch], s=4, linewidths=0)

    palette = plt.get_cmap("tab20", len(cell_types))
    axes[1].scatter(*coordinates.T, c=palette(labels), s=4, linewidths=0)
    for ax, title in zip(axes, ["Colored by batch", "Colored by cell type"]):
        ax.set(title=f"nCoV · OT-tSNE\n{title}", aspect="equal")
        ax.set_box_aspect(1)
        ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.02, wspace=0.08)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Embedded {len(x0)} + {len(x1)} cells; saved {args.output}")


if __name__ == "__main__":
    main()
