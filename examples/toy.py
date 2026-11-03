"""Compare joint t-SNE and OT-tSNE on the README's toy data."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from ne_aligned_utils.methods import (
    compute_ot_tsne_embedding_from_arrays,
    compute_tsne_embedding,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "toy_embedding.png",
        help="Path for the resulting figure (default: figures/toy_embedding.png).",
    )
    args = parser.parse_args()

    # Two unpaired samples from three populations, with a shift in batch 1.
    X, labels = make_blobs(
        n_samples=300,
        centers=[[-4, 0, 0], [0, 4, 0], [4, 0, 0]],
        cluster_std=0.5,
        random_state=0,
    )
    x0, x1 = X[:150].copy(), X[150:].copy()
    x1[:, 2] += 6.0

    unintegrated = compute_tsne_embedding(
        np.vstack([x0, x1]),
        random_state=0,
        perplexity=30,
        n_jobs=4,
    )
    embedding = compute_ot_tsne_embedding_from_arrays(
        x0,
        x1,
        rel_reg=1.0,
        effective_k=15,
        random_state=0,
        n_jobs=4,
    )
    batch = np.repeat([0, 1], [len(x0), len(x1)])

    # Labels are used only for plotting; the method receives just x0 and x1.
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), layout="constrained")
    for col, (method, coordinates) in enumerate(
        [
            ("Joint t-SNE (unintegrated)", unintegrated),
            ("OT-tSNE", embedding),
        ]
    ):
        for row, (groups, palette, prefix) in enumerate(
            [
                (batch, ["#1f77b4", "#ff7f0e"], "Batch"),
                (labels, ["#4c78a8", "#f58518", "#54a24b"], "Population"),
            ]
        ):
            ax = axes[row, col]
            for group, color in enumerate(palette):
                ax.scatter(
                    *coordinates[groups == group].T,
                    color=color,
                    s=16,
                    linewidths=0,
                    alpha=0.85,
                    label=f"{prefix} {group}",
                )
            ax.set(title=f"{method}\nColored by {prefix.lower()}", aspect="equal")
            ax.set_box_aspect(1)
            ax.axis("off")
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=len(palette),
                frameon=False,
                fontsize=9,
                handletextpad=0.3,
                columnspacing=0.8,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
