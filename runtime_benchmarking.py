from __future__ import annotations  # postponed evaluation of annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ne_aligned_utils.methods import (  # noqa: E402
    build_ot_tsne_four_block_matrix,
    compute_ot_plan,
    run_tsne_from_graph,
    sparsify_four_block_top_percentage,
)
from ne_aligned_utils.data import make_fig1_data  # noqa: E402


@dataclass
class RingEmbeddingRun:
    obs0: np.ndarray
    obs1: np.ndarray
    clean0: np.ndarray
    clean1: np.ndarray
    plan: np.ndarray
    graph: np.ndarray
    embedding: np.ndarray
    median_squared_distance: float
    epsilon: float
    data_seconds: float
    sinkhorn_seconds: float
    four_block_seconds: float
    sparsify_seconds: float
    tsne_seconds: float
    total_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ring embedding creation using the same OT -> 4-block graph -> "
            "top-percentage sparsification -> t-SNE flow as notebooks/benchmarking.ipynb."
        )
    )
    parser.add_argument("--n0", type=int, default=1000, help="Number of samples in ring dataset 0.")
    parser.add_argument("--n1", type=int, default=5000, help="Number of samples in ring dataset 1.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used to generate the ring data.")
    parser.add_argument(
        "--tsne-seed",
        type=int,
        default=0,
        help="Seed used for the t-SNE initialization.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=200,
        help="Number of t-SNE optimization epochs.",
    )
    parser.add_argument(
        "--top-pct",
        type=float,
        default=10.0,
        help="Retain the top X percent of neighbors in each 4-block matrix block.",
    )
    reg_group = parser.add_mutually_exclusive_group()
    reg_group.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Explicit Sinkhorn regularization epsilon.",
    )
    reg_group.add_argument(
        "--eps-times",
        type=float,
        default=None,
        help=(
            "Notebook-style epsilon scaling: epsilon = median_squared_distance / eps_times. "
            "Defaults to 1.0 when neither --epsilon nor --eps-times is set."
        ),
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-12,
        help="Sinkhorn stopping tolerance.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=int(1e4),
        help="Maximum number of Sinkhorn iterations.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many full pipeline runs to time.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a quick plot of the observed ring data and the final embedding.",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="Optional path to save the plot instead of only displaying it.",
    )
    args = parser.parse_args()
    if args.epsilon is None and args.eps_times is None:
        args.eps_times = 1.0
    return args


def create_ring_embedding(
    *,
    n0: int,
    n1: int,
    seed: int,
    tsne_seed: int,
    n_epochs: int,
    top_pct: float,
    epsilon: float | None,
    eps_times: float,
    tol: float,
    max_iter: int,
    progress_callback: Callable[[str], None] | None = None,
) -> RingEmbeddingRun:
    def emit_progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    pipeline_start = time.perf_counter()
    emit_progress(
        "starting benchmark pipeline "
        f"(n0={n0}, n1={n1}, seed={seed}, tsne_seed={tsne_seed}, "
        f"epsilon={'auto' if epsilon is None else f'{epsilon:g}'}, top_pct={top_pct:g})"
    )

    emit_progress("starting data generation")
    data_start = time.perf_counter()
    obs0, obs1 = make_fig1_data(n0=n0, n1=n1, seed=seed, clean=False)
    clean0, clean1 = make_fig1_data(n0=n0, n1=n1, seed=seed, clean=True)
    data_seconds = time.perf_counter() - data_start
    emit_progress(f"finished data generation in {data_seconds:.4f}s")

    emit_progress("starting Sinkhorn plan computation")
    sinkhorn_start = time.perf_counter()
    plan, ot_artifacts = compute_ot_plan(
        obs0,
        obs1,
        epsilon=epsilon,
        eps_times=eps_times,
        tol=tol,
        max_iter=max_iter,
        return_artifacts=True,
    )
    median_squared_distance = float(ot_artifacts["median_squared_distance"])
    epsilon_value = float(ot_artifacts["epsilon"])
    sinkhorn_seconds = time.perf_counter() - sinkhorn_start
    emit_progress(
        "finished Sinkhorn in "
        f"{sinkhorn_seconds:.4f}s (median_sq_dist={median_squared_distance:.6g}, "
        f"epsilon={epsilon_value:.6g})"
    )

    emit_progress("starting 4-block matrix construction")
    four_block_start = time.perf_counter()
    four_block_matrix = build_ot_tsne_four_block_matrix(plan)
    four_block_seconds = time.perf_counter() - four_block_start
    emit_progress(f"finished 4-block matrix construction in {four_block_seconds:.4f}s")

    emit_progress("starting graph sparsification")
    sparsify_start = time.perf_counter()
    graph = sparsify_four_block_top_percentage(
        four_block_matrix,
        percentage=float(top_pct) / 100.0,
        n0=n0,
        n1=n1,
    )
    sparsify_seconds = time.perf_counter() - sparsify_start
    emit_progress(
        "finished graph sparsification in "
        f"{sparsify_seconds:.4f}s (graph_nnz={int(np.count_nonzero(graph))})"
    )

    emit_progress("starting t-SNE embedding")
    tsne_start = time.perf_counter()
    embedding = run_tsne_from_graph(graph, random_state=tsne_seed, n_epochs=n_epochs)
    tsne_seconds = time.perf_counter() - tsne_start
    emit_progress(f"finished t-SNE embedding in {tsne_seconds:.4f}s")

    total_seconds = time.perf_counter() - pipeline_start
    emit_progress(f"finished benchmark pipeline in {total_seconds:.4f}s")
    return RingEmbeddingRun(
        obs0=np.asarray(obs0),
        obs1=np.asarray(obs1),
        clean0=np.asarray(clean0),
        clean1=np.asarray(clean1),
        plan=np.asarray(plan),
        graph=np.asarray(graph),
        embedding=np.asarray(embedding),
        median_squared_distance=median_squared_distance,
        epsilon=epsilon_value,
        data_seconds=data_seconds,
        sinkhorn_seconds=sinkhorn_seconds,
        four_block_seconds=four_block_seconds,
        sparsify_seconds=sparsify_seconds,
        tsne_seconds=tsne_seconds,
        total_seconds=total_seconds,
    )


def summarize_runs(runs: list[RingEmbeddingRun]) -> dict[str, tuple[float, float]]:
    timings = {
        "data_seconds": [run.data_seconds for run in runs],
        "sinkhorn_seconds": [run.sinkhorn_seconds for run in runs],
        "four_block_seconds": [run.four_block_seconds for run in runs],
        "sparsify_seconds": [run.sparsify_seconds for run in runs],
        "tsne_seconds": [run.tsne_seconds for run in runs],
        "total_seconds": [run.total_seconds for run in runs],
    }
    summary = {}
    for key, values in timings.items():
        arr = np.asarray(values, dtype=float)
        summary[key] = (
            float(arr.mean()),
            float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        )
    return summary


def print_run_report(run: RingEmbeddingRun, run_idx: int, total_runs: int) -> None:
    nnz = int(np.count_nonzero(run.graph))
    density = float(nnz / run.graph.size)
    print(f"Run {run_idx}/{total_runs}")
    print(
        "  epsilon={:.6g} | median_sq_dist={:.6g} | plan_shape={} | graph_shape={}".format(
            run.epsilon,
            run.median_squared_distance,
            run.plan.shape,
            run.graph.shape,
        )
    )
    print(
        "  graph_nnz={} | graph_density={:.6%} | embedding_shape={}".format(
            nnz,
            density,
            run.embedding.shape,
        )
    )
    print(
        "  timings (s): data={:.4f}, sinkhorn={:.4f}, four_block={:.4f}, sparsify={:.4f}, tsne={:.4f}, total={:.4f}".format(
            run.data_seconds,
            run.sinkhorn_seconds,
            run.four_block_seconds,
            run.sparsify_seconds,
            run.tsne_seconds,
            run.total_seconds,
        )
    )


def print_summary_report(summary: dict[str, tuple[float, float]], repeats: int) -> None:
    if repeats <= 1:
        return
    print("\nSummary across repeats")
    for key in (
        "data_seconds",
        "sinkhorn_seconds",
        "four_block_seconds",
        "sparsify_seconds",
        "tsne_seconds",
        "total_seconds",
    ):
        mean_value, sd_value = summary[key]
        print(f"  {key}: mean={mean_value:.4f}s, sd={sd_value:.4f}s")


def maybe_plot(run: RingEmbeddingRun, *, top_pct: float, save_path: Path | None, show: bool) -> None:
    if not show and save_path is None:
        return

    import matplotlib.pyplot as plt

    colors = np.array(["tab:red"] * run.obs0.shape[0] + ["tab:blue"] * run.obs1.shape[0])
    perm = np.random.RandomState(0).permutation(colors.shape[0])

    fig = plt.figure(figsize=(11, 4.5))
    ax_data = fig.add_subplot(121, projection="3d")
    ax_embed = fig.add_subplot(122)

    ax_data.scatter(*run.obs0.T, c="tab:red", s=6, alpha=0.8, label="dataset 0")
    ax_data.scatter(*run.obs1.T, c="tab:blue", s=3, alpha=0.6, label="dataset 1")
    ax_data.set_title("Observed ring data")
    ax_data.view_init(elev=30, azim=-60)
    ax_data.legend(markerscale=2, loc="upper left")

    ax_embed.scatter(*run.embedding[perm].T, s=2, c=colors[perm], alpha=0.85)
    ax_embed.set_title(f"Ring embedding (top {top_pct:g}%)")
    ax_embed.set_aspect("equal")
    ax_embed.axis("off")

    fig.suptitle(
        f"Ring benchmark | epsilon={run.epsilon:.6g} | total={run.total_seconds:.3f}s",
        y=0.98,
    )
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"\nSaved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n0 <= 0 or args.n1 <= 0:
        raise ValueError("n0 and n1 must both be positive.")
    if args.top_pct <= 0 or args.top_pct > 100:
        raise ValueError(f"top_pct must lie in (0, 100], got {args.top_pct}.")
    if args.n_epochs <= 0:
        raise ValueError(f"n_epochs must be positive, got {args.n_epochs}.")
    if args.repeats <= 0:
        raise ValueError(f"repeats must be positive, got {args.repeats}.")

    runs = []
    for run_idx in range(1, args.repeats + 1):
        run = create_ring_embedding(
            n0=args.n0,
            n1=args.n1,
            seed=args.seed,
            tsne_seed=args.tsne_seed,
            n_epochs=args.n_epochs,
            top_pct=args.top_pct,
            epsilon=args.epsilon,
            eps_times=args.eps_times,
            tol=args.tol,
            max_iter=args.max_iter,
        )
        runs.append(run)
        print_run_report(run, run_idx=run_idx, total_runs=args.repeats)

    summary = summarize_runs(runs)
    print_summary_report(summary, repeats=args.repeats)
    maybe_plot(
        runs[-1],
        top_pct=args.top_pct,
        save_path=args.save_plot,
        show=args.plot,
    )


if __name__ == "__main__":
    main()
