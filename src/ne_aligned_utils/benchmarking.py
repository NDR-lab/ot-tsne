from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from openTSNE import TSNE
from openTSNE.affinity import Affinities

from ne_aligned_utils.data import make_fig1_data, make_joint_gmm
from ne_aligned_utils.metrics import (
    compute_concordance,
    compute_cross_concordance,
    compute_foscttm,
    compute_kmeans_rand,
)
from ne_aligned_utils.transport_plan_sparsify import sparsify_plan_percentage_neighbors
from ne_aligned_utils.utils import compute_ot_plan


METRIC_NAMES = ("concordance", "cross_concordance", "foscttm")
SUMMARY_METRICS = ("concordance", "cross_concordance")
PLOT_COLORS = {
    "concordance": "tab:blue",
    "cross_concordance": "tab:orange",
    "foscttm": "tab:green",
    "rand": "tab:purple",
}
PLOT_LABELS = {
    "concordance": "Concordance",
    "cross_concordance": "Cross-Concordance",
    "foscttm": "FOSCTTM",
    "rand": "Rand Index",
}


def prepare_benchmark_data(
    seed=0,
    n0_ring=1000,
    n1_ring=5000,
    gmm_n0=600,
    gmm_n1=600,
    gmm_p=1000,
    gmm_q=6,
    gmm_theta=3.0,
):
    ring_obs0, ring_obs1 = make_fig1_data(n0=n0_ring, n1=n1_ring, seed=seed, clean=False)
    ring_clean0, ring_clean1 = make_fig1_data(n0=n0_ring, n1=n1_ring, seed=seed, clean=True)

    gmm_obs0, gmm_obs1 = make_joint_gmm(
        n0=gmm_n0,
        n1=gmm_n1,
        p=gmm_p,
        q=gmm_q,
        theta=gmm_theta,
        seed=seed,
        clean=False,
    )
    gmm_clean0, gmm_clean1, clusters0, clusters1 = make_joint_gmm(
        n0=gmm_n0,
        n1=gmm_n1,
        p=gmm_p,
        q=gmm_q,
        theta=gmm_theta,
        seed=seed,
        clean=True,
    )

    sort_idx0 = np.argsort(clusters0)
    sort_idx1 = np.argsort(clusters1)

    return {
        "seed": int(seed),
        "ring": {
            "obs0": np.asarray(ring_obs0),
            "obs1": np.asarray(ring_obs1),
            "clean0": np.asarray(ring_clean0),
            "clean1": np.asarray(ring_clean1),
            "n0": int(n0_ring),
            "n1": int(n1_ring),
        },
        "gmm": {
            "obs0": np.asarray(gmm_obs0),
            "obs1": np.asarray(gmm_obs1),
            "clean0": np.asarray(gmm_clean0[sort_idx0]),
            "clean1": np.asarray(gmm_clean1[sort_idx1]),
            "clusters0": np.asarray(clusters0[sort_idx0]),
            "clusters1": np.asarray(clusters1[sort_idx1]),
            "sort_idx0": np.asarray(sort_idx0),
            "sort_idx1": np.asarray(sort_idx1),
            "n0": int(gmm_n0),
            "n1": int(gmm_n1),
            "p": int(gmm_p),
            "q": int(gmm_q),
            "theta": float(gmm_theta),
        },
    }


def compute_ot_plans_for_eps(data_bundle, eps_times, tol=1e-12, max_iter=int(1e4)):
    ring_plan = compute_ot_plan(
        data_bundle["ring"]["obs0"],
        data_bundle["ring"]["obs1"],
        eps_times=eps_times,
        tol=tol,
        maxIter=max_iter,
    )

    gmm_plan = compute_ot_plan(
        data_bundle["gmm"]["obs0"],
        data_bundle["gmm"]["obs1"],
        eps_times=eps_times,
        tol=tol,
        maxIter=max_iter,
    )
    gmm_plan = gmm_plan[
        np.ix_(data_bundle["gmm"]["sort_idx0"], data_bundle["gmm"]["sort_idx1"])
    ]

    return {
        "eps_times": float(eps_times),
        "ring": {
            "P": np.asarray(ring_plan),
            "lognorm": _lognorm_for_plan(ring_plan),
        },
        "gmm": {
            "P": np.asarray(gmm_plan),
            "lognorm": _lognorm_for_plan(gmm_plan),
        },
    }


def compute_eot_reference_from_plans(
    data_bundle,
    plans,
    t=1,
    embd_dim=2,
    k_conc=50,
    k_cross=50,
    cache_path=None,
):
    out = {
        "t": int(t),
        "embd_dim": int(embd_dim),
        "k_conc": int(k_conc),
        "k_cross": int(k_cross),
    }
    for dataset_name in ("ring", "gmm"):
        dataset = data_bundle[dataset_name]
        embd = build_eot_embedding_from_plan(
            plans[dataset_name]["P"],
            n0=dataset["n0"],
            n1=dataset["n1"],
            embd_dim=embd_dim,
            t=t,
        )
        metrics = metric_triplet(
            dataset["clean0"],
            dataset["clean1"],
            embd,
            k_conc=k_conc,
            k_cross=k_cross,
        )
        out[dataset_name] = {
            "embedding": np.asarray(embd),
            "metrics": metrics,
        }
    if cache_path is not None:
        save_eot_reference_cache(Path(cache_path), out, plans, data_bundle)
    return out


def build_four_block_matrices(plans):
    return {
        "eps_times": float(plans["eps_times"]),
        "ring": {"matrix": build_four_block_matrix(plans["ring"]["P"])},
        "gmm": {"matrix": build_four_block_matrix(plans["gmm"]["P"])},
    }


def run_top_pct_neighbors_benchmark(
    data_bundle,
    block_matrices,
    pct_values,
    tsne_seeds,
    viz_seed,
    n_epochs,
    k_conc=50,
    k_cross=50,
    binarize_graph=False,
    symmetrize_graph=False,
    normalize_graph=False,
    cache_path=None,
    verbose=True,
):
    pct_values = [float(pct) for pct in pct_values]
    tsne_seeds = [int(seed) for seed in tsne_seeds]
    if int(viz_seed) not in tsne_seeds:
        raise ValueError(f"viz_seed={viz_seed} must be included in tsne_seeds={tsne_seeds}")

    results = {}
    for pct in pct_values:
        if verbose:
            print(
                f"[Top % Neighbors | eps_times={block_matrices['eps_times']:g}] "
                f"calculating pct={pct:g}% ..."
            )
        ring_graph = sparsify_top_percentage_neighbors_graph(
            block_matrices["ring"]["matrix"],
            percentage=float(pct) / 100.0,
            n0=data_bundle["ring"]["n0"],
            n1=data_bundle["ring"]["n1"],
        )
        gmm_graph = sparsify_top_percentage_neighbors_graph(
            block_matrices["gmm"]["matrix"],
            percentage=float(pct) / 100.0,
            n0=data_bundle["gmm"]["n0"],
            n1=data_bundle["gmm"]["n1"],
        )
        ring_graph = postprocess_graph_matrix(
            ring_graph,
            binarize=binarize_graph,
            symmetrize=symmetrize_graph,
            normalize=normalize_graph,
        )
        gmm_graph = postprocess_graph_matrix(
            gmm_graph,
            binarize=binarize_graph,
            symmetrize=symmetrize_graph,
            normalize=normalize_graph,
        )

        ring_metrics = {metric_name: [] for metric_name in METRIC_NAMES}
        gmm_metrics = {metric_name: [] for metric_name in METRIC_NAMES}
        ring_embeddings_by_seed = {}
        gmm_embeddings_by_seed = {}
        ring_metrics_by_seed = {}
        gmm_metrics_by_seed = {}
        ring_viz_emb = None
        gmm_viz_emb = None
        ring_viz_metrics = None
        gmm_viz_metrics = None

        for seed in tsne_seeds:
            ring_embd = run_tsne_from_graph(ring_graph, seed=seed, n_epochs=n_epochs)
            gmm_embd = run_tsne_from_graph(gmm_graph, seed=seed, n_epochs=n_epochs)

            ring_metric_row = metric_triplet(
                data_bundle["ring"]["clean0"],
                data_bundle["ring"]["clean1"],
                ring_embd,
                k_conc=k_conc,
                k_cross=k_cross,
            )
            gmm_metric_row = metric_triplet(
                data_bundle["gmm"]["clean0"],
                data_bundle["gmm"]["clean1"],
                gmm_embd,
                k_conc=k_conc,
                k_cross=k_cross,
            )

            ring_embeddings_by_seed[int(seed)] = np.asarray(ring_embd)
            gmm_embeddings_by_seed[int(seed)] = np.asarray(gmm_embd)
            ring_metrics_by_seed[int(seed)] = ring_metric_row
            gmm_metrics_by_seed[int(seed)] = gmm_metric_row

            for metric_name in METRIC_NAMES:
                ring_metrics[metric_name].append(ring_metric_row[metric_name])
                gmm_metrics[metric_name].append(gmm_metric_row[metric_name])

            if int(seed) == int(viz_seed):
                ring_viz_emb = np.asarray(ring_embd)
                gmm_viz_emb = np.asarray(gmm_embd)
                ring_viz_metrics = ring_metric_row
                gmm_viz_metrics = gmm_metric_row

        ring_summary = summarize_metric_lists(ring_metrics)
        gmm_summary = summarize_metric_lists(gmm_metrics)
        results[float(pct)] = {
            "ring_graph": np.asarray(ring_graph),
            "gmm_graph": np.asarray(gmm_graph),
            "ring_viz_emb": ring_viz_emb,
            "gmm_viz_emb": gmm_viz_emb,
            "ring_viz_metrics": ring_viz_metrics,
            "gmm_viz_metrics": gmm_viz_metrics,
            "ring_embeddings_by_seed": ring_embeddings_by_seed,
            "gmm_embeddings_by_seed": gmm_embeddings_by_seed,
            "ring_metrics_by_seed": ring_metrics_by_seed,
            "gmm_metrics_by_seed": gmm_metrics_by_seed,
            "ring_mean": ring_summary["mean"],
            "ring_sd": ring_summary["sd"],
            "gmm_mean": gmm_summary["mean"],
            "gmm_sd": gmm_summary["sd"],
        }

    benchmark = {
        "eps_times": float(block_matrices["eps_times"]),
        "pct_values": pct_values,
        "tsne_seeds": tsne_seeds,
        "viz_seed": int(viz_seed),
        "n_epochs": int(n_epochs),
        "k_conc": int(k_conc),
        "k_cross": int(k_cross),
        "graph_binarized": bool(binarize_graph),
        "graph_symmetrized": bool(symmetrize_graph),
        "graph_normalized": bool(normalize_graph),
        "results": results,
        "cache_path": None if cache_path is None else str(cache_path),
    }

    if cache_path is not None:
        save_benchmark_cache(Path(cache_path), benchmark, data_bundle)

    return benchmark


def prepare_sparse_plan_four_block_matrix(
    P_sparse,
    diag_threshold=1e-3,
):
    P_sparse = np.asarray(P_sparse, dtype=float)
    if P_sparse.ndim != 2:
        raise ValueError(f"Expected a 2D sparse OT plan, got shape {P_sparse.shape}.")
    if np.any(P_sparse < 0):
        raise ValueError("Expected a non-negative sparse OT plan.")

    diag_threshold = float(diag_threshold)
    if diag_threshold < 0.0:
        raise ValueError(f"diag_threshold must be non-negative, got {diag_threshold}.")

    n0, n1 = P_sparse.shape
    four_block = build_four_block_matrix(P_sparse)
    prepared = np.zeros_like(four_block, dtype=float)

    diag00 = np.array(four_block[:n0, :n0], copy=True)
    diag11 = np.array(four_block[n0:, n0:], copy=True)
    if diag_threshold > 0.0:
        if diag00.size > 0:
            diag00_max = float(np.max(diag00))
            if diag00_max > 0.0:
                diag00 = np.where(diag00 >= diag_threshold * diag00_max, diag00, 0.0)
        if diag11.size > 0:
            diag11_max = float(np.max(diag11))
            if diag11_max > 0.0:
                diag11 = np.where(diag11 >= diag_threshold * diag11_max, diag11, 0.0)

    support = P_sparse > 0.0

    prepared[:n0, :n0] = np.where(diag00 > 0.0, diag00, 0.0)
    prepared[n0:, n0:] = np.where(diag11 > 0.0, diag11, 0.0)
    prepared[:n0, n0:] = np.where(support, four_block[:n0, n0:], 0.0)
    prepared[n0:, :n0] = np.where(support.T, four_block[n0:, :n0], 0.0)

    prepared = np.nan_to_num(prepared, nan=0.0, posinf=0.0, neginf=0.0)
    prepared[prepared < 0] = 0.0
    np.fill_diagonal(prepared, 0.0)
    return prepared


def postprocess_graph_matrix(
    matrix,
    binarize=False,
    symmetrize=True,
    normalize=True,
):
    graph = np.asarray(matrix, dtype=float)
    if graph.ndim != 2:
        raise ValueError(f"Expected a 2D 4-block matrix, got shape {graph.shape}.")

    graph = np.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0)
    graph[graph < 0] = 0.0

    if binarize:
        graph = (graph > 0.0).astype(float)

    if symmetrize:
        graph = np.maximum(graph, graph.T)

    np.fill_diagonal(graph, 0.0)

    if normalize:
        total_mass = float(graph.sum())
        if total_mass <= 0.0:
            raise ValueError("Graph has zero total mass after postprocessing.")
        graph = graph / total_mass

    return graph


def postprocess_sparse_plan_four_block_graph(
    matrix,
    binarize=False,
    symmetrize=True,
    normalize=True,
):
    return postprocess_graph_matrix(
        matrix,
        binarize=binarize,
        symmetrize=symmetrize,
        normalize=normalize,
    )


def build_sparse_plan_four_block_graph(
    P_sparse,
    diag_threshold=1e-3,
    binarize=False,
    symmetrize=True,
    normalize=True,
):
    prepared = prepare_sparse_plan_four_block_matrix(
        P_sparse,
        diag_threshold=diag_threshold,
    )
    return postprocess_graph_matrix(
        prepared,
        binarize=binarize,
        symmetrize=symmetrize,
        normalize=normalize,
    )


def run_sparse_plan_top_pct_neighbors_benchmark(
    data_bundle,
    plans,
    pct_values,
    tsne_seeds,
    viz_seed,
    n_epochs,
    k_conc=50,
    k_cross=50,
    diag_threshold=1e-3,
    renormalize_sparse_plan=False,
    binarize_graph=False,
    symmetrize_graph=True,
    normalize_graph=True,
    cache_path=None,
    verbose=True,
):
    pct_values = [float(pct) for pct in pct_values]
    tsne_seeds = [int(seed) for seed in tsne_seeds]
    if int(viz_seed) not in tsne_seeds:
        raise ValueError(f"viz_seed={viz_seed} must be included in tsne_seeds={tsne_seeds}")

    results = {}
    for pct in pct_values:
        if verbose:
            graph_label = "binarized" if binarize_graph else "weighted"
            print(
                f"[Sparse-plan top % neighbors ({graph_label}) | eps_times={plans['eps_times']:g}] "
                f"calculating pct={pct:g}% ..."
            )

        dataset_prepared = {}
        for dataset_name in ("ring", "gmm"):
            P_sparse, sparse_plan_stats = sparsify_plan_percentage_neighbors(
                plans[dataset_name]["P"],
                percentage=float(pct) / 100.0,
                ensure_nonempty=True,
                renormalize=renormalize_sparse_plan,
                return_stats=True,
            )
            graph = build_sparse_plan_four_block_graph(
                P_sparse,
                diag_threshold=diag_threshold,
                binarize=binarize_graph,
                symmetrize=symmetrize_graph,
                normalize=normalize_graph,
            )
            dataset_prepared[dataset_name] = {
                "sparse_plan": np.asarray(P_sparse),
                "sparse_plan_stats": sparse_plan_stats,
                "graph": np.asarray(graph),
                "graph_nnz": int(np.count_nonzero(graph)),
                "graph_density": float(np.count_nonzero(graph) / graph.size),
                "graph_sum": float(np.sum(graph)),
            }

        ring_metrics = {metric_name: [] for metric_name in METRIC_NAMES}
        gmm_metrics = {metric_name: [] for metric_name in METRIC_NAMES}
        ring_embeddings_by_seed = {}
        gmm_embeddings_by_seed = {}
        ring_metrics_by_seed = {}
        gmm_metrics_by_seed = {}
        ring_viz_emb = None
        gmm_viz_emb = None
        ring_viz_metrics = None
        gmm_viz_metrics = None

        for seed in tsne_seeds:
            ring_embd = run_tsne_from_graph(
                dataset_prepared["ring"]["graph"],
                seed=seed,
                n_epochs=n_epochs,
            )
            gmm_embd = run_tsne_from_graph(
                dataset_prepared["gmm"]["graph"],
                seed=seed,
                n_epochs=n_epochs,
            )

            ring_metric_row = metric_triplet(
                data_bundle["ring"]["clean0"],
                data_bundle["ring"]["clean1"],
                ring_embd,
                k_conc=k_conc,
                k_cross=k_cross,
            )
            gmm_metric_row = metric_triplet(
                data_bundle["gmm"]["clean0"],
                data_bundle["gmm"]["clean1"],
                gmm_embd,
                k_conc=k_conc,
                k_cross=k_cross,
            )

            ring_embeddings_by_seed[int(seed)] = np.asarray(ring_embd)
            gmm_embeddings_by_seed[int(seed)] = np.asarray(gmm_embd)
            ring_metrics_by_seed[int(seed)] = ring_metric_row
            gmm_metrics_by_seed[int(seed)] = gmm_metric_row

            for metric_name in METRIC_NAMES:
                ring_metrics[metric_name].append(ring_metric_row[metric_name])
                gmm_metrics[metric_name].append(gmm_metric_row[metric_name])

            if int(seed) == int(viz_seed):
                ring_viz_emb = np.asarray(ring_embd)
                gmm_viz_emb = np.asarray(gmm_embd)
                ring_viz_metrics = ring_metric_row
                gmm_viz_metrics = gmm_metric_row

        ring_summary = summarize_metric_lists(ring_metrics)
        gmm_summary = summarize_metric_lists(gmm_metrics)
        results[float(pct)] = {
            "ring_graph": dataset_prepared["ring"]["graph"],
            "gmm_graph": dataset_prepared["gmm"]["graph"],
            "ring_sparse_plan": dataset_prepared["ring"]["sparse_plan"],
            "gmm_sparse_plan": dataset_prepared["gmm"]["sparse_plan"],
            "ring_sparse_plan_stats": dataset_prepared["ring"]["sparse_plan_stats"],
            "gmm_sparse_plan_stats": dataset_prepared["gmm"]["sparse_plan_stats"],
            "ring_graph_stats": {
                "nnz": dataset_prepared["ring"]["graph_nnz"],
                "density": dataset_prepared["ring"]["graph_density"],
                "sum": dataset_prepared["ring"]["graph_sum"],
            },
            "gmm_graph_stats": {
                "nnz": dataset_prepared["gmm"]["graph_nnz"],
                "density": dataset_prepared["gmm"]["graph_density"],
                "sum": dataset_prepared["gmm"]["graph_sum"],
            },
            "ring_viz_emb": ring_viz_emb,
            "gmm_viz_emb": gmm_viz_emb,
            "ring_viz_metrics": ring_viz_metrics,
            "gmm_viz_metrics": gmm_viz_metrics,
            "ring_embeddings_by_seed": ring_embeddings_by_seed,
            "gmm_embeddings_by_seed": gmm_embeddings_by_seed,
            "ring_metrics_by_seed": ring_metrics_by_seed,
            "gmm_metrics_by_seed": gmm_metrics_by_seed,
            "ring_mean": ring_summary["mean"],
            "ring_sd": ring_summary["sd"],
            "gmm_mean": gmm_summary["mean"],
            "gmm_sd": gmm_summary["sd"],
        }

    benchmark = {
        "eps_times": float(plans["eps_times"]),
        "pct_values": pct_values,
        "tsne_seeds": tsne_seeds,
        "viz_seed": int(viz_seed),
        "n_epochs": int(n_epochs),
        "k_conc": int(k_conc),
        "k_cross": int(k_cross),
        "diag_threshold": float(diag_threshold),
        "renormalize_sparse_plan": bool(renormalize_sparse_plan),
        "graph_binarized": bool(binarize_graph),
        "graph_symmetrized": bool(symmetrize_graph),
        "graph_normalized": bool(normalize_graph),
        "results": results,
        "cache_path": None if cache_path is None else str(cache_path),
    }

    if cache_path is not None:
        save_benchmark_cache(Path(cache_path), benchmark, data_bundle)

    return benchmark


def plot_ot_plan_heatmaps(plans, eps_times=None, figsize=(14, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, dataset_name, title in (
        (axes[0], "ring", "Ring OT plan"),
        (axes[1], "gmm", "GMM OT plan"),
    ):
        plan = np.array(plans[dataset_name]["P"], copy=True)
        plan[plan <= 0] = np.nan
        image = ax.imshow(
            plan,
            cmap="magma",
            interpolation="nearest",
            aspect="auto",
            norm=plans[dataset_name]["lognorm"],
        )
        ax.set_title(title)
        ax.set_xlabel("batch 1 index")
        ax.set_ylabel("batch 0 index")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if eps_times is None:
        eps_times = plans["eps_times"]
    fig.suptitle(f"OT plans for ring and GMM datasets (eps_times={eps_times:g})")
    fig.tight_layout()
    return fig, axes


def plot_pct_embedding_grid(benchmark_results, data_bundle, title=None, max_per_row=8):
    pct_values = list(benchmark_results["pct_values"])
    n_panels = len(pct_values)
    n_blocks = int(np.ceil(n_panels / max_per_row))
    nrows = 2 * n_blocks
    ncols = max_per_row

    ring_colors = np.array(["tab:red"] * data_bundle["ring"]["n0"] + ["tab:blue"] * data_bundle["ring"]["n1"])
    gmm_colors = np.array(["tab:red"] * data_bundle["gmm"]["n0"] + ["tab:blue"] * data_bundle["gmm"]["n1"])

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 3.1 * nrows), squeeze=False)
    for block_idx in range(n_blocks):
        for col_idx in range(ncols):
            panel_idx = block_idx * max_per_row + col_idx
            ax_ring = axes[2 * block_idx, col_idx]
            ax_gmm = axes[2 * block_idx + 1, col_idx]
            if panel_idx >= n_panels:
                ax_ring.axis("off")
                ax_gmm.axis("off")
                continue

            pct = float(pct_values[panel_idx])
            row = benchmark_results["results"][pct]

            ring_perm = np.random.RandomState(10000 + panel_idx).permutation(len(ring_colors))
            gmm_perm = np.random.RandomState(20000 + panel_idx).permutation(len(gmm_colors))

            ax_ring.scatter(*row["ring_viz_emb"][ring_perm].T, s=1, c=ring_colors[ring_perm])
            ax_ring.set_aspect("equal")
            ax_ring.axis("off")
            ax_ring.set_title(f"{pct:g}%")
            ax_ring.text(
                0.02,
                0.98,
                (
                    f"seed {benchmark_results['viz_seed']} conc: "
                    f"{row['ring_viz_metrics']['concordance']:.3f}\n"
                    f"seed {benchmark_results['viz_seed']} x-conc: "
                    f"{row['ring_viz_metrics']['cross_concordance']:.3f}\n"
                    f"seed {benchmark_results['viz_seed']} foscttm: "
                    f"{row['ring_viz_metrics']['foscttm']:.3f}"
                ),
                transform=ax_ring.transAxes,
                va="top",
                ha="left",
                fontsize=7.5,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "none",
                },
            )

            ax_gmm.scatter(*row["gmm_viz_emb"][gmm_perm].T, s=1, c=gmm_colors[gmm_perm])
            ax_gmm.set_aspect("equal")
            ax_gmm.axis("off")
            ax_gmm.text(
                0.02,
                0.98,
                (
                    f"seed {benchmark_results['viz_seed']} conc: "
                    f"{row['gmm_viz_metrics']['concordance']:.3f}\n"
                    f"seed {benchmark_results['viz_seed']} x-conc: "
                    f"{row['gmm_viz_metrics']['cross_concordance']:.3f}\n"
                    f"seed {benchmark_results['viz_seed']} foscttm: "
                    f"{row['gmm_viz_metrics']['foscttm']:.3f}"
                ),
                transform=ax_gmm.transAxes,
                va="top",
                ha="left",
                fontsize=7.5,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "none",
                },
            )

            if col_idx == 0:
                ax_ring.set_ylabel("Ring", fontsize=11)
                ax_gmm.set_ylabel("GMM", fontsize=11)

    if title is None:
        title = (
            "Top % Neighbors embeddings "
            f"(eps_times={benchmark_results['eps_times']:g}, viz seed={benchmark_results['viz_seed']})"
        )
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    return fig, axes


def plot_gmm_pct_embedding_grid_by_cluster(
    benchmark_results,
    data_bundle,
    title=None,
    max_per_row=8,
):
    pct_values = list(benchmark_results["pct_values"])
    n_panels = len(pct_values)
    ncols = min(max_per_row, n_panels)
    n_blocks = int(np.ceil(n_panels / ncols))

    cluster_labels = np.concatenate(
        [
            np.asarray(data_bundle["gmm"]["clusters0"]),
            np.asarray(data_bundle["gmm"]["clusters1"]),
        ]
    )
    all_clusters = np.unique(cluster_labels)
    cmap = plt.get_cmap("tab10", len(all_clusters))
    cluster_colors = {
        int(cluster): cmap(cluster_idx) for cluster_idx, cluster in enumerate(all_clusters)
    }

    fig, axes = plt.subplots(
        n_blocks,
        ncols,
        figsize=(2.8 * ncols, 3.1 * n_blocks),
        squeeze=False,
    )
    for block_idx in range(n_blocks):
        for col_idx in range(ncols):
            panel_idx = block_idx * ncols + col_idx
            ax = axes[block_idx, col_idx]
            if panel_idx >= n_panels:
                ax.axis("off")
                continue

            pct = float(pct_values[panel_idx])
            row = benchmark_results["results"][pct]
            gmm_perm = np.random.RandomState(20000 + panel_idx).permutation(cluster_labels.shape[0])
            gmm_emb = np.asarray(row["gmm_viz_emb"])[gmm_perm]
            gmm_clusters = cluster_labels[gmm_perm]

            for cluster in all_clusters:
                cluster_mask = gmm_clusters == cluster
                ax.scatter(
                    *gmm_emb[cluster_mask].T,
                    s=1,
                    c=[cluster_colors[int(cluster)]],
                    alpha=0.9,
                )

            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{pct:g}%")
            ax.text(
                0.02,
                0.98,
                (
                    f"seed {benchmark_results['viz_seed']} conc: "
                    f"{row['gmm_viz_metrics']['concordance']:.3f}\n"
                    f"seed {benchmark_results['viz_seed']} x-conc: "
                    f"{row['gmm_viz_metrics']['cross_concordance']:.3f}\n"
                    f"seed {benchmark_results['viz_seed']} foscttm: "
                    f"{row['gmm_viz_metrics']['foscttm']:.3f}"
                ),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7.5,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "none",
                },
            )

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=cluster_colors[int(cluster)],
            markersize=5,
            label=f"cluster {int(cluster)}",
        )
        for cluster in all_clusters
    ]
    if title is None:
        title = (
            "Top % Neighbors GMM embeddings "
            f"(eps_times={benchmark_results['eps_times']:g}, "
            f"viz seed={benchmark_results['viz_seed']}, colored by cluster)"
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(all_clusters),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0.06, 1, 0.965])
    return fig, axes


def plot_k50_concordance_summary(
    benchmark_results,
    eot_reference,
    title=None,
    x_label="Top % Neighbors",
):
    pct_values = np.array(sorted(benchmark_results["pct_values"]), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)

    for ax, dataset_name, panel_title in (
        (axes[0], "ring", "Ring"),
        (axes[1], "gmm", "GMM"),
    ):
        for metric_name in SUMMARY_METRICS:
            means = np.array(
                [
                    benchmark_results["results"][float(pct)][f"{dataset_name}_mean"][metric_name]
                    for pct in pct_values
                ],
                dtype=float,
            )
            sds = np.array(
                [
                    benchmark_results["results"][float(pct)][f"{dataset_name}_sd"][metric_name]
                    for pct in pct_values
                ],
                dtype=float,
            )
            color = PLOT_COLORS[metric_name]
            ax.plot(
                pct_values,
                means,
                marker="o",
                lw=1.8,
                color=color,
                label=PLOT_LABELS[metric_name],
            )
            ax.fill_between(pct_values, means - sds, means + sds, color=color, alpha=0.18)
            ax.axhline(
                eot_reference[dataset_name]["metrics"][metric_name],
                color=color,
                linestyle="--",
                linewidth=2.0,
                alpha=0.95,
                label=f"EOT t={eot_reference['t']} {PLOT_LABELS[metric_name]}",
            )

        ax.set_title(panel_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Metric value")
        ax.grid(True, alpha=0.3)
        if pct_values.shape[0] > 1:
            ax.set_xlim(pct_values.min(), pct_values.max())

    if title is None:
        title = (
            "k=50 concordance summary "
            f"(eps_times={benchmark_results['eps_times']:g}, seeds={benchmark_results['tsne_seeds']})"
        )
    axes[0].legend(loc="best")
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    return fig, axes


def compute_gmm_rand_summary(benchmark_results, gt_clusters, cluster_seed=0):
    gt_clusters = np.asarray(gt_clusters)
    if gt_clusters.ndim != 1:
        raise ValueError(f"gt_clusters must be a 1D array, got shape {gt_clusters.shape}.")

    n_clusters = int(np.unique(gt_clusters).size)
    pct_values = np.array(sorted(float(pct) for pct in benchmark_results["pct_values"]), dtype=float)
    means = np.empty(pct_values.size, dtype=float)
    sds = np.empty(pct_values.size, dtype=float)
    rand_by_pct = {}

    for idx, pct in enumerate(pct_values):
        embeddings_by_seed = benchmark_results["results"][float(pct)]["gmm_embeddings_by_seed"]
        rand_by_seed = {}
        rand_values = []
        for seed in sorted(int(seed) for seed in embeddings_by_seed):
            rand_seed = int(seed) if cluster_seed is None else int(cluster_seed)
            rand_value = compute_kmeans_rand(
                embeddings_by_seed[int(seed)],
                gt_clusters,
                n_clusters=n_clusters,
                seed=rand_seed,
            )
            rand_by_seed[int(seed)] = float(rand_value)
            rand_values.append(float(rand_value))

        rand_values = np.asarray(rand_values, dtype=float)
        means[idx] = float(np.mean(rand_values))
        sds[idx] = float(np.std(rand_values, ddof=1)) if rand_values.size > 1 else 0.0
        rand_by_pct[float(pct)] = {
            "by_seed": rand_by_seed,
            "mean": float(means[idx]),
            "sd": float(sds[idx]),
        }

    return {
        "eps_times": float(benchmark_results["eps_times"]),
        "pct_values": pct_values,
        "cluster_seed": None if cluster_seed is None else int(cluster_seed),
        "n_clusters": n_clusters,
        "by_pct": rand_by_pct,
        "mean": means,
        "sd": sds,
    }


def compute_eot_gmm_rand(eot_reference, gt_clusters, cluster_seed=0):
    gt_clusters = np.asarray(gt_clusters)
    n_clusters = int(np.unique(gt_clusters).size)
    rand_value = compute_kmeans_rand(
        eot_reference["gmm"]["embedding"],
        gt_clusters,
        n_clusters=n_clusters,
        seed=0 if cluster_seed is None else int(cluster_seed),
    )
    return float(rand_value)


def plot_gmm_rand_summary(
    benchmark_results,
    eot_reference,
    gt_clusters,
    cluster_seed=0,
    title=None,
    x_label="Top % Neighbors",
):
    rand_summary = compute_gmm_rand_summary(
        benchmark_results,
        gt_clusters=gt_clusters,
        cluster_seed=cluster_seed,
    )
    eot_rand = compute_eot_gmm_rand(
        eot_reference,
        gt_clusters=gt_clusters,
        cluster_seed=cluster_seed,
    )

    pct_values = rand_summary["pct_values"]
    means = rand_summary["mean"]
    sds = rand_summary["sd"]

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    color = PLOT_COLORS["rand"]
    ax.plot(
        pct_values,
        means,
        marker="o",
        lw=1.9,
        color=color,
        label=PLOT_LABELS["rand"],
    )
    ax.fill_between(pct_values, means - sds, means + sds, color=color, alpha=0.18)
    ax.axhline(
        eot_rand,
        color=color,
        linestyle="--",
        linewidth=2.0,
        alpha=0.95,
        label=f"EOT t={eot_reference['t']} {PLOT_LABELS['rand']}",
    )
    ax.set_title("GMM")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Rand index")
    ax.grid(True, alpha=0.3)
    if pct_values.size > 1:
        ax.set_xlim(pct_values.min(), pct_values.max())
    ax.set_ylim(0.0, 1.02)

    if title is None:
        title = (
            "GMM Rand index summary "
            f"(eps_times={benchmark_results['eps_times']:g}, seeds={benchmark_results['tsne_seeds']})"
        )

    ax.legend(loc="best")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, ax, rand_summary, eot_rand


def build_metric_heatmap_data(
    benchmarks_by_eps,
    eot_references_by_eps,
    metric_name,
    dataset_names=("ring", "gmm"),
):
    if metric_name not in METRIC_NAMES:
        raise ValueError(f"Unknown metric_name={metric_name!r}; expected one of {METRIC_NAMES}.")

    eps_times = np.array(sorted(float(eps) for eps in benchmarks_by_eps), dtype=float)
    if eps_times.size == 0:
        raise ValueError("benchmarks_by_eps must contain at least one epsilon entry.")

    first_benchmark = benchmarks_by_eps[float(eps_times[0])]
    pct_values = np.asarray(first_benchmark["pct_values"], dtype=float)

    eot_t = None
    datasets = {}
    for dataset_name in dataset_names:
        metric_matrix = np.empty((eps_times.size, pct_values.size), dtype=float)
        eot_column = np.empty(eps_times.size, dtype=float)

        for row_idx, eps in enumerate(eps_times):
            benchmark = benchmarks_by_eps[float(eps)]
            eot_reference = eot_references_by_eps[float(eps)]
            benchmark_pct_values = np.asarray(benchmark["pct_values"], dtype=float)
            if benchmark_pct_values.shape != pct_values.shape or not np.allclose(
                benchmark_pct_values, pct_values
            ):
                raise ValueError("All benchmark entries must share the same pct_values ordering.")

            metric_matrix[row_idx] = np.asarray(
                [
                    benchmark["results"][float(pct)][f"{dataset_name}_mean"][metric_name]
                    for pct in pct_values
                ],
                dtype=float,
            )
            eot_column[row_idx] = float(eot_reference[dataset_name]["metrics"][metric_name])

            if eot_t is None:
                eot_t = int(eot_reference["t"])
            elif int(eot_reference["t"]) != eot_t:
                raise ValueError("All EOT reference entries must use the same t value.")

        full_matrix = np.full((eps_times.size, pct_values.size + 2), np.nan, dtype=float)
        full_matrix[:, : pct_values.size] = metric_matrix
        full_matrix[:, -1] = eot_column

        datasets[dataset_name] = {
            "top_pct_matrix": metric_matrix,
            "eot_column": eot_column,
            "full_matrix": full_matrix,
        }

    x_labels = [f"{pct:g}%" for pct in pct_values] + ["", f"EOT {eot_t:g}"]
    y_labels = [f"{eps:g}" for eps in eps_times]

    return {
        "metric_name": metric_name,
        "metric_label": PLOT_LABELS[metric_name],
        "eps_times": eps_times,
        "pct_values": pct_values,
        "eot_t": int(eot_t),
        "dataset_names": tuple(dataset_names),
        "datasets": datasets,
        "x_labels": x_labels,
        "y_labels": y_labels,
        "top_pct_column_count": int(pct_values.size),
    }


def compute_shared_dataset_scale_bounds(heatmap_data_items):
    if isinstance(heatmap_data_items, dict):
        if "datasets" in heatmap_data_items:
            items = [heatmap_data_items]
        else:
            items = list(heatmap_data_items.values())
    else:
        items = list(heatmap_data_items)

    if not items:
        raise ValueError("heatmap_data_items must contain at least one heatmap data entry.")

    bounds_by_dataset = {}
    for heatmap_data in items:
        for dataset_name, panel in heatmap_data["datasets"].items():
            panel_values = np.concatenate([panel["top_pct_matrix"].ravel(), panel["eot_column"]])
            panel_vmin = float(np.min(panel_values))
            panel_vmax = float(np.max(panel_values))

            if dataset_name in bounds_by_dataset:
                prev_vmin, prev_vmax = bounds_by_dataset[dataset_name]
                bounds_by_dataset[dataset_name] = (
                    min(prev_vmin, panel_vmin),
                    max(prev_vmax, panel_vmax),
                )
            else:
                bounds_by_dataset[dataset_name] = (panel_vmin, panel_vmax)

    return bounds_by_dataset


def plot_metric_heatmap_grid(
    heatmap_data,
    title=None,
    cmap="viridis",
    save_path=None,
    figsize=None,
    scale_mode="dataset",
    annotate_eot=True,
    scale_bounds_by_dataset=None,
    annotate_cells=False,
):
    dataset_names = tuple(heatmap_data["dataset_names"])
    if figsize is None:
        figsize = (
            1.1 * len(dataset_names) * max(5.0, 0.6 * heatmap_data["top_pct_column_count"]) + 2.2,
            1.25 * len(heatmap_data["eps_times"]) + 2.8,
        )

    valid_scale_modes = {"dataset", "global"}
    if scale_mode not in valid_scale_modes:
        raise ValueError(f"scale_mode must be one of {valid_scale_modes}, got {scale_mode!r}.")

    cmap_obj = plt.get_cmap(cmap)
    global_values = np.concatenate(
        [
            np.concatenate(
                [
                    panel["top_pct_matrix"].ravel(),
                    panel["eot_column"].ravel(),
                ]
            )
            for panel in heatmap_data["datasets"].values()
        ]
    )
    global_vmin = float(np.min(global_values))
    global_vmax = float(np.max(global_values))

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    outer_grid = fig.add_gridspec(1, len(dataset_names), wspace=0.22)
    axes = {}

    for col_idx, dataset_name in enumerate(dataset_names):
        panel = heatmap_data["datasets"][dataset_name]
        panel_grid = outer_grid[0, col_idx].subgridspec(
            1,
            3,
            width_ratios=[heatmap_data["top_pct_column_count"], 1.15, 0.9],
            wspace=0.08,
        )
        ax_main = fig.add_subplot(panel_grid[0, 0])
        ax_eot = fig.add_subplot(panel_grid[0, 1], sharey=ax_main)
        cax = fig.add_subplot(panel_grid[0, 2])

        if scale_bounds_by_dataset is not None and dataset_name in scale_bounds_by_dataset:
            vmin, vmax = scale_bounds_by_dataset[dataset_name]
            vmin = float(vmin)
            vmax = float(vmax)
        elif scale_mode == "global":
            vmin = global_vmin
            vmax = global_vmax
        else:
            panel_values = np.concatenate([panel["top_pct_matrix"].ravel(), panel["eot_column"]])
            vmin = float(np.min(panel_values))
            vmax = float(np.max(panel_values))

        image = ax_main.imshow(
            panel["top_pct_matrix"],
            aspect="auto",
            interpolation="nearest",
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
        )
        eot_image = ax_eot.imshow(
            panel["eot_column"][:, None],
            aspect="auto",
            interpolation="nearest",
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
        )

        ax_main.set_title(dataset_name.upper(), fontsize=13, fontweight="bold", pad=8)
        ax_main.set_xlabel("Top neighbors retained")
        ax_main.set_xticks(np.arange(heatmap_data["top_pct_column_count"]))
        ax_main.set_xticklabels(
            [f"{pct:g}%" for pct in heatmap_data["pct_values"]],
            rotation=45,
            ha="right",
        )
        ax_main.set_yticks(np.arange(len(heatmap_data["y_labels"])))
        ax_main.set_yticklabels(heatmap_data["y_labels"])

        ax_eot.set_xlabel("Reference")
        ax_eot.set_xticks([0])
        ax_eot.set_xticklabels([f"EOT {heatmap_data['eot_t']:g}"], rotation=45, ha="right")
        ax_eot.tick_params(axis="y", left=False, labelleft=False)

        # Notebook sessions often enable global axes grids; turn them off so
        # each heatmap value renders as a single uninterrupted cell.
        for axis in (ax_main, ax_eot):
            axis.grid(False, which="both")
            axis.minorticks_off()

        value_span = max(vmax - vmin, 1e-12)
        if annotate_cells:
            for row_idx in range(panel["top_pct_matrix"].shape[0]):
                for col_idx in range(panel["top_pct_matrix"].shape[1]):
                    value = panel["top_pct_matrix"][row_idx, col_idx]
                    if not np.isfinite(value):
                        continue
                    text_color = "white" if (value - vmin) / value_span < 0.55 else "#1f1f1f"
                    ax_main.text(
                        col_idx,
                        row_idx,
                        f"{value:.2g}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=text_color,
                        fontweight="semibold",
                    )

        if annotate_eot:
            for row_idx, value in enumerate(panel["eot_column"]):
                text_color = "white" if (value - vmin) / value_span < 0.55 else "#1f1f1f"
                ax_eot.text(
                    0,
                    row_idx,
                    f"{value:.2g}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=text_color,
                    fontweight="semibold",
                )

        colorbar = fig.colorbar(eot_image, cax=cax)
        colorbar.set_label("Metric value")
        colorbar.ax.grid(False, which="both")
        colorbar.ax.minorticks_off()

        axes[dataset_name] = {
            "main": ax_main,
            "eot": ax_eot,
            "colorbar": cax,
        }

    axes[dataset_names[0]]["main"].set_ylabel("eps_times")

    if title is None:
        title = (
            f"{heatmap_data['metric_label']} across eps_times and top-neighbor percentages "
            f"(EOT t={heatmap_data['eot_t']})"
        )

    fig.suptitle(title, fontsize=16, y=1.03)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig, axes


def make_eps_cache_path(cache_dir, eps_times):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"top_pct_neighbors_seed_sweep_eps_times_{format_eps_token(eps_times)}.npz"


def make_eps_eot_cache_path(cache_dir, eps_times):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"eot_reference_eps_times_{format_eps_token(eps_times)}.npz"


def format_eps_token(eps_times):
    return f"{float(eps_times):g}".replace("-", "m").replace(".", "p")


def build_eot_embedding_from_plan(P, n0, n1, embd_dim=2, t=1):
    U, singular_values, Vt = np.linalg.svd(np.asarray(P, dtype=float), full_matrices=False)
    left = U[:, 1 : 1 + embd_dim] * (singular_values[1 : 1 + embd_dim] ** t) * np.sqrt(n0)
    right = (
        Vt.T[:, 1 : 1 + embd_dim]
        * (singular_values[1 : 1 + embd_dim] ** t)
        * np.sqrt(n1)
    )
    return np.concatenate([left, right], axis=0)


def build_four_block_matrix(P):
    n0, n1 = P.shape
    U, singular_values, Vt = np.linalg.svd(np.asarray(P, dtype=float), full_matrices=False)
    eot = np.concatenate([U * np.sqrt(n0), Vt.T * np.sqrt(n1)], axis=0)
    matrix = eot @ np.diag(singular_values) @ eot.T / np.sqrt(n0 * n1)
    matrix[matrix < 0] = 0.0
    np.fill_diagonal(matrix, 0.0)
    return matrix


def metric_triplet(clean0, clean1, embd, k_conc=50, k_cross=50):
    clean0 = np.asarray(clean0)
    clean1 = np.asarray(clean1)
    embd = np.asarray(embd)
    n0 = clean0.shape[0]
    n1 = clean1.shape[0]
    clean = np.concatenate([clean0, clean1], axis=0)
    embd0 = embd[:n0]
    embd1 = embd[n0 : n0 + n1]
    return {
        "concordance": float(compute_concordance(clean, embd, k=min(int(k_conc), clean.shape[0]))),
        "cross_concordance": float(
            compute_cross_concordance(
                clean0,
                clean1,
                embd0,
                embd1,
                k=min(int(k_cross), n0, n1),
            )
        ),
        "foscttm": float(compute_foscttm(clean0, clean1, embd0, embd1)),
    }


def summarize_metric_lists(metric_lists):
    summary = {"mean": {}, "sd": {}}
    for metric_name in METRIC_NAMES:
        values = np.asarray(metric_lists[metric_name], dtype=float)
        summary["mean"][metric_name] = float(np.mean(values))
        if values.shape[0] > 1:
            summary["sd"][metric_name] = float(np.std(values, ddof=1))
        else:
            summary["sd"][metric_name] = 0.0
    return summary


def metric_dict_to_row(metric_dict):
    return np.asarray([float(metric_dict[metric_name]) for metric_name in METRIC_NAMES], dtype=float)


def metric_row_to_dict(metric_row, metric_names=METRIC_NAMES):
    metric_names = tuple(str(metric_name) for metric_name in metric_names)
    metric_row = np.asarray(metric_row, dtype=float)
    if metric_row.shape[-1] != len(metric_names):
        raise ValueError(
            f"Expected {len(metric_names)} metric values, got shape {metric_row.shape!r}."
        )
    return {
        metric_name: float(metric_row[idx]) for idx, metric_name in enumerate(metric_names)
    }


def save_benchmark_cache(cache_path, benchmark_results, data_bundle):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    pct_values = np.asarray(benchmark_results["pct_values"], dtype=float)
    tsne_seeds = np.asarray(benchmark_results["tsne_seeds"], dtype=int)

    ring_embeddings = np.stack(
        [
            np.stack(
                [
                    benchmark_results["results"][float(pct)]["ring_embeddings_by_seed"][int(seed)]
                    for seed in tsne_seeds
                ],
                axis=0,
            )
            for pct in pct_values
        ],
        axis=0,
    )
    gmm_embeddings = np.stack(
        [
            np.stack(
                [
                    benchmark_results["results"][float(pct)]["gmm_embeddings_by_seed"][int(seed)]
                    for seed in tsne_seeds
                ],
                axis=0,
            )
            for pct in pct_values
        ],
        axis=0,
    )
    ring_metrics_by_seed = np.stack(
        [
            np.stack(
                [
                    metric_dict_to_row(
                        benchmark_results["results"][float(pct)]["ring_metrics_by_seed"][int(seed)]
                    )
                    for seed in tsne_seeds
                ],
                axis=0,
            )
            for pct in pct_values
        ],
        axis=0,
    )
    gmm_metrics_by_seed = np.stack(
        [
            np.stack(
                [
                    metric_dict_to_row(
                        benchmark_results["results"][float(pct)]["gmm_metrics_by_seed"][int(seed)]
                    )
                    for seed in tsne_seeds
                ],
                axis=0,
            )
            for pct in pct_values
        ],
        axis=0,
    )
    ring_metric_mean = np.stack(
        [
            metric_dict_to_row(benchmark_results["results"][float(pct)]["ring_mean"])
            for pct in pct_values
        ],
        axis=0,
    )
    gmm_metric_mean = np.stack(
        [
            metric_dict_to_row(benchmark_results["results"][float(pct)]["gmm_mean"])
            for pct in pct_values
        ],
        axis=0,
    )
    ring_metric_sd = np.stack(
        [
            metric_dict_to_row(benchmark_results["results"][float(pct)]["ring_sd"])
            for pct in pct_values
        ],
        axis=0,
    )
    gmm_metric_sd = np.stack(
        [
            metric_dict_to_row(benchmark_results["results"][float(pct)]["gmm_sd"])
            for pct in pct_values
        ],
        axis=0,
    )

    np.savez_compressed(
        cache_path,
        eps_times=np.asarray(benchmark_results["eps_times"], dtype=float),
        pct_values=pct_values,
        tsne_seeds=tsne_seeds,
        metric_names=np.asarray(METRIC_NAMES),
        ring_embeddings=ring_embeddings,
        gmm_embeddings=gmm_embeddings,
        ring_metrics_by_seed=ring_metrics_by_seed,
        gmm_metrics_by_seed=gmm_metrics_by_seed,
        ring_metric_mean=ring_metric_mean,
        gmm_metric_mean=gmm_metric_mean,
        ring_metric_sd=ring_metric_sd,
        gmm_metric_sd=gmm_metric_sd,
        ring_clean0=np.asarray(data_bundle["ring"]["clean0"]),
        ring_clean1=np.asarray(data_bundle["ring"]["clean1"]),
        gmm_clean0=np.asarray(data_bundle["gmm"]["clean0"]),
        gmm_clean1=np.asarray(data_bundle["gmm"]["clean1"]),
        viz_seed=np.asarray(benchmark_results["viz_seed"], dtype=int),
        n_epochs=np.asarray(benchmark_results["n_epochs"], dtype=int),
        k_conc=np.asarray(benchmark_results["k_conc"], dtype=int),
        k_cross=np.asarray(benchmark_results["k_cross"], dtype=int),
        data_seed=np.asarray(data_bundle["seed"], dtype=int),
    )


def load_benchmark_cache(cache_path):
    cache_path = Path(cache_path)
    with np.load(cache_path, allow_pickle=False) as cache:
        pct_values = np.asarray(cache["pct_values"], dtype=float)
        tsne_seeds = np.asarray(cache["tsne_seeds"], dtype=int)
        metric_names = tuple(str(metric_name) for metric_name in cache["metric_names"].tolist())
        ring_embeddings = np.asarray(cache["ring_embeddings"], dtype=float)
        gmm_embeddings = np.asarray(cache["gmm_embeddings"], dtype=float)
        ring_metrics_by_seed = np.asarray(cache["ring_metrics_by_seed"], dtype=float)
        gmm_metrics_by_seed = np.asarray(cache["gmm_metrics_by_seed"], dtype=float)
        ring_metric_mean = np.asarray(cache["ring_metric_mean"], dtype=float)
        gmm_metric_mean = np.asarray(cache["gmm_metric_mean"], dtype=float)
        ring_metric_sd = np.asarray(cache["ring_metric_sd"], dtype=float)
        gmm_metric_sd = np.asarray(cache["gmm_metric_sd"], dtype=float)
        eps_times = float(np.asarray(cache["eps_times"]).item())
        viz_seed = int(np.asarray(cache["viz_seed"]).item())
        n_epochs = int(np.asarray(cache["n_epochs"]).item())
        k_conc = int(np.asarray(cache["k_conc"]).item())
        k_cross = int(np.asarray(cache["k_cross"]).item())
        data_seed = int(np.asarray(cache["data_seed"]).item())

    viz_matches = np.where(tsne_seeds == viz_seed)[0]
    if viz_matches.size == 0:
        raise ValueError(f"viz_seed={viz_seed} is not present in tsne_seeds={tsne_seeds.tolist()}.")
    viz_idx = int(viz_matches[0])

    results = {}
    for pct_idx, pct in enumerate(pct_values.tolist()):
        ring_embeddings_by_seed_map = {
            int(seed): np.asarray(ring_embeddings[pct_idx, seed_idx], dtype=float)
            for seed_idx, seed in enumerate(tsne_seeds.tolist())
        }
        gmm_embeddings_by_seed_map = {
            int(seed): np.asarray(gmm_embeddings[pct_idx, seed_idx], dtype=float)
            for seed_idx, seed in enumerate(tsne_seeds.tolist())
        }
        ring_metrics_by_seed_map = {
            int(seed): metric_row_to_dict(
                ring_metrics_by_seed[pct_idx, seed_idx],
                metric_names=metric_names,
            )
            for seed_idx, seed in enumerate(tsne_seeds.tolist())
        }
        gmm_metrics_by_seed_map = {
            int(seed): metric_row_to_dict(
                gmm_metrics_by_seed[pct_idx, seed_idx],
                metric_names=metric_names,
            )
            for seed_idx, seed in enumerate(tsne_seeds.tolist())
        }

        results[float(pct)] = {
            "ring_graph": None,
            "gmm_graph": None,
            "ring_viz_emb": np.asarray(ring_embeddings[pct_idx, viz_idx], dtype=float),
            "gmm_viz_emb": np.asarray(gmm_embeddings[pct_idx, viz_idx], dtype=float),
            "ring_viz_metrics": metric_row_to_dict(
                ring_metrics_by_seed[pct_idx, viz_idx],
                metric_names=metric_names,
            ),
            "gmm_viz_metrics": metric_row_to_dict(
                gmm_metrics_by_seed[pct_idx, viz_idx],
                metric_names=metric_names,
            ),
            "ring_embeddings_by_seed": ring_embeddings_by_seed_map,
            "gmm_embeddings_by_seed": gmm_embeddings_by_seed_map,
            "ring_metrics_by_seed": ring_metrics_by_seed_map,
            "gmm_metrics_by_seed": gmm_metrics_by_seed_map,
            "ring_mean": metric_row_to_dict(
                ring_metric_mean[pct_idx],
                metric_names=metric_names,
            ),
            "ring_sd": metric_row_to_dict(
                ring_metric_sd[pct_idx],
                metric_names=metric_names,
            ),
            "gmm_mean": metric_row_to_dict(
                gmm_metric_mean[pct_idx],
                metric_names=metric_names,
            ),
            "gmm_sd": metric_row_to_dict(
                gmm_metric_sd[pct_idx],
                metric_names=metric_names,
            ),
        }

    return {
        "eps_times": eps_times,
        "pct_values": pct_values.tolist(),
        "tsne_seeds": tsne_seeds.tolist(),
        "viz_seed": viz_seed,
        "n_epochs": n_epochs,
        "k_conc": k_conc,
        "k_cross": k_cross,
        "data_seed": data_seed,
        "results": results,
        "cache_path": str(cache_path),
    }


def load_eot_reference_cache(cache_path):
    cache_path = Path(cache_path)
    with np.load(cache_path, allow_pickle=False) as cache:
        metric_names = tuple(str(metric_name) for metric_name in cache["metric_names"].tolist())
        return {
            "eps_times": float(np.asarray(cache["eps_times"]).item()),
            "t": int(np.asarray(cache["t"]).item()),
            "embd_dim": int(np.asarray(cache["embd_dim"]).item()),
            "k_conc": int(np.asarray(cache["k_conc"]).item()),
            "k_cross": int(np.asarray(cache["k_cross"]).item()),
            "data_seed": int(np.asarray(cache["data_seed"]).item()),
            "ring": {
                "embedding": np.asarray(cache["ring_embedding"], dtype=float),
                "metrics": metric_row_to_dict(
                    cache["ring_metrics"],
                    metric_names=metric_names,
                ),
            },
            "gmm": {
                "embedding": np.asarray(cache["gmm_embedding"], dtype=float),
                "metrics": metric_row_to_dict(
                    cache["gmm_metrics"],
                    metric_names=metric_names,
                ),
            },
            "cache_path": str(cache_path),
        }


def save_eot_reference_cache(cache_path, eot_reference, plans, data_bundle):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        eps_times=np.asarray(plans["eps_times"], dtype=float),
        t=np.asarray(eot_reference["t"], dtype=int),
        embd_dim=np.asarray(eot_reference["embd_dim"], dtype=int),
        k_conc=np.asarray(eot_reference["k_conc"], dtype=int),
        k_cross=np.asarray(eot_reference["k_cross"], dtype=int),
        metric_names=np.asarray(METRIC_NAMES),
        ring_embedding=np.asarray(eot_reference["ring"]["embedding"]),
        gmm_embedding=np.asarray(eot_reference["gmm"]["embedding"]),
        ring_metrics=metric_dict_to_row(eot_reference["ring"]["metrics"]),
        gmm_metrics=metric_dict_to_row(eot_reference["gmm"]["metrics"]),
        ring_clean0=np.asarray(data_bundle["ring"]["clean0"]),
        ring_clean1=np.asarray(data_bundle["ring"]["clean1"]),
        gmm_clean0=np.asarray(data_bundle["gmm"]["clean0"]),
        gmm_clean1=np.asarray(data_bundle["gmm"]["clean1"]),
        data_seed=np.asarray(data_bundle["seed"], dtype=int),
    )


def run_tsne_from_graph(graph, seed=0, n_epochs=200):
    graph = np.asarray(graph, dtype=float)
    graph = np.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0)
    graph[graph < 0] = 0.0
    np.fill_diagonal(graph, 0.0)
    total_mass = float(graph.sum())
    if total_mass <= 0:
        raise ValueError("Graph has zero total mass; cannot run t-SNE.")
    graph = graph / total_mass

    n_points = graph.shape[0]
    init = np.random.RandomState(seed).randn(n_points, 2) * 1e-4
    n_early = int(n_epochs) // 3
    n_late = int(n_epochs) - n_early

    affinities = Affinities()
    affinities.P = sp.csr_matrix(graph)
    embedder = TSNE(
        n_components=2,
        random_state=int(seed),
        verbose=False,
        early_exaggeration_iter=n_early,
        n_iter=n_late,
    )
    return np.asarray(embedder.fit(affinities=affinities, initialization=init))


def sparsify_top_percentage_neighbors_graph(M, percentage, n0, n1, ensure_nonempty=True):
    M = _validate_nonnegative_matrix(M)
    p = _validate_percentage(percentage)
    n0 = int(n0)
    n1 = int(n1)
    if M.shape != (n0 + n1, n0 + n1):
        raise ValueError(
            f"Expected a square 4-block matrix with shape {(n0 + n1, n0 + n1)}, got {M.shape}."
        )

    mask = np.zeros_like(M, dtype=bool)
    for row_slice, col_slice in (
        (slice(0, n0), slice(0, n0)),
        (slice(0, n0), slice(n0, n0 + n1)),
        (slice(n0, n0 + n1), slice(0, n0)),
        (slice(n0, n0 + n1), slice(n0, n0 + n1)),
    ):
        block = M[row_slice, col_slice]
        block_mask = _top_percentage_neighbors_block_mask(block, p)
        if ensure_nonempty:
            block_mask = _ensure_nonempty_mask(block_mask, block)
        mask[row_slice, col_slice] = block_mask

    np.fill_diagonal(mask, False)
    out = np.where(mask, M, 0.0)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out[out < 0] = 0.0
    return out


def _lognorm_for_plan(P):
    values = np.asarray(P)[np.asarray(P) > 0]
    if values.size == 0:
        return None
    vmin = float(np.quantile(values, 0.01))
    vmax = float(np.quantile(values, 0.995))
    vmin = max(vmin, 1e-16)
    if vmax <= vmin:
        vmax = float(values.max()) if float(values.max()) > vmin else vmin * 10.0
    return LogNorm(vmin=vmin, vmax=vmax)


def _validate_nonnegative_matrix(M):
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {M.shape}.")
    if np.any(M < 0):
        raise ValueError("Expected a non-negative matrix.")
    return M


def _validate_percentage(percentage):
    percentage = float(percentage)
    if not (0.0 < percentage <= 1.0):
        raise ValueError(f"percentage must be in (0, 1], got {percentage}.")
    return percentage


def _top_percentage_neighbors_block_mask(block, percentage):
    n_rows, n_cols = block.shape
    row_k = max(1, int(np.ceil(percentage * n_cols)))
    col_k = max(1, int(np.ceil(percentage * n_rows)))
    return _row_or_col_topk_mask(block, row_k=row_k, col_k=col_k)


def _row_or_col_topk_mask(block, row_k, col_k):
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


def _ensure_nonempty_mask(mask, block):
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


__all__ = [
    "METRIC_NAMES",
    "PLOT_COLORS",
    "PLOT_LABELS",
    "build_eot_embedding_from_plan",
    "build_four_block_matrices",
    "build_four_block_matrix",
    "prepare_sparse_plan_four_block_matrix",
    "postprocess_graph_matrix",
    "postprocess_sparse_plan_four_block_graph",
    "build_sparse_plan_four_block_graph",
    "build_metric_heatmap_data",
    "compute_shared_dataset_scale_bounds",
    "compute_eot_gmm_rand",
    "compute_gmm_rand_summary",
    "load_benchmark_cache",
    "load_eot_reference_cache",
    "compute_eot_reference_from_plans",
    "compute_ot_plans_for_eps",
    "format_eps_token",
    "make_eps_cache_path",
    "make_eps_eot_cache_path",
    "metric_triplet",
    "metric_dict_to_row",
    "plot_gmm_pct_embedding_grid_by_cluster",
    "plot_gmm_rand_summary",
    "plot_k50_concordance_summary",
    "plot_metric_heatmap_grid",
    "plot_ot_plan_heatmaps",
    "plot_pct_embedding_grid",
    "prepare_benchmark_data",
    "run_sparse_plan_top_pct_neighbors_benchmark",
    "run_top_pct_neighbors_benchmark",
    "run_tsne_from_graph",
    "save_benchmark_cache",
    "save_eot_reference_cache",
    "summarize_metric_lists",
]
