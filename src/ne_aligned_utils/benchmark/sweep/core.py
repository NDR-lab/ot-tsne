"""Run, aggregate, and score benchmark experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from ne_aligned_utils.benchmark.data import (  # noqa: E402
    canonical_dataset_name as _canonical_dataset_name,
)
from ne_aligned_utils.benchmark.sweep.aggregation import (  # noqa: E402, F401
    _merge_previous_methods,
    aggregate,
    cache_keys,
    score_aggregate,
)
from ne_aligned_utils.benchmark.sweep.config import (  # noqa: E402, F401
    COUNT_LIBRARY_SIZES,
    DEFAULT_COUNT_LIBRARY_SIZE,
    DEFAULT_EFFECTIVE_KS,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_ITER,
    DEFAULT_N_EPOCHS,
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_PERPLEXITY,
    DEFAULT_REL_REG,
    DEFAULT_REL_REG_M,
    DEFAULT_SYMMETRIZATION,
    DEFAULT_TOL,
    DEFAULT_TSNE_EARLY_EXAGGERATION_ITER,
    SYMMETRIZATIONS,
    _resolved_n_neighbors,
    assert_no_parameter_collision,
    assert_output_backend_matches,
    config_token,
    format_rel_reg,
    format_token,
    resolve_aggregate_output_path,
    resolve_output_base,
    resolve_task_output_path,
    resolve_task_root,
)
from ne_aligned_utils.benchmark.sweep.registry import (  # noqa: E402
    CPP_SUBPROCESS_METHODS,
    DEFAULT_METHODS,
    EPS_K_METHODS,
    KNNDIAG_SYMMETRIZATION_METHODS,
    OT_PLAN_METHODS,
    SYMMETRIZATION_METHODS,
    TASK_KINDS,
    UOT_METHODS,
    input_kind_for_method,
)
from ne_aligned_utils.benchmark.sweep.tasks import (  # noqa: E402, F401
    UNPERSISTED_ARTIFACT_KEYS,
    _compute_task,
    common_metadata,
    load_cached_representation,
    run_task,
    save_npz,
)
from ne_aligned_utils.config import (  # noqa: E402
    BACKENDS,
    DEFAULT_BACKEND,
    DEFAULT_MULTIMAP_NATIVE_PROFILE,
    DEFAULT_RC_TSNE_BETA,
    DEFAULT_UMAP_N_NEIGHBORS,
    NO_BACKEND,
    ROOT,
    SUPPORTED_DATASETS,
)

# Compatibility for older tests and callers that imported the former private helper.
_input_kind_for_method = input_kind_for_method


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-cells-per-batch", type=int, default=None)
    parser.add_argument("--max-cells-batch0", type=int, default=None)
    parser.add_argument("--max-cells-batch1", type=int, default=None)
    # Optional benchmark provenance. Defaults preserve the historical meaning of --seed.
    parser.add_argument("--preprocessing-seed", type=int, default=-1)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--method-seed", type=int, default=None)
    parser.add_argument("--visualization-seed", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--preprocessing-hash", default=None)
    parser.add_argument("--sample-hash", default=None)
    parser.add_argument("--representation-family", default=None)
    parser.add_argument("--method-pipeline", default=None)
    parser.add_argument("--benchmark-manifest-hash", default=None)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS)
    # Early exaggeration before the --n-epochs main phase, for both the 50D tails and OT layouts.
    parser.add_argument(
        "--tsne-early-exaggeration-iter", type=int, default=DEFAULT_TSNE_EARLY_EXAGGERATION_ITER
    )
    # Count normalisation divisor for seurat-*, liger-* and ref; see sweep.config.
    parser.add_argument(
        "--count-library-size", choices=COUNT_LIBRARY_SIZES, default=DEFAULT_COUNT_LIBRARY_SIZE
    )
    parser.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS)
    parser.add_argument("--perplexity", type=float, default=DEFAULT_PERPLEXITY)
    # UMAP and OT-UMAP share one neighborhood budget.
    parser.add_argument("--backend", choices=[*BACKENDS, NO_BACKEND], default=DEFAULT_BACKEND)
    parser.add_argument("--n-neighbors", type=int, default=None)
    # Reuse an explicitly named results tree, never an arbitrary path.
    parser.add_argument("--reuse-from", default=None)
    # Let MultiMAP choose its native layout parameters by default; retain an explicit historical
    # controlled profile for reproduction of older runs.
    parser.add_argument(
        "--multimap-native",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MULTIMAP_NATIVE_PROFILE,
    )
    # rc-tSNE calls this paper parameter beta but its driver calls it alpha.
    parser.add_argument("--rc-beta", type=float, default=DEFAULT_RC_TSNE_BETA)
    # Marginal relaxation; infinity is the balanced limit.
    parser.add_argument("--rel-reg-m", type=float, default=DEFAULT_REL_REG_M)
    # OT and kNN diagonal symmetrization are independent choices.
    parser.add_argument(
        "--graph-symmetrization", choices=list(SYMMETRIZATIONS), default=DEFAULT_SYMMETRIZATION
    )
    parser.add_argument(
        "--knndiag-symmetrization", choices=list(SYMMETRIZATIONS), default=DEFAULT_SYMMETRIZATION
    )
    # OT device is an execution choice; backend identifies the visualizer.
    from ne_aligned_utils.methods import CPU_DEVICE, OT_DEVICE_CHOICES

    parser.add_argument("--ot-device", choices=list(OT_DEVICE_CHOICES), default=CPU_DEVICE)
    parser.add_argument("--n-jobs", type=int, default=1)
    # One thread count controls the environment and libraries with explicit worker arguments.
    parser.add_argument("--threads", type=int, default=None)
    # Runtime comparisons are CPU-only when resource control is requested.
    parser.add_argument("--require-cpu", action="store_true", default=None)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    # Initialization applies only to OT-tSNE methods.
    from ne_aligned_utils.methods import OT_TSNE_INIT_CHOICES

    parser.add_argument("--init", choices=list(OT_TSNE_INIT_CHOICES), default="random")


def add_sweep_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rel-reg", nargs="+", type=float, default=list(DEFAULT_REL_REG))
    parser.add_argument(
        "--effective-k",
        "--effective-ks",
        dest="effective_ks",
        nargs="+",
        type=int,
        default=list(DEFAULT_EFFECTIVE_KS),
    )
    parser.add_argument("--methods", nargs="+", choices=TASK_KINDS, default=list(DEFAULT_METHODS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute benchmark embeddings from named repo datasets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    task_parser = subparsers.add_parser("run-task")
    add_common_options(task_parser)
    task_parser.add_argument("--task-kind", choices=TASK_KINDS, required=True)
    task_parser.add_argument("--seed", type=int, required=True)
    task_parser.add_argument("--task-output", type=Path, required=True)
    task_parser.add_argument("--rel-reg", type=float, default=None)
    task_parser.add_argument("--effective-k", type=int, default=None)
    task_parser.add_argument("--force", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate")
    add_common_options(aggregate_parser)
    add_sweep_options(aggregate_parser)
    aggregate_parser.add_argument("--n-seeds", type=int, required=True)
    aggregate_parser.add_argument("--output-cache", type=Path, default=None)
    aggregate_parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "discard methods the existing cache holds that this run was not asked for. "
            "Without it the aggregate is extended, so a partial run cannot drop other methods."
        ),
    )

    score_parser = subparsers.add_parser("score-aggregate")
    score_parser.add_argument("--aggregate-cache", type=Path, required=True)
    score_parser.add_argument("--output-csv", type=Path, default=None)
    score_parser.add_argument("--methods", nargs="+", choices=TASK_KINDS, default=None)

    args = parser.parse_args()
    if hasattr(args, "dataset"):
        args.dataset = _canonical_dataset_name(args.dataset)
    validate_args(parser, args)
    return args


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if hasattr(args, "embedding_dim") and args.embedding_dim <= 0:
        parser.error("--embedding-dim must be positive.")
    if hasattr(args, "n_epochs") and args.n_epochs <= 0:
        parser.error("--n-epochs must be positive.")
    if getattr(args, "tsne_early_exaggeration_iter", 0) < 0:
        parser.error("--tsne-early-exaggeration-iter must be non-negative.")
    if hasattr(args, "pca_components") and args.pca_components <= 0:
        parser.error("--pca-components must be positive.")
    if hasattr(args, "perplexity") and args.perplexity <= 0:
        parser.error("--perplexity must be positive.")
    if hasattr(args, "threads"):
        if args.threads is not None and args.threads < 1:
            parser.error("--threads must be at least 1.")
        # A thread allocation on a CPU task is a controlled CPU run. A task that asks for a GPU
        # still gets threads for its CPU stages, so it must not be turned into a CPU-only run.
        if getattr(args, "require_cpu", None) is None:
            args.require_cpu = args.threads is not None and (
                str(getattr(args, "ot_device", None) or "cpu") == "cpu"
            )
        # Use the thread allocation as n_jobs only when no explicit worker count was supplied.
        if args.threads is not None and getattr(args, "n_jobs", 1) == 1:
            args.n_jobs = int(args.threads)
    if hasattr(args, "n_jobs") and args.n_jobs <= 0:
        parser.error("--n-jobs must be positive.")
    if hasattr(args, "tol") and args.tol <= 0:
        parser.error("--tol must be positive.")
    if hasattr(args, "max_iter") and args.max_iter <= 0:
        parser.error("--max-iter must be positive.")
    if (
        hasattr(args, "max_cells_per_batch")
        and args.max_cells_per_batch is not None
        and args.max_cells_per_batch <= 0
    ):
        parser.error("--max-cells-per-batch must be positive when provided.")
    if (
        hasattr(args, "max_cells_batch0")
        and args.max_cells_batch0 is not None
        and args.max_cells_batch0 <= 0
    ):
        parser.error("--max-cells-batch0 must be positive when provided.")
    if (
        hasattr(args, "max_cells_batch1")
        and args.max_cells_batch1 is not None
        and args.max_cells_batch1 <= 0
    ):
        parser.error("--max-cells-batch1 must be positive when provided.")
    if hasattr(args, "n_seeds") and args.n_seeds <= 0:
        parser.error("--n-seeds must be positive.")
    if hasattr(args, "rel_reg") and args.rel_reg is not None:
        rel_reg_values = args.rel_reg if isinstance(args.rel_reg, list) else [args.rel_reg]
        if any(float(value) <= 0 for value in rel_reg_values):
            parser.error("--rel-reg values must be positive.")
    if hasattr(args, "effective_ks"):
        if any(int(value) <= 0 for value in args.effective_ks):
            parser.error("--effective-k values must be positive.")
    if getattr(args, "n_neighbors", None) is not None and int(args.n_neighbors) < 2:
        parser.error("--n-neighbors must be at least 2.")

    # Use one neighborhood budget for UMAP, OT-UMAP, and MultiMAP.
    # OT-UMAP requires n_neighbors to match the supplied list width.
    task_kind = getattr(args, "task_kind", None)
    methods = set(getattr(args, "methods", ()))
    uses_umap_neighbourhood = (
        str(getattr(args, "backend", DEFAULT_BACKEND)) == "umap"
        or task_kind == "multimap"
        or "multimap" in methods
    )
    if uses_umap_neighbourhood:
        requested = getattr(args, "n_neighbors", None)
        if hasattr(args, "effective_ks"):
            values = tuple(dict.fromkeys(int(value) for value in args.effective_ks))
            if len(values) != 1:
                parser.error(
                    "UMAP runs require exactly one --effective-k per invocation; run each "
                    "neighbourhood value separately."
                )
            effective_k = values[0]
        elif getattr(args, "effective_k", None) is not None:
            effective_k = int(args.effective_k)
        elif requested is not None:
            # Direct UMAP and MultiMAP tasks take their neighborhood budget from n_neighbors.
            effective_k = int(requested)
        else:
            effective_k = DEFAULT_UMAP_N_NEIGHBORS

        if requested is not None and int(requested) != effective_k:
            parser.error(
                f"--n-neighbors ({int(requested)}) must equal --effective-k "
                f"({effective_k}) for UMAP runs."
            )
        args.n_neighbors = effective_k
    if getattr(args, "task_kind", None) in OT_PLAN_METHODS and args.rel_reg is None:
        parser.error("--rel-reg is required for OT-related run-task commands.")
    if getattr(args, "task_kind", None) in EPS_K_METHODS and args.effective_k is None:
        parser.error(
            f"--effective-k is required for --task-kind {getattr(args, 'task_kind', None)}."
        )
    if hasattr(args, "methods") and hasattr(args, "rel_reg"):
        methods = set(args.methods)
        if methods.intersection(EPS_K_METHODS) and not getattr(args, "effective_ks", None):
            parser.error("--effective-k is required when an eps+k OT-tSNE method is requested.")
        if methods.intersection(OT_PLAN_METHODS) and not getattr(args, "rel_reg", None):
            parser.error("--rel-reg is required when OT-related methods are requested.")
    # Reject initialization settings for methods that do not consume them.
    if getattr(args, "init", "random") != "random":
        task_kind = getattr(args, "task_kind", None)
        if task_kind is not None and task_kind not in EPS_K_METHODS:
            parser.error(f"--init is only supported for {', '.join(EPS_K_METHODS)}.")
        if hasattr(args, "methods") and not set(args.methods).issubset(EPS_K_METHODS):
            parser.error(
                "--init applies only to the OT-tSNE family; restrict --methods to "
                f"{', '.join(EPS_K_METHODS)} when requesting a non-random initialization."
            )
    # Validate rc-tsne beta in (0, 1], matching the driver's supported range.
    rc_beta = getattr(args, "rc_beta", DEFAULT_RC_TSNE_BETA)
    if rc_beta is not None and not (0.0 < float(rc_beta) <= 1.0):
        parser.error(f"--rc-beta must be in (0, 1], got {rc_beta}.")
    if rc_beta is not None and float(rc_beta) != DEFAULT_RC_TSNE_BETA:
        task_kind = getattr(args, "task_kind", None)
        if task_kind is not None and task_kind not in CPP_SUBPROCESS_METHODS:
            parser.error(f"--rc-beta is only supported for {', '.join(CPP_SUBPROCESS_METHODS)}.")
        if hasattr(args, "methods") and not set(args.methods).issubset(CPP_SUBPROCESS_METHODS):
            parser.error(
                "--rc-beta applies only to rc-tsne; restrict --methods to "
                f"{', '.join(CPP_SUBPROCESS_METHODS)} when requesting a non-default beta."
            )
        # Require a separate output directory for nondefault beta; cache tokens do not encode it.
        if getattr(args, "output_dir", None) is None:
            parser.error(
                "A non-default --rc-beta must be given an explicit --output-dir: the 2d_outputs "
                "config token does not encode beta, so it would overwrite the reported run."
            )
    # Marginal relaxation applies only to unbalanced OT.
    rel_reg_m = getattr(args, "rel_reg_m", DEFAULT_REL_REG_M)
    if rel_reg_m is not None and not (float(rel_reg_m) > 0.0):
        parser.error(f"--rel-reg-m must be positive, got {rel_reg_m}.")
    if rel_reg_m is not None and float(rel_reg_m) != DEFAULT_REL_REG_M:
        task_kind = getattr(args, "task_kind", None)
        if task_kind is not None and task_kind not in UOT_METHODS:
            parser.error(f"--rel-reg-m is only supported for {', '.join(UOT_METHODS)}.")
        if hasattr(args, "methods") and not set(args.methods).issubset(UOT_METHODS):
            parser.error(
                "--rel-reg-m applies only to the unbalanced-OT variants; restrict --methods to "
                f"{', '.join(UOT_METHODS)} when requesting a non-default relaxation."
            )
        # Require a separate directory because cache tokens do not encode this setting.
        if getattr(args, "output_dir", None) is None:
            parser.error(
                "A non-default --rel-reg-m must be given an explicit --output-dir: the config "
                "token does not encode the relaxation, so it would overwrite another run."
            )
    # Validate graph and within-batch symmetrization independently.
    for flag, value, applicable in (
        (
            "--graph-symmetrization",
            getattr(args, "graph_symmetrization", DEFAULT_SYMMETRIZATION),
            SYMMETRIZATION_METHODS,
        ),
        (
            "--knndiag-symmetrization",
            getattr(args, "knndiag_symmetrization", DEFAULT_SYMMETRIZATION),
            KNNDIAG_SYMMETRIZATION_METHODS,
        ),
    ):
        if value is None or str(value) == DEFAULT_SYMMETRIZATION:
            continue
        task_kind = getattr(args, "task_kind", None)
        if task_kind is not None and task_kind not in applicable:
            parser.error(f"{flag} is only supported for {', '.join(applicable)}.")
        if hasattr(args, "methods") and not set(args.methods).issubset(applicable):
            parser.error(
                f"{flag} applies only to {', '.join(applicable)}; restrict --methods to those "
                "when requesting a non-default symmetrization."
            )
        # Require a separate directory for mutual graphs; cache tokens do not encode symmetrization.
        if getattr(args, "output_dir", None) is None:
            parser.error(
                f"A non-default {flag} must be given an explicit --output-dir: the config token "
                "does not encode the symmetrization, so it would overwrite the 'or' run."
            )
    # OT device applies only to plan-based methods. Collision checks prevent mixed-device caches.
    from ne_aligned_utils.methods import CPU_DEVICE

    ot_device = getattr(args, "ot_device", None)
    if ot_device is not None and str(ot_device) != CPU_DEVICE:
        task_kind = getattr(args, "task_kind", None)
        if task_kind is not None and task_kind not in OT_PLAN_METHODS:
            parser.error(f"--ot-device is only supported for {', '.join(OT_PLAN_METHODS)}.")
        if hasattr(args, "methods") and not set(args.methods).issubset(OT_PLAN_METHODS):
            parser.error(
                "--ot-device applies only to methods that solve an OT plan; restrict --methods to "
                f"{', '.join(OT_PLAN_METHODS)} when requesting a non-default device."
            )
        # Controlled CPU timing excludes accelerator-based OT solves.
        if getattr(args, "require_cpu", False):
            parser.error(
                f"--ot-device {ot_device} contradicts --require-cpu. --require-cpu is what makes "
                "a runtime comparable with the CPU-only benchmark; drop one of the two."
            )


def main() -> None:
    args = parse_args()
    if args.command == "run-task":
        run_task(args)
        return
    if args.command == "aggregate":
        aggregate(args)
        return
    if args.command == "score-aggregate":
        score_aggregate(args)
        return
    raise ValueError(f"Unsupported command {args.command!r}.")


if __name__ == "__main__":
    main()
