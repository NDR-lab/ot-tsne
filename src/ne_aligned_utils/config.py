"""Shared paths and registries for datasets, methods and result locations.

The constants below are the canonical method and dataset registry used by the release pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Repository root and canonical input directory.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class DatasetSpec:
    """Names and storage details for one dataset."""

    directory: str
    label: str | None
    aliases: tuple[str, ...] = ()
    batch_names: tuple[str, str] | None = None


DATASETS: dict[str, DatasetSpec] = {
    "nCoV": DatasetSpec("nCoV", "nCoV"),
    "human_brain": DatasetSpec("human_brain", "Human brain"),
    "human_brain_log": DatasetSpec("human_brain_log", None),
    "lupus": DatasetSpec("lupus", "Lupus"),
    "yao_10x_ent_l2l3": DatasetSpec("yao_10x_ent_l2l3", "Mouse cortex"),
    "PBMC": DatasetSpec("PBMC", "PBMC", batch_names=("10x-3prime", "10x-5prime")),
    "mouse_atlas": DatasetSpec("mouse_atlas", "Mouse atlas", batch_names=("seqwell", "10x")),
    "hematopoietic": DatasetSpec("hematopoietic", "Hematopoietic", batch_names=("b1", "b2")),
    "hematopoietic_shared": DatasetSpec(
        "hematopoietic_shared", "Hematopoietic (shared types)", batch_names=("b1", "b2")
    ),
    "mnist": DatasetSpec("MNIST", "MNIST"),
}

SUPPORTED_DATASETS: tuple[str, ...] = tuple(
    name for key, spec in DATASETS.items() for name in (key, *spec.aliases)
)
DATASET_ALIASES: dict[str, str] = {
    alias: key for key, spec in DATASETS.items() for alias in spec.aliases
}
DATASET_DIRECTORIES: dict[str, str] = {key: spec.directory for key, spec in DATASETS.items()}
TWO_BATCH_DATASETS: dict[str, tuple[str, str]] = {
    key: spec.batch_names for key, spec in DATASETS.items() if spec.batch_names is not None
}
DATASET_LABELS: dict[str, str] = {
    key: spec.label for key, spec in DATASETS.items() if spec.label is not None
}

# Current real datasets used by the benchmark.
BENCHMARK_DATASETS: tuple[str, ...] = (
    "nCoV",
    "human_brain",
    "lupus",
    "yao_10x_ent_l2l3",
    "PBMC",
    "mouse_atlas",
    "hematopoietic",
    "hematopoietic_shared",
)

#: Generated outputs. Set ``NE_ALIGNED_RESULTS_DIR`` to relocate the complete tree.
RESULTS_DIR = Path(os.environ.get("NE_ALIGNED_RESULTS_DIR") or ROOT / "results")

#: Rebuilt datasets used for comparison with the canonical inputs in ``data/``.
PREPROCESSING_STAGING_DIR = RESULTS_DIR / "preprocessing"

#: Working paper figures and final selected figures.
PAPER_FIGURES_DIR = RESULTS_DIR / "paper_figures"
FINAL_FIGURES_DIR = RESULTS_DIR / "paper_figures_final"

#: Results from methods that directly produce their final layout.
OUTPUT_2D_DIR = RESULTS_DIR / "2d_outputs"
MULTIMAP_DIR = RESULTS_DIR / "multimap"

BACKENDS: tuple[str, ...] = ("tsne", "umap")
DEFAULT_BACKEND = "tsne"

#: Stored backend value when a method has no replaceable visualization step.
NO_BACKEND = "na"

# Public names follow the current package: ``liger-*`` is LIGER 2 and ``liger1-*`` is the
# historical adapter. Keep ``liger2`` in current cache filenames so old LIGER 1 files named
# ``liger_*`` can never be mistaken for current results.
STORAGE_NAME_OVERRIDES: dict[str, str] = {
    # Existing ``ref_*`` and ``seurat_*`` files belong to the historical methods.
    "ref": "ref_policar",
    "ref-our": "ref",
    "seurat-2d": "seurat5-2d",
    "seurat-50": "seurat5-50",
    "seurat3-2d": "seurat-2d",
    "seurat3-50": "seurat-50",
    "liger-2d": "liger2-2d",
    "liger-50": "liger2-50",
}


def canonical_name(method: str, backend: str = DEFAULT_BACKEND) -> str:
    """Return the stored method name, including its backend when it has one."""
    if not method_spec(method).backend_axis:
        return method
    return f"{method}-{resolve_backend(method, backend)}"


def split_name(name: str) -> tuple[str, str]:
    """Split a stored name into its base method and backend."""
    if name in METHOD_SPECS:
        return name, resolve_backend(name)
    for backend in BACKENDS:
        suffix = f"-{backend}"
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            if base in METHOD_SPECS:
                return base, backend
    raise KeyError(
        f"{name!r} is not a canonical method name. Names are built by canonical_name(base, "
        "backend) from METHOD_SPECS; nothing else is a valid identity."
    )


def run_uses_perplexity(method: str, backend: str = DEFAULT_BACKEND) -> bool:
    """Whether this run uses a t-SNE-style perplexity parameter."""
    spec = method_spec(method)
    if not spec.uses_perplexity:
        return False
    if not spec.backend_axis:
        return True
    return backend == DEFAULT_BACKEND


def result_dir(backend: str, *parts: str | Path) -> Path:
    """Return a directory inside one backend's result tree."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}.")
    return RESULTS_DIR.joinpath(backend, *parts)


def slides_dir(backend: str) -> Path:
    """Where the slide builders write their ``.tex`` and ``.pdf`` for one backend."""
    return result_dir(backend, "slides")


def figures_dir(backend: str) -> Path:
    """Per-figure PDFs and PNGs that one backend's decks include."""
    return result_dir(backend, "slides", "figures")


def scored_rows_csv(backend: str) -> Path:
    """Return the canonical scored metrics table for one backend."""
    return result_dir(backend, "rank_aggregation", "scored_rows.csv")


# Method names under a backend

#: Default t-SNE perplexity and UMAP neighborhood size.
DEFAULT_PERPLEXITY = 30.0
# openTSNE's default schedule, used by every t-SNE layout we optimise (50D tails and OT-tSNE):
# 250 early-exaggeration iterations, then 500 main-phase iterations.
DEFAULT_TSNE_EARLY_EXAGGERATION_ITER = 250
DEFAULT_TSNE_N_ITER = 500
DEFAULT_UMAP_N_NEIGHBORS = 15

# Method input and training defaults used by the benchmark adapters.
# The supplied feature panel is normally the shared 1,000-gene panel (593 genes for Yao).
DEFAULT_SEURAT_PRESELECTED_FEATURES = True
DEFAULT_SEURAT_WORKFLOW = "v5_cca"
DEFAULT_LIGER_PRESELECTED_FEATURES = True
DEFAULT_REF_TSNE_PRESELECTED_FEATURES = True
DEFAULT_SCVI_MAX_EPOCHS: int | None = None
DEFAULT_SCVI_BATCH_SIZE = 128

#: Canonical MultiMAP runs delegate layout settings to MultiMAP's native profile.
DEFAULT_MULTIMAP_NATIVE_PROFILE = True
#: Controlled-profile values retained for explicit historical reproduction.
DEFAULT_MULTIMAP_MIN_DIST = 0.1
DEFAULT_MULTIMAP_SPREAD = 1.0
DEFAULT_MULTIMAP_N_EPOCHS = 200
DEFAULT_MULTIMAP_INIT = "spectral"
# Let MultiMAP call Scanpy's own per-batch scaling implementation.
DEFAULT_MULTIMAP_USE_NATIVE_SCALING = True

# rc-tsne (revised conditional t-SNE), prespecified

#: Prespecified rc-tSNE settings. See ``vendor/rctsne/VENDOR.md`` for parameter conventions.
DEFAULT_RC_TSNE_BETA = 0.1

RC_TSNE_MAX_ITER = 750
RC_TSNE_STOP_EARLY_EXAG_ITER = 250
RC_TSNE_THETA = 0.5

RC_TSNE_KEEP_NONZERO = 3

BACKEND_TOKEN = "tsne"

#: t-SNE algorithms whose visualization step cannot be replaced with UMAP.
NO_BACKEND_AXIS_METHODS: tuple[str, ...] = ("bc-tsne", "rc-tsne", "ref")


# Method registry
@dataclass(frozen=True)
class MethodSpec:
    """A method's backend behavior and result tree."""

    backend_axis: bool
    tree: str
    #: Whether the computation uses a t-SNE-style perplexity.
    uses_perplexity: bool = True


#: All benchmark methods, keyed by backend-neutral base name.
METHOD_SPECS: dict[str, MethodSpec] = {
    # -- separable embedder: integration to k dimensions, then t-SNE or UMAP ------------------
    "ot": MethodSpec(True, "embeddings"),
    "eot-50": MethodSpec(True, "embeddings"),
    "harmony-50": MethodSpec(True, "embeddings"),
    "scvi-50": MethodSpec(True, "embeddings"),
    "cca-50": MethodSpec(True, "embeddings"),
    "cca-50-uncentered": MethodSpec(True, "embeddings"),
    "joint": MethodSpec(True, "embeddings"),
    "ot-knndiag": MethodSpec(True, "embeddings_variants"),
    "ot-selfotmedian": MethodSpec(True, "embeddings_variants"),
    "ot-gramdiag": MethodSpec(True, "embeddings_variants"),
    # Unbalanced OT: topk changes the solver; mass also scales cross-batch neighbor budgets.
    "ot-uot-topk": MethodSpec(True, "embeddings_uot"),
    "ot-uot-mass": MethodSpec(True, "embeddings_uot"),
    # Zero-diagonal OT uses only cross-batch edges and a separate result tree.
    "ot-zerodiag": MethodSpec(True, "embeddings_zerodiag"),
    # Split-budget zero-diagonal OT preserves the base cross edges. It is t-SNE-only
    # because unequal batch quotas cannot fit UMAP's fixed-width neighbor lists.
    "ot-zerodiag-split": MethodSpec(True, "embeddings_zerodiag"),
    "scanorama-50": MethodSpec(True, "embeddings_newmethods"),
    "seurat-50": MethodSpec(True, "embeddings_newmethods"),
    "seurat3-50": MethodSpec(True, "embeddings_newmethods"),
    "liger1-50": MethodSpec(True, "embeddings_newmethods"),
    "liger-50": MethodSpec(True, "embeddings_newmethods"),
    # ``ref`` is Poličar's fixed procedure; ``ref-our`` is the historical adapter.
    "ref-our": MethodSpec(True, "embeddings_newmethods"),
    # -- no separable embedder: coordinates come out of the method itself --------------------
    # Direct and closed-form 2D layouts.
    "joint-pca": MethodSpec(False, "2d_outputs", uses_perplexity=False),
    "eot-t0": MethodSpec(False, "2d_outputs", uses_perplexity=False),
    "cca-2d": MethodSpec(False, "2d_outputs", uses_perplexity=False),
    "cca-2d-uncentered": MethodSpec(False, "2d_outputs", uses_perplexity=False),
    "harmony-2d": MethodSpec(False, "2d_outputs", uses_perplexity=False),
    "scvi-2d": MethodSpec(False, "2d_outputs", uses_perplexity=False),
    "scanorama-2d": MethodSpec(False, "2d_outputs", uses_perplexity=False),
    "seurat-2d": MethodSpec(False, "2d_outputs", uses_perplexity=False),
    "seurat3-2d": MethodSpec(False, "2d_outputs", uses_perplexity=False),
    "liger1-2d": MethodSpec(False, "2d_outputs", uses_perplexity=False),
    "liger-2d": MethodSpec(False, "2d_outputs", uses_perplexity=False),
    "ref": MethodSpec(False, "reftsne", uses_perplexity=False),
    # Complete integration-and-visualization methods.
    "multimap": MethodSpec(False, "multimap", uses_perplexity=False),
    "rc-tsne": MethodSpec(False, "rctsne"),
    "bc-tsne": MethodSpec(False, "bctsne"),
}


def method_spec(method: str) -> MethodSpec:
    """The registry entry for ``method``; raises rather than guessing at an unknown name."""
    try:
        return METHOD_SPECS[method]
    except KeyError:
        raise KeyError(
            f"{method!r} is not in METHOD_SPECS. Add it there rather than inferring its "
            "capabilities from its name or from where its results happen to sit."
        ) from None


def resolve_backend(method: str, requested: str = DEFAULT_BACKEND) -> str:
    """Return the requested backend, or ``na`` when the method has no backend axis."""
    return requested if method_spec(method).backend_axis else NO_BACKEND


def result_location(method: str, dataset: str, *, backend: str = DEFAULT_BACKEND) -> Path:
    """Return the canonical result directory for a method and dataset."""
    spec = method_spec(method)
    if spec.backend_axis:
        return result_dir(backend, spec.tree, dataset)
    return RESULTS_DIR / spec.tree / dataset


_INIT_TREES: tuple[str, ...] = (
    "tsne/embeddings_init_random",
    "tsne/embeddings_init_spectral",
    "tsne/embeddings_init_eot",
    "tsne/embeddings_variants",
)

#: Legacy locations and initialization-ablation trees that readers must also search.
EXTRA_RESULT_TREES: dict[str, tuple[str, ...]] = {
    # superseded
    "scanorama-2d": ("2d_outputs",),
    # initialization ablation, an axis of its own
    "ot": _INIT_TREES,
    "ot-knndiag": _INIT_TREES,
    "ot-selfotmedian": _INIT_TREES,
    "ot-gramdiag": _INIT_TREES,
}


def result_locations(method: str, dataset: str, *, backend: str = DEFAULT_BACKEND) -> list[Path]:
    """Return the canonical result directory followed by compatible older locations."""
    found = [result_location(method, dataset, backend=backend)]
    for tree in EXTRA_RESULT_TREES.get(method, ()):
        candidate = RESULTS_DIR.joinpath(*tree.split("/"), dataset)
        if candidate not in found:
            found.append(candidate)
    return found


#: Methods stored outside the t-SNE and UMAP backend trees.
OUTPUT_2D_METHODS: tuple[str, ...] = (
    "joint-pca",
    "eot-t0",
    "cca-2d",
    "cca-2d-uncentered",
    "harmony-2d",
    "scvi-2d",
    "scanorama-2d",
    "seurat-2d",
    "seurat3-2d",
    "liger1-2d",
    "liger-2d",
)
MULTIMAP_METHODS: tuple[str, ...] = ("multimap",)
RCTSNE_METHODS: tuple[str, ...] = ("rc-tsne",)
BCTSNE_METHODS: tuple[str, ...] = ("bc-tsne",)
REFTSNE_METHODS: tuple[str, ...] = ("ref",)
SHARED_RESULT_TREES: dict[str, tuple[str, ...]] = {
    "2d_outputs": OUTPUT_2D_METHODS,
    "multimap": MULTIMAP_METHODS,
    "rctsne": RCTSNE_METHODS,
    "bctsne": BCTSNE_METHODS,
    "reftsne": REFTSNE_METHODS,
}


def shared_result_tree(method: str) -> str | None:
    """Canonical top-level results tree for a method with no replaceable backend."""
    for tree, methods in SHARED_RESULT_TREES.items():
        if method in methods:
            return tree
    return None


#: Stored integration representations that can be reused by another backend.
REPRESENTATION_ARTIFACT_KEYS: dict[str, str] = {
    "cca-50": "cca_embedding",
    "cca-50-uncentered": "cca_embedding",
    "eot-50": "eot_embedding",
    "harmony-50": "harmony_embedding",
    "scvi-50": "scvi_embedding",
    "scanorama-50": "scanorama_embedding",
    "seurat-50": "integration_embedding",
    "seurat3-50": "integration_embedding",
    "liger1-50": "integration_embedding",
    "liger-50": "integration_embedding",
}


# Thread allocation

#: Environment variables used to pin numerical libraries to one thread count.
THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def thread_env(threads: int | None) -> dict[str, str]:
    """Return environment values that pin a benchmark task to ``threads`` threads."""
    if threads is None:
        return {}
    threads = int(threads)
    if threads < 1:
        raise ValueError(f"threads must be at least 1, got {threads}.")
    return {name: str(threads) for name in THREAD_ENV_VARS}


def observed_thread_env() -> dict[str, str]:
    """Return the thread settings visible to this process."""
    return {name: os.environ.get(name, "") for name in THREAD_ENV_VARS}


def has_backend_axis(method: str) -> bool:
    """Whether the method's final visualization backend can be replaced."""
    return method_spec(method).backend_axis


def method_backend_label(method: str, backend: str) -> str:
    """Return the stored method name for a backend."""
    return canonical_name(method, backend)


def method_stem(method: str, backend: str = DEFAULT_BACKEND) -> str:
    """Return the underscore-separated filename stem for a method run."""
    storage_method = STORAGE_NAME_OVERRIDES.get(method, method)
    if has_backend_axis(method):
        return f"{storage_method}-{resolve_backend(method, backend)}".replace("-", "_")
    return storage_method.replace("-", "_")


#: Frozen reference artifacts from the original t-SNE run.
FROZEN_BASELINE_DIR = ROOT / "baseline"

#: Original aggregate caches used as reproduction references.
ORIGINAL_CACHES: dict[str, Path] = {
    "mnist": DATA_DIR
    / "benchmark_embeddings/mnist/n_per_batch_all__q_2__epochs_750__tsne_pca_50"
    / "mnist_benchmark_embeddings.npz",
    "nCoV": DATA_DIR
    / "benchmark_embeddings/nCoV/n_per_batch_all__q_2__epochs_200__tsne_pca_50"
    / "nCoV_benchmark_embeddings.npz",
    # This reference predates the corrected human-brain gene panel.
    "human_brain_log": DATA_DIR
    / "benchmark_embeddings/human_brain_log/n_per_batch_5000__q_2__epochs_200__tsne_pca_50"
    / "human_brain_log_benchmark_embeddings.npz",
    "lupus": DATA_DIR
    / "benchmark_embeddings/lupus/n_per_batch_all__q_2__epochs_200__tsne_pca_50"
    / "lupus_benchmark_embeddings.npz",
    "yao_10x_ent_l2l3": DATA_DIR
    / "benchmark_embeddings/yao_10x_ent_l2l3/n_per_batch_5000_10000__q_2__epochs_200__tsne_pca_50"
    / "yao_10x_ent_l2l3_benchmark_embeddings.npz",
}

# Benchmark registry

#: Methods stored in each aggregate-cache tree.
TREE_METHODS: dict[str, tuple[str, ...]] = {
    "embeddings": (
        "ot",
        "eot-50",
        "harmony-50",
        "scvi-50",
        "cca-50",
        # Uncentered CCA is retained for comparison; cca-2d and cca-50 use centering.
        "cca-50-uncentered",
        "joint",
    ),
    "embeddings_variants": (
        "ot-knndiag",
        "ot-selfotmedian",
        "ot-gramdiag",
    ),
    "embeddings_uot": (
        "ot-uot-topk",
        "ot-uot-mass",
    ),
    "embeddings_zerodiag": ("ot-zerodiag", "ot-zerodiag-split"),
    "embeddings_newmethods": (
        "ref-our",
        "scanorama-50",
        "seurat-50",
        "seurat3-50",
        "liger1-50",
        "liger-50",
    ),
    **SHARED_RESULT_TREES,
}

# Slide decks include the reported synthetic dataset and all current real datasets.
DECK_DATASETS: tuple[str, ...] = ("mnist", *BENCHMARK_DATASETS)


def _main(argv: list[str] | None = None) -> int:
    """Print configured paths for shell scripts that cannot import Python values."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m ne_aligned_utils.config")
    sub = parser.add_subparsers(dest="command", required=True)

    p_result = sub.add_parser("result-dir", help="print result_dir(backend, *parts)")
    p_result.add_argument("backend", choices=BACKENDS)
    p_result.add_argument("parts", nargs="*", default=[])

    for name, help_text in (
        ("slides-dir", "print slides_dir(backend)"),
        ("figures-dir", "print figures_dir(backend)"),
        ("scored-rows-csv", "print scored_rows_csv(backend)"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("backend", choices=BACKENDS)

    for name, help_text in (
        ("root", "print ROOT"),
        ("data-dir", "print DATA_DIR"),
        ("results-dir", "print RESULTS_DIR"),
        ("preprocessing-dir", "print PREPROCESSING_STAGING_DIR"),
    ):
        sub.add_parser(name, help=help_text)

    args = parser.parse_args(argv)
    resolved = {
        "result-dir": lambda: result_dir(args.backend, *args.parts),
        "slides-dir": lambda: slides_dir(args.backend),
        "figures-dir": lambda: figures_dir(args.backend),
        "scored-rows-csv": lambda: scored_rows_csv(args.backend),
        "root": lambda: ROOT,
        "data-dir": lambda: DATA_DIR,
        "results-dir": lambda: RESULTS_DIR,
        "preprocessing-dir": lambda: PREPROCESSING_STAGING_DIR,
    }[args.command]()
    print(resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
