"""Check method/backend routing and discovery of compatible stored caches.

Methods without a replaceable backend resolve to na; supported older cache locations
must remain discoverable.
"""

from __future__ import annotations

import numpy as np
import pytest

from ne_aligned_utils.benchmark.sweep import TASK_KINDS
from ne_aligned_utils.config import (
    BACKENDS,
    BENCHMARK_DATASETS,
    DATASET_ALIASES,
    DATASET_DIRECTORIES,
    DATASET_LABELS,
    DATASETS,
    DECK_DATASETS,
    METHOD_SPECS,
    NO_BACKEND,
    RESULTS_DIR,
    SHARED_RESULT_TREES,
    SUPPORTED_DATASETS,
    TWO_BATCH_DATASETS,
    has_backend_axis,
    method_spec,
    resolve_backend,
    result_location,
    result_locations,
    split_name,
)

BACKEND_FREE = [m for m, s in METHOD_SPECS.items() if not s.backend_axis]
WITH_BACKEND = [m for m, s in METHOD_SPECS.items() if s.backend_axis]


def test_dataset_views_are_derived_from_one_registry():
    assert set(DATASET_DIRECTORIES) == set(DATASETS)
    assert set(DATASET_ALIASES).issubset(SUPPORTED_DATASETS)
    assert TWO_BATCH_DATASETS["PBMC"] == ("10x-3prime", "10x-5prime")
    assert DATASET_LABELS["yao_10x_ent_l2l3"] == "Mouse cortex"
    assert DECK_DATASETS == ("mnist", *BENCHMARK_DATASETS)
    assert set(BENCHMARK_DATASETS) == {
        "nCoV",
        "human_brain",
        "lupus",
        "yao_10x_ent_l2l3",
        "PBMC",
        "mouse_atlas",
        "hematopoietic",
        "hematopoietic_shared",
    }
    assert not {"human_brain_log", "mnist"} & set(BENCHMARK_DATASETS)


def test_bc_tsne_has_its_own_shared_result_tree():
    assert SHARED_RESULT_TREES["bctsne"] == ("bc-tsne",)
    assert "bc-tsne" not in SHARED_RESULT_TREES["2d_outputs"]
    assert result_location("bc-tsne", "nCoV") == RESULTS_DIR / "bctsne" / "nCoV"


def test_registry_covers_every_task_kind_exactly():
    """A method missing from the registry must be an error, not a silent default."""
    assert set(METHOD_SPECS) == set(TASK_KINDS)


@pytest.mark.parametrize("method", BACKEND_FREE)
def test_backend_free_methods_resolve_to_na(method):
    """Whatever backend is requested, a method with no embedder records ``na``.

    Recording ``tsne`` here is what made the 2D layouts and MultiMAP claim a t-SNE step that never
    ran, and -- because ``backend`` gates aggregate merging -- silently replaced caches.
    """
    assert has_backend_axis(method) is False
    for backend in (*BACKENDS, NO_BACKEND):
        assert resolve_backend(method, backend) == NO_BACKEND


@pytest.mark.parametrize("method", WITH_BACKEND)
def test_methods_with_an_axis_follow_the_requested_backend(method):
    assert has_backend_axis(method) is True
    for backend in BACKENDS:
        assert resolve_backend(method, backend) == backend


def test_historical_ref_tsne_has_a_backend_axis_because_both_arms_exist():
    """``ref-our`` has both t-SNE and UMAP arms; Poličar's ``ref`` does not.

    It fits a PCA on batch 0 and then embeds, and that embedder has actually been swapped: the
    repository holds ``ref_umap_*`` task files whose geometry differs from the t-SNE arm. Filing it
    as backend-free would collapse two distinct results into one location.
    """
    assert has_backend_axis("ref-our") is True
    assert result_location("ref-our", "nCoV") != result_location("ref-our", "nCoV", backend="umap")
    assert has_backend_axis("ref") is False


def test_capabilities_are_not_inferred_from_the_name():
    """The registry, not the spelling: ``rc-tsne`` and ``bc-tsne`` carry ``tsne``, no axis."""
    for method in ("rc-tsne", "bc-tsne"):
        assert "tsne" in method.split("-")
        assert has_backend_axis(method) is False


def test_backend_free_methods_are_stored_outside_the_backend_trees():
    for method in BACKEND_FREE:
        location = result_location(method, "nCoV")
        assert not set(location.parts) & set(BACKENDS), (
            f"{method} has no backend axis, so its results must not sit under a backend tree"
        )


@pytest.mark.skipif(not RESULTS_DIR.exists(), reason="no local results tree")
def test_every_existing_cache_is_still_discoverable():
    """No cached result may become invisible because the canonical layout changed.

    Readers consult ``result_locations``, which lists the canonical directory first and then every
    legacy one, so the refactor needs no file to move. This walks what is actually on disk and
    asserts each cache is reachable for at least one of the methods it holds.
    """
    unreachable = []
    for cache in RESULTS_DIR.rglob("*_benchmark_embeddings.npz"):
        rel = cache.relative_to(RESULTS_DIR)
        # Superseded and prerename backup trees intentionally retain noncanonical layouts.
        if rel.parts[0] == "superseded" or any("prerename" in p for p in rel.parts):
            continue
        dataset = rel.parts[-3]
        methods = [str(m) for m in np.load(cache, allow_pickle=True)["methods"]]
        # Pass actual backend trees; decompose stored canonical names before lookup.
        bases = []
        for stored in methods:
            try:
                bases.append(split_name(stored))
            except KeyError:
                continue
        reachable = any(
            cache.parent.parent in result_locations(base, dataset, backend=b)
            for base, _stored_backend in bases
            for b in BACKENDS
        )
        if not reachable:
            unreachable.append(str(rel))
    assert not unreachable, (
        f"{len(unreachable)} caches unreachable through result_locations, e.g. {unreachable[:3]}"
    )


def test_spec_lookup_rejects_an_unknown_method():
    with pytest.raises(KeyError, match="METHOD_SPECS"):
        method_spec("not-a-method")


def test_backend_is_not_a_collision_for_backend_free_methods(tmp_path, monkeypatch):
    """A stale ``backend`` label must not block writing beside an existing 2D result.

    ``assert_no_parameter_collision`` guards parameters the config token does not encode, because
    two such runs would overwrite each other. That reasoning needs two runs to exist: a method with
    no backend axis has one version of each result, so a sibling recording a different backend is
    the same computation with an out-of-date label. 444 files in ``2d_outputs`` still record
    ``tsne`` or nothing, written before the axis was explicit, and treating those as collisions
    blocked every new 2D run in those directories.
    """
    from argparse import Namespace

    import numpy as np

    from ne_aligned_utils.benchmark.sweep import assert_no_parameter_collision

    directory = tmp_path / "2d_outputs" / "nCoV" / "n_per_batch_all__q_2__nn_na"
    directory.mkdir(parents=True)
    np.savez(
        directory / "eot_t0_seed_0_rel_reg_1.npz",
        backend=np.asarray("tsne"),
        perplexity=np.asarray(30.0),
        n_neighbors=np.asarray(-1),
    )
    args = Namespace(
        perplexity=30.0,
        backend=NO_BACKEND,
        task_kind="eot-t0",
        n_neighbors=None,
        methods=(),
        output_dir=directory,
    )
    # backend-free: the stale sibling label is ignored
    assert_no_parameter_collision(directory / "eot_t0_seed_1_rel_reg_1.npz", args)

    # a method that does have an axis still gets the protection
    args_with_axis = Namespace(
        perplexity=30.0,
        backend="umap",
        task_kind="ot",
        n_neighbors=None,
        methods=(),
        output_dir=directory,
    )
    with pytest.raises(ValueError, match="already holds"):
        assert_no_parameter_collision(directory / "ot_tsne_seed_0.npz", args_with_axis)


@pytest.mark.parametrize(
    ("task_kind", "backend", "expected"),
    [
        ("cca-50", "umap", "cca-50-umap"),
        # t-SNE is spelled out too. Leaving it unmarked would make `cca-50` mean both the base
        # identity and one particular run of it, which is the implicit-default thinking that
        # produced the original mislabelling. Every stored file agrees: method `cca-50-tsne`.
        ("cca-50", "tsne", "cca-50-tsne"),
        ("ot", "umap", "ot-umap"),
        ("eot-t0", NO_BACKEND, "eot-t0"),
        ("multimap", NO_BACKEND, "multimap"),
    ],
)
def test_metadata_records_what_actually_ran(task_kind, backend, expected):
    """A UMAP run of ``cca-50`` is a ``cca-50-umap`` run, and must say so.

    The filename, ``backend`` and ``embedding_backend`` all agreed on this already; ``task_kind``
    alone dissented, so a stored file contradicted itself. ``task_kind`` stays backend-neutral --
    it is the lookup key the aggregate and every reader index by -- and ``method`` carries what ran.
    """
    from argparse import Namespace

    import numpy as np

    from ne_aligned_utils.benchmark.data import BenchmarkData
    from ne_aligned_utils.benchmark.sweep import common_metadata

    data = BenchmarkData(
        "nCoV",
        np.zeros((4, 3)),
        np.zeros((3, 3)),
        None,
        None,
        (),
        ("a", "b"),
        np.arange(4),
        np.arange(3),
    )
    args = Namespace(
        backend=backend,
        task_kind=task_kind,
        seed=0,
        embedding_dim=2,
        n_epochs=200,
        pca_components=50,
        perplexity=30.0,
        max_cells_per_batch=None,
        max_cells_batch0=None,
        max_cells_batch1=None,
        n_neighbors=None,
        init="random",
        methods=(),
        output_dir="/tmp/x",
    )
    metadata = common_metadata(data, args, task_kind=task_kind)
    assert metadata["method"] == expected
    assert metadata["task_kind"] == task_kind
