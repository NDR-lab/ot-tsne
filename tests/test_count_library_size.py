"""All-gene library sizes for the count-normalised methods (seurat-*, liger-*, ref)."""

import argparse

import h5py
import numpy as np
import pytest

from ne_aligned_utils.benchmark.data import BenchmarkData, all_gene_library_sizes
from ne_aligned_utils.benchmark.sweep.tasks import _count_library_sizes
from ne_aligned_utils.methods.direct_integration_methods import (
    compute_policar_ref_tsne_embedding_from_arrays,
)
from ne_aligned_utils.methods.external.r import write_r_input


def _views(seed=0, n0=40, n1=50, n_selected=30, n_other=200):
    """Processed and raw-count views built the way preprocessing builds them."""
    rng = np.random.default_rng(seed)
    full = rng.poisson(1.5, size=(n0 + n1, n_selected + n_other)).astype(float)
    full[:, :n_selected] *= rng.integers(1, 6, size=(n0 + n1, 1))  # cell-dependent share
    totals = full.sum(axis=1)
    counts = full[:, :n_selected]
    processed = np.log1p(counts / totals[:, None] * 1e4)

    def view(x):
        return BenchmarkData(
            dataset="toy",
            x0=x[:n0],
            x1=x[n0:],
            labels0=None,
            labels1=None,
            label_names=(),
            dataset_names=("a", "b"),
            sample_indices0=np.arange(n0),
            sample_indices1=np.arange(n1),
        )

    return view(processed), view(counts), totals


def test_all_gene_library_sizes_recovers_preprocessing_totals():
    processed, counts, totals = _views()
    sizes0, sizes1 = all_gene_library_sizes(processed, counts)
    np.testing.assert_allclose(np.concatenate([sizes0, sizes1]), totals, rtol=1e-12)


def test_all_gene_library_sizes_rejects_mismatched_views():
    processed, counts, _ = _views()
    shuffled = BenchmarkData(**{**counts.__dict__, "x0": counts.x0[::-1]})
    with pytest.raises(ValueError):
        all_gene_library_sizes(processed, shuffled)


def test_selected_genes_mode_supplies_no_library_sizes():
    _, counts, _ = _views()
    args = argparse.Namespace(count_library_size="selected_genes")
    assert _count_library_sizes(args, counts) is None


def test_r_input_carries_library_sizes(tmp_path):
    x0, x1 = np.ones((3, 4)), np.ones((2, 4))
    path = tmp_path / "input.h5"
    batch_ids = np.array([0, 0, 0, 1, 1])
    write_r_input(path, x0, x1, batch_ids, {"dim": 2}, library_sizes=np.arange(1, 6))
    with h5py.File(path) as handle:
        np.testing.assert_array_equal(handle["library_size"][()], np.arange(1, 6))
    write_r_input(path, x0, x1, batch_ids, {"dim": 2})
    with h5py.File(path) as handle:
        assert "library_size" not in handle


def test_ref_tsne_own_totals_reproduce_default_and_all_gene_totals_differ():
    _, counts, totals = _views(n0=120, n1=60)
    kwargs = dict(random_state=0, return_artifacts=True)
    base, base_artifacts = compute_policar_ref_tsne_embedding_from_arrays(
        counts.x0, counts.x1, **kwargs
    )
    same, _ = compute_policar_ref_tsne_embedding_from_arrays(
        counts.x0, counts.x1, reference_library_sizes=counts.x0.sum(axis=1), **kwargs
    )
    new, new_artifacts = compute_policar_ref_tsne_embedding_from_arrays(
        counts.x0, counts.x1, reference_library_sizes=totals[:120], **kwargs
    )
    np.testing.assert_array_equal(base, same)
    assert not np.array_equal(base, new)
    assert base_artifacts["library_size_source"] == "selected_genes"
    assert new_artifacts["library_size_source"] == "all_genes"
