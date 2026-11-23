"""An OT-tSNE initialization is only reported when the layouts actually started from it."""

import numpy as np
import pytest

import ne_aligned_utils.benchmark.scoring as benchmark_metrics
from ne_aligned_utils.benchmark.sweep.aggregation import _check_init_used, cache_keys


def _stem(method):
    return cache_keys(method)[0].removesuffix("_embeddings")


def test_aggregate_refuses_tasks_that_started_from_another_init():
    stem = _stem("ot")
    _check_init_used({f"{stem}_init_used": "spectral"}, "ot", "tsne", "spectral")
    with pytest.raises(ValueError, match="started from init 'random'"):
        _check_init_used({f"{stem}_init_used": "random"}, "ot", "tsne", "spectral")
    # eot runs made before the label was recorded say "array"; they must be rerun.
    with pytest.raises(ValueError, match="'array'"):
        _check_init_used({f"{stem}_init_used": "array"}, "ot", "tsne", "eot")
    # Task files without the record predate it and ran with random coordinates.
    _check_init_used({}, "ot", "tsne", "random")
    with pytest.raises(ValueError, match="started from init 'random'"):
        _check_init_used({}, "ot", "tsne", "pca")


def _ot_cache(init, init_used):
    stem = _stem("ot")
    return {
        "dataset": np.asarray("toy"),
        "methods": np.asarray(["ot"]),
        "seeds": np.asarray([0]),
        "rel_reg": np.asarray([1.0]),
        "effective_ks": np.asarray([15]),
        "init": np.asarray(init),
        f"{stem}_init_used": np.asarray(init_used),
        "labels_available": np.asarray(True),
        "batch_ids": np.asarray([0, 0, 1, 1]),
        "labels0": np.asarray([[0, 1]]),
        "labels1": np.asarray([[0, 1]]),
        f"{stem}_embeddings": np.zeros((1, 1, 1, 4, 2)) + np.arange(8).reshape(4, 2),
    }


def test_scored_rows_carry_init_used_and_refuse_a_mismatch(monkeypatch):
    monkeypatch.setattr(benchmark_metrics, "compute_neighbor_purity", lambda *_a, **_k: 0.5)
    rows = benchmark_metrics.score_aggregate_cache(
        _ot_cache("pca", "pca"), metric_names=("neighborhood_purity_score",)
    )
    assert rows["init"].tolist() == ["pca"]
    assert rows["init_used"].tolist() == ["pca"]

    with pytest.raises(ValueError, match="started from 'random'"):
        benchmark_metrics.score_aggregate_cache(
            _ot_cache("spectral", "random"), metric_names=("neighborhood_purity_score",)
        )


def test_init_jitter_zero_gives_the_same_init_for_every_seed():
    import numpy as np
    import pytest

    from ne_aligned_utils.methods import _finalize_tsne_init, current_init_jitter, use_init_jitter

    coords = np.random.RandomState(3).randn(50, 2)
    jittered = [_finalize_tsne_init(coords, random_state=s) for s in (0, 1)]
    assert not np.array_equal(*jittered)
    with use_init_jitter(0.0):
        assert current_init_jitter() == 0.0
        plain = [_finalize_tsne_init(coords, random_state=s) for s in (0, 1)]
    np.testing.assert_array_equal(*plain)
    assert current_init_jitter() == 0.01
    with pytest.raises(ValueError), use_init_jitter(-1.0):
        pass
