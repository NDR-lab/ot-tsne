"""Exercise the parallel notebook's actual scientific cells and shared artifact interfaces."""

import hashlib
import json
import subprocess
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from test_synthetic_experiment import recorded_construction

from ne_aligned_utils.evaluation.metrics import compute_kbet, compute_neighbor_purity
from ne_aligned_utils.experiments.parallel import (
    ParallelConfig,
    ParallelExperiment,
    load_parallel_run,
)
from ne_aligned_utils.experiments.synthetic import ExperimentPaths, label_methods

NOTEBOOK = Path(__file__).parents[1] / "notebooks/mnist_parallel_batch_effect.ipynb"


def stage(namespace, tag):
    notebook = json.loads(NOTEBOOK.read_text())
    cell = next(c for c in notebook["cells"] if tag in c.get("metadata", {}).get("tags", []))
    exec(compile("".join(cell["source"]), f"parallel:{tag}", "exec"), namespace)


@pytest.fixture
def setup(tmp_path):
    config = ParallelConfig()
    config = replace(
        config,
        data=replace(config.data, n_per_digit=8),
        corruption=replace(config.corruption, magnitudes=(1.0, 4.0), corruption_seeds=(0, 1)),
        ot=replace(config.ot, rel_reg=(1.0,), ot_effective_k=5),
        competitors=replace(config.competitors, cross_batch_k=5, run_scanorama_50_tsne=False),
        embedding=replace(config.embedding, embedding_seeds=(0,), n_epochs=5),
        evaluation=replace(config.evaluation, kbet_k=5),
    )
    audit = pd.DataFrame(
        [
            dict(
                mode=mode,
                corruption_seed=seed,
                batch=batch,
                digit=digit,
                class_multiplier=weight,
                batch_sign=sign,
                pc1_coefficient=sign * weight,
            )
            for seed in config.corruption.corruption_seeds
            for mode in config.data.modes
            for batch, sign in enumerate(config.corruption.batch_multipliers)
            for digit, weight in enumerate(
                np.ones(10)
                if mode == "constant"
                else np.random.default_rng(seed).uniform(*config.corruption.multiplier_range, 10)
            )
        ]
    )
    paths = ExperimentPaths(tmp_path, tmp_path / "data")
    e = ParallelExperiment(config, paths)
    ns = dict(
        config=config,
        experiment=e,
        paths=paths,
        RUN_EXPERIMENT=True,
        BUILD_FROM_MNIST=False,
        source_construction=recorded_construction(paths.data_root, config.data.n_per_digit),
        SMOKE=True,
        source_audit=audit,
        source_manifest={"variant": "test_fixture"},
        SOURCE_RUN_DIR=tmp_path,
        ROOT=Path(__file__).parents[1],
        benchmark_settings={},
        hashlib=hashlib,
        subprocess=subprocess,
        TSNE_N_JOBS=1,
        np=np,
        pd=pd,
        json=json,
        PCA=PCA,
        display=lambda *args: None,
        partial=partial,
        label_methods=label_methods,
        compute_neighbor_purity=compute_neighbor_purity,
        compute_kbet=compute_kbet,
    )
    cells = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"]]
    for helper in ("def full_plan_record(", "def edge_overlap_record("):
        exec(next(cell for cell in cells if helper in cell), ns)
    return ns


def construct(ns):
    for tag in ("batches", "pca", "corruption"):
        stage(ns, tag)
    return ns["experiment"]


def test_opposite_pc1_shifts_balance_and_nonseparable_heterogeneity(setup):
    e = construct(setup)
    assert not np.intersect1d(e.batch0_indices, e.batch1_indices).size
    for labels in (e.labels0, e.labels1):
        np.testing.assert_array_equal(np.bincount(labels), np.full(10, 8))
    for mode in e.config.data.modes:
        x0, x1 = e.make_batches(mode, 4.0, 0)
        weights = e.class_multipliers[mode, 0]
        np.testing.assert_allclose(
            x0 - e.x0_images, -4 * weights[e.labels0, None] * e.pc1, atol=1e-15
        )
        np.testing.assert_allclose(
            x1 - e.x1_images, 4 * weights[e.labels1, None] * e.pc1, atol=1e-15
        )
    np.testing.assert_array_equal(e.class_multipliers["constant", 0], np.ones(10))
    np.testing.assert_array_equal(
        e.class_multipliers["heterogeneous", 0], np.random.default_rng(0).uniform(-2, 2, 10)
    )
    assert not np.array_equal(
        e.class_multipliers["heterogeneous", 0], e.class_multipliers["heterogeneous", 1]
    )
    stage(setup, "cost_terms")
    stage(setup, "cost")
    cost = e.cost_diagnostics
    assert cost.decomposition_identity_error_max_abs.max() < 1e-9
    assert cost.query("mode == 'heterogeneous'").nonseparable_cost_rms.max() > 0.01
    stage(setup, "affinity")
    assert set(e.affinity_diagnostics["mode"]) == {"constant", "heterogeneous"}
    assert set(e.knn_diagnostics_table["mode"]) == {"constant", "heterogeneous"}
    assert e.knn_diagnostics_table.mutual_nonempty_row_fraction.between(0, 1).all()


@pytest.mark.parametrize("profile", ["default", "k15"])
def test_both_multimap_variants_and_strict_saved_run(setup, profile):
    config = replace(
        setup["config"], competitors=replace(setup["config"].competitors, multimap_profile=profile)
    )
    setup["config"] = config
    setup["experiment"] = ParallelExperiment(config, setup["paths"])
    e = construct(setup)
    stage(setup, "cost_terms")
    stage(setup, "cost")
    stage(setup, "affinity")
    coordinates = np.random.default_rng(8).normal(size=(len(e.labels), 2))
    calls = []

    def multimap(x0, x1, **kwargs):
        calls.append(kwargs)
        return coordinates

    setup.update(
        compute_multimap_embedding_from_arrays=multimap,
        compute_ot_tsne_embedding_from_plan=lambda *args, **kw: coordinates,
        prepare_joint_gallery=lambda: None,
    )
    for tag in ("score_embedding", "ot_embeddings", "scanorama_embeddings", "multimap_embeddings"):
        stage(setup, tag)
    stage(setup, "embedding_grid")
    stage(setup, "aggregation")
    expected = {} if profile == "default" else {"n_neighbors": 15}
    assert {json.dumps(call, sort_keys=True) for call in calls} == {
        json.dumps({"random_state": 0, **expected}, sort_keys=True),
        json.dumps(
            {
                "random_state": 0,
                **expected,
                "scale": False,
                "center_batches": False,
                "use_native_scaling": False,
            },
            sort_keys=True,
        ),
    }
    loaded = e.load_saved()
    pd.testing.assert_frame_equal(loaded.summary, e.summary, check_dtype=False)
    assert set(loaded.summary["mode"]) == {"constant", "heterogeneous"}
    assert loaded.summary.query("method == 'multimap'").method_id.nunique() == 2
    assert set(loaded.summary.query("method == 'multimap'").preprocessing) == {
        "centered",
        "uncentered",
    }
    with pytest.raises(ValueError, match="configuration"):
        load_parallel_run(e.run_dir, expected_config={})
    wrong = replace(config, corruption=replace(config.corruption, multiplier_range=(-1.0, 1.0)))
    assert ParallelExperiment(wrong, e.paths).run_dir != e.run_dir

    def forbidden(*args, **kwargs):
        raise AssertionError("compatible cache must not recompute")

    setup["compute_multimap_embedding_from_arrays"] = forbidden
    stage(setup, "embedding_grid")
    assert load_parallel_run(e.run_dir).manifest["variant"] == "parallel_pc1"

    # The canonical builder and CLI can render both modes solely from saved artifacts.
    import matplotlib.pyplot as plt

    from ne_aligned_utils.paper.figures.parallel_synthetic import (
        ParallelPlotConfig,
        build_parallel_figures,
    )

    for mode in config.data.modes:
        path = e.cache_path(mode, 4.0, "joint-tsne", "tsne", 0, 0)
        e.load_or_compute(path, lambda: coordinates, {"method": "joint-tsne"})
    output = e.run_dir / "figures"
    figures = build_parallel_figures(
        e.run_dir, output_dir=output, selection=ParallelPlotConfig(magnitude=4.0)
    )
    assert len(list(output.glob("*.png"))) == 4
    assert len(figures["cross_batch_affinities"].axes) == 4
    assert len(figures["embedding_metrics"].axes) == 4
    assert len(figures["embeddings"].axes) == 16  # 2 modes × 2 colorings × 4 methods
    for fig in figures.values():
        plt.close(fig)
