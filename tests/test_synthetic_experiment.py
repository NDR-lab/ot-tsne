"""Scientific invariants and strict artifact resume/load behavior for the walkthrough."""

import ast
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from ne_aligned_utils import methods
from ne_aligned_utils.evaluation.metrics import compute_kbet, compute_neighbor_purity
from ne_aligned_utils.experiments import synthetic
from ne_aligned_utils.methods.external import multimap as multimap_backend

NOTEBOOK = Path(__file__).parents[1] / "notebooks/mnist_orthogonal_batch_effect.ipynb"


def recorded_construction(directory, n_per_digit):
    """Small standalone artifact matching the notebooks' recorded-construction input."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "test_source_construction.npz"
    rng = np.random.default_rng(817)
    labels = np.repeat(np.arange(10), n_per_digit)
    clean0 = rng.random((len(labels), 784))
    clean1 = rng.random((len(labels), 784))
    pca = PCA(n_components=None, svd_solver="full").fit(np.vstack([clean0, clean1]))
    np.savez_compressed(
        path,
        clean0=clean0,
        clean1=clean1,
        labels0=labels,
        labels1=labels,
        batch0_indices=np.arange(len(labels)),
        batch1_indices=np.arange(len(labels), 2 * len(labels)),
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_variance=pca.explained_variance_,
    )
    return path


def run_stage(experiment, stage):
    """Exercise maintained notebook cells rather than retaining a second implementation."""
    cells = [
        "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
        if cell["cell_type"] == "code"
    ]
    markers = {
        "batches": "    with np.load(source_construction,",
        "subspaces": "    experiment.x_images =",
        "corruptions": "    experiment.lead_basis =",
        "cost": "    (cost_records, experiment.reference_plans) =",
        "affinity": "    DIAGNOSTIC_MAGNITUDES =",
        "embedding": "    records, embedding_plan_cache =",
        "summary": '    if "preprocessing" not in experiment.results:',
    }
    namespace = getattr(experiment, "_notebook_namespace", None)
    if namespace is None:
        namespace = dict(
            vars(synthetic),
            np=np,
            pd=pd,
            json=json,
            PCA=PCA,
            partial=partial,
            display=lambda *args: None,
            RUN_EXPERIMENT=True,
            BUILD_FROM_MNIST=False,
            experiment=experiment,
            paths=experiment.paths,
            source_construction=recorded_construction(
                experiment.paths.data_root, experiment.config.data.n_per_digit
            ),
            SMOKE=False,
            TSNE_N_JOBS=1,
            compute_tsne_embedding_from_arrays=lambda *a, **k: np.zeros(
                (len(experiment.labels), 2)
            ),
        )
        experiment._notebook_namespace = namespace
        for helper in ("def full_plan_record(", "def edge_overlap_record("):
            exec(next(cell for cell in cells if helper in cell), namespace)
    namespace.update(
        config=experiment.config,
        compute_neighbor_purity=compute_neighbor_purity,
        compute_kbet=compute_kbet,
        compute_ot_tsne_embedding_from_plan=methods.compute_ot_tsne_embedding_from_plan,
        compute_scanorama_then_tsne_embedding_from_arrays=methods.compute_scanorama_then_tsne_embedding_from_arrays,
        compute_multimap_embedding_from_arrays=multimap_backend.compute_multimap_embedding_from_arrays,
    )
    if stage == "corruptions":
        exec(
            next(cell for cell in cells if cell.startswith("def make_class_corruptions(")),
            namespace,
        )
    if stage == "embedding":
        for name in (
            "score_embedding",
            "ot_embeddings",
            "scanorama_embeddings",
            "multimap_embeddings",
        ):
            exec(next(cell for cell in cells if cell.startswith(f"def {name}(")), namespace)
    cell = next(
        cell for cell in cells if cell.startswith("if RUN_EXPERIMENT:") and markers[stage] in cell
    )
    tree = ast.parse(cell)
    if stage == "embedding":
        # This test covers metric/cache orchestration; gallery fitting is independently validated.
        class SkipGallery(ast.NodeTransformer):
            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                    if node.value.func.attr == "prepare_paper_gallery":
                        return None
                return node

        tree = SkipGallery().visit(tree)
    exec(compile(tree, f"{NOTEBOOK.name}:{stage}", "exec"), namespace)
    return namespace.get("counts") if stage == "batches" else None


@pytest.fixture
def experiment(tmp_path):
    config = synthetic.ExperimentConfig(
        data=synthetic.DataConfig(n_per_digit=8, signal_rank=10, tail_pool_size=5),
        corruption=synthetic.CorruptionConfig(magnitudes=(1.0, 10.0), corruption_seeds=(0,)),
        ot=synthetic.OtConfig(rel_reg=(1.0,)),
        competitors=synthetic.CompetitorsConfig(
            cross_batch_k=5, run_multimap=False, run_scanorama_50_tsne=False
        ),
        embedding=synthetic.EmbeddingConfig(embedding_seeds=(0,), n_epochs=5),
        evaluation=synthetic.EvaluationConfig(kbet_k=5),
    )
    return synthetic.SyntheticExperiment(
        config, synthetic.ExperimentPaths(tmp_path, tmp_path / "data")
    )


def test_construction_cost_identity_and_independent_scale_streams(experiment):
    e = experiment
    counts = run_stage(e, "batches")
    assert counts.eq(8).all().all()
    assert not np.intersect1d(e.batch0_indices, e.batch1_indices).size
    run_stage(e, "subspaces")
    run_stage(e, "corruptions")
    audit = e.audit.pivot(index="class", columns="batch", values="configured_scale")
    np.testing.assert_array_equal(np.sort(audit[0]), np.sort(audit[1]))
    assert not np.array_equal(audit[0], audit[1])
    e.initialize_run()
    run_stage(e, "cost")
    e.verify_costs()
    assert e.cost_diagnostics.decomposition_identity_error_max_abs.max() < 1e-9
    # Original data retain a real clean/nuisance interaction; the projected control removes it.
    assert e.cost_diagnostics.query("mode == 'original'").signal_interaction_rms.max() > 0.01
    assert e.cost_diagnostics.query("mode == 'projected'").signal_interaction_rms.max() < 1e-10
    run_stage(e, "affinity")
    assert e.knn_diagnostics_table.mutual_nonempty_row_fraction.between(0, 1).all()
    assert e.affinity_diagnostics.eot_row_nonempty_row_fraction.eq(1).all()


def test_resume_metadata_and_load_only_never_run_methods(experiment, monkeypatch):
    e = experiment
    run_stage(e, "batches")
    run_stage(e, "subspaces")
    run_stage(e, "corruptions")
    e.initialize_run()
    run_stage(e, "cost")
    e.save_stage("cost")
    run_stage(e, "affinity")
    e.save_stage("affinity")
    # Deterministic coordinates isolate cache/load behavior from stochastic optimization.
    coordinates = np.random.default_rng(1).normal(size=(160, 2))
    monkeypatch.setattr(methods, "compute_ot_tsne_embedding_from_plan", lambda *a, **k: coordinates)
    run_stage(e, "embedding")
    e.save_stage("embedding")
    run_stage(e, "summary")
    e.save_stage("summary")

    def forbidden(*args, **kwargs):
        raise AssertionError("load/resume attempted recomputation")

    monkeypatch.setattr(methods, "compute_ot_tsne_embedding_from_plan", forbidden)
    run_stage(e, "embedding")  # compatible cache must be reused
    monkeypatch.setattr(synthetic, "compute_ot_plan", forbidden)
    monkeypatch.setattr(synthetic, "MNIST", forbidden)
    loaded = synthetic.SyntheticExperiment(e.config, e.paths).load_saved()
    pd.testing.assert_frame_equal(loaded.summary, e.summary, check_dtype=False)
    assert set(loaded.summary.method_id) == {
        "ot_tsne__fixed_epsilon__rel_reg_1__k_15",
        "ot_tsne__adaptive_epsilon__rel_reg_1__k_15",
    }
    wrong = replace(e.config, evaluation=synthetic.EvaluationConfig(kbet_k=6))
    explicit = replace(e.paths, run_name=e.run_dir.name)
    with pytest.raises(ValueError, match="configuration"):
        synthetic.SyntheticExperiment(wrong, explicit).load_saved()
    cache = next(e.run_dir.rglob("*.npz"))
    if cache.name == "construction.npz":
        cache = next((e.run_dir / "original/embeddings").rglob("*.npz"))
    cache.with_suffix(".json").write_text("{}")
    with pytest.raises(ValueError, match="metadata"):
        e.load_or_compute(cache, forbidden, {})


def test_topk_support_coverage_and_names(experiment):
    e = experiment
    plan = np.array([[9.0, 3.0, 1.0], [8.0, 7.0, 2.0]])
    rows = e.rectangular_topk_support(plan, 1, axis=1)
    columns = e.rectangular_topk_support(plan, 1, axis=0)
    np.testing.assert_array_equal(rows & columns, [[True, False, False], [False, False, False]])
    purity, coverage, _ = e.support_stats(rows & columns, np.array([0, 1]), np.array([0, 1, 2]))
    assert purity == 1 and coverage == 0.5
    assert e.run_dir == synthetic.run_directory(e.config, e.paths)
    assert "mnist_orthogonal_batch_effect__seed0__" in e.run_dir.name
    with pytest.raises(ValueError, match="directory name"):
        synthetic.run_directory(e.config, replace(e.paths, run_name="../escape"))
    with pytest.raises(ValueError, match="independent"):
        synthetic.SyntheticExperiment(
            replace(
                e.config, corruption=replace(e.config.corruption, batch1_scale_seed_offset=20000)
            ),
            e.paths,
        )


@pytest.mark.parametrize("profile,overrides", [("default", {}), ("k15", {"n_neighbors": 15})])
def test_multimap_profiles_only_override_neighbors(experiment, monkeypatch, profile, overrides):
    """The inline notebook must leave all other MultiMAP settings to the wrapper."""
    import json
    from functools import partial
    from pathlib import Path

    from ne_aligned_utils.paper.inputs.synthetic import GALLERY_FILES, gallery_path

    e = experiment
    config = replace(
        e.config,
        competitors=replace(e.config.competitors, run_multimap=True, multimap_profile=profile),
    )
    e = synthetic.SyntheticExperiment(config, e.paths)
    run_stage(e, "batches")
    run_stage(e, "subspaces")
    run_stage(e, "corruptions")
    e.initialize_run()
    coordinates = np.random.default_rng(10).normal(size=(len(e.labels), 2))
    calls = []

    def multimap(x0, x1, **kwargs):
        calls.append(kwargs)
        return coordinates

    monkeypatch.setattr(multimap_backend, "compute_multimap_embedding_from_arrays", multimap)
    monkeypatch.setattr(methods, "compute_ot_tsne_embedding_from_plan", lambda *a, **k: coordinates)
    run_stage(e, "embedding")
    assert calls and all(call == {"random_state": 0, **overrides} for call in calls)
    frame = e.results[e.results.method.eq("multimap")]
    expected_label = "MultiMAP" if profile == "default" else "MultiMAP k=15"
    assert set(frame.display_method) == {expected_label}
    method_id = synthetic.embedding_method_id(
        "multimap", "native-defaults", k=overrides.get("n_neighbors")
    )
    assert set(frame.method_id) == {method_id}
    from ne_aligned_utils.paper.figures.synthetic import select_downstream

    scanorama = frame.iloc[[0]].assign(
        method="scanorama-50-tsne",
        method_id="scanorama_50_tsne",
        display_method="Scanorama → t-SNE",
    )
    selected = select_downstream(pd.concat([e.results, scanorama], ignore_index=True))
    assert set(selected.loc[selected.method.eq("multimap"), "method_id"]) == {method_id}
    path = gallery_path(e.mode_dirs["original"], GALLERY_FILES[1])
    assert path == e.cache_path(
        "original", 10, "multimap", "native-defaults", 0, 0, k=overrides.get("n_neighbors")
    )

    # Exercise the actual inline notebook function, with a new seed to avoid cached coordinates.
    notebook = json.loads(
        (Path(__file__).parents[1] / "notebooks/mnist_orthogonal_batch_effect.ipynb").read_text()
    )
    cell = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if "".join(c["source"]).startswith("def multimap_embeddings(")
    )
    namespace = dict(
        config=e.config,
        experiment=e,
        partial=partial,
        compute_multimap_embedding_from_arrays=multimap,
        score_embedding=lambda meta, *args: meta,
    )
    exec(compile(cell, "inline-multimap", "exec"), namespace)
    calls.clear()
    x0, x1 = e.make_batches("original", 10, 0)
    namespace["multimap_embeddings"](x0, x1, "original", 10, 0, 123)
    assert calls == [{"random_state": 123, **overrides}]


def test_multimap_profile_rejects_extra_configuration(experiment):
    invalid = replace(
        experiment.config,
        competitors=replace(experiment.config.competitors, multimap_profile="k30"),
    )
    with pytest.raises(ValueError, match="multimap_profile"):
        synthetic.validate_config(invalid)


def _construction(seed=3):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(10), 4)
    clean0, clean1 = rng.random((40, 20)), rng.random((40, 20))
    pca = PCA(n_components=None, svd_solver="full").fit(np.vstack([clean0, clean1]))
    return dict(
        batch0_indices=np.arange(40),
        batch1_indices=np.arange(40, 80),
        labels0=labels,
        labels1=labels,
        clean0=clean0,
        clean1=clean1,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_variance=pca.explained_variance_,
    )


def test_compare_construction_accepts_identical_and_sign_flipped_pca():
    recorded = _construction()
    assert synthetic.compare_construction(dict(recorded), recorded) == "identical"
    flipped = dict(recorded, pca_components=-recorded["pca_components"])
    assert synthetic.compare_construction(flipped, recorded) == "pca within rounding"


def test_compare_construction_rejects_other_samples_and_other_pca():
    recorded = _construction()
    other_images = dict(recorded, clean0=recorded["clean0"] + 1e-12)
    with pytest.raises(ValueError, match="clean0"):
        synthetic.compare_construction(other_images, recorded)
    other_pca = dict(recorded, pca_mean=recorded["pca_mean"] + 1e-3)
    with pytest.raises(ValueError, match="PCA"):
        synthetic.compare_construction(other_pca, recorded)


@pytest.mark.parametrize("experiment", ["orthogonal", "parallel"])
def test_recorded_mnist_construction_is_rebuilt_from_mnist(experiment):
    root = Path(__file__).resolve().parents[1]
    mnist = root / "data/torchvision"
    if not (mnist / "MNIST/raw").is_dir():
        pytest.skip("MNIST is not in data/torchvision; the notebooks download it on request")
    source = root / "data/mnist" / experiment
    data = json.loads((source / "config.json").read_text())["config"]["data"]
    built = synthetic.build_mnist_construction(mnist, data["n_per_digit"], data["random_seed"])
    with np.load(source / "data/construction.npz", allow_pickle=False) as recorded:
        assert synthetic.compare_construction(built, dict(recorded)) in {
            "identical",
            "pca within rounding",
        }
