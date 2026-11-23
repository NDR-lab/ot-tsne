"""Synthetic plotting reads explicit saved stages without running benchmark methods."""

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ne_aligned_utils.paper.inputs import synthetic


def _source():
    source = os.environ.get("NE_ALIGNED_SYNTHETIC_SOURCE")
    if source is None:
        pytest.skip("set NE_ALIGNED_SYNTHETIC_SOURCE to the checkout holding the synthetic runs")
    root = Path(source)
    if not (root / synthetic.DEFAULT_RUNS["two_sided_independent_scales"]["original"]).exists():
        pytest.skip("exact local synthetic source inputs are unavailable")
    return root


def test_import_does_not_create_output_directories(monkeypatch):
    from ne_aligned_utils.paper.figures import synthetic as figures

    def forbidden(*args, **kwargs):
        raise AssertionError("import attempted to create an output directory")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    importlib.reload(figures)


@pytest.mark.parametrize("relation", ("inside", "ancestor"))
def test_output_cannot_overlap_the_source_checkout(tmp_path, relation):
    source = tmp_path / "source"
    output = source / "output" if relation == "inside" else tmp_path
    with pytest.raises(ValueError, match="overlap"):
        synthetic.preflight("one_sided", source_root=source, output_dir=output)


def test_saved_coordinates_are_required_and_validated(tmp_path):
    missing = tmp_path / "missing.npz"
    with pytest.raises(FileNotFoundError):
        synthetic.load_coordinates(missing, 2)
    np.savez(missing, embedding=np.ones((3, 2)))
    with pytest.raises(ValueError, match="coordinates"):
        synthetic.load_coordinates(missing, 2)
    np.savez(missing, embedding=np.array([[np.nan, 0.0], [1.0, 1.0]]))
    with pytest.raises(ValueError, match="coordinates"):
        synthetic.load_coordinates(missing, 2)


def test_duplicate_directed_cache_rows_are_rejected(tmp_path):
    runs = SimpleNamespace(
        variant="test", configs={"original": {"corruption_seeds": [0], "magnitudes": [10.0]}}
    )
    rows = []
    for magnitude in (0.0, 10.0):
        row = {
            "corruption_seed": 0,
            "magnitude": magnitude,
            "rel_reg": 1.0,
            "top_k": 15,
            "batch_sampling": synthetic.SAMPLING,
        }
        for prefix in ("row", "column", "shared_or", "mutual_and"):
            row[f"eot_{prefix}_digit_purity"] = 0.5
            row[f"eot_{prefix}_nonempty_row_fraction"] = 1.0
        rows.append(row)
    path = tmp_path / "test_directed_ot_top15.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    assert len(synthetic.load_directed(runs, tmp_path)) == 2
    pd.DataFrame([*rows, rows[0]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="diagnostics"):
        synthetic.load_directed(runs, tmp_path)


@pytest.mark.parametrize("variant", ("one_sided", "two_sided_correlated_scales"))
def test_legacy_pairs_are_not_silently_substituted_or_relabeled(tmp_path, variant):
    output = tmp_path / "figures"
    with pytest.raises(ValueError, match="legacy paired-image"):
        synthetic.preflight(variant, source_root=_source(), output_dir=output)
    assert not output.exists()


def test_one_sided_archive_scores_are_not_relabelled_as_symmetric(tmp_path):
    from scripts.figures.synthetic import main

    with pytest.raises(ValueError, match="Symmetric graph purity requires a structured run"):
        main(["--source-root", str(_source()), "--output-dir", str(tmp_path / "figures"),
              "--variant", "two_sided_independent_scales", "--dry-run"])


def test_structured_symmetric_graph_figure_rebuild(tmp_path, monkeypatch):
    from ne_aligned_utils import methods
    from scripts.figures.synthetic import main

    root = Path(__file__).resolve().parents[1]
    run = Path(os.environ.get("NE_ALIGNED_SYNTHETIC_RUN", str(root /
        ("results/synthetic_eps_ablation/benchmark-defaults-eps0-20261108-v2/"
         "mnist_orthogonal_batch_effect/mnist_orthogonal_batch_effect__seed0__c70fb5e4"))))
    if not (run / "ablation/symmetric_graph_purity/cross_batch_knn_purity_raw.csv").is_file():
        pytest.skip("saved structured symmetric diagnostics are unavailable")
    def forbidden(*args, **kwargs):
        raise AssertionError("figure rendering attempted OT or embedding fitting")
    monkeypatch.setattr(methods, "compute_ot_plan", forbidden)
    monkeypatch.setattr(methods, "compute_tsne_embedding_from_arrays", forbidden)
    source = Path(os.environ.get("NE_ALIGNED_SYNTHETIC_SOURCE", root))
    if not (source / "data/torchvision/MNIST/raw").exists():
        pytest.skip("MNIST raw data are unavailable")
    output = tmp_path / "figures"
    arguments = ["--source-root", str(source), "--output-dir", str(output),
                 "--variant", "two_sided_independent_scales", "--run-root", str(run)]
    main([*arguments, "--dry-run"])
    assert not output.exists()
    main(arguments)
    manifest = json.loads((output / "source_runs.json").read_text())
    assert manifest["methods_executed"] == []
    assert "equal-batch" in manifest["graph_purity"]
    assert any(name.endswith("cross_batch_knn_purity_raw.csv") for name in manifest["inputs"])
    assert len(manifest["outputs"]) == 5
    for name, digest in manifest["inputs"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_missing_symmetric_metrics_are_rejected(tmp_path):
    runs = synthetic.SyntheticRuns("two_sided_independent_scales",
        {"original": tmp_path / "original", "projected": tmp_path / "signal_projected"},
        {"original": {"schema_version": 1}, "projected": {"schema_version": 1}},
        *[pd.DataFrame() for _ in range(5)])
    with pytest.raises(ValueError, match="Missing symmetric graph-purity diagnostics"):
        synthetic.with_symmetric_graph_purity(runs)
