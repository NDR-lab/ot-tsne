"""Methods figures consume saved stages and preserve the pre-migration export."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ne_aligned_utils.paper.figures.methods import MethodsComposer, build_methods, preflight_methods
from ne_aligned_utils.paper.inputs.methods import ARRAY_NAMES, load_method_arrays


@pytest.fixture
def array_path(tmp_path):
    rng = np.random.default_rng(1)
    source = tmp_path / "inputs"
    source.mkdir()
    path = source / "arrays.npz"
    np.savez_compressed(
        path,
        x0=rng.normal(size=(8, 3)),
        x1=rng.normal(size=(4, 3)),
        plan=rng.uniform(0.1, 1, size=(8, 4)),
        affinity=rng.uniform(0.1, 1, size=(12, 12)),
        sparse=np.eye(12),
        embedding=rng.normal(size=(12, 2)),
    )
    return path


def test_composer_does_not_compute_or_mutate_arrays_and_restores_style(array_path, monkeypatch):
    import ne_aligned_utils.methods as methods

    def forbidden(*args, **kwargs):
        pytest.fail("rendering recomputed a method stage")

    monkeypatch.setattr(methods, "compute_ot_plan", forbidden)
    monkeypatch.setattr(methods, "compute_ot_tsne_embedding_from_plan", forbidden)
    arrays = load_method_arrays(array_path)
    before = {key: value.copy() for key, value in arrays.items()}
    rc = dict(plt.rcParams)
    fig = MethodsComposer().compose(*(arrays[name] for name in ARRAY_NAMES))
    try:
        assert dict(plt.rcParams) == rc
        fig.canvas.draw()
        assert len(fig.axes) == 6
        # The original set_layout overrides the initial subplot size.
        assert tuple(fig.get_size_inches()) == (16.0, 4.0)
        for name in ARRAY_NAMES:
            assert np.array_equal(arrays[name], before[name])
    finally:
        plt.close(fig)


def test_preflight_is_read_only_and_protects_reference_directory(array_path, tmp_path):
    output = tmp_path / "output"
    prepared = preflight_methods(array_path, output)
    assert not output.exists()
    assert prepared.source == array_path
    with pytest.raises(ValueError, match="overlap"):
        preflight_methods(array_path, array_path.parent / "plots")
    assert not (array_path.parent / "plots").exists()


def test_composer_uses_explicit_canvas_without_global_backend_creation(array_path, monkeypatch):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    def forbidden(*args, **kwargs):
        pytest.fail("composer delegated canvas creation to the application's pyplot backend")

    monkeypatch.setattr(plt, "subplots", forbidden)
    monkeypatch.setattr(plt, "figure", forbidden)
    backend = matplotlib.get_backend()
    arrays = load_method_arrays(array_path)
    fig = MethodsComposer().compose(*(arrays[name] for name in ARRAY_NAMES))
    assert isinstance(fig.canvas, FigureCanvasAgg)
    assert fig.dpi == 100
    assert matplotlib.get_backend() == backend


def test_invalid_saved_stages_fail_without_computation(tmp_path):
    path = tmp_path / "missing.npz"
    np.savez_compressed(path, x0=np.zeros((1, 2)))
    with pytest.raises(ValueError, match="missing"):
        load_method_arrays(path)


def test_changed_arrays_fail_before_writing(array_path, tmp_path):
    prepared = preflight_methods(array_path, tmp_path / "output")
    array_path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed after preflight"):
        build_methods(prepared)
    assert not prepared.output.exists()


def test_writer_failure_restores_metadata_epoch_without_success_manifest(
    array_path, tmp_path, monkeypatch
):
    from ne_aligned_utils.paper.figures import methods as methods_reporting

    prepared = preflight_methods(array_path, tmp_path / "output")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "123")

    def fail(fig, **kwargs):
        plt.close(fig)
        raise RuntimeError("save failed")

    monkeypatch.setattr(methods_reporting, "save_method_figure", fail)
    with pytest.raises(RuntimeError, match="save failed"):
        build_methods(prepared)
    import os

    assert os.environ["SOURCE_DATE_EPOCH"] == "123"
    assert not (prepared.output / "figure-manifest.json").exists()


def test_notebook_keeps_the_original_numerical_cell_and_delegates_drawing():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/method_figure.ipynb").read_text())
    baseline = json.loads(
        (root / "tests" / "fixtures" / "figures" / "METHODS_FIGURE_BASELINE.json").read_text()
    )
    source = "".join(notebook["cells"][2]["source"])
    assert hashlib.sha256(source.encode()).hexdigest() == baseline["numerical_cell_sha256"]
    assert "".join(notebook["cells"][3]["source"]).strip() == (
        "from ne_aligned_utils.paper.figures.methods import create_figure"
    )
