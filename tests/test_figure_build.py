"""Unified builds preflight every requested input and never overwrite results."""

import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from ne_aligned_utils.paper import build as pipeline
from ne_aligned_utils.paper.selection.benchmark import CURRENT_BENCHMARK


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    frame = pd.DataFrame(
        [
            dict(
                dataset="nCoV",
                method=key,
                method_key=key,
                method_label=key,
                family=family,
                seed=seed,
                backend="tsne",
                rel_reg=1.0,
                effective_k=15.0,
                purity=0.4 + seed * 0.1,
                silhouette=0.3,
                ilisi=0.5,
                kbet=0.6,
            )
            for key, family in (("ot", "ot"), ("multimap", "dedicated"), ("joint", "reference"))
            for seed in range(3)
        ]
    )
    frame.to_csv(root / "metrics.csv", index=False)
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return frame.copy(deep=True)

    monkeypatch.setattr(pipeline, "load_canonical_frame", load)
    return root, frame, calls


def test_metric_only_preflight_needs_no_embedding_caches_or_output(inputs, tmp_path):
    root, frame, calls = inputs
    target = tmp_path / "new"
    prepared = pipeline.preflight(root, target, datasets=("nCoV",), families=("metrics",))
    assert not target.exists()
    assert calls[0]["require_source_caches"] is False
    pd.testing.assert_frame_equal(prepared.frame, frame)
    assert prepared.summary is not None
    assert prepared.timing is None
    assert prepared.galleries == {}
    assert list(prepared.pooled["method_key"]) == ["joint", "multimap", "ot"]
    assert prepared.input_hashes[str(root / "metrics.csv")]


@pytest.mark.parametrize("datasets", ((), ("nCoV", "nCoV"), ("..",), ("a/b",)))
def test_invalid_dataset_requests_write_nothing(inputs, tmp_path, datasets):
    root, _frame, _calls = inputs
    with pytest.raises(ValueError, match="dataset"):
        pipeline.preflight(root, tmp_path / "output", datasets=datasets)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("families", ((), ("metrics", "metrics"), ("synthetic",)))
def test_only_verified_family_requests_are_accepted(inputs, tmp_path, families):
    root, _frame, _calls = inputs
    with pytest.raises(ValueError, match="families"):
        pipeline.preflight(root, tmp_path / "output", datasets=("nCoV",), families=families)


def test_output_overlap_and_existing_data_are_protected(inputs, tmp_path):
    root, _frame, _calls = inputs
    for target in (root, root.parent, root / "nCoV" / "plots", root / "metrics" / "plots"):
        with pytest.raises(ValueError, match="overlap"):
            pipeline.preflight(root, target, datasets=("nCoV",))
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("user data")
    with pytest.raises(FileExistsError, match="new or empty"):
        pipeline.preflight(root, existing, datasets=("nCoV",))
    assert marker.read_text() == "user data"


def test_missing_gallery_cache_fails_before_any_output(inputs, tmp_path, monkeypatch):
    root, _frame, calls = inputs

    def missing(*args, **kwargs):
        raise FileNotFoundError("missing stored embedding")

    monkeypatch.setattr(pipeline, "load_reported_embedding", missing)
    with pytest.raises(FileNotFoundError, match="stored embedding"):
        pipeline.preflight(
            root, tmp_path / "output", datasets=("nCoV",), families=("metrics", "qualitative")
        )
    assert calls[0]["require_source_caches"] is True
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("fail_second", (False, True))
def test_manifest_only_certifies_success_and_records_hashes(
    inputs, tmp_path, monkeypatch, fail_second
):
    root, _frame, _calls = inputs
    target = tmp_path / "output"
    prepared = pipeline.preflight(root, target, datasets=("nCoV",), families=("metrics",))
    source_hash = hashlib.sha256((root / "metrics.csv").read_bytes()).hexdigest()
    rendered = []

    def render(summary, stem, *, render_context, **kwargs):
        rendered.append(stem)
        if fail_second and len(rendered) == 2:
            raise RuntimeError("injected render failure")
        path = render_context.output_root / f"{stem}.png"
        Image.new("RGB", (10, 8), "white").save(path, dpi=(600, 600))
        return path

    monkeypatch.setattr(pipeline.benchmark_figures, "dataset_metric_grid", render)
    monkeypatch.setattr(pipeline.benchmark_figures, "pooled_metric_grid", render)
    if fail_second:
        with pytest.raises(RuntimeError, match="injected render failure"):
            pipeline.build(prepared)
        assert not (target / "figure-manifest.json").exists()
    else:
        manifest_path = pipeline.build(prepared)
        manifest = json.loads(manifest_path.read_text())
        assert manifest["families"] == ["metrics"]
        assert manifest["frames"]["selected-metric-rows.csv"]["rows"] == 9
        assert manifest["frames"]["metric-summary.csv"]["rows"] == 3
        assert manifest["frames"]["metric-summary-pooled.csv"]["rows"] == 3
        assert manifest["pooled_datasets"] == ["nCoV"]
        assert set(rendered) == {stem for stem, *_ in pipeline.METRIC_GRIDS} | {
            "f_pooled_benchmark"
        }
        for name, record in manifest["outputs"].items():
            assert hashlib.sha256((target / name).read_bytes()).hexdigest() == record["sha256"]
        assert manifest["outputs"][f"{rendered[0]}.png"]["dimensions_px"] == [10, 8]
        with pytest.raises(FileExistsError):
            pipeline.build(prepared)
    assert hashlib.sha256((root / "metrics.csv").read_bytes()).hexdigest() == source_hash


def test_changed_input_after_preflight_is_refused_before_output(inputs, tmp_path):
    root, _frame, _calls = inputs
    target = tmp_path / "output"
    prepared = pipeline.preflight(root, target, datasets=("nCoV",), families=("metrics",))
    (root / "metrics.csv").write_text("changed")
    with pytest.raises(RuntimeError, match="input changed after preflight"):
        pipeline.build(prepared)
    assert not target.exists()


def test_cli_dry_run_never_calls_writer(monkeypatch, tmp_path):
    from scripts.figures import build as cli

    calls = []

    def preflight(root, target, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            frame=[1], datasets=("nCoV",), pool=("nCoV",), families=("metrics",), output=target
        )

    def forbidden(*args):
        pytest.fail("dry-run called a writer")

    monkeypatch.setattr(cli, "preflight", preflight)
    monkeypatch.setattr(cli, "build", forbidden)
    cli.main(["--results-root", str(tmp_path), "--datasets", "nCoV", "--dry-run"])
    assert calls == [
        dict(
            datasets=("nCoV",),
            families=pipeline.FAMILIES,
            supplementary_root=None,
            top_r=pipeline.DEFAULT_TOP_R,
            policy=CURRENT_BENCHMARK,
        )
    ]
    # Any changed setting builds the same figure set for that configuration, named for it.
    calls.clear()
    cli.main(
        ["--results-root", str(tmp_path), "--datasets", "nCoV", "--dry-run", "--effective-k", "30"]
    )
    policy = calls[0]["policy"]
    assert (policy.rel_reg, policy.effective_k, policy.graph_symmetrization) == (1.0, 30.0, "or")
    assert policy.name == "rel_reg=1 k=30 sym=or"


def test_top_r_must_be_positive_and_unique(inputs, tmp_path):
    root, _frame, _calls = inputs
    for top_r in ((), (0,), (3, 3)):
        with pytest.raises(ValueError, match="top_r"):
            pipeline.preflight(root, tmp_path / "output", datasets=("nCoV",), top_r=top_r)


def test_overlapping_subset_is_never_pooled(inputs, tmp_path, monkeypatch):
    root, frame, _calls = inputs
    both = pd.concat([frame, frame.assign(dataset="hematopoietic_shared")], ignore_index=True)
    monkeypatch.setattr(pipeline, "load_canonical_frame", lambda *a, **k: both)
    prepared = pipeline.preflight(
        root,
        tmp_path / "output",
        datasets=("nCoV", "hematopoietic_shared"),
        families=("metrics", "ranks"),
        top_r=(1,),
    )
    assert prepared.pool == ("nCoV",)
    assert set(prepared.pooled["n"]) == {1}
    assert set(prepared.ranks["raw"]["n_datasets"]) == {1}
