"""Runtime panel recipe, summaries, validation and the figure writer, on synthetic timings."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ne_aligned_utils import config
from ne_aligned_utils.paper import layout, registry
from ne_aligned_utils.paper.figures import runtime as runtime_figures
from ne_aligned_utils.paper.inputs.current import CANONICAL_DATASETS
from ne_aligned_utils.paper.rendering import runtime as presentation, runtime_drawing, styles
from ne_aligned_utils.paper.summaries import runtime as summaries
from ne_aligned_utils.paper.summaries.task_timing import RuntimeComparabilityError

CONTEXT = presentation.runtime_render_context()
PLOTTED = [
    key for family in presentation.FAMILY_ORDER for key in presentation.RUNTIME_FAMILIES[family]
]


def _frame(**overrides) -> pd.DataFrame:
    row = dict(
        method_key="ot",
        task_kind="ot",
        dataset="nCoV",
        seed=0,
        rel_reg=1.0,
        effective_k=15,
        backend="tsne",
        timing_schema="ot",
        time_total=7.0,
        time_pre_embedding=2.5,
        time_embedding=4.5,
        n_obs=4060,
        n_features_input=50,
        threads_requested=4,
        n_jobs=4,
        device="cpu",
        blas_name="openblas",
        slurm_cpus_per_task=4,
        env_fingerprint="abc",
    )
    row.update(overrides)
    return pd.DataFrame([row])


def _full_frame(datasets=("nCoV", "lupus")) -> pd.DataFrame:
    rows = []
    for index, key in enumerate(PLOTTED):
        for dataset in datasets:
            for seed in range(3):
                total = float(10 + index + seed)
                rows.append(
                    _frame(
                        method_key=key,
                        task_kind=key,
                        dataset=dataset,
                        seed=seed,
                        time_total=total,
                        time_pre_embedding=0.4 * total,
                        time_embedding=0.6 * total,
                        n_obs=1000 if dataset == "nCoV" else 2000,
                    )
                )
    return pd.concat(rows, ignore_index=True)


def test_every_reported_method_is_in_exactly_one_panel():
    assert len(PLOTTED) == len(set(PLOTTED)) == 23
    assert "multimap" in PLOTTED and "multimap-k15" not in PLOTTED
    for key in PLOTTED:
        registry.resolve(key)


def test_panels_3a_and_3b_split_integration_by_backend_axis():
    for key in presentation.RUNTIME_FAMILIES["3a"]:
        assert config.has_backend_axis(registry.METHODS[key].method)
    for key in presentation.RUNTIME_FAMILIES["3b"]:
        assert not config.has_backend_axis(registry.METHODS[key].method)


def test_3b_mirrors_3a_row_for_row():
    members = presentation.RUNTIME_FAMILIES["3a"]
    gmeans = {key: float(i) for i, key in enumerate(reversed(members))}
    a = summaries.ordered_methods("3a", gmeans, context=CONTEXT)
    b = summaries.ordered_methods("3b", gmeans, context=CONTEXT)
    assert [CONTEXT.method_tools[k] for k in a] == [CONTEXT.method_tools[k] for k in b]


def test_other_panels_sort_by_geometric_mean_and_panel_one_keeps_its_order():
    members = presentation.RUNTIME_FAMILIES["2"]
    gmeans = {key: float(len(members) - i) for i, key in enumerate(members)}
    assert summaries.ordered_methods("2", gmeans, context=CONTEXT) == list(reversed(members))
    declared = presentation.RUNTIME_FAMILIES["1"]
    shuffled = {key: float(i) for i, key in enumerate(reversed(declared))}
    assert summaries.ordered_methods("1", shuffled, context=CONTEXT) == list(declared)


def test_every_dataset_has_a_colour_and_a_marker_from_the_categorical_palette():
    style = presentation.runtime_figure_style()
    for dataset in CANONICAL_DATASETS:
        assert style.palette[dataset] == styles.DATASET_COLORS[dataset]
        assert dataset in presentation.DATASET_MARKERS


def test_dataset_label_carries_its_cell_count():
    labels = registry.DATASET_LABELS
    assert runtime_drawing.dataset_label("nCoV", 4060, labels=labels) == "nCoV (4,060)"
    assert runtime_drawing.dataset_label("nCoV", labels=labels) == "nCoV"


def test_annotation_keeps_unplotted_rows_with_a_reason():
    frame = pd.concat([_frame(), _frame(method_key="multimap-k15")], ignore_index=True)
    out = summaries.annotate(frame, context=CONTEXT)
    assert out["included_in_plot"].tolist() == [True, False]
    assert out.loc[1, "exclusion_reason"]
    assert out.loc[0, "family"] == "1"
    assert out.loc[0, "method_label"] == registry.METHODS["ot"].label


def test_medians_and_geometric_means_use_time_total():
    frame = pd.concat(
        [_frame(seed=s, time_total=t, total_seconds=999.0) for s, t in enumerate((6.0, 7.0, 20.0))],
        ignore_index=True,
    )
    annotated = summaries.annotate(frame, context=CONTEXT)
    medians = summaries.dataset_medians(annotated, context=CONTEXT)
    assert float(medians.loc[0, "time_total"]) == 7.0
    assert int(medians.loc[0, "n_runs"]) == 3
    assert "total_seconds" not in medians
    assert summaries.geometric_means(medians)["ot"] == pytest.approx(7.0)


def test_axis_ignores_stages_that_are_never_drawn():
    medians = pd.DataFrame(
        [
            dict(method_key="ref", family="2", time_total=10.0, time_pre_embedding=1e-6),
            dict(method_key="ot", family="1", time_total=100.0, time_pre_embedding=2.0),
        ]
    )
    low, high = runtime_drawing.axis_limits(medians, phase_families=presentation.PHASE_FAMILIES)
    assert 1e-3 < low < 2.0 and high > 100.0


def test_ot_breakdown_falls_back_to_one_run_when_phase_medians_disagree():
    frame = pd.concat(
        [
            _frame(seed=0, time_total=10.0, time_pre_embedding=1.0, time_embedding=9.0),
            _frame(seed=1, time_total=11.0, time_pre_embedding=8.0, time_embedding=3.0),
            _frame(seed=2, time_total=30.0, time_pre_embedding=25.0, time_embedding=5.0),
        ],
        ignore_index=True,
    )
    table = summaries.ot_breakdown_table(
        summaries.annotate(frame, context=CONTEXT), datasets=("nCoV",), context=CONTEXT
    )
    row = table.iloc[0]
    assert row.median_source == "median-total run"
    assert (row.time_pre_embedding, row.time_embedding) == (8.0, 3.0)


def test_validation_refuses_runs_from_different_allocations(tmp_path):
    frame = pd.concat([_frame(), _frame(seed=1, threads_requested=8)], ignore_index=True)
    with pytest.raises(RuntimeComparabilityError, match="threads_requested"):
        summaries.validate(
            summaries.annotate(frame, context=CONTEXT),
            root=tmp_path,
            datasets=("nCoV",),
            context=CONTEXT,
        )


@pytest.mark.parametrize("gpu", (False, True))
def test_writer_produces_figures_and_audit_tables(tmp_path, gpu):
    gpu_medians = None
    if gpu:
        gpu_medians = pd.DataFrame(
            [
                dict(
                    method_key="ot",
                    dataset="nCoV",
                    time_total=2.0,
                    time_pre_embedding=1.0,
                    time_embedding=1.0,
                    n_runs=3,
                    n_obs=1000,
                    device="cuda",
                )
            ]
        )
    with layout.paper_style():
        written = runtime_figures.write_runtime_figures(
            _full_frame(),
            root=tmp_path,
            output_dir=tmp_path / "runtime",
            datasets=("nCoV", "lupus"),
            gpu_medians=gpu_medians,
        )
    plt.close("all")
    names = {path.name for path in (tmp_path / "runtime").iterdir()}
    assert {
        "runtime_all_methods.png",
        "runtime_ot_tsne_breakdown.png",
        "runtime_summary.csv",
        "selected-timing-rows.csv",
        "dataset-medians.csv",
        "runtime_validation.txt",
        "runtime_captions.md",
    } <= names
    device = {f"{name}.png" for name in presentation.DEVICE_VARIANTS}
    assert (device <= names) == gpu
    assert all(path.exists() for path in written.values())
    summary = pd.read_csv(tmp_path / "runtime" / "runtime_summary.csv")
    assert list(summary.columns) == list(summaries.SUMMARY_COLUMNS)
    assert np.isfinite(summary["time_total"]).all()


def test_save_figure_writes_both_formats_and_refuses_to_drop_the_png(tmp_path):
    with layout.paper_style():
        fig, ax = plt.subplots(figsize=(layout.FULL_WIDTH, 1.6))
        ax.plot([1.0, 2.0], [1.0, 2.0])
        path = layout.save_figure(fig, "pair", root=tmp_path, preview=False, formats=("png", "pdf"))
    assert path.suffix == ".png" and path.with_suffix(".pdf").exists()
    fig = plt.figure(figsize=(2.0, 1.0))
    with pytest.raises(ValueError, match="omits 'png'"):
        layout.save_figure(fig, "solo", root=tmp_path, preview=False, formats=("pdf",))
    plt.close(fig)
