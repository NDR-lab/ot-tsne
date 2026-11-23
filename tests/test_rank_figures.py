"""Rank and top-r summaries and their figures."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ne_aligned_utils.paper import layout
from ne_aligned_utils.paper.rendering import presentation
from ne_aligned_utils.paper.summaries import ranks

METHODS = ("ot", "multimap", "seurat-50", "joint")


def _frame(methods=METHODS):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        [
            dict(
                dataset=dataset,
                # The metric tables name both MultiMAP arms "multimap"; method_key tells them apart.
                method="multimap" if key.startswith("multimap") else key,
                method_key=key,
                rel_reg=1.0 if key == "ot" else float("nan"),
                effective_k=15.0 if key == "ot" else float("nan"),
                seed=seed,
                **{metric: float(rng.uniform()) for metric in ranks.METRICS},
            )
            for dataset in ("d1", "d2", "d3")
            for key in methods
            for seed in range(3)
        ]
    )


def test_aggregate_ranks_cover_all_bio_and_mixing_blocks():
    table = ranks.aggregate_ranks(
        _frame(), datasets=("d1", "d2"), normalization="raw", rel_reg=1.0, effective_k=15.0
    )
    assert set(table["block"]) == {"all", "bio", "mixing"}
    assert len(table) == 3 * len(METHODS)
    assert set(table["n_datasets"]) == {2} and set(table["n_competitors"]) == {len(METHODS)}
    # Mean ranks over a field of four sum to 1 + 2 + 3 + 4 in every block.
    assert table.groupby("block")["aggregate_rank"].sum().tolist() == [10.0, 10.0, 10.0]


def test_minmax_and_raw_are_the_only_normalizations():
    with pytest.raises(ValueError, match="normalization"):
        ranks.aggregate_ranks(
            _frame(), datasets=("d1",), normalization="z", rel_reg=1.0, effective_k=15.0
        )


def test_top_k_counts_one_bio_and_one_mixing_placing_per_dataset():
    table = ranks.top_k(
        _frame(),
        datasets=("d1", "d2", "d3"),
        normalization="minmax",
        k=2,
        rel_reg=1.0,
        effective_k=15.0,
    )
    assert set(table["top_k"]) == {2} and set(table["n_cells"]) == {2}
    assert (table["n_top_k"] == table["n_top_k_bio"] + table["n_top_k_mixing"]).all()
    # Two of four methods place in each half on each dataset.
    assert table.groupby("dataset")["n_top_k_bio"].sum().eq(2).all()
    with pytest.raises(ValueError, match="k must"):
        ranks.top_k(
            _frame(), datasets=("d1",), normalization="raw", k=0, rel_reg=1.0, effective_k=15.0
        )


def _panels(fig):
    return [ax for ax in fig.axes if ax.collections]


def test_rank_figure_defaults_to_own_order_columns_with_printed_values():
    table = ranks.aggregate_ranks(
        _frame(), datasets=("d1", "d2", "d3"), normalization="raw", rel_reg=1.0, effective_k=15.0
    )
    composer = presentation.rank_composer()
    try:
        with layout.paper_style():
            fig = composer.rank_aggregate(table)
        panels = _panels(fig)
        assert [ax.get_title() for ax in panels] == list(composer.block_titles.values())
        for ax, block in zip(panels, ("all", "bio", "mixing")):
            block_rows = table[table["block"] == block].sort_values("aggregate_rank")
            names = [t.get_text() for t in ax.get_yticklabels()]
            assert names == [composer.method_labels[k] for k in block_rows["method_key"]]
            values = [t.get_text() for t in ax.texts]
            assert values == [f"{v:.1f}" for v in block_rows["aggregate_rank"]]
    finally:
        plt.close("all")


def test_rank_figure_can_share_the_first_columns_order_without_values():
    table = ranks.aggregate_ranks(
        _frame(), datasets=("d1", "d2"), normalization="raw", rel_reg=1.0, effective_k=15.0
    )
    try:
        with layout.paper_style():
            fig = presentation.rank_composer().rank_aggregate(
                table, order_per_block=False, show_values=False
            )
        panels = _panels(fig)
        assert all(not ax.texts for ax in panels)
        assert [t.get_text() for t in panels[1].get_yticklabels()] == [""] * len(METHODS)
    finally:
        plt.close("all")


def test_rank_pool_must_be_selected_before_ranking():
    table = ranks.aggregate_ranks(
        _frame(), datasets=("d1",), normalization="raw", rel_reg=1.0, effective_k=15.0
    )
    with pytest.raises(ValueError, match="Exclude before ranking"):
        presentation.rank_composer().rank_aggregate(table[table["method_key"] != "joint"])


def test_top_k_figure_reads_r_from_its_table():
    table = ranks.top_k(
        _frame(),
        datasets=("d1", "d2", "d3"),
        normalization="raw",
        k=2,
        rel_reg=1.0,
        effective_k=15.0,
    )
    table = table.assign(dataset=table["dataset"].map({"d1": "nCoV", "d2": "lupus", "d3": "PBMC"}))
    try:
        with layout.paper_style():
            fig = presentation.top_k_composer().top_k_finishes(table)
        assert fig._supxlabel.get_text() == "Top-2 finishes"
    finally:
        plt.close("all")


def test_ranks_use_native_multimap_and_refuse_the_matched_k_arm():
    matched = _frame(("ot", "multimap-k15", "seurat-50", "joint"))
    with pytest.raises(ValueError, match="native MultiMAP only"):
        ranks.aggregate_ranks(
            matched, datasets=("d1",), normalization="raw", rel_reg=1.0, effective_k=15.0
        )
    with pytest.raises(ValueError, match="native MultiMAP only"):
        ranks.top_k(
            matched, datasets=("d1",), normalization="raw", k=2, rel_reg=1.0, effective_k=15.0
        )
    both = pd.concat([_frame(), matched[matched["method_key"] == "multimap-k15"]])
    with pytest.raises(ValueError, match="native MultiMAP only"):
        ranks.aggregate_ranks(
            both, datasets=("d1",), normalization="raw", rel_reg=1.0, effective_k=15.0
        )
    without = _frame(("ot", "seurat-50", "joint"))
    with pytest.raises(ValueError, match="native MultiMAP is missing"):
        ranks.aggregate_ranks(
            without, datasets=("d1",), normalization="raw", rel_reg=1.0, effective_k=15.0
        )


def test_a_method_missing_from_one_dataset_is_refused_not_dropped():
    frame = _frame()
    frame = frame[~((frame["dataset"] == "d2") & (frame["method_key"] == "joint"))]
    with pytest.raises(ValueError, match="missing from some dataset.*joint"):
        ranks.aggregate_ranks(
            frame, datasets=("d1", "d2"), normalization="raw", rel_reg=1.0, effective_k=15.0
        )


def test_ranks_match_score_aggregation_directly():
    """The figure tables are score_aggregation's numbers, only reshaped."""
    from ne_aligned_utils.evaluation import score_aggregation as sa
    from ne_aligned_utils.paper.registry import METRIC_SOURCE_COLUMNS

    frame = _frame()
    rows = frame.drop(columns=["method"]).rename(
        columns={**{v: k for k, v in METRIC_SOURCE_COLUMNS.items()}, "method_key": "method"}
    )
    for normalization in ranks.NORMALIZATIONS:
        expected = sa.aggregate_scores(
            sa.score_within(
                rows,
                metrics=tuple(METRIC_SOURCE_COLUMNS),
                rel_reg_values=(1.0,),
                effective_k=None,
                normalization=normalization,
            )
        ).set_index("method")
        table = ranks.aggregate_ranks(
            frame,
            datasets=("d1", "d2", "d3"),
            normalization=normalization,
            rel_reg=1.0,
            effective_k=15.0,
        )
        for block, score in ranks.BLOCKS.items():
            got = table[table["block"] == block].set_index("method_key")["aggregate_rank"]
            assert got.to_dict() == expected[f"{score}_rank"].to_dict()
