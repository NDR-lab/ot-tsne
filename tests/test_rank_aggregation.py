import functools

"""Check metric directions, fair ties, common competitor pools, and input-order invariance."""

import numpy as np
import pandas as pd
import pytest

from ne_aligned_utils.evaluation.rank_aggregation import (
    DEPRECATED_METHODS,
    HANDICAPPED_2D_METHODS,
    METRIC_DIRECTION,
    METRIC_FAMILY,
    RANK_METRICS,
    SHARED_REL_REG,
    STATISTICS,
    _practical_ranks,
    aggregate_ranks,
    bootstrap_aggregate_ranks,
    coverage_matrix,
    family_rollup,
    leave_one_out_matrix,
    leave_one_out_ranks,
    per_metric_table,
    rank_matrix,
    rank_within as _rank_within,
    read_rel_reg_table,
    resolve_method_pool,
    top_k_frequency,
)

#: The functions take no default configuration; these tests use the presets explicitly.
rank_within = functools.partial(_rank_within, rel_reg_values=SHARED_REL_REG, effective_k=15.0)


def _rows(methods=("a", "b", "c"), datasets=("d1", "d2"), rel_reg=(1.0, 1 / 10), seeds=(0, 1)):
    """Synthetic scored rows: every method rel_reg-swept, value increasing with method index."""
    recs = []
    for ds_i, ds in enumerate(datasets):
        for m_i, m in enumerate(methods):
            for e in rel_reg:
                for s in seeds:
                    recs.append(
                        {
                            "dataset": ds,
                            "method": m,
                            "seed": s,
                            "rel_reg": e,
                            "effective_k": 15.0,
                            # deterministic and distinct per method, with a small seed wobble
                            "neighborhood_purity_score": 0.1 * (m_i + 1) + 0.01 * s + 0.001 * ds_i,
                            "silhouette_score": 0.1 * (m_i + 1),
                            "ilisi_score": 0.1 * (m_i + 1),
                            "kbet_score": 0.1 * (m_i + 1),
                        }
                    )
    return pd.DataFrame(recs)


# direction


def test_higher_is_better_metrics_rank_descending():
    ranked = rank_within(_rows(), rel_reg_values=(1.0, 1 / 10))
    one = ranked[
        (ranked.dataset == "d1") & (ranked.rel_reg == 1.0) & (ranked.metric == "kbet_score")
    ]
    best = one.loc[one["rank"].idxmin()]
    # method "c" has the largest value, so it must take rank 1
    assert best["method"] == "c"
    assert best["value"] == pytest.approx(one["value"].max())


def test_every_rank_metric_has_a_direction_and_family():
    for metric in RANK_METRICS:
        assert metric in METRIC_DIRECTION
        assert METRIC_FAMILY[metric] in {"bio", "batch"}
    # the aggregate is only balanced if both sides are actually represented
    families = {METRIC_FAMILY[m] for m in RANK_METRICS}
    assert families == {"bio", "batch"}


# ties and ordering


def test_ties_share_the_average_rank():
    """Two methods tied for best must both get 1.5, not 1 and 2 by row order."""
    rows = _rows(methods=("a", "b", "c"))
    rows.loc[rows.method.isin(["b", "c"]), "kbet_score"] = 0.9  # b and c tie at the top
    ranked = rank_within(rows, metrics=("kbet_score",), rel_reg_values=(1.0,))
    one = ranked[ranked.dataset == "d1"].set_index("method")["rank"]
    assert one["b"] == pytest.approx(1.5)
    assert one["c"] == pytest.approx(1.5)
    assert one["a"] == pytest.approx(3.0)


def test_aggregate_is_invariant_to_row_order():
    rows = _rows()
    a = aggregate_ranks(rank_within(rows))
    b = aggregate_ranks(rank_within(rows.sample(frac=1.0, random_state=0)))
    merged = a.merge(b, on=["method", "rel_reg"], suffixes=("_a", "_b"))
    assert len(merged) == len(a)
    np.testing.assert_allclose(merged.aggregate_rank_a, merged.aggregate_rank_b)


# method pool


def test_pool_is_the_intersection_across_datasets():
    rows = _rows(methods=("a", "b", "c"))
    rows = rows[~((rows.dataset == "d2") & (rows.method == "c"))]  # c missing on d2
    assert resolve_method_pool(rows, ["d1", "d2"]) == ("a", "b")
    assert resolve_method_pool(rows, ["d1"]) == ("a", "b", "c")


def test_pool_drops_deprecated_and_optionally_handicapped():
    methods = ("ot", "harmony-2d", "scvi-2d", "harmony-50", *DEPRECATED_METHODS)
    rows = _rows(methods=methods)
    full = resolve_method_pool(rows, ["d1", "d2"])
    assert not set(full) & set(DEPRECATED_METHODS)
    assert set(HANDICAPPED_2D_METHODS) <= set(full)

    fair = resolve_method_pool(rows, ["d1", "d2"], fair_subset=True)
    assert not set(fair) & set(HANDICAPPED_2D_METHODS)
    # the fair form of each tool survives
    assert "harmony-50" in fair
    assert set(full) - set(fair) == set(HANDICAPPED_2D_METHODS)


def test_pool_raises_on_a_dataset_with_no_rows():
    with pytest.raises(ValueError, match="no scored rows"):
        resolve_method_pool(_rows(), ["d1", "does-not-exist"])


# non-rel_reg broadcast


def test_config_free_methods_enter_every_epsilon_table():
    """A method with no epsilon must be ranked at every epsilon, not dropped."""
    rows = _rows(methods=("swept",), rel_reg=(1.0, 1 / 10))
    flat = _rows(methods=("flat",), rel_reg=(1.0,))
    flat["rel_reg"] = np.nan  # no epsilon knob
    ranked = rank_within(
        pd.concat([rows, flat], ignore_index=True),
        metrics=("kbet_score",),
        rel_reg_values=(1.0, 1 / 10),
    )
    seen = ranked[ranked.method == "flat"]["rel_reg"].unique()
    assert sorted(seen) == [1 / 10, 1.0]
    # and its own value is identical in both tables
    vals = ranked[ranked.method == "flat"]["value"].unique()
    assert len(vals) == 1


# aggregation and coverage


def test_coverage_counts_are_reported():
    rows = _rows(datasets=("d1", "d2", "d3"))
    agg = aggregate_ranks(rank_within(rows, datasets=("d1", "d2", "d3")))
    assert (agg["n_datasets"] == 3).all()
    assert (agg["n_metrics"] == len(RANK_METRICS)).all()


def test_family_weighting_balances_bio_and_batch():
    """Three bio metrics and one batch metric must still weight the two sides equally."""
    rows = _rows(methods=("bio_winner", "batch_winner"), datasets=("d1",), rel_reg=(1.0,))
    # bio_winner tops the two bio metrics; batch_winner tops the two batch metrics
    for m in ("neighborhood_purity_score", "silhouette_score"):
        rows.loc[rows.method == "bio_winner", m] = 1.0
        rows.loc[rows.method == "batch_winner", m] = 0.0
    for m in ("ilisi_score", "kbet_score"):
        rows.loc[rows.method == "bio_winner", m] = 0.0
        rows.loc[rows.method == "batch_winner", m] = 1.0
    agg = aggregate_ranks(rank_within(rows, datasets=("d1",), rel_reg_values=(1.0,)))
    ranks = agg.set_index("method")["aggregate_rank"]
    # symmetric wins => identical aggregate rank
    assert ranks["bio_winner"] == pytest.approx(ranks["batch_winner"])


def test_rank_matrix_orders_rows_by_best_rank():
    agg = aggregate_ranks(rank_within(_rows()))
    mat = rank_matrix(agg)
    best = mat.min(axis=1).to_numpy()
    assert (np.diff(best) >= 0).all(), "rows must be sorted by best achieved rank"


def test_family_rollup_counts_members():
    methods = ("ot", "ot-knndiag", "harmony-50", "joint-pca")
    agg = aggregate_ranks(rank_within(_rows(methods=methods)))
    roll = family_rollup(agg)
    ot = roll[(roll.family == "OT") & (roll.rel_reg == 1.0)].iloc[0]
    assert ot["n_methods"] == 2  # the two OT entrants above
    assert ot["best_rank"] <= ot["mean_rank"]


def test_per_metric_table_is_method_by_metric():
    ranked = rank_within(_rows())
    table = per_metric_table(ranked, rel_reg=1.0)
    assert list(table.columns) == ["Purity", "Silhouette", "iLISI", "kBET"]
    assert set(table.index) == {"a", "b", "c"}


def test_coverage_matrix_flags_a_missing_cell():
    rows = _rows(methods=("a", "b"))
    rows = rows[~((rows.dataset == "d2") & (rows.method == "b"))]
    cov = coverage_matrix(rows)
    assert bool(cov.loc["a", "d2"]) is True
    assert bool(cov.loc["b", "d2"]) is False


def test_shared_eps_is_the_grid_all_methods_have():
    # documents why the grid is four values rather than the nine the plain OT methods used
    assert SHARED_REL_REG == (1.0, 1 / 10, 1 / 100, 1 / 300)


def test_effective_k_filter_keeps_methods_without_a_k():
    rows = _rows(methods=("swept",))
    flat = _rows(methods=("flat",))
    flat["effective_k"] = np.nan  # baselines carry no k
    ranked = rank_within(pd.concat([rows, flat], ignore_index=True), effective_k=15.0)
    assert set(ranked["method"].unique()) == {"swept", "flat"}


# practical ties


def test_practical_ranks_with_zero_tolerance_matches_pandas_average():
    values = np.array([0.5, 0.5, 0.3, 0.9])
    expected = pd.Series(values).rank(ascending=False, method="average").to_numpy()
    np.testing.assert_allclose(_practical_ranks(values, 0.0), expected)


def test_practical_ranks_ties_values_within_the_tolerance():
    # 0.653 / 0.650 / 0.648 with a 0.003 tolerance: the first two tie, the third is separate
    # from the group's best by more than the tolerance
    ranks = _practical_ranks(np.array([0.653, 0.650, 0.648]), 0.003)
    assert ranks[0] == pytest.approx(1.5)
    assert ranks[1] == pytest.approx(1.5)
    assert ranks[2] == pytest.approx(3.0)


def test_practical_ranks_group_width_is_bounded_by_the_tolerance():
    """A chain of small gaps must not collapse into one wide tie."""
    values = np.array([1.00, 0.99, 0.98, 0.97, 0.96])  # every consecutive gap is 0.01
    ranks = _practical_ranks(values, 0.01)
    # single linkage would give all five the same rank; grouping from each group's own best
    # ties only {1.00, 0.99} and then {0.98, 0.97} and then {0.96}
    assert ranks[0] == pytest.approx(1.5)
    assert ranks[2] == pytest.approx(3.5)
    assert ranks[4] == pytest.approx(5.0)


def test_seed_std_tolerance_ties_differences_below_the_seed_noise():
    rows = _rows(methods=("a", "b"), datasets=("d1",), rel_reg=(1.0,), seeds=(0, 1, 2))
    # b differs from a by far less than the seed wobble the fixture builds into purity
    rows.loc[rows.method == "b", "neighborhood_purity_score"] = (
        rows.loc[rows.method == "a", "neighborhood_purity_score"].to_numpy() + 1e-6
    )
    strict = rank_within(rows, metrics=("neighborhood_purity_score",), rel_reg_values=(1.0,))
    tolerant = rank_within(
        rows,
        metrics=("neighborhood_purity_score",),
        rel_reg_values=(1.0,),
        tie_tolerance="seed_std",
    )
    assert sorted(strict["rank"]) == [1.0, 2.0]
    assert sorted(tolerant["rank"]) == [1.5, 1.5]


def test_tie_tolerance_falls_back_to_exact_when_there_is_one_seed():
    rows = _rows(methods=("a", "b"), datasets=("d1",), rel_reg=(1.0,), seeds=(0,))
    ranked = rank_within(rows, rel_reg_values=(1.0,), tie_tolerance="seed_std")
    assert set(ranked["tie_tolerance"]) == {0.0}


def test_tie_tolerance_rejects_an_unknown_string():
    with pytest.raises(ValueError, match="tie_tolerance"):
        rank_within(_rows(), tie_tolerance="whatever")


# robustness views


def test_bootstrap_point_estimate_equals_the_plain_aggregate():
    ranked = rank_within(_rows())
    boot = bootstrap_aggregate_ranks(ranked, rel_reg=1.0, n_boot=50, seed=0)
    agg = aggregate_ranks(ranked)
    agg = agg[agg.rel_reg == 1.0].set_index("method")["aggregate_rank"]
    merged = boot.set_index("method")["aggregate_rank"]
    np.testing.assert_allclose(merged.loc[agg.index], agg)


def test_bootstrap_interval_contains_the_point_estimate_and_is_reproducible():
    ranked = rank_within(_rows())
    a = bootstrap_aggregate_ranks(ranked, rel_reg=1.0, n_boot=200, seed=7)
    b = bootstrap_aggregate_ranks(ranked, rel_reg=1.0, n_boot=200, seed=7)
    assert (a["lo"] <= a["aggregate_rank"] + 1e-9).all()
    assert (a["hi"] >= a["aggregate_rank"] - 1e-9).all()
    np.testing.assert_allclose(a["lo"], b["lo"])  # seeded, so byte-identical


def test_top_k_frequency_counts_cells_not_averages():
    ranked = rank_within(_rows())
    freq = top_k_frequency(ranked, rel_reg=1.0, k=1)
    # "c" holds the best value on every metric of the fixture, so it is top-1 in every cell
    best = freq.set_index("method").loc["c"]
    assert best["frequency"] == pytest.approx(1.0)
    assert best["n_top"] == best["n_cells"]
    # and the frequencies are computed out of the same number of cells for everyone
    assert freq["n_cells"].nunique() == 1


def test_leave_one_out_drops_exactly_one_level():
    ranked = rank_within(_rows())
    loo = leave_one_out_ranks(ranked, rel_reg=1.0, by="dataset")
    assert set(loo["held_out"]) == set(ranked["dataset"].unique())
    per_method = loo.groupby("method").size()
    assert (per_method == ranked["dataset"].nunique()).all()


def test_leave_one_out_matrix_first_column_is_the_full_aggregate():
    ranked = rank_within(_rows())
    matrix = leave_one_out_matrix(ranked, rel_reg=1.0)
    agg = aggregate_ranks(ranked)
    agg = agg[agg.rel_reg == 1.0].set_index("method")["aggregate_rank"]
    assert matrix.columns[0] == "all"
    np.testing.assert_allclose(matrix["all"], agg.loc[matrix.index])
    # one column per dataset and per metric, plus "all"
    assert matrix.shape[1] == 1 + ranked["dataset"].nunique() + ranked["metric"].nunique()


def test_leave_one_out_rejects_an_unknown_axis():
    with pytest.raises(ValueError, match="by must be"):
        leave_one_out_ranks(rank_within(_rows()), rel_reg=1.0, by="seed")


# mean vs median
#
# The whole point of offering both is a method whose placings disagree with each other: the mean
# carries the disagreement into the aggregate, the median discards it. The fixture above cannot
# show that (its values are monotone in the method index on every dataset, so every method holds
# one rank everywhere), hence a second fixture built to disagree.


def _disagreeing_rows():
    """Three datasets where "a" is best on two and worst on the third, and "c" the reverse."""
    order = {"d1": ("a", "b", "c"), "d2": ("a", "b", "c"), "d3": ("c", "b", "a")}
    recs = []
    for ds, ranking in order.items():
        for position, method in enumerate(ranking):
            value = 0.9 - 0.1 * position  # first in the ranking holds the largest value
            recs.append(
                {
                    "dataset": ds,
                    "method": method,
                    "seed": 0,
                    "rel_reg": 1.0,
                    "effective_k": 15.0,
                    "neighborhood_purity_score": value,
                    "silhouette_score": value,
                    "ilisi_score": value,
                    "kbet_score": value,
                }
            )
    return pd.DataFrame(recs)


def test_median_aggregate_discards_the_one_dataset_that_disagrees():
    ranked = rank_within(_disagreeing_rows(), rel_reg_values=(1.0,))
    mean = aggregate_ranks(ranked).set_index("method")["aggregate_rank"]
    median = aggregate_ranks(ranked, statistic="median").set_index("method")["aggregate_rank"]
    # "a" placed 1st, 1st, 3rd: the mean is dragged off 1 by the third dataset, the median is not
    assert mean["a"] == pytest.approx(5 / 3)
    assert median["a"] == pytest.approx(1.0)
    assert mean["c"] == pytest.approx(7 / 3)
    assert median["c"] == pytest.approx(3.0)
    # "b" is 2nd everywhere, so the two rules cannot disagree about it
    assert mean["b"] == pytest.approx(median["b"])


def test_aggregate_records_which_statistic_produced_it():
    """The figures read this column to label themselves, so it has to be on every row."""
    for statistic in STATISTICS:
        agg = aggregate_ranks(rank_within(_rows()), statistic=statistic)
        assert (agg["statistic"] == statistic).all()


def test_aggregate_rejects_an_unknown_statistic():
    with pytest.raises(ValueError, match="statistic"):
        aggregate_ranks(rank_within(_rows()), statistic="mode")


def test_median_bootstrap_point_estimate_equals_the_median_aggregate():
    """A replicate is only comparable to the point estimate if both use the same rule."""
    ranked = rank_within(_disagreeing_rows(), rel_reg_values=(1.0,))
    boot = bootstrap_aggregate_ranks(ranked, rel_reg=1.0, n_boot=50, statistic="median")
    agg = aggregate_ranks(ranked, statistic="median").set_index("method")["aggregate_rank"]
    np.testing.assert_allclose(boot.set_index("method")["aggregate_rank"].loc[agg.index], agg)
    assert (boot["statistic"] == "median").all()


def test_median_leave_one_out_matrix_agrees_with_the_median_aggregate():
    ranked = rank_within(_disagreeing_rows(), rel_reg_values=(1.0,))
    matrix = leave_one_out_matrix(ranked, rel_reg=1.0, statistic="median")
    agg = aggregate_ranks(ranked, statistic="median").set_index("method")["aggregate_rank"]
    np.testing.assert_allclose(matrix["all"], agg.loc[matrix.index])


def test_per_metric_table_median_differs_from_the_mean_and_reorders_rows():
    ranked = rank_within(_disagreeing_rows(), rel_reg_values=(1.0,))
    mean = per_metric_table(ranked, 1.0)
    median = per_metric_table(ranked, 1.0, statistic="median")
    assert list(mean.columns) == list(median.columns)
    assert mean.loc["a", "Purity"] == pytest.approx(5 / 3)
    assert median.loc["a", "Purity"] == pytest.approx(1.0)


def test_per_metric_table_rejects_an_unknown_statistic():
    with pytest.raises(ValueError, match="statistic"):
        per_metric_table(rank_within(_rows()), 1.0, statistic="mode")


def test_the_sweep_grid_survives_a_csv_round_trip(tmp_path):
    """Preserve the exact binary value of 1/300 through CSV serialization.

    Equality-based grid filters would drop the value if decimal parsing rounded it differently.
    """
    path = tmp_path / "rows.csv"
    pd.DataFrame({"rel_reg": list(SHARED_REL_REG)}).to_csv(path, index=False)

    good = read_rel_reg_table(path)
    assert good["rel_reg"].tolist() == list(SHARED_REL_REG)
    assert int(good["rel_reg"].isin(SHARED_REL_REG).sum()) == len(SHARED_REL_REG)

    # The failure mode this guards against, stated so the guard cannot be mistaken for decoration.
    naive = pd.read_csv(path)
    assert int(naive["rel_reg"].isin(SHARED_REL_REG).sum()) < len(SHARED_REL_REG)
