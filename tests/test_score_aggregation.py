"""Check score-before-rank aggregation and its distinction from rank-before-score.

Constructed examples verify ordering differences; numeric references protect the
rank-first pipeline's independent contract.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from ne_aligned_utils.evaluation.rank_aggregation import (
    RANK_METRICS,
    aggregate_ranks,
    rank_within as _rank_within,
)
from ne_aligned_utils.evaluation.score_aggregation import (
    NORMALIZATIONS,
    REPORTED_REL_REG,
    SCORES,
    aggregate_scores,
    compare_rankings,
    leaders,
    metric_sides,
    score_within as _score_within,
)

#: The functions take no default configuration; these tests use the presets explicitly.
score_within = functools.partial(_score_within, rel_reg_values=REPORTED_REL_REG, effective_k=15.0)
rank_within = functools.partial(_rank_within, rel_reg_values=REPORTED_REL_REG, effective_k=15.0)

PURITY, SILHOUETTE, ILISI, KBET = RANK_METRICS
METRICS = list(RANK_METRICS)
COLUMNS = ["dataset", "method", "rel_reg", "seed", "effective_k", *METRICS]

#: The two settings the analysis is reported at. Two rather than one throughout, because several
#: of these properties are per (dataset, rel_reg) cell and a single value cannot show that.
REL_REG = (1.0, 0.1)


def _rows(spec, *, seeds=(0,), rel_reg_values=REL_REG, free=()):
    """A complete frame from ``{(dataset, method): (purity, silhouette, ilisi, kbet)}``.

    Every method is swept over ``rel_reg_values`` so the competitor pool is complete, which
    :func:`score_within` requires. Methods named in ``free`` get ``rel_reg = NaN`` instead, which
    is how the scored tables record a method with no regularization knob.
    """
    records = []
    for (dataset, method), values in spec.items():
        settings = [float("nan")] if method in free else list(rel_reg_values)
        for rel_reg in settings:
            for seed in seeds:
                records.append([dataset, method, rel_reg, seed, 15.0, *values])
    return pd.DataFrame(records, columns=COLUMNS)


def _uniform(methods, datasets=("d1", "d2"), overrides=None):
    """A spec where every method scores its own constant on every metric and dataset.

    ``methods`` maps a name to that constant; ``overrides`` replaces individual cells, keyed
    ``(dataset, method)``. Enough for most of these tests, which care about ordering rather than
    about realistic values.
    """
    spec = {
        (dataset, method): (value, value, value, value)
        for dataset in datasets
        for method, value in methods.items()
    }
    spec.update(overrides or {})
    return spec


def _at(scored, dataset, method, column, rel_reg=1.0):
    """One cell of a scored frame, as a float."""
    row = scored[
        (scored["dataset"] == dataset)
        & (scored["method"] == method)
        & (scored["rel_reg"] == rel_reg)
    ]
    assert len(row) == 1, f"expected one row for {dataset}/{method}/{rel_reg}, got {len(row)}"
    return float(row[column].iloc[0])


# 1. Seeds are averaged before anything is ranked


def _seed_split_rows():
    """``a`` wins on seed 0 and loses on seed 1; on the seed mean it wins.

    Purity: a is 0.9 then 0.1, mean 0.50; b is 0.45 on both seeds, mean 0.45. Ranking each seed
    and averaging the placings gives the two methods 1.5 each -- a dead heat. Averaging the values
    first and ranking once puts a ahead. The pipeline must do the second.
    """
    records = []
    for dataset in ("d1", "d2"):
        for rel_reg in REL_REG:
            for seed, a_purity in ((0, 0.9), (1, 0.1)):
                records.append([dataset, "a", rel_reg, seed, 15.0, a_purity, 0.5, 0.5, 0.5])
                records.append([dataset, "b", rel_reg, seed, 15.0, 0.45, 0.5, 0.5, 0.5])
    return pd.DataFrame(records, columns=COLUMNS)


def test_seeds_are_averaged_before_any_ranking():
    scored = score_within(_seed_split_rows(), normalization="raw")
    # the value that reaches the composite is the seed mean, not either seed
    assert _at(scored, "d1", "a", PURITY) == pytest.approx(0.5)
    assert _at(scored, "d1", "b", PURITY) == pytest.approx(0.45)
    # and the ordering follows the mean, where a per-seed ranking would have tied
    assert _at(scored, "d1", "a", "bio_rank") == 1.0
    assert _at(scored, "d1", "b", "bio_rank") == 2.0


def test_the_seed_column_does_not_survive_into_the_scores():
    """A surviving seed axis would mean something downstream could still rank per seed."""
    scored = score_within(_seed_split_rows())
    assert "seed" not in scored.columns
    assert len(scored) == 2 * 2 * len(REL_REG)  # methods x datasets x settings


# 2-4. The composites are built from values


def _asymmetric_rows():
    """Three methods whose two bio metrics disagree, so a mean of values is not a mean of ranks."""
    spec = {
        ("d1", "a"): (0.90, 0.10, 0.50, 0.50),
        ("d1", "b"): (0.55, 0.50, 0.60, 0.40),
        ("d1", "c"): (0.10, 0.90, 0.40, 0.60),
        ("d2", "a"): (0.80, 0.20, 0.55, 0.45),
        ("d2", "b"): (0.50, 0.55, 0.65, 0.35),
        ("d2", "c"): (0.20, 0.80, 0.45, 0.65),
    }
    return _rows(spec)


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_bio_score_is_the_mean_of_the_purity_and_silhouette_values(normalization):
    scored = score_within(_asymmetric_rows(), normalization=normalization)
    expected = scored[[f"{PURITY}__norm", f"{SILHOUETTE}__norm"]].mean(axis=1)
    assert np.allclose(scored["bio_score"], expected)
    if normalization == "raw":
        # in the raw variant the normalized column *is* the value, so this is the plain formula
        assert np.allclose(scored["bio_score"], scored[[PURITY, SILHOUETTE]].mean(axis=1))


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_mixing_score_is_the_mean_of_the_ilisi_and_kbet_values(normalization):
    scored = score_within(_asymmetric_rows(), normalization=normalization)
    expected = scored[[f"{ILISI}__norm", f"{KBET}__norm"]].mean(axis=1)
    assert np.allclose(scored["mixing_score"], expected)
    if normalization == "raw":
        assert np.allclose(scored["mixing_score"], scored[[ILISI, KBET]].mean(axis=1))


def test_bio_score_is_not_the_mean_of_the_two_metric_ranks():
    """The distinction the whole pipeline exists for, at the level of one composite.

    On ``d1`` the purity order is a > b > c and the silhouette order is c > b > a, so every method
    has a mean metric rank of exactly 2 and a rank-first bio score cannot separate them. The value
    means are 0.500, 0.525 and 0.500, which puts b first.
    """
    scored = score_within(_asymmetric_rows(), normalization="raw")
    on_d1 = scored[(scored["dataset"] == "d1") & (scored["rel_reg"] == 1.0)]
    ranked = rank_within(
        _asymmetric_rows(),
        metrics=(PURITY, SILHOUETTE),
        rel_reg_values=REL_REG,
    )
    mean_metric_rank = (
        ranked[(ranked["dataset"] == "d1") & (ranked["rel_reg"] == 1.0)]
        .groupby("method")["rank"]
        .mean()
    )
    assert set(mean_metric_rank.unique()) == {2.0}, "the fixture must tie every mean metric rank"
    assert on_d1.set_index("method")["bio_rank"].to_dict() == {"a": 2.5, "b": 1.0, "c": 2.5}


def test_overall_score_is_the_equal_weighting_of_the_two_composites():
    scored = score_within(_asymmetric_rows())
    expected = 0.5 * scored["bio_score"] + 0.5 * scored["mixing_score"]
    assert np.allclose(scored["overall_score"], expected)


def test_bio_weight_moves_the_overall_score_and_defaults_to_a_half():
    scored = score_within(_asymmetric_rows(), bio_weight=0.75)
    expected = 0.75 * scored["bio_score"] + 0.25 * scored["mixing_score"]
    assert np.allclose(scored["overall_score"], expected)
    assert set(scored["bio_weight"]) == {0.75}
    assert set(score_within(_asymmetric_rows())["bio_weight"]) == {0.5}


def test_bio_weight_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match="bio_weight"):
        score_within(_asymmetric_rows(), bio_weight=1.5)


# 5. The overall rank comes from the overall score


def _overall_disagreement_rows():
    """A cell where ranking the overall score and averaging the two ranks disagree.

    Every method is first on one side and third on the other or second on both, so all three have
    a mean rank of exactly 2 and averaging the two placings cannot order them at all. By score they
    separate cleanly -- 0.500, 0.475, 0.310 -- because a's bio lead is larger than its mixing
    deficit and c's is not.
    """
    spec = {
        ("d1", "a"): (1.00, 1.00, 0.00, 0.00),
        ("d1", "b"): (0.45, 0.45, 0.50, 0.50),
        ("d1", "c"): (0.00, 0.00, 0.62, 0.62),
        ("d2", "a"): (1.00, 1.00, 0.00, 0.00),
        ("d2", "b"): (0.45, 0.45, 0.50, 0.50),
        ("d2", "c"): (0.00, 0.00, 0.62, 0.62),
    }
    return _rows(spec)


def test_overall_rank_ranks_the_score_and_does_not_average_the_two_ranks():
    scored = score_within(_overall_disagreement_rows(), normalization="raw")
    cell = scored[(scored["dataset"] == "d1") & (scored["rel_reg"] == 1.0)].set_index("method")

    by_score = cell["overall_score"].rank(ascending=False, method="average")
    assert cell["overall_rank"].to_dict() == by_score.to_dict()

    averaged = ((cell["bio_rank"] + cell["mixing_rank"]) / 2).rank(method="average")
    assert set(averaged) == {2.0}, "the fixture must tie every method on averaged ranks"
    assert cell["overall_rank"].to_dict() == {"a": 1.0, "b": 2.0, "c": 3.0}, (
        "ranking the score must separate the methods that averaging the ranks ties"
    )


def test_every_score_is_ranked_over_the_whole_field():
    """Placings lie in [1, n] and sum to n(n+1)/2, however the ties inside a cell fall.

    Not "the best placing is 1": with averaged ties a three-way tie for first gives every method a
    2.0 and no method holds rank 1, which is correct and is exactly what this fixture produces on
    the mixing side.
    """
    scored = score_within(_asymmetric_rows())
    for side in SCORES:
        by_cell = scored.groupby(["dataset", "rel_reg"])[f"{side}_rank"]
        assert by_cell.min().min() >= 1.0
        assert by_cell.max().max() <= 3.0
        assert np.allclose(by_cell.sum(), 6.0)


# 6-7. Normalisation: per cell, and the two variants stay apart


def test_minmax_is_computed_independently_within_each_dataset_rel_reg_metric_cell():
    scored = score_within(_asymmetric_rows(), normalization="minmax")
    for metric in METRICS:
        by_cell = scored.groupby(["dataset", "rel_reg"])[f"{metric}__norm"]
        assert np.allclose(by_cell.min(), 0.0)
        assert np.allclose(by_cell.max(), 1.0)


def test_rescaling_one_dataset_leaves_the_other_datasets_normalised_values_alone():
    """The cell is the unit of normalization, so a change inside one cell cannot leave it."""
    base = _asymmetric_rows()
    moved = base.copy()
    mask = moved["dataset"] == "d2"
    moved.loc[mask, METRICS] = moved.loc[mask, METRICS] * 0.25 + 0.6

    columns = ["dataset", "method", "rel_reg", *[f"{m}__norm" for m in METRICS]]
    before = score_within(base)[columns].set_index(["dataset", "method", "rel_reg"])
    after = score_within(moved)[columns].set_index(["dataset", "method", "rel_reg"])
    d1 = before.xs("d1", level="dataset"), after.xs("d1", level="dataset")
    assert np.allclose(d1[0].to_numpy(), d1[1].to_numpy()), "d1 must not see d2's rescaling"
    # a positive affine map inside a cell is exactly what min-max divides out, so d2's normalized
    # values are unchanged too -- which is the property, not a coincidence
    d2 = before.xs("d2", level="dataset"), after.xs("d2", level="dataset")
    assert np.allclose(d2[0].to_numpy(), d2[1].to_numpy())


def test_a_cell_where_every_method_ties_normalises_to_a_constant():
    """No range to divide by. The limit of the formula, and it must not be inf or NaN."""
    scored = score_within(_rows(_uniform({"a": 0.5, "b": 0.5, "c": 0.5})))
    assert np.isfinite(scored[[f"{m}__norm" for m in METRICS]].to_numpy()).all()
    assert set(scored[f"{PURITY}__norm"]) == {0.0}
    assert set(scored["bio_rank"]) == {2.0}, "an all-tie cell must place every method equal"


def test_the_two_variants_are_labelled_and_differ():
    rows = _asymmetric_rows()
    raw = score_within(rows, normalization="raw")
    minmax = score_within(rows, normalization="minmax")
    assert set(raw["normalization"]) == {"raw"}
    assert set(minmax["normalization"]) == {"minmax"}
    assert not np.allclose(raw[f"{PURITY}__norm"], minmax[f"{PURITY}__norm"])
    # the raw variant passes the metric through untouched; the min-max variant does not
    assert np.allclose(raw[f"{PURITY}__norm"], raw[PURITY])


def test_an_unknown_normalization_is_refused():
    with pytest.raises(ValueError, match="normalization must be one of"):
        score_within(_asymmetric_rows(), normalization="zscore")


def test_aggregating_a_mix_of_variants_is_refused():
    rows = _asymmetric_rows()
    mixed = pd.concat(
        [score_within(rows, normalization=n) for n in NORMALIZATIONS], ignore_index=True
    )
    with pytest.raises(ValueError, match="mix normalisation variants"):
        aggregate_scores(mixed)


# 8-10. Ranking inside a dataset, then averaging across them


def test_ranks_are_computed_inside_each_dataset():
    """``a`` is best on d1 and worst on d2, and takes both placings rather than one average."""
    spec = _uniform(
        {"a": 0.9, "b": 0.5, "c": 0.1},
        overrides={
            ("d2", "a"): (0.1, 0.1, 0.1, 0.1),
            ("d2", "c"): (0.9, 0.9, 0.9, 0.9),
        },
    )
    scored = score_within(_rows(spec), normalization="raw")
    assert _at(scored, "d1", "a", "overall_rank") == 1.0
    assert _at(scored, "d2", "a", "overall_rank") == 3.0


def test_cross_dataset_ranks_are_the_mean_of_the_dataset_level_ranks():
    spec = _uniform(
        {"a": 0.9, "b": 0.5, "c": 0.1},
        overrides={
            ("d2", "a"): (0.1, 0.1, 0.1, 0.1),
            ("d2", "c"): (0.9, 0.9, 0.9, 0.9),
        },
    )
    scored = score_within(_rows(spec), normalization="raw")
    agg = aggregate_scores(scored)
    for side in SCORES:
        by_hand = scored.groupby(["method", "rel_reg"])[f"{side}_rank"].mean()
        assert np.allclose(
            agg.set_index(["method", "rel_reg"])[f"{side}_rank"].sort_index(),
            by_hand.sort_index(),
        )
    # pinned, so the arithmetic is stated and not merely self-consistent: a is 1 then 3
    at_one = agg[(agg["method"] == "a") & (agg["rel_reg"] == 1.0)]
    assert float(at_one["overall_rank"].iloc[0]) == pytest.approx(2.0)
    assert int(at_one["n_datasets"].iloc[0]) == 2


def test_exact_ties_receive_the_average_of_the_ranks_they_span():
    spec = _uniform({"a": 0.7, "b": 0.7, "c": 0.2})
    scored = score_within(_rows(spec), normalization="raw")
    cell = scored[(scored["dataset"] == "d1") & (scored["rel_reg"] == 1.0)].set_index("method")
    assert cell["overall_rank"].to_dict() == {"a": 1.5, "b": 1.5, "c": 3.0}


# 11. Methods with no regularization knob


def test_a_method_without_a_rel_reg_holds_one_score_in_every_table():
    """Its value is fixed across the grid; its placing still moves, because its rivals' are not."""
    spec = {
        ("d1", "free"): (0.50, 0.50, 0.50, 0.50),
        ("d1", "swept"): (0.60, 0.60, 0.60, 0.60),
        ("d2", "free"): (0.50, 0.50, 0.50, 0.50),
        ("d2", "swept"): (0.60, 0.60, 0.60, 0.60),
    }
    rows = _rows(spec, free=("free",))
    # the swept method is worse at rel_reg 0.1, which is what moves the free method's placing
    weak = rows["rel_reg"] == 0.1
    rows.loc[weak & (rows["method"] == "swept"), METRICS] = 0.40

    scored = score_within(rows, normalization="raw")
    assert set(scored[scored["method"] == "free"]["rel_reg"]) == set(REL_REG)
    assert set(scored[scored["method"] == "free"]["overall_score"]) == {0.5}
    assert _at(scored, "d1", "free", "overall_rank", rel_reg=1.0) == 2.0
    assert _at(scored, "d1", "free", "overall_rank", rel_reg=0.1) == 1.0


def test_a_method_missing_from_one_dataset_leaves_the_pool_everywhere():
    """The common pool, exactly as ``rank_aggregation.resolve_method_pool`` defines it.

    Dropping it from *every* dataset rather than from the one it is missing on is what keeps
    min-max honest: the denominator is then computed against the same field in every cell. A
    method present on one dataset and not the other would otherwise rescale its rivals' scores on
    the dataset it does appear on.
    """
    spec = _uniform({"a": 0.9, "b": 0.5, "c": 0.1})
    del spec[("d2", "c")]
    scored = score_within(_rows(spec))
    assert set(scored["method"]) == {"a", "b"}
    assert set(scored["n_competitors"]) == {2}


def test_a_method_missing_at_one_rel_reg_is_refused():
    """The pool resolver intersects over datasets and is blind to the regularization axis."""
    rows = _rows(_uniform({"a": 0.9, "b": 0.5, "c": 0.1}))
    rows = rows[~((rows["method"] == "c") & (rows["rel_reg"] == 0.1) & (rows["dataset"] == "d2"))]
    with pytest.raises(ValueError, match="different number"):
        score_within(rows)


def test_a_missing_metric_value_is_refused():
    rows = _rows(_uniform({"a": 0.9, "b": 0.5, "c": 0.1}))
    rows.loc[rows.index[0], PURITY] = np.nan
    with pytest.raises(ValueError, match="missing metric"):
        score_within(rows)


def test_asking_for_a_rel_reg_the_frame_does_not_hold_is_refused():
    with pytest.raises(ValueError, match="no rows at rel_reg"):
        score_within(_asymmetric_rows(), rel_reg_values=(0.01,))


def test_metric_sides_needs_both_sides_populated():
    assert metric_sides(RANK_METRICS) == {
        "bio": (PURITY, SILHOUETTE),
        "mixing": (ILISI, KBET),
    }
    with pytest.raises(ValueError, match="no mixing metric"):
        metric_sides((PURITY, SILHOUETTE))
    with pytest.raises(ValueError, match="METRIC_FAMILY"):
        metric_sides((PURITY, "made_up_score"))


# The methodological example


def test_average_of_metric_ranks_and_rank_of_average_values_can_disagree():
    """Show that tied mean metric ranks can coexist with unequal mean scores.

    For purity/silhouette pairs (0.9, 0.1), (0.55, 0.5), and (0.1, 0.9), all mean
    ranks are 2, but the middle method has the highest mean score, 0.525.
    """
    rows = _asymmetric_rows()

    ranked = rank_within(rows, metrics=(PURITY, SILHOUETTE), rel_reg_values=(1.0,))
    on_d1 = ranked[ranked["dataset"] == "d1"]
    rank_first = on_d1.groupby("method")["rank"].mean().to_dict()
    assert rank_first == {"a": 2.0, "b": 2.0, "c": 2.0}

    scored = score_within(rows, rel_reg_values=(1.0,), normalization="raw")
    values_first = scored[scored["dataset"] == "d1"].set_index("method")["bio_score"].to_dict()
    assert values_first["b"] > values_first["a"] == values_first["c"]

    # and the two orderings really are different objects, not one rounded differently
    assert set(rank_first.values()) == {2.0}
    assert len(set(values_first.values())) == 2


# The existing pipeline is unchanged


def _regression_rows():
    """A fixture with no ties and no symmetry, so a pinned number is sensitive to any change."""
    spec = {
        ("d1", "a"): (0.90, 0.70, 0.30, 0.40),
        ("d1", "b"): (0.60, 0.80, 0.50, 0.20),
        ("d1", "c"): (0.30, 0.20, 0.90, 0.95),
        ("d2", "a"): (0.20, 0.65, 0.35, 0.45),
        ("d2", "b"): (0.75, 0.15, 0.55, 0.25),
        ("d2", "c"): (0.40, 0.85, 0.85, 0.90),
    }
    return _rows(spec)


def test_the_rank_first_pipeline_still_gives_exactly_what_it_used_to():
    """A numeric pin on ``rank_within`` + ``aggregate_ranks``, computed before this work began.

    The score-before-rank pipeline imports ``rank_aggregation``'s direction table, family split and
    epsilon broadcast. Those imports are what keep the two procedures from drifting apart, and they
    are also a way for a change made for the new pipeline to reach the old one. This pins the old
    one's output so that cannot happen silently. ``tests/test_rank_aggregation.py`` covers the same
    module property by property; this is the end-to-end number.
    """
    ranked = rank_within(_regression_rows(), rel_reg_values=(1.0,))
    agg = aggregate_ranks(ranked, weight="family", statistic="mean")
    actual = agg.set_index("method")["aggregate_rank"].round(6).to_dict()
    assert actual == {"a": 2.25, "b": 2.125, "c": 1.625}

    per_metric = (
        ranked.groupby(["method", "metric"])["rank"].mean().unstack().round(6).to_dict("index")
    )
    assert per_metric["a"] == {PURITY: 2.0, SILHOUETTE: 2.0, ILISI: 3.0, KBET: 2.0}
    assert per_metric["b"] == {PURITY: 1.5, SILHOUETTE: 2.0, ILISI: 2.0, KBET: 3.0}
    assert per_metric["c"] == {PURITY: 2.5, SILHOUETTE: 2.0, ILISI: 1.0, KBET: 1.0}


# The comparison layer


def test_leaders_returns_more_than_k_only_when_the_boundary_is_tied():
    ranks = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
    assert leaders(ranks, 3) == ("a", "b", "c")
    tied = pd.Series({"a": 1.0, "b": 2.0, "c": 3.5, "d": 3.5})
    assert leaders(tied, 3) == ("a", "b", "c", "d")


def test_compare_rankings_reports_the_coefficient_and_the_leaders_together():
    left = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
    right = pd.Series({"a": 2.0, "b": 1.0, "c": 3.0, "d": 4.0})
    out = compare_rankings(left, right)
    assert out["n_methods"] == 4
    assert out["spearman_r"] == pytest.approx(0.8)
    assert out["top1_left"] == "a" and out["top1_right"] == "b"
    assert out["top1_changed"] is True
    assert out["top3_changed"] is False


def test_compare_rankings_refuses_to_correlate_different_method_sets():
    left = pd.Series({"a": 1.0, "b": 2.0})
    right = pd.Series({"a": 1.0, "c": 2.0})
    with pytest.raises(ValueError, match="same methods"):
        compare_rankings(left, right)


def test_matched_k_multimap_and_ambiguous_method_columns_are_refused():
    rows = _seed_split_rows()
    k15 = rows.assign(
        method=rows["method"].where(rows["method"] != rows["method"].iloc[0], "multimap-k15")
    )
    with pytest.raises(ValueError, match="matched-k MultiMAP"):
        score_within(k15)
    with pytest.raises(ValueError, match="both 'method' and 'method_key'"):
        score_within(rows.assign(method_key=rows["method"]))


# Runs are never averaged together


def _two_k_rows(device_for_30="cpu"):
    """``ot`` at k=15 and k=30 (one run each), plus a method without k, over two seeds."""
    records = []
    for dataset in ("d1", "d2"):
        for seed in (0, 1):
            for k, purity, device in ((15.0, 0.9, "cpu"), (30.0, 0.1, device_for_30)):
                records.append(
                    dict(
                        dataset=dataset,
                        method="ot",
                        rel_reg=1.0,
                        effective_k=k,
                        device=device,
                        seed=seed,
                        **{m: 0.5 for m in METRICS},
                    )
                    | {PURITY: purity}
                )
            records.append(
                dict(
                    dataset=dataset,
                    method="seurat-50",
                    rel_reg=float("nan"),
                    effective_k=float("nan"),
                    device="cpu",
                    seed=seed,
                    **{m: 0.5 for m in METRICS},
                )
            )
    return pd.DataFrame(records)


def test_two_k_values_are_an_error_not_an_average():
    for function in (_score_within, _rank_within):
        with pytest.raises(ValueError, match="several configurations.*effective_k"):
            function(_two_k_rows(), rel_reg_values=(1.0,), effective_k=None)


def test_choosing_a_k_ranks_that_run_alone():
    for k, purity in ((15.0, 0.9), (30.0, 0.1)):
        scored = _score_within(_two_k_rows(), rel_reg_values=(1.0,), effective_k=k)
        assert _at(scored, "d1", "ot", PURITY) == purity
        assert len(scored) == 4  # two methods on two datasets


def test_rows_that_differ_only_in_device_are_kept_apart():
    rows = _two_k_rows(device_for_30="cuda").assign(effective_k=15.0)
    rows.loc[rows["method"] == "seurat-50", "effective_k"] = float("nan")
    with pytest.raises(ValueError, match="several configurations.*device"):
        _score_within(rows, rel_reg_values=(1.0,), effective_k=15.0)


def test_runs_that_no_column_tells_apart_are_refused():
    rows = _two_k_rows().drop(columns=["effective_k"])
    with pytest.raises(ValueError, match="same seed more than once"):
        _score_within(rows, rel_reg_values=(1.0,), effective_k=None)


def test_unequal_seeds_and_missing_seed_column_are_refused():
    rows = _two_k_rows()
    short = rows[~((rows["method"] == "seurat-50") & (rows["seed"] == 1))]
    with pytest.raises(ValueError, match="different seeds"):
        _score_within(short, rel_reg_values=(1.0,), effective_k=15.0)
    with pytest.raises(ValueError, match="'seed' column"):
        _score_within(rows.drop(columns=["seed"]), rel_reg_values=(1.0,), effective_k=15.0)


def test_there_is_no_default_configuration():
    with pytest.raises(TypeError, match="rel_reg_values|effective_k"):
        _score_within(_two_k_rows())
    with pytest.raises(TypeError, match="rel_reg_values|effective_k"):
        _rank_within(_two_k_rows())
