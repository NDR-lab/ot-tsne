import json

import numpy as np
import pandas as pd
import pytest

from ne_aligned_utils.paper import layout
from ne_aligned_utils.paper.figures.sensitivity import PAIRS, line_styles, trajectory_grid
from ne_aligned_utils.paper.summaries.sensitivity import sweep_points
from scripts.figures import sensitivity as cli

SOURCE = ("neighborhood_purity_score", "silhouette_score", "ilisi_score", "kbet_score")


def _rows(methods=("ot",), rel_regs=(0.1, 1.0, 10.0), ks=(5, 15), seeds=(0, 1, 2)):
    rng = np.random.default_rng(0)
    rows = []
    for dataset in ("PBMC", "lupus"):
        for method in methods:
            for rel_reg in rel_regs:
                for k in ks:
                    for seed in seeds:
                        row = {
                            "dataset": dataset,
                            "method": method,
                            "seed": seed,
                            "rel_reg": rel_reg,
                            "effective_k": float(k),
                            "graph_symmetrization": "or",
                            "backend": "tsne",
                            "device": "cpu",
                            "method_settings": json.dumps({"perplexity": 30.0}),
                        }
                        row.update({c: rng.uniform(0.2, 0.9) for c in SOURCE})
                        rows.append(row)
    return pd.DataFrame(rows)


def test_unfixed_settings_become_separate_trajectories_in_numeric_order():
    frame = _rows(ks=(5, 10, 60))
    points = sweep_points(frame, sweep="rel_reg", datasets=["PBMC", "lupus"])
    lines = points.drop_duplicates("line").sort_values("line")
    assert list(lines["effective_k"]) == [5.0, 10.0, 60.0]
    assert list(lines["line_label"]) == ["ot, k=5", "ot, k=10", "ot, k=60"]
    assert set(points["n_seeds"]) == {3}


def test_seed_means_and_composites_per_point():
    frame = _rows()
    points = sweep_points(
        frame, sweep="rel_reg", datasets=["PBMC", "lupus"], fixed={"effective_k": 15}
    )
    assert points["line"].nunique() == 1
    one = frame[(frame.dataset == "PBMC") & (frame.rel_reg == 1.0) & (frame.effective_k == 15)]
    point = points[(points.dataset == "PBMC") & (points.rel_reg == 1.0)].iloc[0]
    assert point["kbet_mean"] == pytest.approx(one["kbet_score"].mean())
    bio = one[["neighborhood_purity_score", "silhouette_score"]].mean(axis=1)
    assert point["bio_mean"] == pytest.approx(bio.mean())
    assert point["bio_sd"] == pytest.approx(bio.std(ddof=1))


def test_rows_of_two_runs_with_identical_settings_are_refused():
    frame = _rows()
    with pytest.raises(ValueError, match="seed"):
        sweep_points(pd.concat([frame, frame]), sweep="rel_reg", datasets=["PBMC"])


def test_fixing_an_unknown_column_or_value_is_an_error():
    frame = _rows()
    with pytest.raises(ValueError, match="no such column"):
        sweep_points(frame, sweep="rel_reg", datasets=["PBMC"], fixed={"k": 15})
    with pytest.raises(ValueError, match="no rows left"):
        sweep_points(frame, sweep="rel_reg", datasets=["PBMC"], fixed={"effective_k": 7})


def test_sweep_over_a_method_setting():
    frame = _rows(methods=("rc-tsne",), rel_regs=(np.nan,), ks=(np.nan,))
    frame = pd.concat(
        [
            frame.assign(method_settings=json.dumps({"beta": beta, "pca_components": 50}))
            for beta in (0.0001, 0.1)
        ],
        ignore_index=True,
    )
    points = sweep_points(frame, sweep="beta", datasets=["PBMC", "lupus"])
    assert points["line"].nunique() == 1
    assert sorted(points["beta"].unique()) == [0.0001, 0.1]


def test_variants_keep_registry_colours_and_one_method_gets_a_ramp():
    variants = sweep_points(
        _rows(methods=("ot", "ot-knndiag")),
        sweep="rel_reg",
        datasets=["PBMC"],
        fixed={"effective_k": 15},
    )
    styles = line_styles(variants).set_index("method")
    assert styles.loc["ot", "color"] == "#4a1b04"
    assert styles.loc["ot-knndiag", "dashes"] == "--"
    ramp = line_styles(sweep_points(_rows(ks=(5, 15, 30)), sweep="rel_reg", datasets=["PBMC"]))
    assert ramp["color"].nunique() == 3
    assert ramp["color"].iloc[-1] == "#4a1b04"


@pytest.mark.parametrize("sweep", ["rel_reg", "effective_k"])
def test_every_pair_draws(tmp_path, sweep):
    points = sweep_points(_rows(), sweep=sweep, datasets=["PBMC", "lupus"])
    with layout.paper_style():
        for pair in PAIRS:
            path = trajectory_grid(
                points,
                f"{sweep}-{pair}",
                sweep=sweep,
                pair=pair,
                datasets=["PBMC", "lupus"],
                output_root=tmp_path,
                reported={"method": "ot", "rel_reg": 1.0, "effective_k": 15},
            )
            assert path.is_file()


def test_cli_writes_points_and_figures(tmp_path):
    table = tmp_path / "metrics.csv"
    _rows().to_csv(table, index=False)
    out = tmp_path / "out"
    cli.main(
        [
            "--metrics",
            str(tmp_path),
            "--sweep",
            "effective_k",
            "--fix",
            "rel_reg=1",
            "--output-dir",
            str(out),
        ]
    )
    assert sorted(p.name for p in out.iterdir()) == [
        "sensitivity-effective_k-points.csv",
        "sensitivity-effective_k-purity-kbet.png",
        "sensitivity-effective_k-silhouette-ilisi.png",
    ]
