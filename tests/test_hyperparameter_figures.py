"""Hyperparameter and initialisation figures: change tables, rank grid, galleries and the CLI."""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest

from ne_aligned_utils.config import ROOT
from ne_aligned_utils.paper.summaries import hyperparameters as summary

SOURCE = {
    "purity": "neighborhood_purity_score",
    "silhouette": "silhouette_score",
    "ilisi": "ilisi_score",
    "kbet": "kbet_score",
}
SETTINGS = json.dumps({"ot_initialization": "random", "perplexity": 30.0}, sort_keys=True)


def _row(dataset, method, seed, *, rel_reg=1.0, k=15, init="random", value=0.5, settings=None):
    row = dict(
        dataset=dataset,
        method=method,
        seed=seed,
        seed_index=seed,
        rel_reg=rel_reg,
        effective_k=k,
        graph_symmetrization="or",
        knndiag_symmetrization="or",
        transport="balanced",
        rel_reg_m=np.nan,
        backend="tsne",
        device="cpu",
        method_settings=settings
        or json.dumps({"ot_initialization": init, "perplexity": 30.0}, sort_keys=True),
        init=init,
        source_cache="unused",
    )
    row.update({column: value for column in SOURCE.values()})
    return row


def _sweep(datasets=("nCoV", "PBMC"), rel_regs=(0.1, 1.0, 10.0), ks=(5, 15, 60)):
    rows = []
    for d in datasets:
        for r in rel_regs:
            for k in ks:
                # Lower rel_reg: better mixing, worse bio. Seeds differ by +-0.001.
                shift = 0.05 * np.log10(r)
                for seed, noise in enumerate((-0.001, 0.0, 0.001)):
                    row = _row(d, "ot", seed, rel_reg=r, k=k)
                    row[SOURCE["purity"]] = 0.6 + shift + noise
                    row[SOURCE["silhouette"]] = 0.5 + shift + noise
                    row[SOURCE["ilisi"]] = 0.7 - shift + noise
                    row[SOURCE["kbet"]] = 0.8 - shift + noise
                    rows.append(row)
    return pd.DataFrame(rows)


def test_grid_changes_are_relative_to_the_default_point():
    changes = summary.grid_changes(_sweep(), datasets=["nCoV", "PBMC"])
    default = changes[(changes["rel_reg"] == 1.0) & (changes["effective_k"] == 15)]
    assert np.allclose(default["delta"], 0.0) and (default["direction"] == 0).all()
    low = changes[(changes["rel_reg"] == 0.1) & (changes["metric"] == "bio")]
    assert np.allclose(low["delta"], -0.05) and (low["direction"] == -1).all()
    mixing = changes[(changes["rel_reg"] == 0.1) & (changes["metric"] == "mixing")]
    assert (mixing["direction"] == 1).all()
    pooled = summary.pooled_grid(changes, ["nCoV", "PBMC"])
    cell = pooled[(pooled["rel_reg"] == 0.1) & (pooled["effective_k"] == 5)]
    bio = cell[cell["metric"] == "bio"].iloc[0]
    assert bio["n_down"] == 2 and bio["n_up"] == 0
    line = summary.grid_line(changes, "effective_k")
    assert set(line["rel_reg"]) == {1.0} and set(line["effective_k"]) == {5, 15, 60}


def test_noise_rules_and_minimum_change():
    frame = _sweep()
    # Shift one point by 0.008 (below 1 point) and another by 0.03 with default-SD noise.
    small = (frame["rel_reg"] == 10.0) & (frame["effective_k"] == 5) & (frame["dataset"] == "nCoV")
    for column in SOURCE.values():
        frame.loc[small, column] = (
            frame.loc[
                (frame["rel_reg"] == 1.0)
                & (frame["effective_k"] == 15)
                & (frame["dataset"] == "nCoV"),
                column,
            ].to_numpy()
            + 0.008
        )
    changes = summary.grid_changes(frame, datasets=["nCoV", "PBMC"])
    cell = changes[(changes["dataset"] == "nCoV") & (changes["rel_reg"] == 10.0)]
    cell = cell[(cell["effective_k"] == 5) & (cell["metric"] == "overall")].iloc[0]
    assert cell["delta"] == pytest.approx(0.008) and cell["direction"] == 0
    loose = summary.grid_changes(
        frame, datasets=["nCoV", "PBMC"], noise="default_sd", noise_factor=2.0, min_change=0.0
    )
    row = loose[(loose["dataset"] == "nCoV") & (loose["rel_reg"] == 10.0)]
    row = row[(row["effective_k"] == 5) & (row["metric"] == "overall")].iloc[0]
    # Seeds differ by +-0.001, so the default's SD is 0.001 and 0.008 is beyond 2 SD.
    assert row["direction"] == 1 and set(loose["noise_rule"]) == {"default_sd"}
    with pytest.raises(ValueError, match="noise must be"):
        summary.grid_changes(frame, datasets=["nCoV"], noise="sd")


def test_grid_refuses_to_average_other_settings_or_bad_seeds():
    frame = _sweep()
    other = frame.copy()
    other["graph_symmetrization"] = "and"
    with pytest.raises(ValueError, match="graph_symmetrization"):
        summary.grid_changes(pd.concat([frame, other]), datasets=["nCoV", "PBMC"])
    with pytest.raises(ValueError, match="same seed"):
        summary.grid_changes(pd.concat([frame, frame]), datasets=["nCoV", "PBMC"])
    with pytest.raises(ValueError, match="default point"):
        summary.grid_changes(frame[frame["rel_reg"] != 1.0], datasets=["nCoV", "PBMC"])


def _inits(datasets=("nCoV", "PBMC"), methods=("ot", "ot-gramdiag")):
    rows = []
    for d in datasets:
        for m in methods:
            for init, base in (("random", 0.5), ("spectral", 0.52), ("eot", 0.5)):
                for seed, noise in enumerate((-0.001, 0.0, 0.001)):
                    rows.append(_row(d, m, seed, init=init, value=base + noise))
    return pd.DataFrame(rows)


def test_init_changes_compare_each_init_with_random():
    changes = summary.init_changes(
        _inits(), datasets=["nCoV", "PBMC"], methods=["ot", "ot-gramdiag"]
    )
    spectral = changes[(changes["init"] == "spectral") & (changes["metric"] == "purity")]
    assert np.allclose(spectral["delta"], 0.02) and (spectral["direction"] == 1).all()
    eot = changes[(changes["init"] == "eot")]
    assert np.allclose(eot["delta"], 0.0)
    pooled = summary.pooled_init(changes, ["nCoV", "PBMC"])
    assert {"mean_sd", "n_up", "n_down"} <= set(pooled.columns)
    # The mean change is judged like one dataset's: 0.02 is beyond 2 SE of the mean and >= 1
    # point; no change is not.
    by_init = pooled[pooled["metric"] == "purity"].groupby("init")["direction"].unique()
    assert list(by_init["spectral"]) == [1] and list(by_init["eot"]) == [0]
    assert summary.init_order(["pca", "eot", "random", "spectral"]) == [
        "random",
        "spectral",
        "eot",
        "pca",
    ]


def test_init_refuses_runs_that_differ_in_more_than_the_init():
    frame = _inits()
    epochs = json.dumps({"ot_initialization": "spectral", "perplexity": 50.0}, sort_keys=True)
    frame.loc[frame["init"] == "spectral", "method_settings"] = epochs
    with pytest.raises(ValueError, match="settings_besides_init"):
        summary.init_changes(frame, datasets=["nCoV", "PBMC"], methods=["ot", "ot-gramdiag"])
    no_random = _inits()
    no_random = no_random[no_random["init"] != "random"]
    with pytest.raises(ValueError, match="random"):
        summary.init_changes(no_random, datasets=["nCoV", "PBMC"], methods=["ot"])


def test_neighbour_agreement_ignores_rotation_and_scale():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 2))
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    assert summary.neighbour_agreement(a, 3.0 * a @ rotation) == pytest.approx(1.0)
    assert summary.neighbour_agreement(a, rng.normal(size=(200, 2))) < 0.3


def _paper_and_sweep(datasets=("nCoV", "PBMC"), rel_regs=((1.0, 0.6), (0.1, 0.4)), ks=(15,)):
    rows = []
    rows = []
    for d in datasets:
        for method, value in (
            ("ot", 0.6),
            ("multimap", 0.55),
            ("joint", 0.5),
            ("harmony-50", 0.45),
        ):
            for seed in range(3):
                row = _row(d, method, seed, value=value + 0.001 * seed)
                if method != "ot":
                    row.update(rel_reg=np.nan, effective_k=np.nan, init="random")
                rows.append(row)
    paper = pd.DataFrame(rows).rename(columns={v: k for k, v in SOURCE.items()})
    paper["method_key"] = paper["method"]
    sweep = []
    for d in datasets:
        for r, value in rel_regs:
            for k in ks:
                for seed in range(3):
                    sweep.append(_row(d, "ot", seed, rel_reg=r, k=k, value=value + 0.001 * seed))
    return paper, pd.DataFrame(sweep)


def test_ot_rank_grid_moves_only_ot_tsne():
    datasets = ["nCoV", "PBMC"]
    paper, sweep = _paper_and_sweep(datasets)
    ranks = summary.ot_rank_grid(paper, sweep, datasets=datasets, normalization="raw")
    overall = ranks[ranks["block"] == "all"].set_index(["source", "rel_reg"])
    assert overall.loc[("reported", np.nan), "aggregate_rank"] == 1.0
    assert overall.loc[("sweep", 1.0), "placing"] == 1
    # 0.6 vs multimap 0.55: one rank ahead of the best other method.
    assert overall.loc[("sweep", 1.0), "lead"] == 1.0
    assert overall.loc[("sweep", 1.0), "best_other"] == "multimap"
    assert overall.loc[("sweep", 0.1), "aggregate_rank"] == 4.0
    assert overall.loc[("sweep", 0.1), "placing"] == 4
    assert overall.loc[("sweep", 0.1), "lead"] == -3.0


def test_context_figures_render(tmp_path):
    from ne_aligned_utils.paper import layout
    from ne_aligned_utils.paper.figures import hyperparameters as draw

    datasets = ["nCoV", "PBMC"]
    grid = [(r, 0.6 - 0.1 * abs(np.log10(r))) for r in (0.1, 0.5, 1.0, 5.0, 10.0)]
    paper, sweep = _paper_and_sweep(datasets, rel_regs=grid, ks=(5, 15, 60))
    ranks = summary.ot_rank_grid(paper, sweep, datasets=datasets, normalization="raw")
    changes = summary.grid_changes(sweep, datasets=datasets)
    context = summary.context_scores(paper, datasets=datasets)
    assert set(context["method_key"]) == {"ot", "multimap", "joint", "harmony-50"}
    with layout.paper_style():
        paths = [
            draw.rank_grids(ranks, output_root=tmp_path),
            draw.placing_lines(ranks, output_root=tmp_path),
            draw.robustness_strip(
                changes, context, "overall", datasets=datasets, output_root=tmp_path
            ),
            draw.absolute_lines(changes, context, "bio", datasets=datasets, output_root=tmp_path),
        ]
    assert all(path.is_file() for path in paths)


@pytest.fixture(scope="module")
def cli():
    path = ROOT / "scripts/figures/hyperparameters.py"
    spec = importlib.util.spec_from_file_location("hyperparameter_figures", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_grid_cli_writes_every_figure(tmp_path, cli):
    root = tmp_path / "sweep"
    root.mkdir()
    _sweep().to_csv(root / "metrics.csv", index=False)
    out = tmp_path / "figures"
    cli.main(["grid", "--metrics", str(root), "--output-dir", str(out)])
    names = {p.name for p in out.iterdir()}
    for name in (
        "sensitivity-pooled.png",
        "sensitivity-pooled-metrics.png",
        "sensitivity-lines.png",
        "sensitivity-lines-metrics.png",
        "sensitivity-datasets-bio.png",
        "sensitivity-datasets-kbet.png",
        "sensitivity-changes.csv",
        "sensitivity-pooled.csv",
    ):
        assert name in names
    with pytest.raises(FileExistsError):
        cli.main(["grid", "--metrics", str(root), "--output-dir", str(out)])


def test_init_cli_writes_every_figure_and_galleries(tmp_path, cli, monkeypatch):
    root = tmp_path / "init"
    root.mkdir()
    _inits().to_csv(root / "metrics.csv", index=False)
    rng = np.random.default_rng(1)
    cache = {
        "batch_ids": np.repeat([0, 1], 30),
        "labels0": np.array([np.arange(30) % 3]),
        "labels1": np.array([np.arange(30) % 3]),
        "label_names": np.array(["a", "b", "c"]),
    }

    def fake_layouts(frame, *, dataset, method, init):
        return {seed: rng.normal(size=(60, 2)) for seed in range(3)}, cache

    monkeypatch.setattr(summary, "seed_layouts", fake_layouts)
    out = tmp_path / "figures"
    cli.main(
        [
            "init",
            "--metrics",
            str(root),
            "--methods",
            "ot",
            "ot-gramdiag",
            "--output-dir",
            str(out),
            "--galleries",
            "--agreement",
        ]
    )
    names = {p.name for p in out.iterdir()}
    for name in (
        "init-pooled.png",
        "init-pooled-metrics.png",
        "init-datasets-bio.png",
        "init-seed-spread.png",
        "init-dots.png",
        "init-seeds-ot-nCoV.png",
        "init-seeds-ot-PBMC.png",
        "init-agreement.csv",
        "init-changes.csv",
    ):
        assert name in names
