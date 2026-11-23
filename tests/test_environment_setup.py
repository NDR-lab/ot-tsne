"""Contracts for the external-environment preflight."""

from __future__ import annotations

import subprocess

from scripts import check_environment


def test_missing_launcher_is_reported_by_variable_name(monkeypatch):
    monkeypatch.delenv("NE_ALIGNED_MULTIMAP_PYTHON", raising=False)

    check = check_environment.check_multimap()

    assert check.ok is False
    assert "NE_ALIGNED_MULTIMAP_PYTHON is unset" in check.detail


def test_r_preflight_checks_interpreter_and_package_versions(monkeypatch):
    captured = []
    monkeypatch.setattr(
        check_environment,
        "_launcher",
        lambda _variable: (["Rscript"], "Rscript"),
    )
    monkeypatch.setattr(
        check_environment,
        "_run",
        lambda command, **_kwargs: captured.append(command)
        or subprocess.CompletedProcess(command, 0, "R version 4.5.3", ""),
    )

    check = check_environment.check_r("R methods", "NE_ALIGNED_RSCRIPT", {"Seurat": "5.5.1"})

    assert check.ok is True
    expression = captured[0][-1]
    assert "R version 4.5.3" in expression
    assert 'Seurat="5.5.1"' in expression


def test_multimap_preflight_checks_pinned_commit(monkeypatch):
    captured = []
    monkeypatch.setattr(
        check_environment,
        "_launcher",
        lambda _variable: (["python"], "python"),
    )
    monkeypatch.setattr(
        check_environment,
        "_run",
        lambda command, **_kwargs: captured.append(command)
        or subprocess.CompletedProcess(command, 0, "", ""),
    )

    check = check_environment.check_multimap()

    assert check.ok is True
    expression = captured[0][-1]
    assert "(3, 11, 15)" in expression
    assert "681e608c45cdb6b139dfb6700e40c7520bc6096d" in expression
