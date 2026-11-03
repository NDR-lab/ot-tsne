#!/usr/bin/env python3
"""Verify the main environment and every external benchmark launcher."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _run(command: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def _failure_detail(completed: subprocess.CompletedProcess[str]) -> str:
    output = (completed.stderr or completed.stdout).strip().splitlines()
    return output[-1] if output else f"exit status {completed.returncode}"


def _launcher(variable: str) -> tuple[list[str] | None, str]:
    value = os.environ.get(variable, "")
    tokens = shlex.split(value)
    if not tokens:
        return None, f"MISSING: source .ne-aligned-env.sh ({variable} is unset)"
    executable = shutil.which(tokens[0]) or (tokens[0] if Path(tokens[0]).is_file() else None)
    if executable is None:
        return None, f"BROKEN: executable not found: {tokens[0]}"
    return tokens, value


def check_main_python() -> Check:
    try:
        import anndata  # noqa: F401
        import numpy  # noqa: F401
        import ot  # noqa: F401
        import scanpy  # noqa: F401
        import sklearn  # noqa: F401
    except Exception as error:  # pragma: no cover - exercised by real installation failures
        return Check("Main Python", False, f"{type(error).__name__}: {error}")
    return Check("Main Python", True, sys.executable)


def check_scvi() -> Check:
    try:
        import scvi
    except Exception as error:  # pragma: no cover - exercised by real installation failures
        return Check("scVI", False, f"{type(error).__name__}: {error}")
    return Check("scVI", True, scvi.__version__)


def check_ot_tsne() -> Check:
    try:
        from ne_aligned_utils.methods import compute_ot_plan  # noqa: F401
    except Exception as error:  # pragma: no cover - exercised by real installation failures
        return Check("OT-tSNE", False, f"{type(error).__name__}: {error}")
    return Check("OT-tSNE", True, "main Python environment")


def check_r(name: str, variable: str, packages: dict[str, str]) -> Check:
    launcher, display = _launcher(variable)
    if launcher is None:
        return Check(name, False, display)
    pins = ",".join(f'{package}="{version}"' for package, version in packages.items())
    expression = (
        f"pins <- c({pins}); "
        "if (getRversion() != numeric_version('4.5.3')) "
        "stop(paste(R.version.string, '!= R version 4.5.3')); "
        "for (p in names(pins)) { "
        "if (!requireNamespace(p, quietly=TRUE)) stop(paste('missing', p)); "
        "have <- as.character(packageVersion(p)); "
        "if (have != pins[[p]]) stop(paste(p, have, '!=', pins[[p]])); }; "
        "cat(R.version.string)"
    )
    completed = _run([*launcher, "-e", expression])
    if completed.returncode != 0:
        return Check(name, False, _failure_detail(completed))
    return Check(name, True, display)


def check_multimap() -> Check:
    launcher, display = _launcher("NE_ALIGNED_MULTIMAP_PYTHON")
    if launcher is None:
        return Check("MultiMAP", False, display)
    expression = (
        "import sys, numpy, scanpy, MultiMAP, importlib.metadata, json; "
        "assert sys.version_info[:3] == (3, 11, 15), sys.version; "
        "assert numpy.__version__ == '1.26.4', numpy.__version__; "
        "assert scanpy.__version__ == '1.11.5', scanpy.__version__; "
        "d=importlib.metadata.distribution('multimap'); "
        "u=json.loads(d.read_text('direct_url.json')); "
        "assert u['vcs_info']['commit_id'] == "
        "'681e608c45cdb6b139dfb6700e40c7520bc6096d', u"
    )
    completed = _run([*launcher, "-c", expression])
    if completed.returncode != 0:
        return Check("MultiMAP", False, _failure_detail(completed))
    return Check("MultiMAP", True, display)


def check_rctsne() -> Check:
    launcher, display = _launcher("NE_ALIGNED_RCTSNE_BIN")
    if launcher is None:
        return Check("rc-tSNE", False, display)
    completed = _run([*launcher, "wrong-version"], timeout=30)
    output = f"{completed.stdout}\n{completed.stderr}"
    if "t-SNE v1.2.1" not in output or "wrong version number" not in output:
        return Check("rc-tSNE", False, _failure_detail(completed))
    return Check("rc-tSNE", True, display)


def run_checks() -> list[Check]:
    return [
        check_main_python(),
        check_scvi(),
        check_ot_tsne(),
        check_r(
            "Seurat/LIGER 1/bc-tSNE",
            "NE_ALIGNED_RSCRIPT",
            {
                "hdf5r": "1.3.12",
                "Seurat": "5.5.1",
                "rliger": "1.0.1",
                "bcTSNE": "0.11.1",
            },
        ),
        check_r(
            "LIGER 2",
            "NE_ALIGNED_LIGER_RSCRIPT",
            {"hdf5r": "1.3.12", "RcppPlanc": "2.0.16", "rliger": "2.2.1"},
        ),
        check_multimap(),
        check_rctsne(),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-missing-external",
        action="store_true",
        help="return success when only dedicated external environments are unavailable",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = run_checks()
    width = max(len(check.name) for check in checks)
    for check in checks:
        status = "OK" if check.ok else "NOT READY"
        print(f"{check.name:<{width}}  {status:<9}  {check.detail}")
    required = checks[:3] if args.allow_missing_external else checks
    if not all(check.ok for check in required):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
