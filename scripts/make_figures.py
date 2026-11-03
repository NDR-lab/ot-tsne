#!/usr/bin/env python3
"""Build verified current figure families from completed artifacts."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    builder = Path(__file__).resolve().parents[1] / "scripts/figures/build.py"
    runpy.run_path(
        str(builder),
        run_name="__main__",
    )
