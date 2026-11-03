#!/usr/bin/env python3
"""Compare two figure builds by their manifests: which outputs are identical, changed, new or gone.

    uv run python scripts/figures/compare_manifests.py old/figure-manifest.json new/figure-manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(old: dict, new: dict) -> dict[str, list[str]]:
    a, b = old["outputs"], new["outputs"]
    names = sorted((set(a) | set(b)) - {"figure-manifest.json"})
    result = {"identical": [], "changed": [], "new": [], "missing": []}
    for name in names:
        if name not in b:
            result["missing"].append(name)
        elif name not in a:
            result["new"].append(name)
        elif a[name]["sha256"] == b[name]["sha256"]:
            result["identical"].append(name)
        else:
            result["changed"].append(name)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    args = parser.parse_args(argv)
    result = compare(json.loads(args.old.read_text()), json.loads(args.new.read_text()))
    print(f"identical: {len(result['identical'])}")
    for kind in ("changed", "new", "missing"):
        for name in result[kind]:
            print(f"{kind:8s} {name}")
    return 0 if not (result["changed"] or result["missing"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
