"""Check palette contrast and color-vision accessibility.

Example: ``uv run python scripts/figures/palette_validation.py "#2a78d6,#eb6834"``
"""

from __future__ import annotations

import argparse
import math

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
CVD = {
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}

def hex2srgb(h):
    h = h.strip().lstrip("#")
    return [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)]

def s2lin(c): return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
def lin2s(c):
    c = min(1.0, max(0.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
def lin(h): return [s2lin(c) for c in hex2srgb(h)]
def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
def contrast(a, b):
    hi, lo = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)

def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1/3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1/3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1/3)
    return [0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
            1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
            0.0259040371*l + 0.7827717662*m - 0.8086757660*s]

def oklab(h): return oklab_from_lin(lin(h))
def oklch(h):
    L, a, b = oklab(h)
    return L, math.hypot(a, b)

def simulate(h, kind):
    m, v = CVD[kind], lin(h)
    out = [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]
    return oklab_from_lin([min(1.0, max(0.0, c)) for c in out])

def delta_e(h1, h2, kind=None):
    a = simulate(h1, kind) if kind else oklab(h1)
    b = simulate(h2, kind) if kind else oklab(h2)
    return 100 * math.dist(a, b)

ORDINAL_MIN_DL = 0.06
ORDINAL_LIGHT_FLOOR = 2.0


def okhue(h):
    _, a, b = oklab(h)
    return math.degrees(math.atan2(b, a)) % 360


def validate_ordinal(palette, mode="light", surface=None):
    """Check an ordinal ramp for hue, lightness, separation, and contrast."""
    surface = surface or {"light": "#fcfcfb", "dark": "#1a1a19"}[mode]
    rows, ok = [], True
    lightness = [oklch(c)[0] for c in palette]

    order = sorted(range(len(lightness)), key=lambda i: lightness[i])
    mono = order == list(range(len(order))) or order == list(reversed(range(len(order))))
    ok &= mono
    rows.append(("Lightness monotone", "PASS" if mono else "FAIL",
                 "steps read light->dark" if mono
                 else f"out of order: {[round(v, 3) for v in lightness]}"))

    gaps = [abs(lightness[i + 1] - lightness[i]) for i in range(len(lightness) - 1)]
    thin = [(palette[i], palette[i + 1], round(g, 3))
            for i, g in enumerate(gaps) if g < ORDINAL_MIN_DL]
    ok &= not thin
    rows.append(("Adjacent dL", "PASS" if not thin else "FAIL",
                 f"steps too close: {thin}" if thin else f"all gaps >= {ORDINAL_MIN_DL}"))

    by_l = sorted(palette, key=lambda c: oklch(c)[0])
    lightest = by_l[-1] if mode == "light" else by_l[0]
    cr = contrast(lightest, surface)
    ok &= cr >= ORDINAL_LIGHT_FLOOR
    rows.append(("Light-end contrast", "PASS" if cr >= ORDINAL_LIGHT_FLOOR else "FAIL",
                 f"{lightest} at {cr:.2f}:1 vs surface (floor {ORDINAL_LIGHT_FLOOR}:1)"))

    hues = [okhue(c) for c in palette]
    spread = max(hues) - min(hues)
    if spread > 180:
        spread = 360 - spread
    one_hue = spread <= 40
    ok &= one_hue
    rows.append(("Single hue", "PASS" if one_hue else "FAIL",
                 f"hue spread {spread:.0f} deg (max 40)"))
    return rows, ok


def validate(palette, mode="light", surface=None, pairs="adjacent"):
    surface = surface or {"light": "#fcfcfb", "dark": "#1a1a19"}[mode]
    lo, hi = BAND[mode]
    rows, ok = [], True

    off = [(c, round(oklch(c)[0], 3)) for c in palette if not (lo <= oklch(c)[0] <= hi)]
    ok &= not off
    rows.append(("Lightness band", "PASS" if not off else "FAIL",
                 f"outside {lo}-{hi}: {off}" if off else f"all {len(palette)} inside L {lo}-{hi}"))

    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not lowc
    rows.append(("Chroma floor", "PASS" if not lowc else "FAIL",
                 f"below floor: {lowc}" if lowc else f"all {len(palette)} >= {CHROMA_FLOOR}"))

    n = len(palette)
    pairlist = ([(i, j) for i in range(n) for j in range(i + 1, n)] if pairs == "all"
                else [(i, i + 1) for i in range(n - 1)])
    worst = min(((delta_e(palette[i], palette[j], k), k, palette[i], palette[j])
                 for k in ("protan", "deutan") for i, j in pairlist), key=lambda t: t[0])
    tri = min(delta_e(palette[i], palette[j], "tritan") for i, j in pairlist)
    state = "PASS" if worst[0] >= CVD_TARGET else ("FLOOR" if worst[0] >= CVD_FLOOR else "FAIL")
    ok &= state != "FAIL"
    rows.append(("CVD separation", state,
                 f"worst {pairs} {worst[3]}<->{worst[2]} dE {worst[0]:.1f} ({worst[1]}) . tritan {tri:.1f}"))

    nworst = min(((delta_e(palette[i], palette[j]), palette[i], palette[j]) for i, j in pairlist),
                 key=lambda t: t[0])
    nstate = "PASS" if nworst[0] >= NORMAL_FLOOR else "FAIL"
    ok &= nstate == "PASS"
    rows.append(("Normal-vision floor", nstate,
                 f"worst {pairs} {nworst[2]}<->{nworst[1]} dE {nworst[0]:.1f} (normal)"))

    low = [(c, round(contrast(c, surface), 2)) for c in palette if contrast(c, surface) < CONTRAST_MIN]
    rows.append(("Contrast vs surface", "RELIEF" if low else "PASS",
                 f"below {CONTRAST_MIN}:1, relief required (visible labels or table): {low}"
                 if low else f"all {len(palette)} >= {CONTRAST_MIN}:1"))
    return rows, ok

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("palette", help="comma-separated hex colours, in slot order")
    parser.add_argument("--mode", choices=("light", "dark"), default="light")
    parser.add_argument(
        "--surface", default=None, help="surface the marks sit on; defaults to the mode's"
    )
    parser.add_argument(
        "--pairs",
        choices=("adjacent", "all"),
        default="adjacent",
        help="adjacent for stacks, bars and lines; all for scatter, bubble and small multiples",
    )
    parser.add_argument(
        "--ordinal",
        action="store_true",
        help="check as a one-hue ordered ramp instead of as categorical hues",
    )
    args = parser.parse_args()
    palette = [c.strip() for c in args.palette.split(",") if c.strip()]
    if args.ordinal:
        rows, ok = validate_ordinal(palette, args.mode, args.surface)
    else:
        rows, ok = validate(palette, args.mode, args.surface, args.pairs)
    kind = "ordinal" if args.ordinal else f"categorical, pairs={args.pairs}"
    print(
        f"palette n={len(palette)} mode={args.mode} "
        f"surface={args.surface or 'default'} ({kind})"
    )
    for name, state, detail in rows:
        print(f"  {state:<6} {name:<22} {detail}")
    print("OK" if ok else "NOT OK")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
