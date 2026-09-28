"""Checks for the smooth-window multiscale engine (mtpi.multiscale).

1. Full resolution vs brute force: at sample pixels, the weighted mean/std
   under the explicit 3-pass kernel (three fractional boxes convolved, per axis)
   against the engine's running-sum implementation.
2. Pyramid levels: scales just above each level threshold are recomputed one
   level finer (the more exact grid) and compared with the production level.

    python scripts/validate_multiscale.py S24W047 N03W067
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import mtpi.multiscale as M
from mtpi.tiled import DY, N, cell_name, dem_cells, row_lats


def kernel_1d(w: float) -> np.ndarray:
    k = int(math.floor(w)); f = w - k
    box = np.r_[f, np.ones(2 * k + 1), f]
    return np.convolve(np.convolve(box, box), box)


def brute(cell, L, rng, samples=12):
    lat = cell[0]
    w_y, _ = M._widths(L, np.array([lat + 0.5]), 1)
    halo = 3 * M._reach(w_y / max(math.cos(math.radians(abs(lat) + 1)), 1e-3)) + 4
    z = M.read_padded_multi(cell, halo, halo)
    _, devs, _ = M.multiscale_dev(cell, [L], force_level={L: 1})
    got = devs[L]
    errs = []
    for i, j in zip(rng.integers(0, N, samples), rng.integers(0, N, samples)):
        if np.isnan(got[i, j]):
            continue
        wy, wx = M._widths(L, row_lats(lat, np.array([i])), 1)
        ky, kx = kernel_1d(wy), kernel_1d(float(wx[0]))
        ry, rx = len(ky) // 2, len(kx) // 2
        win = z[halo + i - ry:halo + i + ry + 1, halo + j - rx:halo + j + rx + 1].astype(np.float64)
        wgt = np.outer(ky, kx) * np.isfinite(win)
        v = np.nan_to_num(win)
        m = (wgt * v).sum() / wgt.sum()
        sd = math.sqrt(max((wgt * v * v).sum() / wgt.sum() - m * m, 0))
        want = (z[halo + i, halo + j] - m) / sd if sd > 1e-6 else 0.0
        errs.append(abs(want - got[i, j]))
    return np.array(errs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cells", nargs="+")
    p.add_argument("--T", type=float, nargs="+", default=[12.0])
    p.add_argument("--brute", action="store_true", help="also run the brute-force kernel check")
    args = p.parse_args()
    by_name = {cell_name(c): c for c in dem_cells()}
    rng = np.random.default_rng(0)
    for name in args.cells:
        cell = by_name[name]
        print(f"== {name}")
        for L in (300.0, 950.0, 2040.0) if args.brute else ():
            e = brute(cell, L, rng)
            print(f"  brute force {L / 1000:5.2f} km (level 1): max |err| {e.max():.2e} over {len(e)} px")
        for T in args.T:
            for b in M.LEVELS[1:]:
                L = 1.02 * 2 * DY * b * T  # just above the threshold: the coarsest use of level b
                finer = M.LEVELS[M.LEVELS.index(b) - 1]
                _, prod, _ = M.multiscale_dev(cell, [L], force_level={L: b})
                _, ref, _ = M.multiscale_dev(cell, [L], force_level={L: finer})
                ok = np.isfinite(ref[L])
                col = lambda d: np.clip(np.abs(d), 0, 2) / 2 * 255  # rendered channel, clip 2
                d = np.abs(col(prod[L]) - col(ref[L]))[ok]
                print(f"  T={T:>2} {L / 1000:6.1f} km: level {b:>3} vs {finer:>3}: colour-level error "
                      f"max {d.max():5.1f}  p99.9 {np.percentile(d, 99.9):4.1f}  p99 {np.percentile(d, 99):4.1f}"
                      f"  median {np.median(d):.2f}")


if __name__ == "__main__":
    main()
