"""Brute-force check of ``mtpi.tiled`` DEV output at sampled pixels.

For each target cell, reads a plain mosaic of the raw FABDEM tiles around it
(wide enough for the largest window), then at random pixels -- plus pixels on
the cell's edges, where halo/seam bugs would show -- computes mean and std over
the window directly (no integral images, no block stats) and compares with the
int16 DEV COG the pipeline wrote (``--keep-dev``).

    python scripts/validate_tiled.py --out-dir data/out/tiled_val S26W050 N61E006
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import rasterio

from mtpi.tiled import (
    DEFAULT_SCALES_M, DY, N, cell_name, dem_cells, read_cell, wrap_lon,
)


def mosaic(cell, ky: int, kx: int) -> np.ndarray:
    lat, lon = cell
    out = np.full(((2 * ky + 1) * N, (2 * kx + 1) * N), np.nan, np.float32)
    for dlat in range(-ky, ky + 1):
        if not -90 <= lat + dlat < 90:
            continue
        for dlon in range(-kx, kx + 1):
            r, c = (ky - dlat) * N, (kx + dlon) * N
            out[r:r + N, c:c + N] = read_cell((lat + dlat, wrap_lon(lon + dlon)))
    return out


def brute_dev(z: np.ndarray, r: int, c: int, length: float, lat: float) -> float:
    hy = max(1, round(length / 2 / DY))
    hx = max(1, round(length / 2 / (DY * math.cos(math.radians(lat)))))
    w = z[r - hy:r + hy + 1, c - hx:c + hx + 1].astype(np.float64)
    w = w[~np.isnan(w)]
    sd = w.std()
    return 0.0 if sd <= 1e-6 else (z[r, c] - w.mean()) / sd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cells", nargs="+")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    scales = DEFAULT_SCALES_M
    by_name = {cell_name(c): c for c in dem_cells()}

    for name in args.cells:
        cell = by_name[name]
        lat = cell[0]
        far = abs(lat) + 1 if lat >= 0 else abs(lat)  # poleward edge
        half = max(scales.values()) / 2
        ky = math.ceil(half / DY / N)
        kx = math.ceil(half / (DY * math.cos(math.radians(far))) / N)
        print(f"== {name}: mosaic {2 * ky + 1} x {2 * kx + 1} cells", flush=True)
        z = mosaic(cell, ky, kx)
        with rasterio.open(args.out_dir / "dev" / f"{name}.tif") as src:
            got = src.read().astype(np.float64) / 1000
            got_valid = src.read_masks(1) > 0

        # Random interior pixels + edge pixels (seam neighbours).
        pts = list(zip(rng.integers(0, N, args.samples), rng.integers(0, N, args.samples)))
        pts += [(0, rng.integers(N)), (N - 1, rng.integers(N)),
                (rng.integers(N), 0), (rng.integers(N), N - 1), (0, 0), (N - 1, N - 1)]
        errs = {k: [] for k in scales}
        mask_mismatch = 0
        for i, j in pts:
            r, c = ky * N + i, kx * N + j
            valid = not np.isnan(z[r, c])
            mask_mismatch += valid != got_valid[i, j]
            if not valid:
                continue
            row_lat = lat + 1 - i / 3600
            for b, (k, length) in enumerate(scales.items()):
                errs[k].append(abs(brute_dev(z, r, c, length, row_lat) - got[b, i, j]))
        for k, e in errs.items():
            e = np.array(e)
            print(f"  {k:5s} n={len(e):3d}  max|err|={e.max():.4f}  "
                  f"p95={np.percentile(e, 95):.4f}  median={np.median(e):.4f}")
        print(f"  land-mask mismatches: {mask_mismatch}/{len(pts)}")


if __name__ == "__main__":
    main()
