"""Check global-scale DEV (e.g. 3000 km) against a finer-block reference.

A 3000 km window can't be brute-forced at 30 m (~1e10 px per sample), so this
compares the production path -- the global ~10 km block grid (``global_dev``)
-- with the same statistics computed from the ~1 km per-cell block stats of
every cell the window reaches (``coarse_dev`` with the scale forced regional).
The difference is the cost of the coarser block snapping and interpolation.

    python scripts/validate_global.py S26W050 N40E044 N66E179 N79E015
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np

from mtpi.tiled import Config, cell_name, coarse_dev, dem_cells, global_dev, read_cell


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cells", nargs="+")
    p.add_argument("--scale", type=float, default=3_000_000.0)
    p.add_argument("--global-block", type=int, default=Config.global_block)
    args = p.parse_args()
    by_name = {cell_name(c): c for c in dem_cells()}
    label = f"{args.scale / 1000:g}km"
    prod = Config(scales_m={label: args.scale}, global_block=args.global_block)
    ref_cfg = dataclasses.replace(prod, global_above_m=float("inf"))

    for name in args.cells:
        cell = by_name[name]
        z = read_cell(cell)
        ref = float(np.nanmean(z))
        zc = z - ref
        got = global_dev(cell, zc, ref, prod)[label]
        want = coarse_dev(cell, zc, ref, ref_cfg)[label]
        d = np.abs(got - want)[np.isfinite(zc)]
        print(f"{name}: |DEV global - DEV 1km-blocks|  max={d.max():.4f}  "
              f"p99={np.percentile(d, 99):.4f}  median={np.median(d):.4f}  "
              f"(DEV range {np.nanmin(want):+.2f}..{np.nanmax(want):+.2f})", flush=True)


if __name__ == "__main__":
    main()
