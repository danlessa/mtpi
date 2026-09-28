"""World run of the published mTPI versions from one smooth-window pass per cell.

Each cell's DEV is computed once (``mtpi.multiscale``) at every window size
the products need, reduced to DEVmax per band, and written as every product:

    v1   R = DEVmax 30-300 km,   G = 3-30 km,   B = 0.3-3 km
    v2   R = DEVmax 300-3000 km, G = 30-300 km, B = 3-30 km

per product: ``rgb/<cell>.tif`` (RGBA COG), ``dev/<cell>.tif`` (int16 DEVmax x
100, i.e. 0.01 z steps -- finer than one colour level -- with GDAL scale 0.01
and band descriptions naming the ranges) and XYZ tile fragments for
``mtpi.xyz``.  Band sampling density is per range: dense where scales are
cheap (coarse grids), sparser where they cost full-resolution passes.

Streaming to R2, resume (a product is done when its RGB COG exists at the
destination), per-cell fault tolerance and pass 1 come from ``mtpi.tiled``.

    python -m mtpi.products --world --dest-root s3://mtpi-fabdem
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rasterio.enums import ColorInterp

from . import tiled
from .multiscale import _reach, _widths, level_for, log_scales, multiscale_dev
from .render import _stretch_abs
from .tiled import DATA, N, Cell, cell_name, dem_cells, wrap_lon

# band name -> (smallest, largest window in metres, window sizes per decade)
BANDS: dict[str, tuple[float, float, int]] = {
    "0.3-3km": (300.0, 3e3, 6),
    "3-30km": (3e3, 3e4, 8),
    "30-300km": (3e4, 3e5, 12),
    "300-3000km": (3e5, 3e6, 12),
}


DEV_SCALE = 100  # stored int16 = round(DEVmax * DEV_SCALE)


@dataclass(frozen=True)
class Product:
    name: str
    rgb: tuple[str, str, str]  # band names for R, G, B
    dev: tuple[str, ...]  # band names kept in dev/


PRODUCTS = (
    Product("v1", rgb=("30-300km", "3-30km", "0.3-3km"), dev=("0.3-3km", "3-30km", "30-300km")),
    Product("v2", rgb=("300-3000km", "30-300km", "3-30km"), dev=("300-3000km",)),
)


def scales_for(bands: dict) -> list[float]:
    """Union of every band's log-spaced window sizes (shared endpoints once)."""
    out: list[float] = []
    for lo, hi, per_decade in bands.values():
        for L in log_scales(lo, hi, per_decade):
            if not any(abs(L - x) <= 1e-6 * x for x in out):
                out.append(float(L))
    return sorted(out)


@dataclass
class Job:
    products: tuple[Product, ...] = PRODUCTS
    bands: dict = field(default_factory=lambda: dict(BANDS))
    clip: float = 2.0
    codec: str = "zstd"  # rgb COGs: zstd (fast, lossless) | webp-lossless | webp (lossy)
    dest_root: str | None = None  # s3://bucket -> products at s3://bucket/<name>/
    out_root: Path = DATA / "out" / "products"
    frag_root: Path = DATA / "interim" / "xyz_frag_ms"
    xyz_zoom: int | None = 9
    overwrite: bool = False
    base: tiled.Config = field(default_factory=tiled.Config)  # coarse cache, global grid

    def cfg(self, p: Product) -> tiled.Config:
        """tiled.Config for one product (paths, destination) -- reuses tiled's I/O helpers."""
        return dataclasses.replace(
            self.base, out_dir=self.out_root / p.name, frag_dir=self.frag_root / p.name,
            dest=f"{self.dest_root.rstrip('/')}/{p.name}" if self.dest_root else None,
            xyz_zoom=self.xyz_zoom, keep_dev=True, overwrite=self.overwrite)


def _rgb_opts(codec: str) -> dict:
    return {"zstd": dict(compress="zstd", predictor="yes", level=9),
            "webp-lossless": dict(compress="webp", quality=100),
            "webp": dict(compress="webp", quality=90)}[codec]


def _product_done(job: Job, p: Product, cell: Cell) -> bool:
    cfg = job.cfg(p)
    frags = job.xyz_zoom is None or tiled._frag_marker(cfg, cell).exists()
    return not job.overwrite and frags and tiled._done(cfg, tiled.rgb_path(cfg, cell))


def run_cell(cell: Cell, job: Job) -> tuple[Cell, float]:
    t = time.perf_counter()
    todo = [p for p in job.products if not _product_done(job, p, cell)]
    if not todo:
        return cell, 0.0
    ranges = {k: (lo, hi) for k, (lo, hi, _) in job.bands.items()}
    zc, _, mx = multiscale_dev(cell, scales_for(job.bands), cfg=job.base, keep=[], bands=ranges)
    valid = np.isfinite(zc)
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    for p in todo:
        cfg = job.cfg(p)
        dev = np.stack([np.where(valid, np.round(mx[k] * DEV_SCALE).clip(-32767, 32767), -32768)
                        for k in p.dev]).astype(np.int16)
        tiled._write_cog(tiled.dev_path(cfg, cell), dev, cell, nodata=-32768,
                         descriptions=[f"DEVmax {k}" for k in p.dev],
                         scales=[1 / DEV_SCALE] * len(p.dev), compress="zstd", predictor="yes")
        tiled._publish(cfg, tiled.dev_path(cfg, cell))
        rgba = np.stack([_stretch_abs(np.nan_to_num(mx[k]), valid, job.clip) for k in p.rgb] + [alpha])
        if job.xyz_zoom is not None:
            from .xyz import cell_fragments, write_fragments

            write_fragments(cell_fragments(rgba, cell, job.xyz_zoom), cell_name(cell),
                            cfg.frag_dir, job.xyz_zoom)
            tiled._frag_marker(cfg, cell).parent.mkdir(parents=True, exist_ok=True)
            tiled._frag_marker(cfg, cell).touch()
        # RGB last: its presence at the destination marks the product done for this cell.
        tiled._write_cog(tiled.rgb_path(cfg, cell), rgba, cell,
                         colorinterp=[ColorInterp.red, ColorInterp.green, ColorInterp.blue,
                                      ColorInterp.alpha],
                         descriptions=[f"|DEVmax {k}|" for k in p.rgb] + ["land"],
                         **_rgb_opts(job.codec))
        tiled._publish(cfg, tiled.rgb_path(cfg, cell))
    return cell, time.perf_counter() - t


def pass1_cells(targets: list[Cell], job: Job) -> list[Cell]:
    """Cells whose block stats (30 px cache) the targets' level-30 windows reach."""
    scales = [L for L in scales_for(job.bands) if level_for(L) == 30]
    if not scales:
        return []
    need: set[Cell] = set()
    nb = N // 30
    for lat, lon in targets:
        py = max(3 * _reach(_widths(L, np.zeros(1), 30)[0]) for L in scales) + 2
        far = tiled.row_lats(lat, np.array([-py, nb + py]), 30)
        px = max(3 * _reach(float(_widths(L, far, 30)[1].max())) for L in scales) + 2
        ky, kx = math.ceil(py / nb), min(math.ceil(px / nb), 180)
        need |= {(lat + a, wrap_lon(lon + b)) for a in range(-ky, ky + 1)
                 for b in range(-kx, kx + 1)} & dem_cells()
    return sorted(need)


def run(targets: list[Cell], job: Job, workers: int) -> list[Cell]:
    p1 = pass1_cells(targets, job)
    print(f"[plan] {len(targets)} target cells; block stats for {len(p1)} cells; "
          f"{len(scales_for(job.bands))} window sizes; products {[p.name for p in job.products]}")
    failed1 = set(tiled._map(tiled.run_coarse, p1, job.base, workers, "coarse"))
    if any(level_for(L) == 120 for L in scales_for(job.bands)):
        # The global grid reads every cell's block stats (built from the full cache).
        print(f"[global] {tiled.build_global(job.base)}", flush=True)
    blocked = {c for c in targets if failed1 & set(pass1_cells([c], job))}
    failed2 = tiled._map(run_cell, [c for c in targets if c not in blocked], job, workers, "dev")
    failed = sorted(failed1 | blocked | set(failed2))
    if failed:
        print(f"[run] {len(failed)} cells failed or blocked: "
              f"{' '.join(cell_name(c) for c in failed)}", flush=True)
    return failed


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="World run of the mTPI products (v1, v2).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    g.add_argument("--cells", nargs="+", metavar="CELL")
    g.add_argument("--world", action="store_true")
    p.add_argument("--dest-root", metavar="s3://BUCKET", help="stream products to BUCKET/<name>/")
    p.add_argument("--out-root", type=Path, default=Job.out_root)
    p.add_argument("--frag-root", type=Path, default=Job.frag_root)
    p.add_argument("--codec", choices=["zstd", "webp-lossless", "webp"], default="zstd")
    p.add_argument("--xyz-zoom", type=int, default=9, help="-1 disables tile fragments")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args(argv)

    if args.world:
        targets = sorted(dem_cells())
    elif args.bbox:
        targets = tiled.cells_in_bbox(*args.bbox)
    else:
        by_name = {cell_name(c): c for c in dem_cells()}
        targets = sorted(by_name[n] for n in args.cells)
    job = Job(codec=args.codec, dest_root=args.dest_root, out_root=args.out_root,
              frag_root=args.frag_root, xyz_zoom=None if args.xyz_zoom < 0 else args.xyz_zoom,
              overwrite=args.overwrite)
    if run(targets, job, args.workers):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
