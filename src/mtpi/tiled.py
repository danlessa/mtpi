"""Tiled, global multiscale DEV on the native FABDEM 1-arc-second grid.

The whole-raster pipeline (``mtpi.pipeline``) warps one AOI to UTM and hands it
to WhiteboxTools, which holds everything in RAM -- fine for a region, impossible
for the globe (~2e11 cells).  This module instead works cell by cell (1 deg x 1
deg, 3600 x 3600, one FABDEM COG each) in two passes:

1. **Coarse stats** -- every land cell is reduced to ``block`` x ``block``
   blocks (default 30 px, ~1 km) of (count, mean, variance) and cached.
2. **DEV** -- per target cell:

   * *fine scales* (micro, meso: windows up to a few tens of km) are exact:
     the cell plus a halo read from its 8 neighbours' COGs, windowed sums via
     integral images (O(1) per cell regardless of window size);
   * *coarse scales* (macro, 300 km: a 5000-px halo would multiply reads ~14x)
     come from the block stats of the surrounding cells, then bilinearly
     upsampled.  Window edges snap to whole blocks -- ~0.3 % of a 300 km window.
   * *global scales* (>= 1000 km, e.g. 3000 km) would reach ~1,600 cells, so
     pass 1's block stats are first merged into one global grid of coarser
     blocks (default 120 px, ~4 km: 10800 x 5400, ~700 MB, memory-mapped so
     workers share it).  Edge snapping ~0.1 % of a 3000 km window: within
     0.02 z of a 1 km-block reference (scripts/validate_global.py).  Beyond
     ~67 deg the window runs past the pole and its half-width is capped at the
     full circle.

No reprojection: windows are square *in metres*, so the half-height is fixed
(1" of latitude ~ 30.9 m) and the half-width grows per row as 1/cos(lat).
Longitude wraps across the antimeridian.

Sea is null: FABDEM stores sea inside land tiles as exactly 0 m and has no file
for all-ocean cells; both are excluded from every window's statistics and are
transparent in the output.  The Caspian, which FABDEM stores as a flat -28.0 m
surface, is treated as sea too (see `flat_surface`).  FABDEM's Armenia/
Azerbaijan gap is filled from Copernicus GLO-30 2023_1 (`mtpi.fill`).

Outputs, one file per cell under ``out_dir`` (so regional runs accumulate into
one global tree)::

    rgb/<cell>.tif   RGBA uint8 COG: |DEV| per scale, largest scale -> R,
                     smallest -> B (default R=300 km, G=30 km, B=3 km), A=land
    dev/<cell>.tif   optional int16 COG, DEV x 1000 per scale (band descriptions
                     name the scales)

With ``--xyz-zoom Z`` each cell also leaves Web Mercator tile fragments for
``mtpi.xyz`` to assemble into an XYZ pyramid afterwards.

With ``--dest s3://bucket/prefix`` each file is uploaded as soon as it is
written and the local copy deleted, so local disk only holds the block-stats
cache (~3 GB for the globe).  Works with Cloudflare R2 (or any S3 API) via
boto3's standard environment: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
AWS_ENDPOINT_URL_S3 (``https://<account>.r2.cloudflarestorage.com``) and
AWS_DEFAULT_REGION=auto.

Run ``python -m mtpi.tiled --help``.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
import rasterio.shutil
from rasterio.enums import ColorInterp
from rasterio.errors import RasterioIOError
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.windows import Window

from .data import (
    FABDEM_NODATA, FABDEM_SUFFIX, _tile_name, dem_cells, tile_source,
)
from .render import _stretch_abs

# Remote COG reads: don't list the "directory", retry transient HTTP errors.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
# GDAL's defaults (429/500/502/503/504) miss Cloudflare's 52x timeouts, seen from R2.
os.environ.setdefault("GDAL_HTTP_RETRY_CODES", "429,500,502,503,504,520,521,522,523,524")

N = 3600  # pixels per cell side
R_EARTH = 6_371_008.8
DY = math.pi * R_EARTH / 180 / 3600  # metres per arc-second of latitude (~30.9)

DEFAULT_SCALES_M: dict[str, float] = {"micro": 3000.0, "meso": 30000.0, "macro": 300000.0}
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"

Cell = tuple[int, int]  # (lat, lon) of the SW corner, degrees


@dataclass
class Config:
    scales_m: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCALES_M))
    coarse_above_m: float = 100_000.0  # scales >= this use the block stats
    global_above_m: float = 1_000_000.0  # scales >= this use the global block grid
    block: int = 30  # coarse block side, px (must divide N)
    global_block: int = 120  # global grid block side, px (multiple of block, divides N)
    clip: float = 2.0
    codec: str = "webp"  # rgb COG compression: webp (lossy) | deflate (lossless)
    quality: int = 90
    keep_dev: bool = False
    dev_scales: tuple[str, ...] | None = None  # labels to keep in dev/ (None = all)
    coarse_dir: Path = DATA / "interim" / "coarse"
    out_dir: Path = DATA / "out" / "tiled"
    dest: str | None = None  # s3://bucket/prefix to stream outputs to (then delete local)
    xyz_zoom: int | None = None  # also cut XYZ tile fragments at this zoom (see mtpi.xyz)
    frag_dir: Path = DATA / "interim" / "xyz_frag"
    overwrite: bool = False

    def fine_scales(self) -> dict[str, float]:
        return {k: v for k, v in self.scales_m.items() if v < self.coarse_above_m}

    def coarse_scales(self) -> dict[str, float]:
        return {k: v for k, v in self.scales_m.items()
                if self.coarse_above_m <= v < self.global_above_m}

    def global_scales(self) -> dict[str, float]:
        return {k: v for k, v in self.scales_m.items() if v >= self.global_above_m}

    def rgb_order(self) -> list[str]:
        """Scale labels for (R, G, B): largest scale red, smallest blue."""
        return sorted(self.scales_m, key=self.scales_m.get, reverse=True)


# --------------------------------------------------------------------------- grid


def cell_name(cell: Cell) -> str:
    return _tile_name(*cell).removesuffix(FABDEM_SUFFIX)


def wrap_lon(lon: int) -> int:
    return (lon + 180) % 360 - 180


def cell_transform(cell: Cell) -> Affine:
    """Pixel-is-area transform matching FABDEM (centres on whole arc-seconds)."""
    lat, lon = cell
    px = 1 / 3600
    return Affine(px, 0, lon - px / 2, 0, -px, lat + 1 + px / 2)


def row_lats(lat: int, rows: np.ndarray, block: int = 1) -> np.ndarray:
    """Centre latitude of (block-)rows of cell `lat`; row 0 is the north edge."""
    return lat + 1 - (rows * block + (block - 1) / 2) / 3600


def half_widths(length_m: float, lats: np.ndarray, block: int = 1) -> tuple[int, np.ndarray]:
    """Window half-sizes in (block-)pixels for a square window of side `length_m`.

    Returns (hy, hx_per_row): the window is (2*hy+1) x (2*hx+1).
    """
    d = DY * block
    hy = max(1, round(length_m / 2 / d))
    cos = np.maximum(np.cos(np.radians(lats)), 1e-3)
    hx = np.maximum(1, np.round(length_m / 2 / (d * cos))).astype(np.int64)
    return hy, hx


# --------------------------------------------------------------------------- reading


def _read_retry(path: str, window: Window | None, tries: int = 4) -> np.ndarray:
    """Band 1 of `path` (optionally a window); retries the open *and* the read,
    since a remote COG can fail mid-read after opening fine."""
    for i in range(tries):
        try:
            with rasterio.open(path) as src:
                return src.read(1, window=window)
        except RasterioIOError:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


# FABDEM stores the Caspian as a flat surface at exactly this level.  Land below
# sea level around it can hit the same value, but never flat over ~2 km.
CASPIAN_LEVEL = -28.0
CASPIAN_CELLS = (36, 48, 46, 56)  # lat [36, 48) x lon [46, 56): cells touching it
FLAT_SIDE = 65  # px (~2 km): side of the all-level square that marks a water surface


def _square_sums(m: np.ndarray, side: int) -> np.ndarray:
    """Sums over every side x side square, indexed by its top-left pixel."""
    I = _integral(m)
    return I[side:, side:] - I[:-side, side:] - I[side:, :-side] + I[:-side, :-side]


def flat_surface(z: np.ndarray, level: float, side: int = FLAT_SIDE) -> np.ndarray:
    """Pixels inside some side x side square lying entirely at exactly `level`.

    A morphological opening of ``z == level``: keeps large water surfaces and
    drops small patches of land that happen to share the value.  Uses only
    `z`, so a whole tile always gets the same mask however it is read.
    """
    full = _square_sums(z == level, side) == side * side
    return _square_sums(np.pad(full, side - 1), side) > 0


def _in_caspian(cell: Cell) -> bool:
    lat0, lat1, lon0, lon1 = CASPIAN_CELLS
    return lat0 <= cell[0] < lat1 and lon0 <= cell[1] < lon1


def _clean(z: np.ndarray) -> np.ndarray:
    """FABDEM nodata and sea (exactly 0 m) -> NaN, in place."""
    z[(z <= FABDEM_NODATA + 1) | (z == 0)] = np.nan
    return z


def read_cell(cell: Cell, window: Window | None = None) -> np.ndarray:
    """Float32 elevations of `cell` (optionally a window), invalid as NaN."""
    if cell not in dem_cells():
        h, w = (N, N) if window is None else (window.height, window.width)
        return np.full((h, w), np.nan, np.float32)
    caspian = _in_caspian(cell)
    # The Caspian mask needs the whole tile to be the same from every reader.
    z = _read_retry(tile_source(*cell), None if caspian else window).astype(np.float32)
    if caspian:
        z[flat_surface(z, CASPIAN_LEVEL)] = np.nan
        if window is not None:
            z = z[window.toslices()]
    return _clean(z)


def read_padded(cell: Cell, hy: int, hx: int) -> np.ndarray:
    """`cell` with a (hy, hx) halo from its 8 neighbours; NaN where invalid."""
    if hy > N or hx > N:
        raise ValueError(f"halo {hy}x{hx} px exceeds one neighbouring cell")
    lat, lon = cell
    H, W = N + 2 * hy, N + 2 * hx
    out = np.full((H, W), np.nan, np.float32)
    for dlat in (1, 0, -1):
        nlat = lat + dlat
        if not -90 <= nlat < 90:
            continue
        r_off = hy - dlat * N  # row of `out` holding the neighbour's row 0
        r0, r1 = max(0, r_off), min(H, r_off + N)
        for dlon in (-1, 0, 1):
            c_off = hx + dlon * N
            c0, c1 = max(0, c_off), min(W, c_off + N)
            if r0 >= r1 or c0 >= c1:
                continue
            win = Window(c0 - c_off, r0 - r_off, c1 - c0, r1 - r0)
            out[r0:r1, c0:c1] = read_cell((nlat, wrap_lon(lon + dlon)), win)
    return out


# --------------------------------------------------------------------------- box sums


def _integral(a: np.ndarray) -> np.ndarray:
    """Summed-area table I (float64, (H+1) x (W+1)): I[r, c] = a[:r, :c].sum()."""
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    np.cumsum(a, axis=0, out=out[1:, 1:])
    np.cumsum(out[1:, 1:], axis=1, out=out[1:, 1:])
    return out


def _window_sums(I: np.ndarray, r0: int, c0: int, h: int, w: int,
                 hy: int, hx_rows: np.ndarray) -> np.ndarray:
    """Windowed sums for the h x w outputs whose centres start at array (r0, c0).

    Row i's window spans rows [r-hy, r+hy] and columns [c-hx_rows[i], c+hx_rows[i]].
    hx changes slowly with latitude, so rows are processed in runs of equal hx.
    """
    out = np.empty((h, w))
    breaks = np.flatnonzero(np.diff(hx_rows)) + 1
    for i0, i1 in zip(np.r_[0, breaks], np.r_[breaks, h]):
        hx = int(hx_rows[i0])
        top = slice(r0 + i0 - hy, r0 + i1 - hy)
        bot = slice(r0 + i0 + hy + 1, r0 + i1 + hy + 1)
        left = slice(c0 - hx, c0 - hx + w)
        right = slice(c0 + hx + 1, c0 + hx + 1 + w)
        out[i0:i1] = I[bot, right] - I[top, right] - I[bot, left] + I[top, left]
    return out


def _mean_std(n: np.ndarray, s: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Window mean and (population) std from count, sum and sum of squares."""
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / n
        sd = np.sqrt(np.maximum(q / n - m * m, 0.0))
    return m, sd


def _dev(z: np.ndarray, m: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """(z - m) / sd; 0 where the window is flat or empty, NaN where z is."""
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (z - m) / sd
    d[~(sd > 1e-6)] = 0.0
    d[np.isnan(z)] = np.nan
    return d.astype(np.float32)


# --------------------------------------------------------------------------- pass 1


def coarse_path(cfg: Config, cell: Cell) -> Path:
    return cfg.coarse_dir / f"b{cfg.block}" / f"{cell_name(cell)}.npy"


def coarse_stats(z: np.ndarray, block: int) -> np.ndarray:
    """(3, nb, nb) float32: valid count, mean and variance per block."""
    nb = N // block
    zb = z.reshape(nb, block, nb, block)
    valid = ~np.isnan(zb)
    n = valid.sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(zb, axis=(1, 3), dtype=np.float64) / n
        var = np.nansum((zb - mean[:, None, :, None]) ** 2, axis=(1, 3)) / n
    return np.stack([n, np.nan_to_num(mean), np.nan_to_num(var)]).astype(np.float32)


def run_coarse(cell: Cell, cfg: Config) -> tuple[Cell, float]:
    t = time.perf_counter()
    path = coarse_path(cfg, cell)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        stats = coarse_stats(read_cell(cell), cfg.block)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, stats)
        tmp.replace(path)
    return cell, time.perf_counter() - t


def _load_coarse(cfg: Config, cell: Cell) -> np.ndarray:
    nb = N // cfg.block
    if cell not in dem_cells():
        return np.zeros((3, nb, nb), np.float32)
    return np.load(coarse_path(cfg, cell))


def coarse_reach(cell: Cell, cfg: Config) -> tuple[int, int]:
    """(ky, kx): cells on each side whose block stats `cell`'s coarse windows reach.

    Covers the target's blocks plus the 1-block interpolation margin, padded by
    the largest coarse half-window (which widens poleward).
    """
    nb = N // cfg.block
    lats = row_lats(cell[0], np.arange(-1, nb + 1), cfg.block)
    hy, hx = 0, 0
    for length in cfg.coarse_scales().values():
        h, x = half_widths(length, lats, cfg.block)
        hy, hx = max(hy, h), max(hx, int(x.max()))
    return math.ceil((hy + 1) / nb), min(math.ceil((hx + 1) / nb), 180)


def pass1_cells(targets: list[Cell], cfg: Config) -> list[Cell]:
    """Land cells whose block stats the targets' coarse windows reach."""
    need: set[Cell] = set()
    for lat, lon in targets:
        ky, kx = coarse_reach((lat, lon), cfg)
        for dlat in range(-ky, ky + 1):
            for dlon in range(-kx, kx + 1):
                c = (lat + dlat, wrap_lon(lon + dlon))
                if c in dem_cells():
                    need.add(c)
    return sorted(need)


# --------------------------------------------------------------------------- pass 2


def _upsample(a: np.ndarray, block: int) -> np.ndarray:
    """Bilinear (nb+2)^2 block-centre grid (1-block margin) -> N x N pixels."""
    u = (np.arange(N) - (block - 1) / 2) / block + 1
    i0 = np.floor(u).astype(np.int64)
    w = u - i0
    rows = a[i0] * (1 - w)[:, None] + a[i0 + 1] * w[:, None]
    return rows[:, i0] * (1 - w)[None, :] + rows[:, i0 + 1] * w[None, :]


def coarse_dev(cell: Cell, zc: np.ndarray, ref: float, cfg: Config) -> dict[str, np.ndarray]:
    """DEV at the coarse scales from block stats. `zc` is the cell minus `ref`."""
    if not cfg.coarse_scales():
        return {}
    nb, B = N // cfg.block, cfg.block
    lat, lon = cell
    ky, kx = coarse_reach(cell, cfg)

    # Mosaic block stats of the (2ky+1) x (2kx+1) cells around the target.
    mos = np.zeros((3, (2 * ky + 1) * nb, (2 * kx + 1) * nb), np.float32)
    for dlat in range(-ky, ky + 1):
        if not -90 <= lat + dlat < 90:
            continue
        r = (ky - dlat) * nb
        for dlon in range(-kx, kx + 1):
            c = (kx + dlon) * nb
            mos[:, r:r + nb, c:c + nb] = _load_coarse(cfg, (lat + dlat, wrap_lon(lon + dlon)))

    n = mos[0].astype(np.float64)
    mean = mos[1] - ref  # centre on the cell's mean: keeps sum-of-squares well conditioned
    s = n * mean
    q = n * (mos[2] + mean * mean)

    # Window centres: the target's blocks plus a 1-block margin.
    r0, c0 = ky * nb - 1, kx * nb - 1
    lats = row_lats(lat, np.arange(-1, nb + 1), B)
    out = {}
    for label, length in cfg.coarse_scales().items():
        hy, hx = half_widths(length, lats, B)
        sums = [_window_sums(_integral(a), r0, c0, nb + 2, nb + 2, hy, hx) for a in (n, s, q)]
        m, sd = _mean_std(*sums)
        m, sd = np.nan_to_num(m), np.nan_to_num(sd)
        out[label] = _dev(zc, _upsample(m, B), _upsample(sd, B))
    return out


def global_path(cfg: Config) -> Path:
    return cfg.coarse_dir / f"global_b{cfg.global_block}.npy"


def build_global(cfg: Config) -> Path:
    """Merge every cell's block stats into one global (3, 180*k, 360*k) grid.

    Blocks of ``global_block`` px (k = N // global_block per degree) combine the
    finer blocks exactly: counts add, and mean/variance follow from the summed
    first and second moments.  Row 0 is the north edge (90 deg), column 0 is
    -180 deg.
    """
    k, g = N // cfg.global_block, cfg.global_block // cfg.block
    grid = np.zeros((3, 180 * k, 360 * k), np.float32)
    missing = []
    for lat, lon in sorted(dem_cells()):
        path = coarse_path(cfg, (lat, lon))
        if not path.exists():
            missing.append(cell_name((lat, lon)))
            continue
        n, m, v = np.load(path).astype(np.float64)
        n_ = n.reshape(k, g, k, g).sum(axis=(1, 3))
        s_ = (n * m).reshape(k, g, k, g).sum(axis=(1, 3))
        q_ = (n * (v + m * m)).reshape(k, g, k, g).sum(axis=(1, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s_ / n_
            var = np.maximum(q_ / n_ - mean * mean, 0.0)
        r, c = (89 - lat) * k, (lon + 180) * k
        grid[:, r:r + k, c:c + k] = np.stack([n_, np.nan_to_num(mean), np.nan_to_num(var)])
    if missing:
        raise FileNotFoundError(f"no block stats for {len(missing)} cells "
                                f"(rerun pass 1): {' '.join(missing[:20])}")
    path = global_path(cfg)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, grid)
    tmp.replace(path)
    return path


_GLOBAL: dict[str, np.ndarray] = {}


def global_dev(cell: Cell, zc: np.ndarray, ref: float, cfg: Config) -> dict[str, np.ndarray]:
    """DEV at the global scales from the global block grid (see `build_global`)."""
    if not cfg.global_scales():
        return {}
    key = str(global_path(cfg))
    if key not in _GLOBAL:  # memory-mapped: workers share one copy via the page cache
        _GLOBAL[key] = np.load(key, mmap_mode="r")
    grid = _GLOBAL[key]
    H, W = grid.shape[1:]
    B, nb = cfg.global_block, N // cfg.global_block
    lat, lon = cell
    lats = row_lats(lat, np.arange(-1, nb + 1), B)
    r_t, c_t = (89 - lat) * nb - 1, (lon + 180) * nb - 1  # first centre, incl. margin
    out = {}
    for label, length in cfg.global_scales().items():
        hy, hx = half_widths(length, lats, B)
        hx = np.minimum(hx, (W - 1) // 2)  # never wider than the full circle
        py, px = hy, int(hx.max())
        rows = np.arange(r_t - py, r_t + nb + 2 + py)
        cols = np.arange(c_t - px, c_t + nb + 2 + px) % W  # wrap the antimeridian
        sub = np.zeros((3, rows.size, cols.size))
        inside = (rows >= 0) & (rows < H)  # beyond the poles: no land
        sub[:, inside] = grid[:, rows[inside]][:, :, cols]
        n = sub[0]
        mean = sub[1] - ref
        s, q = n * mean, n * (sub[2] + mean * mean)
        sums = [_window_sums(_integral(a), py, px, nb + 2, nb + 2, hy, hx) for a in (n, s, q)]
        m, sd = _mean_std(*sums)
        out[label] = _dev(zc, _upsample(np.nan_to_num(m), B), _upsample(np.nan_to_num(sd), B))
    return out


def fine_dev(cell: Cell, cfg: Config) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    """Exact DEV at the fine scales. Returns (cell - ref, ref, {label: dev})."""
    lats = row_lats(cell[0], np.arange(N))
    halves = {k: half_widths(v, lats) for k, v in cfg.fine_scales().items()}
    hy = max(h for h, _ in halves.values())
    hx = max(int(x.max()) for _, x in halves.values())

    z = read_padded(cell, hy, hx)
    center = z[hy:hy + N, hx:hx + N]
    ref = float(np.nanmean(center)) if np.isfinite(center).any() else 0.0
    valid = ~np.isnan(z)
    zc = np.where(valid, z - ref, 0.0)
    del z

    sums: dict[str, list[np.ndarray]] = {k: [] for k in halves}
    for a in (valid, zc, zc * zc):
        I = _integral(a)
        for k, (h, x) in halves.items():
            sums[k].append(_window_sums(I, hy, hx, N, N, h, x))
        del I
    zc_center = np.where(valid[hy:hy + N, hx:hx + N], zc[hy:hy + N, hx:hx + N], np.nan)
    return zc_center, ref, {k: _dev(zc_center, *_mean_std(*s)) for k, s in sums.items()}


def _write_cog(path: Path, arr: np.ndarray, cell: Cell, colorinterp=None,
               nodata=None, tags: dict | None = None, descriptions=None, **opts) -> None:
    profile = dict(driver="GTiff", width=N, height=N, count=arr.shape[0], dtype=arr.dtype,
                   crs="EPSG:4326", transform=cell_transform(cell), nodata=nodata)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.tif")
    with MemoryFile() as mem, mem.open(**profile) as ds:
        ds.write(arr)
        if colorinterp:
            ds.colorinterp = colorinterp
        if tags:
            ds.update_tags(**tags)
        for i, d in enumerate(descriptions or [], 1):
            ds.set_band_description(i, d)
        rasterio.shutil.copy(ds, tmp, driver="COG", blocksize=512,
                             overview_resampling="average", **opts)
    tmp.replace(path)


def rgb_path(cfg: Config, cell: Cell) -> Path:
    return cfg.out_dir / "rgb" / f"{cell_name(cell)}.tif"


def dev_path(cfg: Config, cell: Cell) -> Path:
    return cfg.out_dir / "dev" / f"{cell_name(cell)}.tif"


_S3 = None


def _s3():
    """One boto3 client per worker process."""
    global _S3
    if _S3 is None:
        import boto3
        from botocore.config import Config as BotoConfig

        _S3 = boto3.client("s3", config=BotoConfig(retries={"mode": "standard",
                                                            "max_attempts": 10}))
    return _S3


def _remote(cfg: Config, path: Path) -> tuple[str, str]:
    """(bucket, key) that local output `path` maps to under `cfg.dest`."""
    bucket, _, prefix = cfg.dest.removeprefix("s3://").partition("/")
    rel = path.relative_to(cfg.out_dir).as_posix()
    return bucket, f"{prefix.strip('/')}/{rel}" if prefix.strip("/") else rel


def _done(cfg: Config, path: Path) -> bool:
    if path.exists():
        return True
    if not cfg.dest:
        return False
    from botocore.exceptions import ClientError

    try:
        _s3().head_object(Bucket=(r := _remote(cfg, path))[0], Key=r[1])
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def _publish(cfg: Config, path: Path) -> None:
    """Upload `path` to `cfg.dest` and delete it locally (no-op without dest)."""
    if not cfg.dest:
        return
    bucket, key = _remote(cfg, path)
    _s3().upload_file(str(path), bucket, key, ExtraArgs={"ContentType": "image/tiff"})
    path.unlink()


def _frag_marker(cfg: Config, cell: Cell) -> Path:
    return cfg.frag_dir / "done" / str(cfg.xyz_zoom) / cell_name(cell)


def run_cell(cell: Cell, cfg: Config) -> tuple[Cell, float]:
    t = time.perf_counter()
    out = rgb_path(cfg, cell)
    # The RGB COG is written (and uploaded) last, so its presence means done.
    frags_done = cfg.xyz_zoom is None or _frag_marker(cfg, cell).exists()
    if not cfg.overwrite and frags_done and _done(cfg, out):
        return cell, 0.0

    zc, ref, devs = fine_dev(cell, cfg)
    devs |= coarse_dev(cell, zc, ref, cfg)
    devs |= global_dev(cell, zc, ref, cfg)
    valid = ~np.isnan(zc)

    if cfg.keep_dev:
        keep = [k for k in cfg.scales_m if cfg.dev_scales is None or k in cfg.dev_scales]
        dev = np.stack([np.where(valid, np.round(devs[k] * 1000).clip(-32767, 32767), -32768)
                        for k in keep]).astype(np.int16)
        _write_cog(dev_path(cfg, cell), dev, cell, nodata=-32768, descriptions=keep,
                   compress="zstd", predictor="yes")
        _publish(cfg, dev_path(cfg, cell))

    rgb = [_stretch_abs(devs[k], valid, cfg.clip) for k in cfg.rgb_order()]
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    rgba = np.stack(rgb + [alpha])
    if cfg.xyz_zoom is not None:
        from .xyz import cell_fragments, write_fragments

        write_fragments(cell_fragments(rgba, cell, cfg.xyz_zoom), cell_name(cell),
                        cfg.frag_dir, cfg.xyz_zoom)
        _frag_marker(cfg, cell).parent.mkdir(parents=True, exist_ok=True)
        _frag_marker(cfg, cell).touch()

    opts = (dict(compress="webp", quality=cfg.quality) if cfg.codec == "webp"
            else dict(compress="deflate", predictor="yes"))
    _write_cog(out, rgba, cell,
               colorinterp=[ColorInterp.red, ColorInterp.green, ColorInterp.blue,
                            ColorInterp.alpha], **opts)
    _publish(cfg, out)
    return cell, time.perf_counter() - t


# --------------------------------------------------------------------------- driver


def cells_in_bbox(west: float, south: float, east: float, north: float) -> list[Cell]:
    return sorted(
        (lat, lon)
        for lat in range(math.floor(south), math.ceil(north))
        for lon in range(math.floor(west), math.ceil(east))
        if (lat, wrap_lon(lon)) in dem_cells()
    )


def _star(args):
    fn, cell, cfg = args
    try:
        return (*fn(cell, cfg), None)
    except Exception as e:  # one bad cell (network, corrupt tile) must not stop the run
        return cell, 0.0, f"{type(e).__name__}: {e}"


def _map(fn, cells: list[Cell], cfg: Config, workers: int, label: str) -> list[Cell]:
    """Run `fn` over `cells`; returns the cells that failed (rerun to retry them)."""
    failed, t0 = [], time.perf_counter()
    with Pool(workers) as pool:
        jobs = [(fn, c, cfg) for c in cells]
        for i, (cell, dt, err) in enumerate(pool.imap_unordered(_star, jobs), 1):
            if err:
                failed.append(cell)
                print(f"[{label}] {i}/{len(cells)} {cell_name(cell)} FAILED {err}", flush=True)
            else:
                print(f"[{label}] {i}/{len(cells)} {cell_name(cell)} {dt:.1f}s", flush=True)
    print(f"[{label}] {len(cells)} cells in {time.perf_counter() - t0:.0f}s wall "
          f"({workers} workers, {len(failed)} failed)", flush=True)
    return failed


def run(targets: list[Cell], cfg: Config, workers: int) -> list[Cell]:
    """Both passes; returns failed cells (rerunning resumes and retries them)."""
    p1 = pass1_cells(targets, cfg)
    print(f"[plan] {len(targets)} target cells; block stats for {len(p1)} cells")
    failed1 = set(_map(run_coarse, p1, cfg, workers, "coarse"))
    if cfg.global_scales():
        # Global scales read every cell's block stats: all of pass 1 must exist.
        print(f"[global] {build_global(cfg)}", flush=True)
    # A target whose macro window reaches a cell with no block stats can't be done.
    blocked = {c for c in targets if failed1 & set(pass1_cells([c], cfg))}
    failed2 = _map(run_cell, [c for c in targets if c not in blocked], cfg, workers, "dev")
    failed = sorted(failed1 | blocked | set(failed2))
    if failed:
        print(f"[run] {len(failed)} cells failed or blocked: "
              f"{' '.join(cell_name(c) for c in failed)}", flush=True)
    return failed


def preview(targets: list[Cell], cfg: Config, png: Path, factor: int = 8) -> Path:
    """Mosaic the targets' RGB COGs (read from overviews) into one PNG."""
    import matplotlib.image as mpimg

    lats = [c[0] for c in targets]
    lons = [c[1] for c in targets]
    s = N // factor
    canvas = np.zeros(((max(lats) - min(lats) + 1) * s, (max(lons) - min(lons) + 1) * s, 4),
                      np.uint8)
    for cell in targets:
        with rasterio.open(rgb_path(cfg, cell)) as src:
            a = src.read(out_shape=(4, s, s))
        r = (max(lats) - cell[0]) * s
        c = (cell[1] - min(lons)) * s
        canvas[r:r + s, c:c + s] = np.moveaxis(a, 0, -1)
    png.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(png, canvas)
    return png


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Tiled global multiscale DEV on the FABDEM grid.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    g.add_argument("--cells", nargs="+", metavar="CELL", help="e.g. S26W050 N61E006")
    g.add_argument("--world", action="store_true", help="every cell with elevation data")
    p.add_argument("--scales", nargs=3, type=float, metavar="M",
                   help="three window sizes in metres (default 3000 30000 300000); "
                        "custom scales are labelled like 30km, largest -> red")
    p.add_argument("--block", type=int, default=30, help="coarse block side, px")
    p.add_argument("--clip", type=float, default=2.0)
    p.add_argument("--codec", choices=["webp", "deflate"], default="webp")
    p.add_argument("--keep-dev", action="store_true", help="also write int16 DEV COGs")
    p.add_argument("--dev-scales", nargs="+", metavar="LABEL",
                   help="with --keep-dev, only these scales (labels as in --scales, e.g. 3000km)")
    p.add_argument("--out-dir", type=Path, default=Config.out_dir)
    p.add_argument("--coarse-dir", type=Path, default=Config.coarse_dir)
    p.add_argument("--dest", metavar="s3://BUCKET/PREFIX",
                   help="upload each output (e.g. to R2) and delete the local copy")
    p.add_argument("--xyz-zoom", type=int, help="also cut XYZ tile fragments at this zoom")
    p.add_argument("--frag-dir", type=Path, default=Config.frag_dir)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--preview", type=Path,
                   help="write a PNG mosaic of the targets (local outputs only, not with --dest)")
    args = p.parse_args(argv)
    if args.preview and args.dest:
        p.error("--preview reads local outputs; it can't be combined with --dest")

    if args.world:
        targets = sorted(dem_cells())
    elif args.bbox:
        targets = cells_in_bbox(*args.bbox)
    else:
        by_name = {cell_name(c): c for c in dem_cells()}
        targets = sorted(by_name[n] for n in args.cells)

    scales = dict(DEFAULT_SCALES_M) if args.scales is None else {
        f"{m / 1000:g}km": m for m in args.scales}
    if args.dev_scales and not set(args.dev_scales) <= set(scales):
        p.error(f"--dev-scales must be among {list(scales)}")
    cfg = Config(scales_m=scales, block=args.block, clip=args.clip, codec=args.codec,
                 keep_dev=args.keep_dev,
                 dev_scales=tuple(args.dev_scales) if args.dev_scales else None, out_dir=args.out_dir, coarse_dir=args.coarse_dir,
                 dest=args.dest, xyz_zoom=args.xyz_zoom, frag_dir=args.frag_dir,
                 overwrite=args.overwrite)
    failed = run(targets, cfg, args.workers)
    if args.preview:
        print(f"[preview] {preview(targets, cfg, args.preview)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
