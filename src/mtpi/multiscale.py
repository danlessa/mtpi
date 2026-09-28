"""Smooth-window multiscale DEV (and DEVmax) on a block pyramid.

Replaces the square windows of ``mtpi.tiled`` -- whose hard edges cast
sharp square "shadows" of strong relief into flat terrain -- with a smooth,
near-Gaussian window: three successive box filters, each ``1/sqrt(3)`` the
half-width of the square window, which keeps the same spread (sigma) as the
square window of that size.  Box half-widths are fractional (the end samples
get a partial weight), so the window grows continuously with latitude instead
of in whole-pixel steps.

Any number of scales costs about the same per scale, because each scale's
window statistics are computed on a block grid with >= ``T`` blocks per
half-window and bilinearly upsampled:

    level   1 px   full resolution (cell + halo read from neighbouring COGs)
    level   3 px   3 x 3 sums of that full-resolution halo
    level  10 px   10 x 10 sums of that full-resolution halo
    level  30 px   per-cell block-stats cache (pass 1)
    level 120 px   global grid (``tiled.build_global``)

The heavy loops (box passes, upsample + DEV) are compiled with numba.
DEVmax over a scale range is the DEV with the largest magnitude among the
scales in that range (sign kept).
"""

from __future__ import annotations

import math
import time

import numpy as np
from numba import njit
from rasterio.windows import Window

from .data import dem_cells
from .tiled import (
    DY, N, Cell, Config, coarse_path, global_path, read_cell, row_lats, wrap_lon,
)

LEVELS = (1, 3, 10, 30, 120)
FROM_FULL_RES = (1, 3, 10)  # levels aggregated from the full-resolution halo
ROOT3 = math.sqrt(3)


def log_scales(lo: float, hi: float, per_decade: int = 6) -> np.ndarray:
    n = round(per_decade * math.log10(hi / lo)) + 1
    return np.geomspace(lo, hi, n)


def level_for(length_m: float, T: float = 20) -> int:
    """Coarsest block size keeping >= T blocks per (square-equivalent) half-window."""
    h = length_m / 2 / DY
    return max([b for b in LEVELS if h / b >= T] or [1])


def _widths(length_m: float, lats: np.ndarray, b: int) -> tuple[float, np.ndarray]:
    """Per-pass fractional half-widths in blocks: (w_y, w_x per row)."""
    h = length_m / 2 / (DY * b)
    cos = np.maximum(np.cos(np.radians(lats)), 1e-3)
    return h / ROOT3, h / ROOT3 / cos


def _reach(w: float) -> int:
    """Samples on each side touched by one pass of fractional half-width w."""
    return int(math.floor(w)) + 1


# --------------------------------------------------------------------------- kernels


@njit(cache=True)
def _box_y(a, w):
    """Valid-mode box along rows, fractional half-width w; a is (C, H, W)."""
    C, H, W = a.shape
    k = int(math.floor(w))
    f = w - k
    K = k + 1
    Ho = H - 2 * K
    out = np.empty((C, Ho, W))
    core = np.zeros((C, W))
    for c in range(C):
        for r in range(1, 2 * K):
            for j in range(W):
                core[c, j] += a[c, r, j]
    for i in range(Ho):
        for c in range(C):
            for j in range(W):
                out[c, i, j] = core[c, j] + f * (a[c, i, j] + a[c, i + 2 * K, j])
        if i + 1 < Ho:
            for c in range(C):
                for j in range(W):
                    core[c, j] += a[c, i + 2 * K, j] - a[c, i + 1, j]
    return out


@njit(cache=True)
def _box_x(a, w_rows, Km):
    """Valid-mode box along columns, fractional per-row half-width, crop Km per side."""
    C, H, W = a.shape
    Wo = W - 2 * Km
    out = np.empty((C, H, Wo))
    for r in range(H):
        k = int(math.floor(w_rows[r]))
        f = w_rows[r] - k
        K = k + 1
        off = Km - K
        for c in range(C):
            core = 0.0
            for t in range(off + 1, off + 2 * K):
                core += a[c, r, t]
            for j in range(Wo):
                s = j + off
                out[c, r, j] = core + f * (a[c, r, s] + a[c, r, s + 2 * K])
                core += a[c, r, s + 2 * K] - a[c, r, s + 1]
    return out


@njit(cache=True)
def _dev_upsampled(zc, m, sd, b):
    """DEV of full-res `zc` against block-centre mean/std (1-block margin), bilinear."""
    n = zc.shape[0]
    out = np.empty((n, n), np.float32)
    j0 = np.empty(n, np.int64)
    wx = np.empty(n)
    for j in range(n):
        v = (j - (b - 1) / 2) / b + 1
        j0[j] = int(math.floor(v))
        wx[j] = v - j0[j]
    for i in range(n):
        u = (i - (b - 1) / 2) / b + 1
        i0 = int(math.floor(u))
        wy = u - i0
        for j in range(n):
            z = zc[i, j]
            if math.isnan(z):
                out[i, j] = np.nan
                continue
            a0, x = j0[j], wx[j]
            mm = (1 - wy) * ((1 - x) * m[i0, a0] + x * m[i0, a0 + 1]) \
                + wy * ((1 - x) * m[i0 + 1, a0] + x * m[i0 + 1, a0 + 1])
            ss = (1 - wy) * ((1 - x) * sd[i0, a0] + x * sd[i0, a0 + 1]) \
                + wy * ((1 - x) * sd[i0 + 1, a0] + x * sd[i0 + 1, a0 + 1])
            out[i, j] = (z - mm) / ss if ss > 1e-6 else 0.0
    return out


@njit(cache=True)
def _update_max(best, dev):
    """best <- dev where |dev| > |best| (NaN-safe), in place."""
    n0, n1 = dev.shape
    for i in range(n0):
        for j in range(n1):
            d = dev[i, j]
            if math.isnan(d):
                best[i, j] = np.nan
            elif abs(d) > abs(best[i, j]):
                best[i, j] = d


@njit(cache=True)
def _dev_from_sums(zc, n, s, q):
    """DEV from windowed count/sum/sum-of-squares; n may be (h, w) or (h, 1)."""
    h, w = zc.shape
    wide = n.shape[1] > 1
    out = np.empty((h, w), np.float32)
    for i in range(h):
        for j in range(w):
            z = zc[i, j]
            if math.isnan(z):
                out[i, j] = np.nan
                continue
            nn = n[i, j] if wide else n[i, 0]
            m = s[i, j] / nn
            v = q[i, j] / nn - m * m
            sd = math.sqrt(v) if v > 0 else 0.0
            out[i, j] = (z - m) / sd if sd > 1e-6 else 0.0
    return out


def smooth_sums(stack: np.ndarray, w_y: float, w_x: np.ndarray, oy: int, ox: int,
                out_h: int, out_w: int, all_valid: bool = False, w_cap: float = math.inf):
    """Windowed (count, sum, sum of squares) under the 3-pass window.

    Outputs start at (oy, ox) of `stack` = (count, sum, sum of squares) per
    sample; `w_x` gives the per-pass half-width for every row of `stack`.  With
    `all_valid` the count is the kernel applied to ones, computed on a
    one-column strip (shape (out_h, 1)) instead of the full array.

    Every row is filtered with its own latitude's width (no dependence on which
    cell is computing), so overlapping cells agree exactly; `w_cap` only bounds
    polar rows (e.g. to less than the full circle on the global grid).
    """
    Ky = _reach(w_y)
    ry = 3 * Ky
    w_x = np.minimum(w_x[max(oy - ry, 0):oy + out_h + ry], w_cap)
    Km = _reach(float(w_x.max()))
    rx = 3 * Km
    if oy < ry or ox < rx or oy + out_h + ry > stack.shape[1] or ox + out_w + rx > stack.shape[2]:
        raise ValueError(f"halo too small: need {ry}x{rx}, have {oy}x{ox} in {stack.shape}")
    rows = slice(oy - ry, oy + out_h + ry)
    a = np.ascontiguousarray(stack[1 if all_valid else 0:, rows, ox - rx:ox + out_w + rx])
    ones = np.ones((1, a.shape[1], 2 * rx + 1)) if all_valid else None
    wr = w_x
    for _ in range(3):
        a = _box_y(a, w_y)
        ones = _box_y(ones, w_y) if all_valid else None
        wr = np.ascontiguousarray(wr[Ky:len(wr) - Ky])
        a = _box_x(a, wr, Km)
        ones = _box_x(ones, wr, Km) if all_valid else None
    return (ones[0], a[0], a[1]) if all_valid else (a[0], a[1], a[2])


def smooth_mean_sd(stack, w_y, w_x, oy, ox, out_h, out_w, w_cap=math.inf):
    """Mean/std under the 3-pass window (see `smooth_sums`)."""
    n, s, q = smooth_sums(stack, w_y, w_x, oy, ox, out_h, out_w, w_cap=w_cap)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / n
        sd = np.sqrt(np.maximum(q / n - m * m, 0.0))
    return np.nan_to_num(m), np.nan_to_num(sd)


# --------------------------------------------------------------------------- inputs


def read_padded_multi(cell: Cell, hy: int, hx: int) -> np.ndarray:
    """`cell` with a (hy, hx) px halo from as many neighbouring cells as needed."""
    lat, lon = cell
    H, W = N + 2 * hy, N + 2 * hx
    out = np.full((H, W), np.nan, np.float32)
    ky, kx = math.ceil(hy / N), math.ceil(hx / N)
    for dlat in range(ky, -ky - 1, -1):
        nlat = lat + dlat
        if not -90 <= nlat < 90:
            continue
        r_off = hy - dlat * N
        r0, r1 = max(0, r_off), min(H, r_off + N)
        for dlon in range(-kx, kx + 1):
            c_off = hx + dlon * N
            c0, c1 = max(0, c_off), min(W, c_off + N)
            if r0 >= r1 or c0 >= c1:
                continue
            win = Window(c0 - c_off, r0 - r_off, c1 - c0, r1 - r0)
            out[r0:r1, c0:c1] = read_cell((nlat, wrap_lon(lon + dlon)), win)
    return out


def _stats_grid(cfg: Config, cell: Cell, b: int, py: int, px: int) -> np.ndarray:
    """(3, nb+2py, nb+2px) float64 (count, mean, var) around `cell` at level 30 or 120."""
    nb = N // b
    lat, lon = cell
    if b == 120:
        g = np.load(global_path(cfg), mmap_mode="r")
        H, W = g.shape[1:]
        rows = np.arange((89 - lat) * nb - py, (89 - lat) * nb + nb + py)
        cols = np.arange((lon + 180) * nb - px, (lon + 180) * nb + nb + px) % W
        out = np.zeros((3, rows.size, cols.size))
        ok = (rows >= 0) & (rows < H)
        out[:, ok] = g[:, rows[ok]][:, :, cols]
        return out
    ky, kx = math.ceil(py / nb), min(math.ceil(px / nb), 180)
    mos = np.zeros((3, (2 * ky + 1) * nb, (2 * kx + 1) * nb))
    for dlat in range(-ky, ky + 1):
        if not -90 <= lat + dlat < 90:
            continue
        r = (ky - dlat) * nb
        for dlon in range(-kx, kx + 1):
            c = (kx + dlon) * nb
            nc = (lat + dlat, wrap_lon(lon + dlon))
            if nc in dem_cells():
                mos[:, r:r + nb, c:c + nb] = np.load(coarse_path(cfg, nc))
    return mos[:, ky * nb - py:ky * nb + nb + py, kx * nb - px:kx * nb + nb + px]


# --------------------------------------------------------------------------- driver


def multiscale_dev(cell: Cell, scales_m, cfg: Config = Config(), T: float = 20,
                   timing: dict | None = None, force_level: dict | None = None,
                   keep=None, bands: dict | None = None):
    """Smooth-window DEV of `cell` at every scale in `scales_m`.

    Returns (zc, devs, maxes): zc is the cell's elevation minus its mean;
    devs maps scale -> DEV for scales in `keep` (default: all); maxes maps each
    name in `bands` ({name: (lo, hi)}) to DEVmax over scales in [lo, hi].
    """
    t0 = time.perf_counter()
    lat = cell[0]
    force_level = force_level or {}
    by_level: dict[int, list[float]] = {}
    for L in scales_m:
        by_level.setdefault(force_level.get(float(L), level_for(L, T)), []).append(float(L))

    # Full-resolution halo: enough for every level aggregated from it.
    fine = [(L, b) for b in FROM_FULL_RES for L in by_level.get(b, [])] or [(min(scales_m), 1)]
    hy = max(b * (3 * _reach(_widths(L, np.zeros(1), b)[0]) + 2) for L, b in fine)
    hy = -(-hy // 30) * 30  # whole blocks at every level
    far = row_lats(lat, np.array([-hy, N + hy]))  # widest rows the passes will touch
    hx = max(b * (3 * _reach(float(_widths(L, far, b)[1].max())) + 2) for L, b in fine)
    hx = -(-hx // 30) * 30
    z = read_padded_multi(cell, hy, hx)
    valid = ~np.isnan(z)
    center = z[hy:hy + N, hx:hx + N]
    ref = float(np.nanmean(center)) if np.isfinite(center).any() else 0.0
    zc_pad = np.where(valid, z - ref, 0.0)
    zc = np.where(valid[hy:hy + N, hx:hx + N], zc_pad[hy:hy + N, hx:hx + N], np.nan)
    full = np.stack([valid.astype(np.float64), zc_pad, zc_pad * zc_pad])
    del z, zc_pad
    if timing is not None:
        timing["read"] = time.perf_counter() - t0

    keep = set(map(float, scales_m)) if keep is None else set(map(float, keep))
    bands = bands or {}
    devs: dict[float, np.ndarray] = {}
    maxes = {k: np.zeros((N, N), np.float32) for k in bands}

    def emit(L, dev):
        k = next((k for k in keep if abs(L - k) <= 1e-6 * k), None)  # float-tolerant match
        if k is not None:
            devs[k] = dev
        for k, (lo, hi) in bands.items():
            if lo * (1 - 1e-9) <= L <= hi * (1 + 1e-9):
                _update_max(maxes[k], dev)

    # Global grid: keep each window's support (3 passes) under the full circle.
    w_cap = {b: math.inf for b in LEVELS}
    w_cap[120] = (360 * (N // 120) / 2 - 4) / 3 - 1
    for b in sorted(by_level):
        t = time.perf_counter()
        nb = N // b
        if b == 1:
            lats = row_lats(lat, np.arange(-hy, N + hy))
            for L in by_level[b]:
                w_y, w_x = _widths(L, lats, 1)
                ry = 3 * _reach(w_y)
                rx = 3 * _reach(float(w_x[hy:hy + N].max()))
                all_valid = bool(valid[hy - ry:hy + N + ry, hx - rx:hx + N + rx].all())
                n, s, q = smooth_sums(full, w_y, w_x, hy, hx, N, N, all_valid)
                emit(L, _dev_from_sums(zc, n, s, q))
        else:
            if b in FROM_FULL_RES:
                py, px = hy // b, hx // b
                grid = full.reshape(3, full.shape[1] // b, b, full.shape[2] // b, b).sum(axis=(2, 4))
            else:
                py = max(3 * _reach(_widths(L, np.zeros(1), b)[0]) for L in by_level[b]) + 2
                lats_p = row_lats(lat, np.arange(-py, nb + py), b)
                px = max(3 * _reach(min(float(_widths(L, lats_p, b)[1].max()), w_cap[b]))
                         for L in by_level[b]) + 2
                g = _stats_grid(cfg, cell, b, py, px)
                cnt, mean = g[0], g[1] - ref
                grid = np.stack([cnt, cnt * mean, cnt * (g[2] + mean * mean)])
            lats = row_lats(lat, np.arange(-py, nb + py), b)
            for L in by_level[b]:
                w_y, w_x = _widths(L, lats, b)
                m, sd = smooth_mean_sd(grid, w_y, w_x, py - 1, px - 1, nb + 2, nb + 2, w_cap[b])
                emit(L, _dev_upsampled(zc, m, sd, b))
        if timing is not None:
            timing[f"level {b:>3} ({len(by_level[b])} scales)"] = time.perf_counter() - t
    return zc, devs, maxes
