"""Web Mercator XYZ tiles (256 px WebP) from the per-cell RGBA.

Two steps, so the COGs never have to be read back:

1. During the tiled run (``mtpi.tiled --xyz-zoom Z``) each cell is cut into
   *fragments* of the zoom-Z tiles it overlaps.  Every tile pixel belongs to
   the one cell containing its centre and is the area average of the 30 m
   pixels inside its footprint (clipped to that cell), weighted by alpha so
   coasts don't darken.  Fragments are RGBA PNGs, zero where the cell doesn't
   own the pixel, under ``<frag_dir>/<z>/<x>/<y>/<cell>.png``.
2. ``python -m mtpi.xyz`` merges fragments into zoom-Z tiles (ownership is
   disjoint, so merging is a sum), averages 2 x 2 children down to zoom 0, and
   writes ``<z>/<x>/<y>.webp`` locally or to ``--dest`` (e.g. R2).
   Fully transparent tiles are skipped.

    python -m mtpi.xyz --frag-dir data/interim/xyz_frag --dest s3://mtpi-fabdem/v1/xyz
"""

from __future__ import annotations

import argparse
import io
import math
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

TILE = 256


# --------------------------------------------------------------------------- mercator


def lon_to_x(lon: float | np.ndarray, z: int):
    return (np.asarray(lon) + 180.0) / 360.0 * 2**z


def lat_to_y(lat: float | np.ndarray, z: int):
    r = np.radians(np.clip(lat, -85.05112878, 85.05112878))
    return (1 - np.log(np.tan(r) + 1 / np.cos(r)) / math.pi) / 2 * 2**z


def y_to_lat(y: np.ndarray, z: int) -> np.ndarray:
    return np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * np.asarray(y) / 2**z))))


def cell_tiles(cell: tuple[int, int], z: int) -> list[tuple[int, int]]:
    """Zoom-z tiles overlapping the 1 deg cell."""
    lat, lon = cell
    n = 2**z
    x0, x1 = int(lon_to_x(lon, z)), int(math.ceil(lon_to_x(lon + 1, z))) - 1
    y0, y1 = int(lat_to_y(lat + 1, z)), int(math.ceil(lat_to_y(lat, z))) - 1
    return [(x, y) for x in range(max(x0, 0), min(x1, n - 1) + 1)
            for y in range(max(y0, 0), min(y1, n - 1) + 1)]


# --------------------------------------------------------------------------- step 1


def _edges(coord: np.ndarray, n: int) -> np.ndarray:
    """Pixel-centre coordinates -> integer source-pixel edges, clipped to [0, n]."""
    return np.clip(np.round(coord + 0.5), 0, n).astype(np.int64)


def cell_fragments(rgba: np.ndarray, cell: tuple[int, int], z: int) -> dict:
    """{(x, y): uint8 RGBA 256x256} fragments of `rgba` (4, N, N) for zoom z."""
    lat, lon = cell
    N = rgba.shape[1]
    a = rgba[3].astype(np.float64) / 255
    # Summed-area tables of alpha and premultiplied colour.
    tables = []
    for band in (a, *(rgba[k] * a for k in range(3))):
        t = np.zeros((N + 1, N + 1))
        np.cumsum(band, axis=0, out=t[1:, 1:])
        np.cumsum(t[1:, 1:], axis=1, out=t[1:, 1:])
        tables.append(t)

    out = {}
    for x, y in cell_tiles(cell, z):
        # Tile pixel edges -> source pixel coordinates (centre-based: col j at lon + j/N).
        ex = (np.arange(TILE + 1) / TILE + x) / 2**z * 360 - 180
        ey = y_to_lat(np.arange(TILE + 1) / TILE + y, z)
        cx, cy = (ex - lon) * N, (lat + 1 - ey) * N
        mx, my = (cx[:-1] + cx[1:]) / 2, (cy[:-1] + cy[1:]) / 2
        own_x = (mx >= -0.5) & (mx < N - 0.5)
        own_y = (my >= -0.5) & (my < N - 0.5)
        if not own_x.any() or not own_y.any():
            continue
        c0, c1 = _edges(cx[:-1], N), _edges(cx[1:], N)
        r0, r1 = _edges(cy[:-1], N), _edges(cy[1:], N)
        # Footprints thinner than a source pixel (high zooms): take the nearest one.
        thin = c1 <= c0
        c0[thin] = np.clip(np.floor(mx[thin] + 0.5).astype(np.int64), 0, N - 1)
        c1[thin] = c0[thin] + 1
        thin = r1 <= r0
        r0[thin] = np.clip(np.floor(my[thin] + 0.5).astype(np.int64), 0, N - 1)
        r1[thin] = r0[thin] + 1
        count = np.outer(r1 - r0, c1 - c0)
        sums = [t[np.ix_(r1, c1)] - t[np.ix_(r0, c1)] - t[np.ix_(r1, c0)] + t[np.ix_(r0, c0)]
                for t in tables]
        sa = sums[0]
        frag = np.zeros((TILE, TILE, 4), np.uint8)
        with np.errstate(invalid="ignore", divide="ignore"):
            for k in range(3):
                frag[..., k] = np.where(sa > 0, sums[k + 1] / sa, 0).round().astype(np.uint8)
        frag[..., 3] = (sa / count * 255).round().astype(np.uint8)
        frag[~np.outer(own_y, own_x)] = 0
        frag[frag[..., 3] == 0] = 0
        if frag[..., 3].any():
            out[x, y] = frag
    return out


def write_fragments(frags: dict, cell_name: str, frag_dir: Path, z: int) -> None:
    for (x, y), frag in frags.items():
        path = frag_dir / str(z) / str(x) / str(y) / f"{cell_name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.png")
        Image.fromarray(frag, "RGBA").save(tmp)
        tmp.replace(path)


# --------------------------------------------------------------------------- step 2


def _merge(dirs: list[Path]) -> np.ndarray:
    acc = np.zeros((TILE, TILE, 4), np.uint16)
    for d in dirs:
        for f in d.glob("*.png"):
            acc += np.asarray(Image.open(f).convert("RGBA"), np.uint16)
    return acc.clip(0, 255).astype(np.uint8)


def _downsample(children: dict, x: int, y: int) -> np.ndarray:
    """Parent tile (x, y) from its 4 children: alpha-weighted 2x2 average."""
    big = np.zeros((2 * TILE, 2 * TILE, 4), np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            c = children.get((2 * x + dx, 2 * y + dy))
            if c is not None:
                big[dy * TILE:(dy + 1) * TILE, dx * TILE:(dx + 1) * TILE] = c
    a = big[..., 3:] / 255
    pm = np.concatenate([big[..., :3] * a, a], axis=-1)
    pm = pm.reshape(TILE, 2, TILE, 2, 4).mean(axis=(1, 3))
    out = np.zeros((TILE, TILE, 4), np.uint8)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[..., :3] = np.where(pm[..., 3:] > 0, pm[..., :3] / pm[..., 3:], 0).round()
    out[..., 3] = (pm[..., 3] * 255).round()
    return out


class Sink:
    """Writes tiles to a local directory or an S3 prefix."""

    def __init__(self, out_dir: Path | None, dest: str | None, quality: int):
        self.out_dir, self.dest, self.quality = out_dir, dest, quality
        self._s3 = None

    def put(self, z: int, x: int, y: int, tile: np.ndarray) -> None:
        buf = io.BytesIO()
        if self.quality >= 100:  # lossless: no per-tile colour casts at tile edges
            Image.fromarray(tile, "RGBA").save(buf, "WEBP", lossless=True, quality=80, method=4)
        else:
            Image.fromarray(tile, "RGBA").save(buf, "WEBP", quality=self.quality, method=4)
        rel = f"{z}/{x}/{y}.webp"
        if self.dest:
            if self._s3 is None:
                import boto3
                from botocore.config import Config as BotoConfig

                self._s3 = boto3.client("s3", config=BotoConfig(
                    retries={"mode": "standard", "max_attempts": 10}))
            bucket, _, prefix = self.dest.removeprefix("s3://").partition("/")
            key = f"{prefix.strip('/')}/{rel}".lstrip("/")
            self._s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue(),
                                ContentType="image/webp",
                                CacheControl="public, max-age=31536000, immutable")
        else:
            path = self.out_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(buf.getvalue())


_SINK: Sink | None = None


def _init(out_dir, dest, quality):
    global _SINK
    _SINK = Sink(out_dir, dest, quality)


def _leaf(args) -> tuple[tuple[int, int], np.ndarray | None]:
    z, (x, y), dirs = args
    tile = _merge(dirs)
    if not tile[..., 3].any():
        return (x, y), None
    _SINK.put(z, x, y, tile)
    return (x, y), tile


def _parent(args) -> tuple[tuple[int, int], np.ndarray | None]:
    z, (x, y), children = args
    tile = _downsample(children, x, y)
    if not tile[..., 3].any():
        return (x, y), None
    _SINK.put(z, x, y, tile)
    return (x, y), tile


def build_pyramid(frag_dir: Path, zoom: int, out_dir: Path | None, dest: str | None,
                  quality: int = 85, workers: int = os.cpu_count() or 1) -> None:
    root = frag_dir / str(zoom)
    leaves = sorted((int(d.parent.name), int(d.name), d) for d in root.glob("*/*") if d.is_dir())
    jobs = [(zoom, (x, y), [d]) for x, y, d in leaves]
    t0 = time.perf_counter()
    with Pool(workers, initializer=_init, initargs=(out_dir, dest, quality)) as pool:
        level = {k: t for k, t in pool.imap_unordered(_leaf, jobs, chunksize=64)
                 if t is not None}
        print(f"[xyz] z{zoom}: {len(level)} tiles ({time.perf_counter() - t0:.0f}s)", flush=True)
        for z in range(zoom - 1, -1, -1):
            parents = defaultdict(dict)
            for (x, y), t in level.items():
                parents[x // 2, y // 2][x, y] = t
            jobs = [(z, k, ch) for k, ch in parents.items()]
            level = {k: t for k, t in pool.imap_unordered(_parent, jobs, chunksize=16)
                     if t is not None}
            print(f"[xyz] z{z}: {len(level)} tiles ({time.perf_counter() - t0:.0f}s)", flush=True)


def main(argv: list[str] | None = None) -> None:
    from .tiled import DATA

    p = argparse.ArgumentParser(description="Build the XYZ WebP pyramid from cell fragments.")
    p.add_argument("--frag-dir", type=Path, default=DATA / "interim" / "xyz_frag")
    p.add_argument("--zoom", type=int, default=9, help="fragment (max) zoom")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--out-dir", type=Path, help="write tiles locally")
    g.add_argument("--dest", metavar="s3://BUCKET/PREFIX", help="upload tiles (e.g. to R2)")
    p.add_argument("--quality", type=int, default=85, help="WebP quality; 100 = lossless")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args(argv)
    build_pyramid(args.frag_dir, args.zoom, args.out_dir, args.dest, args.quality, args.workers)


if __name__ == "__main__":
    main()
