"""Fast XYZ zoom level straight from the published RGBA COGs.

Work unit = one *canvas*: a zoom-(z-5) tile, i.e. 32 x 32 zoom-z tiles
(8192 px square).  Every zoom-z pixel takes its value from exactly one cell --
the cell containing its centre, area-averaged over that cell's 30 m pixels
(the rule of ``mtpi.xyz.cell_fragments``) -- so cells paint disjoint parts of
the canvas: no fragments on disk, no merging.  Cells on a canvas edge are read
only over the window the canvas needs (HTTP range reads of the COG).  Tiles
are cropped from the canvas, encoded as lossless WebP at the fastest effort,
staged in /dev/shm and uploaded with s5cmd (much less CPU per object than
Python HTTP clients).

    python -m mtpi.xyz_fast --version v1 --version v2 --zoom 11 --workers 12
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from numba import njit
from PIL import Image
from rasterio.windows import Window

from .tiled import DATA, dem_cells
from .xyz import TILE, cell_tiles, lon_to_x, lat_to_y, y_to_lat

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_HTTP_RETRY_CODES", "429,500,502,503,504,520,521,522,523,524")

COG_URL = "https://mtpi.pedalhidrografi.co/{version}/rgb/{name}.tif"
BUCKET = "mtpi-fabdem"
SPAN = 32  # zoom-z tiles per canvas side (canvas zoom = z - 5)
S5CMD = os.path.expanduser("~/.local/bin/s5cmd")
STAGE = Path("/dev/shm/mtpi_xyz_fast")
MARKERS = DATA / "interim" / "xyz_fast"


def _cell_name(lat: int, lon: int) -> str:
    return f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"


# --------------------------------------------------------------------------- geometry


def _footprints(c: np.ndarray, N: int):
    """Per output pixel from its edge coordinates `c` (source centre coords):
    owned mask and source footprint [start, end), as in cell_fragments."""
    m = (c[:-1] + c[1:]) / 2
    own = (m >= -0.5) & (m < N - 0.5)
    s = np.clip(np.round(c[:-1] + 0.5), 0, N).astype(np.int64)
    e = np.clip(np.round(c[1:] + 0.5), 0, N).astype(np.int64)
    thin = e <= s  # footprint thinner than a source pixel: take the nearest one
    s[thin] = np.clip(np.floor(m[thin] + 0.5), 0, N - 1).astype(np.int64)
    e[thin] = s[thin] + 1
    return own, s, e


def cell_plan(cell, z: int, px0: int, py0: int, size: int, N: int = 3600):
    """Owned zoom-z pixels of `cell` inside the canvas [px0, px0+size) x [py0, py0+size).

    Returns None, or (ax0, ay0, cs, ce, rs, re) with the first owned global
    pixel column/row and per-pixel source footprints.
    """
    lat, lon = cell
    W = TILE * 2**z
    gx = np.arange(max(int(math.floor(float(lon_to_x(lon, z)) * TILE)) - 1, px0),
                   min(int(math.ceil(float(lon_to_x(lon + 1, z)) * TILE)) + 2, px0 + size) + 1)
    gy = np.arange(max(int(math.floor(float(lat_to_y(lat + 1, z)) * TILE)) - 1, py0),
                   min(int(math.ceil(float(lat_to_y(lat, z)) * TILE)) + 2, py0 + size) + 1)
    if len(gx) < 2 or len(gy) < 2:
        return None
    ownx, cs, ce = _footprints((gx / W * 360 - 180 - lon) * N, N)
    owny, rs, re = _footprints((lat + 1 - y_to_lat(gy / TILE, z)) * N, N)
    if not ownx.any() or not owny.any():
        return None
    ix, iy = np.flatnonzero(ownx), np.flatnonzero(owny)
    ix0, ix1, iy0, iy1 = ix[0], ix[-1] + 1, iy[0], iy[-1] + 1  # owned runs are contiguous
    return (int(gx[ix0]), int(gy[iy0]), cs[ix0:ix1], ce[ix0:ix1], rs[iy0:iy1], re[iy0:iy1])


@njit(cache=True, nogil=True)
def _paint(rgba, rs, re, cs, ce, out):
    """out[i, j] = alpha-weighted mean colour and mean alpha over each footprint."""
    acc = np.zeros((len(cs), 4))
    for i in range(len(rs)):
        acc[:] = 0.0
        for r in range(rs[i], re[i]):
            for j in range(len(cs)):
                for c in range(cs[j], ce[j]):
                    a = rgba[3, r, c]
                    if a:
                        acc[j, 0] += a
                        acc[j, 1] += rgba[0, r, c] * a
                        acc[j, 2] += rgba[1, r, c] * a
                        acc[j, 3] += rgba[2, r, c] * a
        nr = re[i] - rs[i]
        for j in range(len(cs)):
            sa = acc[j, 0]
            if sa > 0:
                al = round(sa / (nr * (ce[j] - cs[j])))
                if al > 0:
                    out[i, j, 0] = round(acc[j, 1] / sa)
                    out[i, j, 1] = round(acc[j, 2] / sa)
                    out[i, j, 2] = round(acc[j, 3] / sa)
                    out[i, j, 3] = al


def canvas_cells(X: int, Y: int, cz: int) -> list:
    """Land cells overlapping canvas tile (X, Y) at zoom cz."""
    lon0, lon1 = X / 2**cz * 360 - 180, (X + 1) / 2**cz * 360 - 180
    lat1, lat0 = float(y_to_lat(Y, cz)), float(y_to_lat(Y + 1, cz))
    cells = dem_cells()
    return [(la, lo) for la in range(math.floor(lat0) - 1, math.ceil(lat1) + 1)
            for lo in range(math.floor(lon0) - 1, math.ceil(lon1) + 1) if (la, lo) in cells]


def render_canvas(version: str, X: int, Y: int, z: int, threads: int = 6) -> np.ndarray:
    """(8192, 8192, 4) uint8 RGBA canvas for canvas tile (X, Y) at zoom z - 5.

    Cells are read and painted by a thread pool: GDAL and the (nogil) kernel
    release the GIL, and each cell paints a disjoint part of the canvas.
    """
    size = SPAN * TILE
    px0, py0 = X * size, Y * size
    canvas = np.zeros((size, size, 4), np.uint8)
    plans = [(cell, p) for cell in canvas_cells(X, Y, z - 5)
             if (p := cell_plan(cell, z, px0, py0, size)) is not None]

    def one(item):
        cell, (ax0, ay0, cs, ce, rs, re) = item
        c_lo, c_hi, r_lo, r_hi = int(cs.min()), int(ce.max()), int(rs.min()), int(re.max())
        url = "/vsicurl/" + COG_URL.format(version=version, name=_cell_name(*cell))
        for attempt in range(4):
            try:
                with rasterio.open(url) as src:
                    rgba = src.read(window=Window(c_lo, r_lo, c_hi - c_lo, r_hi - r_lo))
                break
            except rasterio.errors.RasterioIOError:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        sub = canvas[ay0 - py0:ay0 - py0 + len(rs), ax0 - px0:ax0 - px0 + len(cs)]
        _paint(rgba, rs - r_lo, re - r_lo, cs - c_lo, ce - c_lo, sub)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, plans))  # re-raises the first failure
    return canvas


# --------------------------------------------------------------------------- job


def _marker(version: str, z: int, X: int, Y: int) -> Path:
    return MARKERS / version / str(z) / f"{X}_{Y}"


def _init_worker() -> None:
    """Few malloc arenas: threaded readers otherwise keep GBs of freed memory per process."""
    import ctypes
    try:
        ctypes.CDLL("libc.so.6").mallopt(-8, 2)  # M_ARENA_MAX = 2
    except OSError:
        pass


def render_job(job) -> tuple:
    """Worker: render and stage one canvas's tiles. Returns (job, seconds, n_tiles, stage, error)."""
    version, X, Y, z, local_out, threads = job
    t = time.perf_counter()
    stage = Path(local_out) if local_out else STAGE / f"{version}_{z}_{X}_{Y}"
    try:
        canvas = render_canvas(version, X, Y, z, threads)
        n = 0
        for ty in range(SPAN):
            for tx in range(SPAN):
                tile = canvas[ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE]
                if not tile[..., 3].any():
                    continue
                path = stage / str(z) / str(X * SPAN + tx) / f"{Y * SPAN + ty}.webp"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(tile, "RGBA").save(path, "WEBP", lossless=True, quality=0, method=0)
                n += 1
        return job, time.perf_counter() - t, n, str(stage), None
    except Exception as e:  # one bad canvas must not stop the run; rerun retries it
        if not local_out:
            shutil.rmtree(stage, ignore_errors=True)
        return job, time.perf_counter() - t, 0, None, f"{type(e).__name__}: {e}"


def upload(job, stage: str, n: int, upload_workers: int) -> str | None:
    """Parent thread: s5cmd the staged tiles, then mark the canvas done. Returns error or None."""
    version, X, Y, z = job[:4]
    try:
        if n:
            r = subprocess.run(
                [S5CMD, "--endpoint-url", os.environ["AWS_ENDPOINT_URL_S3"], "--retry-count", "10",
                 "--numworkers", str(upload_workers), "cp", "--content-type", "image/webp",
                 "--cache-control", "public, max-age=604800",
                 f"{stage}/*", f"s3://{BUCKET}/{version}/xyz/"],
                capture_output=True, text=True)
            if r.returncode != 0:
                return f"s5cmd exit {r.returncode}: {r.stderr.strip()[-300:]}"
        _marker(version, z, X, Y).parent.mkdir(parents=True, exist_ok=True)
        _marker(version, z, X, Y).touch()
        return None
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def canvases(z: int) -> list:
    """Canvas tiles (zoom z - 5) overlapping any land cell."""
    return sorted({t for cell in dem_cells() for t in cell_tiles(cell, z - 5)})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Fast XYZ zoom level from the published RGBA COGs.")
    p.add_argument("--version", action="append", required=True, help="v1, v2 (repeatable, in order)")
    p.add_argument("--zoom", type=int, required=True)
    p.add_argument("--canvas", nargs=2, type=int, action="append", metavar=("X", "Y"),
                   help="only these canvas tiles (zoom-5); default: all with land")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--read-threads", type=int, default=8, help="cell read/paint threads per worker")
    p.add_argument("--upload-workers", type=int, default=48, help="s5cmd workers per canvas")
    p.add_argument("--upload-slots", type=int, default=6, help="canvases uploading at once")
    p.add_argument("--local-out", type=Path, help="write tiles here instead of uploading (testing)")
    args = p.parse_args(argv)
    todo = args.canvas or canvases(args.zoom)
    local = str(args.local_out) if args.local_out else None
    jobs = [(v, X, Y, args.zoom, local, args.read_threads) for v in args.version for X, Y in todo
            if local or not _marker(v, args.zoom, X, Y).exists()]
    print(f"[fast] {len(jobs)} canvases to do at z{args.zoom} "
          f"({len(todo) * len(args.version) - len(jobs)} already done)", flush=True)
    t0 = time.perf_counter()
    state = {"tiles": 0, "done": 0, "failed": []}
    lock = threading.Lock()
    inflight = threading.Semaphore(args.workers + args.upload_slots)  # bounds RAM and /dev/shm
    uploads = ThreadPoolExecutor(args.upload_slots)

    def finish(job, dt, n, err):
        with lock:
            state["done"] += 1
            tag = f"{job[0]} {job[1]}/{job[2]}"
            if err:
                state["failed"].append(job)
                print(f"[fast] {state['done']}/{len(jobs)} {tag} FAILED {err}", flush=True)
            else:
                state["tiles"] += n
                el = time.perf_counter() - t0
                print(f"[fast] {state['done']}/{len(jobs)} {tag} {n} tiles {dt:.1f}s | total "
                      f"{state['tiles']} tiles, {state['tiles'] / max(el, 1e-9):.0f} tiles/s, "
                      f"{el / 60:.1f} min", flush=True)
        inflight.release()

    def rendered(res):
        job, dt, n, stage, err = res
        if err or local:
            finish(job, dt, n, err)
            return
        def go():
            try:
                err = upload(job, stage, n, args.upload_workers)
            except Exception as e:  # never leave an in-flight slot unreleased
                err = f"{type(e).__name__}: {e}"
            finish(job, dt, n, err)
        uploads.submit(go)

    def done(fut, job):
        try:
            rendered(fut.result())
        except Exception as e:  # e.g. BrokenProcessPool: a worker died (OOM?) -> fail, rerun retries
            finish(job, 0, 0, f"{type(e).__name__}: {e}")

    with ProcessPoolExecutor(args.workers, initializer=_init_worker, max_tasks_per_child=25) as pool:
        for job in jobs:
            inflight.acquire()
            try:
                fut = pool.submit(render_job, job)
            except Exception as e:  # pool already broken: fail the rest fast
                finish(job, 0, 0, f"{type(e).__name__}: {e}")
                continue
            fut.add_done_callback(lambda f, job=job: done(f, job))
        for _ in range(args.workers + args.upload_slots):  # wait for everything in flight
            inflight.acquire()
    uploads.shutdown(wait=True)
    print(f"[fast] done: {len(jobs)} canvases, {state['tiles']} tiles, {len(state['failed'])} failed, "
          f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    if state["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
