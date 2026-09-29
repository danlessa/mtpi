"""Cut XYZ tile fragments from published RGBA COGs (no terrain recomputation).

The colour COGs are lossless (zstd), so fragments cut from them equal the ones
the world run would have written.  Used to add a zoom level after the fact:

    python -m mtpi.xyz_cut --src s3://mtpi-fabdem/v1 --zoom 10 --frag-dir data/interim/xyz_z10/v1
    python -m mtpi.xyz --zoom 10 --min-zoom 10 --frag-dir data/interim/xyz_z10/v1 \\
        --quality 100 --dest s3://mtpi-fabdem/v1/xyz
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import rasterio
from rasterio.io import MemoryFile

from . import tiled
from .tiled import Cell, cell_name, dem_cells
from .xyz import cell_fragments, write_fragments


@dataclass
class CutJob:
    src: str  # s3://bucket/prefix holding rgb/<cell>.tif
    frag_dir: Path
    zoom: int
    fmt: str = "webp"


def cut_cell(cell: Cell, job: CutJob) -> tuple[Cell, float]:
    t = time.perf_counter()
    marker = job.frag_dir / "done" / str(job.zoom) / cell_name(cell)
    if marker.exists():
        return cell, 0.0
    bucket, _, prefix = job.src.removeprefix("s3://").partition("/")
    key = f"{prefix.strip('/')}/rgb/{cell_name(cell)}.tif".lstrip("/")
    body = tiled._s3().get_object(Bucket=bucket, Key=key)["Body"].read()
    with MemoryFile(body) as mem, mem.open() as src:
        rgba = src.read()
    write_fragments(cell_fragments(rgba, cell, job.zoom), cell_name(cell), job.frag_dir,
                    job.zoom, job.fmt)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return cell, time.perf_counter() - t


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Cut XYZ fragments from published RGBA COGs.")
    p.add_argument("--src", required=True, metavar="s3://BUCKET/PREFIX")
    p.add_argument("--zoom", type=int, required=True)
    p.add_argument("--frag-dir", type=Path, required=True)
    p.add_argument("--cells", nargs="+", metavar="CELL", help="default: every cell")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args(argv)
    cells = sorted(dem_cells())
    if args.cells:
        by_name = {cell_name(c): c for c in cells}
        cells = sorted(by_name[n] for n in args.cells)
    failed = tiled._map(cut_cell, cells, CutJob(args.src, args.frag_dir, args.zoom),
                        args.workers, "cut")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
