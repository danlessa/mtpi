"""DEM mosaicking, reprojection, and multiscale deviation-from-mean-elevation.

The topographic-position primitive here is DEV (Wilson's deviation from mean
elevation): for each cell, ``(z - mean) / stddev`` over a square window.  It is a
standardized, scale-explicit TPI -- positive on ridges/peaks, negative in
valleys, ~0 on uniform slopes -- which makes three scales directly comparable
and stackable into an RGB composite.

DEV is computed in a *metric* CRS so that window sizes expressed in metres map
to a fixed number of cells.  FABDEM is geographic (degrees), so we reproject to
UTM (or any projected CRS) first.
"""

from __future__ import annotations

from pathlib import Path

from osgeo import gdal

from .data import FABDEM_NODATA

gdal.UseExceptions()


def mosaic_and_warp(
    tile_paths: list[Path],
    out_path: Path,
    dst_epsg: int,
    res: float,
    resample: str = "cubic",
) -> Path:
    """Mosaic `tile_paths` and reproject to EPSG:`dst_epsg` at `res` metres."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # gdal.Warp mosaics a list of sources directly -- no separate BuildVRT needed.
    gdal.Warp(
        str(out_path),
        [str(p) for p in tile_paths],
        dstSRS=f"EPSG:{dst_epsg}",
        xRes=res,
        yRes=res,
        resampleAlg=resample,
        srcNodata=FABDEM_NODATA,
        dstNodata=FABDEM_NODATA,
        targetAlignedPixels=True,
        multithread=True,
        # NOTE: no PREDICTOR -- whitebox's GeoTIFF reader rejects floating-point
        # predictors (PREDICTOR=3), which it would need for this Float32 DEM.
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    return out_path


def _odd_cells(dist_m: float, res: float) -> int:
    """Window size in cells for a length `dist_m`, forced odd and >= 3."""
    cells = max(3, round(dist_m / res))
    if cells % 2 == 0:
        cells += 1
    return cells


def multiscale_dev(
    dem_path: Path,
    work_dir: Path,
    scales_m: dict[str, float],
    res: float,
) -> dict[str, Path]:
    """Compute DEV at each named scale (metres). Returns {label: dev_path}.

    Uses WhiteboxTools' ``dev_from_mean_elev``, which evaluates the windowed
    mean/stddev via integral images -- O(1) per cell, so large windows are cheap.
    """
    import whitebox

    work_dir.mkdir(parents=True, exist_ok=True)
    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False
    wbt.set_working_dir(str(work_dir))

    out: dict[str, Path] = {}
    for label, dist_m in scales_m.items():
        cells = _odd_cells(dist_m, res)
        dev_path = work_dir / f"dev_{label}.tif"
        wbt.dev_from_mean_elev(
            str(dem_path),
            str(dev_path),
            filterx=cells,
            filtery=cells,
        )
        print(f"[dev] {label}: {dist_m/1000:.1f} km -> {cells} cells -> {dev_path.name}")
        out[label] = dev_path
    return out
