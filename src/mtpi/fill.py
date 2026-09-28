"""Fill FABDEM's Armenia/Azerbaijan gap from Copernicus GLO-30 2023_1.

FABDEM derives from the 2021 Copernicus GLO-30 release, which withheld Armenia
and Azerbaijan, so those 1 deg cells have no FABDEM file.  The DGED 2023_1
release includes them.  OpenTopography serves it as 3601 x 3601 COGs sharing
FABDEM's pixel centres plus one overlapping row/column, so filling is a crop,
not a resample.  Same sensor (TanDEM-X) and vertical datum (EGM2008) as FABDEM,
and the Caspian is already FABDEM's flat -28.0 m surface.  Caveat: GLO-30 is a
surface model -- FABDEM is the same data with forests and buildings removed --
so forested ground reads a canopy height higher.

Each fill tile is written as a FABDEM tile would be (Float32, NoData -9999),
so it can sit in the FABDEM bucket under FABDEM names for cameratopo, and mtpi
reads it like any other cell.  The GeoTIFF metadata records the true source.

    python -m mtpi.fill [--dest s3://<fabdem-bucket>]
"""

from __future__ import annotations

import argparse

import numpy as np
import rasterio

from .data import FABDEM_NODATA, FILL_CELLS, _tile_name, fill_path
from .tiled import CASPIAN_LEVEL, N, _write_cog

COP30_URL = "https://opentopography.s3.sdsc.edu/raster/COP30/COP30_hh/"
TAGS = {
    "SOURCE": "Copernicus GLO-30 DEM, DGED 2023_1 (via OpenTopography); FABDEM gap fill",
    "COPYRIGHT": "(c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 "
                 "provided under COPERNICUS by the European Union and ESA; all rights reserved",
}


def cop30_url(lat: int, lon: int) -> str:
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    return f"/vsicurl/{COP30_URL}Copernicus_DSM_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM.tif"


def build(cell: tuple[int, int]) -> np.ndarray:
    """COP30 for `cell` on the FABDEM grid (3600 x 3600 float32, NoData -9999)."""
    with rasterio.open(cop30_url(*cell)) as src:
        # Row 0 / col 0 centres match FABDEM's; the last row/col overlap the next cell.
        z = src.read(1, masked=True)[:N, :N].filled(np.nan).astype(np.float32)
    z[~np.isfinite(z)] = FABDEM_NODATA
    return z


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build COP30 fill tiles for FABDEM's gap.")
    p.add_argument("--dest", metavar="s3://BUCKET/PREFIX",
                   help="also upload the fill tiles under FABDEM names (drop-in for the "
                        "FABDEM bucket that cameratopo reads)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    for cell in sorted(FILL_CELLS):
        path = fill_path(*cell)
        if path.exists() and not args.overwrite:
            print(f"[fill] {path.name} exists")
        else:
            z = build(cell)
            _write_cog(path, z[None], cell, nodata=FABDEM_NODATA, tags=TAGS,
                       compress="deflate", predictor="yes")
            print(f"[fill] {path.name}  caspian={np.mean(z == CASPIAN_LEVEL):.1%}  "
                  f"void={np.mean(z == FABDEM_NODATA):.2%}", flush=True)
        if args.dest:
            import boto3
            from botocore.exceptions import ClientError

            s3 = boto3.client("s3")
            bucket, _, prefix = args.dest.removeprefix("s3://").partition("/")
            key = f"{prefix.strip('/')}/{_tile_name(*cell)}".lstrip("/")
            # The destination may be the live FABDEM bucket: never clobber a
            # real FABDEM tile (or an earlier fill) unless asked to.
            try:
                s3.head_object(Bucket=bucket, Key=key)
                if not args.overwrite:
                    print(f"[fill] s3://{bucket}/{key} exists, skipped")
                    continue
            except ClientError as e:
                if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                    raise
            s3.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": "image/tiff"})
            print(f"[fill] uploaded s3://{bucket}/{key}", flush=True)


if __name__ == "__main__":
    main()
