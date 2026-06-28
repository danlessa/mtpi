"""Area-of-interest definition and FABDEM tile handling.

FABDEM is distributed as 1 deg x 1 deg GeoTIFF tiles named by their
south-west corner, e.g. ``S26W050_FABDEM_V1-2.tif`` covers latitude
[-26, -25] and longitude [-50, -49].  The global set lives in the
GCS bucket ``gs://telhas/fabdem``.  Tiles are EPSG:4326, ~1 arc-second
(~30 m), Float32, NoData = -9999.

License note: FABDEM is CC BY-NC-SA 4.0 -- non-commercial use only.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

FABDEM_BUCKET = "gs://telhas/fabdem"
FABDEM_SUFFIX = "_FABDEM_V1-2.tif"
FABDEM_NODATA = -9999.0


def _tile_name(lat: int, lon: int) -> str:
    """FABDEM file name for the tile whose SW corner is (lat, lon) degrees."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}{FABDEM_SUFFIX}"


@dataclass(frozen=True)
class AOI:
    """A geographic area of interest, in degrees (WGS84)."""

    name: str
    west: float
    south: float
    east: float
    north: float

    def tiles(self) -> list[str]:
        """FABDEM tile names (by SW corner) covering this AOI."""
        names = []
        for lat in range(math.floor(self.south), math.ceil(self.north)):
            for lon in range(math.floor(self.west), math.ceil(self.east)):
                names.append(_tile_name(lat, lon))
        return names

    def utm_epsg(self) -> int:
        """Best-fit WGS84/UTM EPSG code for the AOI centroid."""
        lon_c = (self.west + self.east) / 2
        lat_c = (self.south + self.north) / 2
        zone = int((lon_c + 180) / 6) + 1
        return (32700 if lat_c < 0 else 32600) + zone

    @property
    def width_km(self) -> float:
        lat_c = math.radians((self.south + self.north) / 2)
        return (self.east - self.west) * 111.32 * math.cos(lat_c)

    @property
    def height_km(self) -> float:
        return (self.north - self.south) * 110.57


# A few named presets. `parana` is the validation AOI: the Paraná plateau
# (Curitiba) dropping over the Serra do Mar escarpment to the coastal plain --
# strong relief across micro/meso/macro scales.
PRESETS: dict[str, AOI] = {
    "parana": AOI("parana", west=-50.0, south=-26.0, east=-48.0, north=-25.0),
}


def fetch_tiles(aoi: AOI, raw_dir: Path, bucket: str = FABDEM_BUCKET) -> list[Path]:
    """Download the FABDEM tiles covering `aoi` into `raw_dir` (idempotent).

    Tiles that do not exist in the bucket (e.g. all-ocean cells) are skipped
    with a warning rather than aborting the run.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    local: list[Path] = []
    missing: list[str] = []
    for name in aoi.tiles():
        dst = raw_dir / name
        if dst.exists():
            local.append(dst)
            continue
        src = f"{bucket}/{name}"
        # `gsutil -q stat` returns non-zero if the object does not exist.
        if subprocess.run(["gsutil", "-q", "stat", src]).returncode != 0:
            missing.append(name)
            continue
        subprocess.run(["gsutil", "-q", "cp", src, str(dst)], check=True)
        local.append(dst)
    if missing:
        print(f"[fetch] {len(missing)} tile(s) absent in bucket (ocean?): {missing}")
    if not local:
        raise RuntimeError(f"No FABDEM tiles found for AOI {aoi.name}")
    return local
