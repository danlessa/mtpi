"""Area-of-interest definition and FABDEM tile handling.

FABDEM is distributed as 1 deg x 1 deg GeoTIFF tiles named by their
south-west corner, e.g. ``S26W050_FABDEM_V1-2.tif`` covers latitude
[-26, -25] and longitude [-50, -49].  The global set is served as COGs from
Cloudflare R2 at ``FABDEM_BASE_URL`` (the same source cameratopo reads).
Tiles are EPSG:4326, 3600 x 3600 at 1 arc-second (~30 m), Float32,
NoData = -9999.  Pixel *centres* sit on whole arc-seconds (row 0 at the
tile's north edge latitude), so neighbouring tiles abut with no overlap.
Sea inside a land tile is stored as 0 m; all-ocean cells have no file.

FABDEM derives from the 2021 Copernicus GLO-30 release, which withheld Armenia
and Azerbaijan.  Those cells (``FILL_CELLS``) come from the 2023_1 GLO-30
release (which includes them), on the same grid, built by ``python -m mtpi.fill``
and published in the FABDEM bucket under FABDEM names (for cameratopo too).

License note: FABDEM is CC BY-NC-SA 4.0 -- non-commercial use only.
"""

from __future__ import annotations

import math
import shutil
import urllib.request
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from pathlib import Path

FABDEM_BASE_URL = "https://fabdem.pedalhidrografi.co/"
FABDEM_SUFFIX = "_FABDEM_V1-2.tif"
FABDEM_NODATA = -9999.0
FILL_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "fill"
FILL_SUFFIX = "_COP30.tif"

# 1 deg cells (SW corners) with no FABDEM file that COP30 2023_1 covers: the
# Armenia/Azerbaijan gap.
FILL_CELLS: frozenset[tuple[int, int]] = frozenset({
    (38, 45), (38, 46), (38, 48), (38, 49),
    (39, 44), (39, 45), (39, 46), (39, 47), (39, 48), (39, 49),
    (40, 43), (40, 44), (40, 45), (40, 46), (40, 47), (40, 48), (40, 49), (40, 50),
    (41, 43), (41, 44), (41, 45), (41, 46), (41, 47), (41, 48), (41, 49),
})


def _tile_name(lat: int, lon: int) -> str:
    """FABDEM file name for the tile whose SW corner is (lat, lon) degrees."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}{FABDEM_SUFFIX}"


def tile_url(lat: int, lon: int) -> str:
    """HTTPS URL of the FABDEM COG whose SW corner is (lat, lon)."""
    return FABDEM_BASE_URL + _tile_name(lat, lon)


@cache
def land_cells() -> frozenset[tuple[int, int]]:
    """(lat, lon) SW corners of every 1 deg cell that has a FABDEM file.

    Read from the packaged ``fabdem_cells.txt`` (the Earth Engine collection
    listing, shared with cameratopo). A cell outside this set is open ocean.
    """
    cells = set()
    for line in files("mtpi").joinpath("fabdem_cells.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        lat = int(line[1:3]) * (1 if line[0] == "N" else -1)
        lon = int(line[4:7]) * (1 if line[3] == "E" else -1)
        cells.add((lat, lon))
    return frozenset(cells)


def fill_path(lat: int, lon: int) -> Path:
    """Local COP30 fill COG for the cell whose SW corner is (lat, lon)."""
    return FILL_DIR / _tile_name(lat, lon).replace(FABDEM_SUFFIX, FILL_SUFFIX)


@cache
def dem_cells() -> frozenset[tuple[int, int]]:
    """Every cell with elevation data: FABDEM land cells plus the COP30 fill."""
    return land_cells() | FILL_CELLS


def tile_source(lat: int, lon: int) -> str:
    """GDAL path of the elevation tile for (lat, lon).

    FABDEM cells, and fill cells not built locally, come from the FABDEM
    bucket (the fill is published there under FABDEM names).
    """
    if (lat, lon) in FILL_CELLS and (path := fill_path(lat, lon)).exists():
        return str(path)
    return "/vsicurl/" + tile_url(lat, lon)


@dataclass(frozen=True)
class AOI:
    """A geographic area of interest, in degrees (WGS84)."""

    name: str
    west: float
    south: float
    east: float
    north: float

    def cells(self) -> list[tuple[int, int]]:
        """(lat, lon) SW corners of the 1 deg cells covering this AOI."""
        return [
            (lat, lon)
            for lat in range(math.floor(self.south), math.ceil(self.north))
            for lon in range(math.floor(self.west), math.ceil(self.east))
        ]

    def tiles(self) -> list[str]:
        """FABDEM tile names (by SW corner) covering this AOI."""
        return [_tile_name(lat, lon) for lat, lon in self.cells()]

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


def fetch_tiles(aoi: AOI, raw_dir: Path, base_url: str = FABDEM_BASE_URL) -> list[Path]:
    """Download the FABDEM tiles covering `aoi` into `raw_dir` (idempotent).

    Cells without a FABDEM file (all-ocean, per `land_cells`) are skipped
    with a warning rather than aborting the run.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    local: list[Path] = []
    missing: list[str] = []
    for lat, lon in aoi.cells():
        name = _tile_name(lat, lon)
        dst = raw_dir / name
        if dst.exists():
            local.append(dst)
            continue
        if (lat, lon) not in land_cells():
            missing.append(name)
            continue
        tmp = dst.with_suffix(".part")
        # Cloudflare rejects urllib's default User-Agent with 403.
        req = urllib.request.Request(base_url + name, headers={"User-Agent": "mtpi"})
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(dst)
        local.append(dst)
    if missing:
        print(f"[fetch] {len(missing)} tile(s) absent in bucket (ocean?): {missing}")
    if not local:
        raise RuntimeError(f"No FABDEM tiles found for AOI {aoi.name}")
    return local
