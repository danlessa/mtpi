"""End-to-end AOI -> RGB multiscale topographic-position base map."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .data import AOI, fetch_tiles
from .render import ARTWORK_RGB, rgb_composite
from .terrain import mosaic_and_warp, multiscale_dev

# The artwork's canonical scales (metres): micro 3 km, meso 30 km, macro 300 km.
# Mapped to colour by ARTWORK_RGB (R=macro, G=meso, B=micro). The 300 km macro
# needs a regional extent; for small validation AOIs pass reduced --scales.
DEFAULT_SCALES_M: dict[str, float] = {"micro": 3000.0, "meso": 30000.0, "macro": 300000.0}

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"


@dataclass
class RunPaths:
    dem: Path
    devs: dict[str, Path]
    rgb_tif: Path
    rgb_png: Path


@dataclass
class Config:
    aoi: AOI
    res: float = 30.0
    dst_epsg: int | None = None  # default: AOI's best-fit UTM
    scales_m: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCALES_M))
    clip: float = 2.0
    raw_dir: Path = DATA / "raw"
    interim_dir: Path = DATA / "interim"
    out_dir: Path = DATA / "out"


def run(cfg: Config) -> RunPaths:
    aoi = cfg.aoi
    epsg = cfg.dst_epsg or aoi.utm_epsg()
    print(f"[aoi] {aoi.name}  {aoi.width_km:.0f} x {aoi.height_km:.0f} km  -> EPSG:{epsg} @ {cfg.res} m")

    tiles = fetch_tiles(aoi, cfg.raw_dir)
    print(f"[fetch] {len(tiles)} tile(s) ready")

    work = cfg.interim_dir / aoi.name
    dem = mosaic_and_warp(tiles, work / f"{aoi.name}_dem_{epsg}.tif", epsg, cfg.res)
    print(f"[warp] {dem}")

    devs = multiscale_dev(dem, work, cfg.scales_m, cfg.res)

    rgb_tif = cfg.out_dir / f"{aoi.name}_mtpi_rgb.tif"
    rgb_png = cfg.out_dir / f"{aoi.name}_mtpi_rgb.png"
    rgb_composite(devs, rgb_tif, rgb_png, order=ARTWORK_RGB, clip=cfg.clip)
    print(f"[rgb] {rgb_tif}\n[rgb] {rgb_png}")

    return RunPaths(dem=dem, devs=devs, rgb_tif=rgb_tif, rgb_png=rgb_png)
