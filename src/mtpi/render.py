"""Compose multiscale DEV layers into an RGB topographic-position base map.

Matches the established "Pedal Hidrográfico" aesthetic: each channel is the
**absolute** deviation from mean elevation `|DEV|` at one scale, stretched
0..`clip` onto an additive RGB channel over a black background, so:

    micro -> Blue,  meso -> Green,  macro -> Red

A point that matches its neighbours at a scale contributes 0 (black) to that
channel; a strong ridge *or* valley contributes brightly. Mixtures read as
cyan (micro+meso), purple (micro+macro), yellow (meso+macro), white (all),
black (none) -- revealing where geoforms of different sizes coincide.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import rasterio.enums

# (R, G, B) <- scale labels, per the artwork's colour scheme.
ARTWORK_RGB: tuple[str, str, str] = ("macro", "meso", "micro")


def _read_dev(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, valid_mask) for a DEV raster, NoData/NaN masked out."""
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
    data = np.asarray(arr.filled(np.nan), dtype="float32")
    valid = np.isfinite(data)
    return data, valid


def _stretch_abs(dev: np.ndarray, valid: np.ndarray, clip: float) -> np.ndarray:
    """|DEV| clipped to [0, clip] -> uint8 0..255; invalid -> 0 (black)."""
    u = (np.clip(np.abs(dev), 0.0, clip) / clip * 255.0).round()
    return np.where(valid, u, 0).astype("uint8")


def rgb_composite(
    dev_paths: dict[str, Path],
    out_tif: Path,
    out_png: Path,
    order: tuple[str, str, str] = ARTWORK_RGB,
    clip: float = 2.0,
) -> tuple[Path, Path]:
    """Write an RGB GeoTIFF (georeferenced) and an RGBA PNG preview.

    `order` maps three DEV labels onto (R, G, B); `clip` is the |z-score| at
    which each channel saturates to full intensity.
    """
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    chans, valid_all = [], None
    for label in order:
        dev, valid = _read_dev(dev_paths[label])
        chans.append(dev)
        valid_all = valid if valid_all is None else (valid_all & valid)

    rgb = np.stack([_stretch_abs(dev, valid_all, clip) for dev in chans], axis=0)

    # Georeferenced 3-band byte GeoTIFF (for GIS / large-format printing).
    with rasterio.open(dev_paths[order[0]]) as ref:
        profile = ref.profile
    profile.update(
        count=3, dtype="uint8", nodata=None,
        compress="deflate", predictor=2, photometric="RGB",
        tiled=True, blockxsize=256, blockysize=256,  # whitebox DEVs are striped
    )
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(rgb)
        dst.colorinterp = [rasterio.enums.ColorInterp.red,
                           rasterio.enums.ColorInterp.green,
                           rasterio.enums.ColorInterp.blue]

    # RGBA PNG preview with transparent NoData.
    import matplotlib.image as mpimg

    alpha = np.where(valid_all, 255, 0).astype("uint8")
    rgba = np.concatenate([rgb, alpha[None]], axis=0)
    mpimg.imsave(out_png, np.transpose(rgba, (1, 2, 0)))

    return out_tif, out_png
