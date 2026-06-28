"""Command-line entry point.

Examples
--------
    python -m mtpi.cli --preset parana
    python -m mtpi.cli --bbox -50 -26 -48 -25 --name parana --res 30 \
        --scales 500 5000 25000
"""

from __future__ import annotations

import argparse

from .data import PRESETS, AOI
from .pipeline import DEFAULT_SCALES_M, Config, run


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Compute an RGB multiscale topographic-position base map.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--preset", choices=sorted(PRESETS), help="named AOI preset")
    g.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                   help="AOI bounding box in degrees")
    p.add_argument("--name", help="AOI name (required with --bbox)")
    p.add_argument("--res", type=float, default=30.0, help="output resolution, metres")
    p.add_argument("--epsg", type=int, default=None, help="target EPSG (default: AOI UTM)")
    p.add_argument("--scales", nargs=3, type=float, metavar=("MICRO", "MESO", "MACRO"),
                   help="three window sizes in metres (R, G, B)")
    p.add_argument("--clip", type=float, default=2.0, help="|z-score| saturation per channel")
    args = p.parse_args(argv)

    if args.preset:
        aoi = PRESETS[args.preset]
    else:
        if not args.name:
            p.error("--name is required with --bbox")
        aoi = AOI(args.name, *args.bbox)

    scales = DEFAULT_SCALES_M if args.scales is None else dict(
        zip(("micro", "meso", "macro"), args.scales)
    )
    run(Config(aoi=aoi, res=args.res, dst_epsg=args.epsg, scales_m=scales, clip=args.clip))


if __name__ == "__main__":
    main()
