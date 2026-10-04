# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Global multiscale topographic-position maps (MTPI) from FABDEM (30 m): for each pixel, the
deviation from mean elevation (DEV, a z-score) at several window sizes, rendered as RGB with the
largest scale in red and the smallest in blue (the artwork described in README.md, in Portuguese).
Outputs are published to the Cloudflare R2 bucket `mtpi-fabdem` and consumed as XYZ layers by
cameratopo (`../../pedalhidro/cameratopo`, layers "MTPI global v0/v1/v2").

The colour stretch is settled: linear |DEV| from 0 to 2 per channel, gamma 1 (`render._stretch_abs`).

## Setup and commands

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[r2]"      # numpy, rasterio, numba, boto3
# ".[legacy]" adds GDAL + WhiteboxTools for the original single-AOI path (mtpi.cli)
set -a; . ./.env; set +a                                 # R2 S3 credentials (gitignored)
```

`s5cmd` (used for tile uploads) lives in `~/.local/bin`.

There is no test suite or linter. Correctness is checked with the validation scripts:

```bash
.venv/bin/python scripts/validate_multiscale.py S24W047 N03W067 --brute --T 20  # current engine
.venv/bin/python scripts/validate_tiled.py --out-dir DIR CELL ...  # legacy engine, brute force
.venv/bin/python scripts/validate_global.py CELL ...               # legacy 3000 km global grid
```

Without `--dest-root`, a run on a few cells writes locally, which is the quickest way to try a change:

```bash
.venv/bin/python -m mtpi.products --cells S14W044 --products v0 --workers 1 \
    --out-root data/out/test --frag-root data/interim/test
```

World runs are detached, logged to `data/logs/`, and resumable (rerun the same script):
`scripts/run_products.sh` (v1+v2), `scripts/run_v0.sh`, `scripts/run_z11.sh`
(`nohup setsid scripts/X.sh > data/logs/X.log 2>&1 &`).

## Architecture

**Grid and data** (`data.py`, `tiled.py`)
- FABDEM 1° COGs come from `fabdem.pedalhidrografi.co`, on the native 1″ lat/lon grid: 3600² per
  cell, with pixel centres on whole arc-seconds, so neighbouring cells abut exactly.
- `fabdem_cells.txt` lists the land cells; any other cell is ocean.
- Windows are square in metres, so a row's x half-width grows as 1/cos(lat). Longitude wraps
  across the antimeridian.
- Sea is null. Exactly 0 m is sea, and so is the Caspian, a flat −28.0 m surface found by a
  flat-square opening in `tiled.read_cell`. Sea is excluded from all window statistics and is
  transparent in the output.
- The 25 Armenia/Azerbaijan cells missing from FABDEM are filled from Copernicus GLO-30 2023_1
  (`fill.py`). They are published into the FABDEM bucket under FABDEM file names.

**Engines**
- `tiled.py` is the legacy engine with square windows. It uses integral images: fine scales are
  exact (cell plus a halo from neighbouring COGs), coarse scales come from per-cell 30 px block
  stats (pass 1, cached in `data/interim/coarse/b30`), and global scales come from the merged
  `global_b120.npy` grid. Square windows cast sharp square "shadows" of strong relief into flat
  terrain, which is why this engine was replaced.
- `multiscale.py` is the current engine. It uses smooth windows: three fractional box passes with
  the same sigma as the square window. Each scale is evaluated on a block pyramid with at least
  `T`=20 blocks per half-window:
  - 1, 3 and 10 px, aggregated from the full-resolution halo;
  - 30 px, from the pass-1 cache;
  - 120 px, from the global grid.
- The heavy loops are numba kernels.
- Each row is filtered with its own latitude's width, so overlapping cells produce identical
  values (no seams).
- DEVmax is the DEV with the largest magnitude over log-spaced window sizes within a band.
- Pass 1 and the global grid (`tiled.run_coarse`, `tiled.build_global`) serve both engines.

**Products** (`products.py`) compute each cell once and write every selected product:

| product | R | G | B | dev/ bands kept |
|---|---|---|---|---|
| v0 | 3–30 km | 0.3–3 km | 0.03–0.3 km | 0.03–0.3 km |
| v1 | 30–300 km | 3–30 km | 0.3–3 km | all three |
| v2 | 300–3000 km | 30–300 km | 3–30 km | 300–3000 km |

- Bands sample 6, 8 or 12 window sizes per decade (denser where scales are cheap).
- `--products` also limits the computation to the bands those products use.
- R2 layout:
  - `<v>/rgb/<cell>.tif`: RGBA COG, zstd.
  - `<v>/dev/<cell>.tif`: int16 DEVmax × 100, with GDAL scale 0.01.
  - `<v>/xyz/{z}/{x}/{y}.webp`: tiles.
- R2 streaming, per-cell fault tolerance and resume come from `tiled` (`_publish`, `_done`, `_map`).
- A product counts as done for a cell once its fragment marker exists. The marker is written
  after the RGB upload, because older outputs share the same names.

**Tiles**
- Every tile pixel takes its value from the one cell containing its centre, area-averaged over that
  cell's pixels with alpha weighting. This rule is shared by `xyz.cell_fragments` and `xyz_fast`.
- z0–z9: the product run writes fragments, and `mtpi.xyz` merges them and builds the pyramid.
- z10+: `xyz_fast.py` renders straight from the published COGs:
  - one 8192 px canvas per zoom-(z−5) tile;
  - windowed HTTP reads from `mtpi.pedalhidrografi.co`;
  - a numba painter;
  - lossless WebP at effort 0;
  - `s5cmd` uploads from a parent thread pool.

  Its output is bit-identical to the fragment path. It runs at about 250 tiles/s on 4 vCPU.
- `xyz_cut.py` is the older fragment-based path for adding a zoom.

**Legacy single-AOI path**: `cli.py`, `pipeline.py` and `terrain.py`, using a UTM warp and
WhiteboxTools. This produced the original artwork.

## Operational notes

- **cameratopo coupling.** Tiles uploaded before 2026-10-03 carry `Cache-Control: immutable` for
  one year. cameratopo's layer URLs therefore end in `?r=N`; bump N there after re-rendering zooms
  that already exist. When adding a zoom, raise that layer's `zmax` only once its tiles are live.
  Pushing cameratopo's `main` deploys it, and every change to served files needs a `web/sw.js`
  VERSION bump.
- **Clock after a resize.** The box is a Magalu VPS that is often resized (4 ↔ 32 vCPU). After a
  resize the clock can be hours off, and R2 then rejects requests (`RequestTimeTooSkewed` or 403).
  Compare against a server's `Date` header and run `sudo systemctl restart systemd-timesyncd`.
- **Memory.** Threaded readers can get worker processes OOM-killed. Workers cap malloc arenas
  (`MALLOC_ARENA_MAX=2`, or `mallopt`). `xyz_fast` uses `ProcessPoolExecutor`, so a dead worker
  fails its canvas instead of hanging the run; rerun to retry it.
- **Upload concurrency.** Uploads to R2 are latency-bound (about 0.6 s per PUT from the box). Use
  high concurrency: 48 or more s5cmd workers, or 16 or more Python workers.
- **Commit signing.** Commits are SSH-signed; the key's agent is usually at
  `/run/user/1001/ssh-agent.socket`.
