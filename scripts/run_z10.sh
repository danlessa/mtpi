#!/usr/bin/env bash
# Add z10 XYZ tiles to v1 and v2, cut from the published lossless RGBA COGs.
# One version at a time (fragments ~32 GB each); resumable (per-cell markers).
#   nohup scripts/run_z10.sh > data/logs/z10.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
WORKERS=${WORKERS:-$(nproc)}
for v in v1 v2; do
  F=data/interim/xyz_z10/$v
  [ -f "$F/.uploaded" ] && { echo "[z10] $v already done"; continue; }
  .venv/bin/python -m mtpi.xyz_cut --src "s3://mtpi-fabdem/$v" --zoom 10 --frag-dir "$F" --workers "$WORKERS"
  .venv/bin/python -m mtpi.xyz --zoom 10 --min-zoom 10 --frag-dir "$F" --quality 100 \
      --workers "$WORKERS" --dest "s3://mtpi-fabdem/$v/xyz"
  rm -rf "$F/10"; touch "$F/.uploaded"
  echo "[z10] $v uploaded"
done
