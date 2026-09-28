#!/usr/bin/env bash
# Global mTPI run: per-cell RGBA + int16 DEV COGs streamed to R2, then the
# XYZ WebP pyramid (z0-z9). Resumable: rerun the same command after a crash,
# a resize or failed cells -- finished cells are skipped (checked on R2).
#
#   nohup scripts/run_world.sh > data/logs/world.log 2>&1 &
#   tail -f data/logs/world.log
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs
set -a; . ./.env; set +a  # R2 credentials for mtpi-fabdem

DEST=${DEST:-s3://mtpi-fabdem/v1}
WORKERS=${WORKERS:-$(nproc)}
ZOOM=${ZOOM:-9}

.venv/bin/python -m mtpi.tiled --world --keep-dev --workers "$WORKERS" \
    --xyz-zoom "$ZOOM" --dest "$DEST"
.venv/bin/python -m mtpi.xyz --zoom "$ZOOM" --workers "$WORKERS" --dest "$DEST/xyz"
