#!/usr/bin/env bash
# Global mTPI v2: scales 30 / 300 / 3000 km (R = 3000 km, G = 300 km, B = 30 km).
# Reuses v1's block-stats cache (pass 1 is skipped for cached cells); keeps
# only the new 3000 km DEV band (30 and 300 km equal v1's meso and macro).
# Resumable like run_world.sh.
#
#   nohup scripts/run_world_v2.sh > data/logs/world_v2.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs
set -a; . ./.env; set +a  # R2 credentials for mtpi-fabdem

DEST=${DEST:-s3://mtpi-fabdem/v2}
WORKERS=${WORKERS:-$(nproc)}
ZOOM=${ZOOM:-9}
FRAGS=data/interim/xyz_frag_v2

.venv/bin/python -m mtpi.tiled --world --scales 30000 300000 3000000 \
    --keep-dev --dev-scales 3000km --workers "$WORKERS" \
    --xyz-zoom "$ZOOM" --frag-dir "$FRAGS" --out-dir data/out/tiled_v2_world --dest "$DEST"
.venv/bin/python -m mtpi.xyz --zoom "$ZOOM" --frag-dir "$FRAGS" --workers "$WORKERS" \
    --dest "$DEST/xyz"
