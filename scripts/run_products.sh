#!/usr/bin/env bash
# World run of the smooth-window DEVmax products (v1, v2) into R2, then their
# lossless z0-z9 XYZ pyramids. Resumable: rerun to retry failed cells.
#   nohup scripts/run_products.sh > data/logs/products.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
WORKERS=${WORKERS:-$(nproc)}
FRAGS=data/interim/xyz_frag_ms
.venv/bin/python -m mtpi.products --world --dest-root s3://mtpi-fabdem --codec zstd \
    --xyz-zoom 9 --frag-root "$FRAGS" --workers "$WORKERS"
for v in v1 v2; do
  .venv/bin/python -m mtpi.xyz --zoom 9 --frag-dir "$FRAGS/$v" --quality 100 \
      --workers "$WORKERS" --dest "s3://mtpi-fabdem/$v/xyz"
done
