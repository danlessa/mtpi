#!/usr/bin/env bash
# MTPI v0 (R = DEVmax 3-30 km, G = 0.3-3 km, B = 0.03-0.3 km) for the world:
# COGs + DEV to s3://mtpi-fabdem/v0/, z0-z9 tiles from fragments, then z10 and z11
# with the fast canvas renderer. Every step is resumable (markers); retried 3x.
#   nohup setsid scripts/run_v0.sh > data/logs/v0.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export MALLOC_ARENA_MAX=2
FRAGS=data/interim/xyz_frag_v0
retry() { local i; for i in 1 2 3; do "$@" && return 0; echo "[v0] $(date -u +%FT%TZ) attempt $i failed: $*"; done; return 1; }
retry .venv/bin/python -m mtpi.products --world --products v0 --dest-root s3://mtpi-fabdem \
    --codec zstd --xyz-zoom 9 --frag-root "$FRAGS" --workers 3 && echo "[v0] $(date -u +%FT%TZ) COGs done"
retry .venv/bin/python -m mtpi.xyz --zoom 9 --frag-dir "$FRAGS/v0" --quality 100 --workers 16 \
    --dest s3://mtpi-fabdem/v0/xyz && echo "[v0] $(date -u +%FT%TZ) z0-z9 done" && rm -rf "$FRAGS/v0/9"
for z in 10 11; do
  retry .venv/bin/python -m mtpi.xyz_fast --version v0 --zoom $z --workers 5 --read-threads 8 \
      --upload-slots 6 && echo "[v0] $(date -u +%FT%TZ) z$z done"
done
echo "[v0] $(date -u +%FT%TZ) finished"
