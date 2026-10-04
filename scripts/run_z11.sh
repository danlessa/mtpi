#!/usr/bin/env bash
# z11 XYZ tiles for v1 then v2 via mtpi.xyz_fast (canvas renderer + s5cmd uploads).
# Resumable: finished canvases have markers; failed ones are retried (3 attempts).
#   nohup setsid scripts/run_z11.sh > data/logs/z11.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export MALLOC_ARENA_MAX=2
for v in v1 v2; do
  for i in 1 2 3; do
    if .venv/bin/python -m mtpi.xyz_fast --version "$v" --zoom 11 --workers 5 --read-threads 8 --upload-slots 6; then
      echo "[z11] $(date -u +%FT%TZ) $v complete"; break
    fi
    echo "[z11] $(date -u +%FT%TZ) $v attempt $i had failures; retrying"
  done
done
echo "[z11] $(date -u +%FT%TZ) finished"
