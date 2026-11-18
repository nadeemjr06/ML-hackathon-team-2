#!/usr/bin/env bash
# Laptop 1: start the PathSnap API on all interfaces, port 8000.
# Usage: deploy/run_api.sh [/path/to/eval_trips]
set -e
cd "$(dirname "$0")/.."
if [ -n "$1" ]; then export PATHSNAP_EVAL_DIR="$1"; fi
echo "Evaluation trips: ${PATHSNAP_EVAL_DIR:-data/student_test}"
echo "LAN IP (give this to Laptop 2): $(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')"
exec .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
