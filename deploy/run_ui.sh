#!/usr/bin/env bash
# Laptop 2: serve the evaluator UI on port 5500. Usage: deploy/run_ui.sh http://<laptop1-ip>:8000
set -e
cd "$(dirname "$0")/../ui"
API="${1:-http://127.0.0.1:8000}"
echo "Open:  http://localhost:5500/?api=${API}"
exec python3 -m http.server 5500
