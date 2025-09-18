#!/usr/bin/env bash
set -euo pipefail

# Default to current working directory of container
cd /app

# Helpful echo
echo "Starting Jupyter Lab on 0.0.0.0:8888 (no token)"

# Use Jupyter Lab (fast UI); fallback to classic notebook if desired
exec jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --NotebookApp.token='' \
  --NotebookApp.password='' \
  --LabApp.token='' \
  --LabApp.password='' \
  --ServerApp.allow_origin='*' \
  --ServerApp.disable_check_xsrf=True
