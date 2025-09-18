#!/usr/bin/env bash
set -euo pipefail

cd /app
export STREAMLIT_SERVER_PORT=${STREAMLIT_SERVER_PORT:-8501}
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

echo "Starting Streamlit on 0.0.0.0:${STREAMLIT_SERVER_PORT}"
exec streamlit run streamlit_app.py --server.port=${STREAMLIT_SERVER_PORT} --server.address=${STREAMLIT_SERVER_ADDRESS}
