#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")"
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
