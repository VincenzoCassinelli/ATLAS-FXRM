#!/bin/bash
# Double-click to launch ATLAS-FXRM on macOS. First run installs everything (a few minutes).
cd "$(dirname "$0")" || exit 1
PY=$(command -v python3.12 || command -v python3.13 || command -v python3.11 || command -v python3)
if [ -z "$PY" ]; then echo "Python not found. Install it from python.org/downloads, then double-click again."; read -p "Press Enter..."; exit 1; fi
if [ ! -d venv ]; then
  echo "First-time setup using $PY ..."
  "$PY" -m venv venv || { echo "Could not create environment."; read -p "Press Enter..."; exit 1; }
  source venv/bin/activate
  python -m pip install --upgrade pip -q
  python -m pip install -r requirements.txt || { echo "Install failed - see errors above."; rm -rf venv; read -p "Press Enter..."; exit 1; }
else
  source venv/bin/activate
fi
mkdir -p .streamlit; [ -f config.toml ] && mv config.toml .streamlit/
echo; echo "Starting ATLAS-FXRM... your browser will open. Close this window or press Ctrl+C to quit."
streamlit run dashboard.py --server.headless=false
