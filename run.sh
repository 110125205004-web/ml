#!/usr/bin/env bash
# Setup + launch helper.
#   ./run.sh                 -> create .venv if needed and start the webcam app
#   ./run.sh --camera 1      -> any main.py arguments are passed through
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "[setup] creating virtualenv ..."
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
exec "$PY" main.py "$@"
