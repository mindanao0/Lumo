#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
export PYGAME_HIDE_SUPPORT_PROMPT=1
if ! .venv/bin/python -c 'import numpy, pygame, OpenGL, psutil' >/dev/null 2>&1; then
  .venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
fi
exec .venv/bin/python "$PWD/lumo.py" "$@"
