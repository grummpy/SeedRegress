#!/bin/bash
# macOS double-click launcher. Checks Python, then starts SeedRegress.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.11 or newer is required, and python3 was not found."
  echo "Install it from https://www.python.org/downloads/ and then double-click this launcher again."
  echo "Press Return to close."
  read -r _
  exit 1
fi

major="$(python3 -c 'import sys; print(sys.version_info.major)')"
minor="$(python3 -c 'import sys; print(sys.version_info.minor)')"
if [ "$major" -lt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -lt 11 ]; }; then
  echo "Python 3.11 or newer is required. This machine has Python ${major}.${minor}."
  echo "Install a newer Python from https://www.python.org/downloads/ and then double-click this launcher again."
  echo "Press Return to close."
  read -r _
  exit 1
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

stamp=".venv/.requirements.stamp"
if [ ! -f "$stamp" ] || [ requirements.txt -nt "$stamp" ]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  touch "$stamp"
fi

exec python -m seedregress serve
