#!/usr/bin/env bash
set -euo pipefail
# Usage: PYTHON=/path/to/python bash reproduce.sh /path/to/Q1-3-Z3 /path/to/new-work
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
"${PYTHON:-python}" "$script_dir/run.py" --raw-dir "$1" --work "$2"
