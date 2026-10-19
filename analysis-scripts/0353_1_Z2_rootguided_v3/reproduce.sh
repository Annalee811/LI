#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
"${PYTHON:-python}" "$script_dir/run.py" --raw "$1" --work "$2"
