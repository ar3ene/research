#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
requirements_file="$repo_root/requirements.txt"

if [[ ! -x "$python_bin" ]]; then
  echo "Error: expected virtual environment Python at $python_bin" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: scripts/install_python_packages.sh <package> [<package> ...]" >&2
  echo "Example: scripts/install_python_packages.sh scipy seaborn" >&2
  exit 1
fi

"$python_bin" -m pip install "$@"
"$python_bin" -m pip freeze | LC_ALL=C sort > "$requirements_file"

echo "Updated $requirements_file"
