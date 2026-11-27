#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
task_python="${PYTHON_EXECUTABLE:-.venv/bin/python}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PWD}/.cache/matplotlib}"
"$task_python" scripts/download_data.py --train-only
"$task_python" scripts/run_hw3.py "$@"
