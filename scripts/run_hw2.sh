#!/usr/bin/env bash
# Полный повтор HW2 из корня проекта или другой рабочей папки.
set -euo pipefail
cd "$(dirname "$0")/.."
task_python="${PYTHON_EXECUTABLE:-.venv/bin/python}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PWD}/.cache/matplotlib}"
"$task_python" scripts/download_data.py
"$task_python" scripts/run_hw2.py "$@"
"$task_python" scripts/summarize_hw2.py "$@"
"$task_python" scripts/diagnose_hw2.py "$@"
"$task_python" scripts/stress_hw2.py "$@"
