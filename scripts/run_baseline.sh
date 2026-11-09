#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
task_python="${PYTHON_EXECUTABLE:-.venv/bin/python}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${PWD}/.cache/matplotlib"
exec "$task_python" -m spaceship_hw1 run "$@"
