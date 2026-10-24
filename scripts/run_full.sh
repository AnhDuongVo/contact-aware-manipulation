#!/usr/bin/env bash
# Reproduce everything from scratch on one machine.
#   bash scripts/run_full.sh            # committed configuration (about 7 h on 2 cores)
#   CONFIG=configs/full.yaml bash scripts/run_full.sh   # bigger run for a lab node
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/main.yaml}
WORKERS=${WORKERS:-$(python -c "import os; print(os.cpu_count())")}
export MUJOCO_GL=${MUJOCO_GL:-egl}   # use osmesa on machines without a GPU

python -m pytest -q
python scripts/run_experiments.py "$CONFIG" --workers "$WORKERS"
if [ "$CONFIG" = "configs/main.yaml" ]; then
  python scripts/run_experiments.py configs/sample_efficiency.yaml --workers "$WORKERS"
  python scripts/run_experiments.py configs/press_memory.yaml --workers "$WORKERS"
  python -m cambench.analysis.report --main results/main --se results/sample_efficiency \
      --memory results/press_memory --out results
else
  OUT=$(python -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['out'])")
  python -m cambench.analysis.report --main "$OUT" --se none --memory none --out "$OUT"
fi
