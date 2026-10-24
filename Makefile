# One-command entry points. WORKERS defaults to the number of CPU cores.
WORKERS ?= $(shell python -c "import os; print(os.cpu_count())")

.PHONY: install test smoke data main sample-efficiency press-memory report videos all clean-results

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q

smoke:            ## two-minute end-to-end check
	python scripts/run_experiments.py configs/smoke.yaml --workers 2

data:
	python scripts/collect.py --task insertion --episodes 400
	python scripts/collect.py --task press --episodes 300
	python scripts/collect.py --task surface --episodes 300

main:
	python scripts/run_experiments.py configs/main.yaml --workers $(WORKERS)

sample-efficiency:
	python scripts/run_experiments.py configs/sample_efficiency.yaml --workers $(WORKERS)

press-memory:     ## follow-up: longer history on the press task
	python scripts/run_experiments.py configs/press_memory.yaml --workers $(WORKERS)

report:
	python -m cambench.analysis.report --main results/main --se results/sample_efficiency --memory results/press_memory --out results

videos:
	MUJOCO_GL=osmesa python scripts/make_videos.py --task insertion --eval-set heldout:calibration --episodes 0 1 2 3
	MUJOCO_GL=osmesa python scripts/make_videos.py --task press --eval-set heldout:calibration --episodes 0 1
	MUJOCO_GL=osmesa python scripts/make_videos.py --task surface --eval-set heldout:rough_surface --episodes 0 1

all: data main sample-efficiency press-memory report videos

clean-results:
	rm -rf results/main results/sample_efficiency results/press_memory results/figures
