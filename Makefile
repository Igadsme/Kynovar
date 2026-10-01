.PHONY: setup test generate train evaluate ood discover plan falsify revise backend web web-build acceptance smoke

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
NPM ?= npm

export COPYFILE_DISABLE := 1

setup:
	python3 -m venv --copies .venv
	find .venv -name '._*' -delete
	$(PYTHON) -m pip install --upgrade pip
	find .venv -name '._*' -delete
	$(PYTHON) -m pip install --config-settings editable_mode=compat -e ".[dev,web]"
	find .venv -name '._*' -delete

setup-web:
	cd frontend && $(NPM) ci

test:
	$(PYTHON) -m pytest

generate:
	$(PYTHON) scripts/generate_dataset.py --config configs/experiments/development.yaml

train:
	$(PYTHON) scripts/train_dynamics.py --config configs/experiments/development.yaml --model gnn

evaluate:
	$(PYTHON) scripts/run_benchmark.py --config configs/experiments/stable-v1.yaml

ood:
	$(PYTHON) scripts/run_ood.py --config configs/experiments/stable-v1.yaml

discover:
	$(PYTHON) scripts/discover_laws.py

plan:
	$(PYTHON) scripts/active_experiments.py

falsify:
	$(PYTHON) scripts/falsification.py

revise:
	$(PYTHON) scripts/theory_shift.py

backend:
	$(PYTHON) -m uvicorn backend.kynovar_api.app:app --host 127.0.0.1 --port 8000

web:
	cd frontend && $(NPM) run dev

web-build:
	cd frontend && $(NPM) run typecheck && $(NPM) run build

smoke:
	$(PYTHON) scripts/acceptance.py --profile smoke

acceptance:
	$(PYTHON) scripts/acceptance.py --profile full
