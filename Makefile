.PHONY: setup setup-web test verify generate train evaluate ood discover plan falsify revise backend frontend web web-build dev acceptance smoke

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

verify: test
	cd frontend && $(NPM) run typecheck && $(NPM) run build

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

frontend:
	cd frontend && $(NPM) run build && PORT=3000 HOSTNAME=127.0.0.1 KYNOVAR_API_INTERNAL=http://127.0.0.1:8000 $(NPM) run start

web:
	cd frontend && $(NPM) run dev

web-build:
	cd frontend && $(NPM) run typecheck && $(NPM) run build

dev:
	@trap 'kill 0' INT TERM EXIT; $(PYTHON) -m uvicorn backend.kynovar_api.app:app --host 127.0.0.1 --port 8000 & cd frontend && KYNOVAR_API_INTERNAL=http://127.0.0.1:8000 $(NPM) run dev

smoke:
	$(PYTHON) scripts/acceptance.py --profile smoke

acceptance:
	$(PYTHON) scripts/acceptance.py --profile full
