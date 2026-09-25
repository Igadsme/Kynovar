.PHONY: setup test generate train discover evaluate web

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

# ExFAT does not support symlinks and macOS writes AppleDouble sidecar files.
# --copies keeps the virtualenv on the project drive. editable_mode=compat
# installs a .pth file instead of a symlink into site-packages.
export COPYFILE_DISABLE := 1

setup:
	python3 -m venv --copies .venv
	find .venv -name '._*' -delete
	$(PYTHON) -m pip install --upgrade pip
	find .venv -name '._*' -delete
	$(PYTHON) -m pip install --config-settings editable_mode=compat -e ".[dev]"
	find .venv -name '._*' -delete

test:
	$(PYTHON) -m pytest

generate train discover evaluate web:
	@echo "This target is not implemented yet. Current milestone: 1 (universe simulator)." >&2
	@echo "See docs/milestone-1.md." >&2
	@exit 1
