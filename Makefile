.PHONY: setup bootstrap check format lint type test deps-audit all fix clean

AGENT_MODE ?= baseline
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python

setup:
	uv venv $(VENV)
	uv pip install -e '.[dev]' --python $(PYTHON)

bootstrap: setup

format:
	$(VENV)/bin/ruff format .

lint:
	$(VENV)/bin/ruff check .

type:
	$(VENV)/bin/pyright

check:
	$(VENV)/bin/ruff format --check .
	$(VENV)/bin/ruff check .
	$(VENV)/bin/pyright

fix:
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .

test:
	$(VENV)/bin/pytest -q

deps-audit:
	@if [ "$(AGENT_MODE)" = "production" ]; then \
		$(VENV)/bin/pip-audit --progress-spinner off . ; \
	else \
		$(VENV)/bin/pip-audit --progress-spinner off . || echo "pip-audit reported issues (advisory in baseline)" ; \
	fi

all:
	$(MAKE) check
	$(MAKE) test
	$(MAKE) deps-audit

clean:
	rm -rf $(VENV)
