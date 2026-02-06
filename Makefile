SHELL := /bin/sh

UV ?= uv
export MPLBACKEND = Agg

.PHONY: lint test docs docs-sync docs-py

lint:
	$(UV) run pre-commit run --all-files

test:
	$(UV) run pytest

docs: docs-py docs-sync

docs-sync:
	$(UV) run jupytext --sync notebooks/*.py notebooks/*.ipynb

docs-py:
	set -e; \
	for notebook in notebooks/*.py; do \
		MPLBACKEND=Agg $(UV) run python "$$notebook"; \
	done
