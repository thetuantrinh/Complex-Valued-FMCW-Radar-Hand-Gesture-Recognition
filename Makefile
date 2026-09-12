.PHONY: help install install-dev test lint format typecheck check clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install the package with development tooling
	pip install -e ".[dev]"

test:  ## Run the test suite
	pytest

lint:  ## Lint with ruff
	ruff check .
	ruff format --check .

format:  ## Auto-format with ruff
	ruff check --fix .
	ruff format .

typecheck:  ## Type-check with mypy
	mypy

check: lint typecheck test  ## Run every check

clean:  ## Remove build and cache artifacts
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
