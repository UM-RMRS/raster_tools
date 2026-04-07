.PHONY: all dist clean clean-env clean-build clean-pyc clean-docs dev test test-mp upload

all: dist

upload: dist
	python -m twine upload dist/*

dist: clean
	python -m build

dev:
	conda env create -f requirements/dev.yml
	pip install --no-deps -e .

test:
	pytest

test-mp:
	pytest -n 10

clean: clean-env clean-build clean-pyc clean-docs


clean-env:
	rm -rf venv/

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean-docs:
	rm -rf docs/_build
	rm -f docs/generated/*
	rm -f docs/**/generated/*
