.PHONY: all build clean cleandocs dev test test-mp

all: build

dev:
	conda env create -f requirements/dev.yml

install-dev:
	pip install -e .

test:
	pytest

test-mp:
	pytest -n 10

clean:
	rm -rf venv/
	rm -rf raster_tools.egg-info/

cleandocs:
	rm -rf docs/_build
	rm -f docs/generated/*
	rm -f docs/**/generated/*
