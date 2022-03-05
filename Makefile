.PHONY: all build clean cleandocs dev test test-mp

all: build

build:
	python setup.py build_ext -fi

dev:
	conda env create -f requirements/dev.yml

install-dev:
	pip install --no-build-isolation -e .

test:
	nose2

test-mp:
	nose2 --plugin=nose2.plugins.mp -N 10

clean:
	rm -rf build/
	rm -rf venv/
	rm -rf raster_tools.egg-info/
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyd" | xargs rm -f
	find . -name "*.pyx" -exec ./tools/rm_matching_c_cpp_file.sh {} \;

cleandocs:
	rm -rf docs/_build
	rm -f docs/generated/*
	rm -f docs/**/generated/*
