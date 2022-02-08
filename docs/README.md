# raster_tools Documentation

### Building the Docs
Navigate to the `docs` directory and execute the commands below. In order to
build the docs, we need to create a python environment using
`requirements/docs.yml`.

If using conda:

```sh
# Create an environment for building the docs
conda env create -f ../requirements/docs.yml
conda activate rstools-docs
```

If using pip

```sh
# Create an environment for building the docs
python -m venv ../docs-venv
. ../docs-venv/bin/activate
pip install -r ../requirements/docs.txt
```

```sh
# Build them
make html
```

### Viewing the Docs
Once the docs have been built, they can be viewed by opening the generated
html files.

```sh
open _build/html/index.html
```
