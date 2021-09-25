# raster_tools Documentation

### Building the Docs
Navigate to the `docs` directory and execute the commands below. In order to
build the docs, we need to create a python environment using
`ci/requirements/docs.yml`.

```sh
# Create an environment for building the docs
conda env create -f ../ci/requirements/docs.yml
conda activate rstools-docs
# Build them
make html
```

The `ci/requirements/docs.txt` file can be used with pip to create
an environment as well.

### Viewing the Docs
Once the docs have been built, they can be viewed by opening the generated
html files.

```sh
open _build/html/index.html
```
