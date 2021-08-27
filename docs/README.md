# raster_tools Documentation

### Building the Docs
Navigate to the `docs` directory and execute the commands below. In order to
build the docs, we need to create a python environment using
`ci/requirements/docs.yml`

```sh
# Create an environment for building the docs
conda env create -f ../ci/requirements/doc.yml
conda activate rstools-docs
# Build them
make html
```

### Viewing the Docs
Once the docs have build built, then can be viewed by opening the generated
html files.

```sh
open _build/html/index.html
```
