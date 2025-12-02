## Raster Tools
--------------

Python tools for lazy, parallel geospatial raster processing.

## Introduction
---------------

Raster Tools is a Python package that facilitates a wide range of geospatial,
statistical, and machine learning analyses using delayed and automated parallel
processing for rasters. It seeks to bridge the gaps in Python's data stack for
processing raster data and to make building processing and analysis pipelines
easier. With Raster Tools, operations can be easily chained together,
eliminating the need to write intermediate results and saving on storage space.
The use of Dask, also enables [out-of-core
processing](https://en.wikipedia.org/wiki/External_memory_algorithm) so rasters
larger than the available memory can be processed in chunks.

Under the hood, Raster Tools uses Dask to parallelize tasks and delay
operations, [Rasterio](https://github.com/rasterio/rasterio),
[rioxarray](https://github.com/corteva/rioxarray), and
[odc-geo](https://github.com/opendatacube/odc-geo) for geospatial operations,
and [Numba](https://github.com/numba/numba) for accelerating Python code.
Limited support is also provided for working with Vector data using
[dask-geopandas](https://github.com/geopandas/dask-geopandas).

## Install
----------

#### Pip

```sh
pip install raster-tools
```

#### Conda

```sh
conda install -c conda-forge cfgrib dask-geopandas dask-image fiona netcdf4 numba odc-geo pyogrio rioxarray scipy
pip install --no-deps raster-tools
```


## Helpful Links
- [How to Contribute](./CONTRIBUTING.md)
- [Documentation](https://um-rmrs.github.io/raster_tools/)
- [Notebooks & Tutorials](./notebooks/README.md)
- [PyPi link](https://pypi.org/project/raster-tools/)
- [Installation](./notebooks/install_raster_tools.md)

## Package Dependencies
- [cfgrib](https://github.com/ecmwf/cfgrib)
- [dask](https://dask.org/)
- [dask_image](https://image.dask.org/en/latest/)
- [dask-geopandas](https://dask-geopandas.readthedocs.io/en/stable/)
- [fiona](https://fiona.readthedocs.io/en/stable/)
- [geopandas](https://geopandas.org/en/stable/)
- [netcdf](https://unidata.github.io/netcdf4-python/)
- [numba](https://numba.pydata.org/)
- [numpy](https://numpy.org/doc/stable/)
- [odc-geo](https://odc-geo.readthedocs.io/en/latest/)
- [pyproj](https://pyproj4.github.io/pyproj/stable/)
- [rasterio](https://rasterio.readthedocs.io/en/latest/)
- [rioxarray](https://corteva.github.io/rioxarray/stable/)
- [shapely 2](https://shapely.readthedocs.io/en/stable/)
- [scipy](https://docs.scipy.org/doc/scipy/)
- [xarray](https://xarray.pydata.org/en/stable/)
