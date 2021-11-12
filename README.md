# raster_tools
RMRS Raster Utility Project

## Dependencies
* [cython](https://cython.readthedocs.io/en/latest/)
* [dask](https://dask.org/)
* [dask_image](https://image.dask.org/en/latest/)
* [netcdf4](https://unidata.github.io/netcdf4-python/)
* [rasterio](https://rasterio.readthedocs.io/en/latest/)
* [rioxarray](https://corteva.github.io/rioxarray/stable/)
* [xarray](https://xarray.pydata.org/en/stable/)
* [cupy](https://cupy.dev/): optional and experimental

## Before Using
Some of this packages modules use
[cython](https://cython.readthedocs.io/en/latest/) code.  You must compile the
Cython code in order to use this package. To do this, make sure that the cython
package is installed and run the following in the project root:
```sh
python setup.py build_ext --inplace
```
This will compile the necessary shared objects that python can use.

my first edit jsh
