import os
import urllib
from pathlib import Path

import dask
import numpy as np
import rasterio as rio
import rioxarray as xrio
import xarray as xr
from affine import Affine
from dask.array.core import normalize_chunks as dask_chunks

from raster_tools.dtypes import F32, F64, I64, U8, is_bool, is_float, is_int
from raster_tools.exceptions import (
    AffineEncodingError,
    DimensionsError,
    RasterDataError,
    RasterIOError,
)
from raster_tools.utils import to_chunk_dict, validate_path


def _get_extension(path):
    return os.path.splitext(path)[-1].lower()


def _get_chunking_info_from_file(src_file):
    with rio.open(src_file) as src:
        tile_shape = (1, *src.block_shapes[0])
        shape = (src.count, *src.shape)
        dtype = np.dtype(src.dtypes[0])
        return tile_shape, shape, dtype


def _get_chunks(data=None, src_file=None):
    chunks = (1, "auto", "auto")
    if data is None:
        if src_file is None:
            return chunks
        tile_shape, shape, dtype = _get_chunking_info_from_file(src_file)
    else:
        shape = data.shape
        dtype = data.dtype
        tile_shape = None
        if dask.is_dask_collection(data):
            tile_shape = data.chunks
        elif src_file is not None:
            _, tile_shape, _ = _get_chunking_info_from_file(src_file)
    return dask_chunks(chunks, shape, dtype=dtype, previous_chunks=tile_shape)


def chunk(xrs, src_file=None):
    chunks = to_chunk_dict(
        _get_chunks(
            xrs.raster if isinstance(xrs, xr.Dataset) else xrs, src_file
        )
    )
    return xrs.chunk(chunks)


TIFF_EXTS = frozenset((".tif", ".tiff"))
NC_EXTS = frozenset((".cdf", ".nc", ".nc4"))
HDF_EXTS = frozenset((".hdf", ".h4", ".hdf4", ".he2", ".h5", ".hdf5", ".he5"))
GRIB_EXTS = frozenset((".grib", ".grib2", ".grb", ".grb2", ".gb", ".gb2"))
BATCH_EXTS = frozenset((".bch",))

# File extenstions that can't be read in yet
READ_NOT_IMPLEMENTED_EXTS = NC_EXTS | HDF_EXTS | GRIB_EXTS
# File extenstions that can't be written out yet
WRITE_NOT_IMPLEMENTED_EXTS = NC_EXTS | HDF_EXTS | GRIB_EXTS

IO_UNDERSTOOD_TYPES = (str, Path)


def is_batch_file(path):
    return _get_extension(path) in BATCH_EXTS


ESRI_DEFAULT_F32_NV = np.finfo(F32).min


def normalize_null_value(nv, dtype):
    # Make sure that ESRI's default F32 null value is properly
    # registered as F32
    if dtype == F32 and nv is not None and np.isclose(nv, ESRI_DEFAULT_F32_NV):
        nv = F32.type(nv)
    # Some rasters have (u)int dtype and a null value that is a whole number
    # but it gets read in as a float. This can cause a lot of accidental type
    # promotions down the pipeline. Check for this case and correct it.
    if is_int(dtype) and is_float(nv) and float(nv).is_integer():
        nv = int(nv)
    return nv


def open_raster_from_path_or_url(path):
    from raster_tools.raster import (
        _try_to_get_null_value_xarray,
        normalize_xarray_data,
    )

    if type(path) in IO_UNDERSTOOD_TYPES:
        path = str(path)
    else:
        raise RasterIOError(
            f"Could not resolve input to a raster path or URL: '{path}'"
        )
    if urllib.parse.urlparse(path) == "":
        # Assume file path
        validate_path(path)
        ext = _get_extension(path)
    else:
        # URL
        ext = ""

    xrs = None
    # Try to let gdal open anything but NC, HDF, GRIB files
    if not ext or ext not in READ_NOT_IMPLEMENTED_EXTS:
        try:
            xrs = xrio.open_rasterio(
                path, chunks=to_chunk_dict(_get_chunks()), lock=False
            )
        except rio.errors.RasterioIOError as e:
            raise RasterIOError(
                "Could not open given path as a raster."
            ) from e
    elif ext in READ_NOT_IMPLEMENTED_EXTS:
        raise NotImplementedError(
            "Reading of NetCDF, HDF, and GRIB files is not supported at this"
            " time."
        )
    else:
        raise RasterIOError("Unknown file type")
    if isinstance(xrs, xr.Dataset):
        raise RasterDataError("Too many data variables in input data")
    assert isinstance(
        xrs, xr.DataArray
    ), "Resulting data structure must be a DataArray"
    if not dask.is_dask_collection(xrs):
        xrs = chunk(xrs, path)

    xrs = normalize_xarray_data(xrs)

    nv = _try_to_get_null_value_xarray(xrs)
    nv = normalize_null_value(nv, xrs.dtype)
    xrs = xrs.rio.write_nodata(nv)
    return xrs


def write_raster(xrs, path, no_data_value, **rio_gdal_kwargs):
    ext = _get_extension(path)
    rio_is_bool = False
    if ext in TIFF_EXTS or len(ext) == 0:
        if xrs.dtype == I64:
            # GDAL, and thus rioxarray and rasterio, doesn't support I64 so
            # cast up to float. This avoids to_raster throwing a TypeError.
            xrs = xrs.astype(F64)
        elif is_bool(xrs.dtype):
            # GDAL doesn't support boolean dtype either so convert to uint8
            # 0-1 encoding.
            rio_is_bool = True
            xrs = xrs.astype(U8)

    if not ext or ext not in WRITE_NOT_IMPLEMENTED_EXTS:
        kwargs = {"lock": True, "compute": True, **rio_gdal_kwargs}
        if "blockheight" in kwargs:
            value = kwargs.pop("blockheight")
            kwargs["blockysize"] = value
        if "blockwidth" in kwargs:
            value = kwargs.pop("blockwidth")
            kwargs["blockxsize"] = value
        if rio_is_bool:
            # Store each entry using a single bit
            kwargs["nbits"] = 1
        xrs.rio.to_raster(path, **kwargs)
    else:
        # TODO: populate
        raise NotImplementedError()


def _get_valid_variables(meta, ignore_too_many_dims):
    data_vars = list(meta.data_vars)
    valid = []
    for v in data_vars:
        n = meta[v].squeeze().ndim
        if n > 3:
            if ignore_too_many_dims:
                continue
            else:
                raise DimensionsError(
                    f"Too many dimensions for variable {v!r} with "
                    f"{meta[v].ndim}."
                )
        elif n in (2, 3):
            valid.append(v)
        else:
            raise DimensionsError(
                f"Too few dimensions for variable {v!r} with {n}."
            )
    if not valid:
        raise ValueError("No valid raster variables found")
    return valid


def _build_xr_raster(path, variable, affine, crs, xarray_kwargs):
    if affine is None:
        affine = Affine(1, 0, 0, 0, -1, 0, 0)
    kwargs = xarray_kwargs.copy()
    kwargs["chunks"] = "auto"
    var = xr.open_dataset(path, **kwargs)[variable].squeeze()
    var_data = var.data
    if var.ndim == 2:
        var_data = np.expand_dims(var_data, axis=0)
    var_data = var_data.rechunk((1, "auto", "auto"))
    band = np.array(list(range(var_data.shape[0])))
    x = var[var.rio.x_dim].to_numpy()
    y = var[var.rio.y_dim].to_numpy()
    new_var = xr.DataArray(
        var_data, dims=["band", "y", "x"], coords=(band, y, x)
    )
    new_var = new_var.rio.write_transform(affine)
    if crs is not None:
        new_var = new_var.rio.write_crs(crs)
    nv = var._FillValue if "_FillValue" in var.attrs else var.rio.nodata
    if nv is not None:
        new_var = new_var.rio.write_nodata(nv)
    return new_var


def _get_affine(ds):
    try:
        affine = ds.rio.transform()
    except TypeError as err:
        # Some datasets like gridMET improperly encode the transform.
        raise AffineEncodingError(
            "Error reading GeoTransform data:"
            f"{ds.coords[ds.rio.grid_mapping].attrs['GeoTransform']!r}"
        ) from err
    return affine


def open_dataset(
    path,
    crs=None,
    ignore_extra_dim_errors=False,
    xarray_kwargs=None,
):
    """Open a netCDF or GRIB dataset.

    This function opens a netCDF or GRIB dataset file and returns a dictionary
    of Raster objectds where each raster corrersponds to the variables in the
    the file. netCDF/GRIB files can be N-dimensional, while rasters only
    comprehend 2 to 3 dimensions (band, y, x), so it may not be possible to map
    all variables in a file to a raster. See the `ignore_extra_dim_errors`
    option below for more information.

    Parameters
    ----------
    path : str
        THe path to the netCDF or GRIB dataset file.
    crs : str, rasterio.crs.CRS, optional
        A coordinate reference system definition to attach to the dataset. This
        can be an EPSG, PROJ, or WKT string. It can also be a
        `rasterio.crs.CRS` object. netCDF/GRIB files do not always encode a
        CRS. This option allows a CRS to be supplied, if known ahead of time.
        It can also be used to override the CRS encoded in the file.
    ignore_extra_dim_errors : bool, optional
        If ``True``, ignore dataset variables that cannot be mapped to a
        raster. An error is raised, otherwise. netCDF/GRIB files allow
        N-dimensional. Rasters only comprehend 2 or 3 dimensional data so it is
        not always possible to map a variable to a raster. The default is
        ``False``.
    xarray_kwargs : dict, optional
        Keyword arguments to supply to `xarray.open_dataset` when opening the
        file.

    Raises
    ------
    raster_tools.io.AffineEncodingError
        Raised if the affine matrix is improperly encoded.
    ra

    Returns
    -------
    dataset : dict of Raster
        A ``dict`` of Raster objects. The keys are the variable names in the
        dataset file and the values are the corresponding variable data as a
        raster.

    """
    from raster_tools.raster import Raster

    if xarray_kwargs is None:
        xarray_kwargs = {}
    xarray_kwargs["decode_coords"] = "all"
    tmp_ds = xr.open_dataset(path, **xarray_kwargs)
    data_vars = _get_valid_variables(tmp_ds, ignore_extra_dim_errors)
    crs = crs or tmp_ds.rio.crs
    affine = _get_affine(tmp_ds)
    tmp_ds = None
    ds = {}
    for v in data_vars:
        var = _build_xr_raster(path, v, affine, crs, xarray_kwargs)
        ds[v] = Raster(var)
    return ds
