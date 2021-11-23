import os
from pathlib import Path

import dask
import numpy as np
import rasterio as rio
import rioxarray as rxr
import xarray as xr
from dask.array.core import normalize_chunks as dask_chunks

from ._types import F64, I64, U8
from ._utils import create_null_mask, is_bool, validate_file


class RasterIOError(BaseException):
    pass


class RasterDataError(BaseException):
    pass


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
        return dask_chunks(
            chunks, shape, dtype=dtype, previous_chunks=tile_shape
        )
    else:
        shape = data.shape
        dtype = data.dtype
        tile_shape = None
        if dask.is_dask_collection(data):
            tile_shape = data.chunks
        elif src_file is not None:
            _, tile_shape, _ = _get_chunking_info_from_file(src_file)
        return dask_chunks(
            chunks, shape, dtype=dtype, previous_chunks=tile_shape
        )


def chunk(xrs, src_file=None):
    return xrs.chunk(_get_chunks(xrs, src_file))


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


def normalize_xarray_data(xrs):
    # Make sure that x and y are always increasing. xarray will auto align
    # rasters but when a raster is converted to a numpy or dask array, the
    # data may not be aligned. This ensures that rasters converted to
    # non-georeferenecd formats will be oriented the same.
    xdiff = np.diff(xrs.x)
    if len(xdiff) and (xdiff < 0).all():
        xrs = xrs.reindex(x=xrs.x[::-1])
    ydiff = np.diff(xrs.y)
    if len(ydiff) and (ydiff < 0).all():
        xrs = xrs.reindex(y=xrs.y[::-1])
    tf = xrs.rio.transform(True)
    xrs = xrs.rio.write_transform(tf)
    return xrs


def open_raster_from_path(path):
    if type(path) in IO_UNDERSTOOD_TYPES:
        path = str(path)
        path = os.path.abspath(path)
    else:
        raise RasterIOError(
            f"Could not resolve input to a raster path: '{path}'"
        )
    validate_file(path)
    ext = _get_extension(path)

    xrs = None
    # Try to let gdal open anything but NC, HDF, GRIB files
    if not ext or ext not in READ_NOT_IMPLEMENTED_EXTS:
        try:
            xrs = rxr.open_rasterio(path, chunks=_get_chunks())
        except rio.errors.RasterioIOError as e:
            raise RasterIOError(str(e))
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

    nv = xrs.attrs.get("_FillValue", None)
    mask = None
    mask = create_null_mask(xrs, nv)
    return xrs, mask, nv


def write_raster(xrs, path, no_data_value, **rio_gdal_kwargs):
    ext = _get_extension(path)
    rio_is_bool = False
    if ext in TIFF_EXTS:
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
