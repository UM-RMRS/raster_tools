import dask
import os
import numpy as np
import rasterio as rio
import rioxarray as rxr
import xarray as xr
from dask.array.core import normalize_chunks as dask_chunks
from pathlib import Path

from ._types import DEFAULT_NULL, F64, I64, maybe_promote
from ._utils import create_null_mask, is_float, is_scalar, validate_file


class RasterIOError(BaseException):
    pass


def _get_extension(path):
    return os.path.splitext(path)[-1].lower()


def chunk(xrs, src_file=None):
    tile_shape = None
    if src_file is not None and _get_extension(src_file) in TIFF_EXTS:
        with rio.open(src_file) as src:
            tile_shape = (1, *src.block_shapes[0])
    elif dask.is_dask_collection(xrs):
        tile_shape = xrs.chunks
    chunks = dask_chunks(
        (1, "auto", "auto"),
        xrs.shape,
        dtype=xrs.dtype,
        previous_chunks=tile_shape,
    )
    return xrs.chunk(chunks)


TIFF_EXTS = frozenset((".tif", ".tiff"))
BATCH_EXTS = frozenset((".bch",))
NC_EXTS = frozenset((".nc",))


IO_UNDERSTOOD_TYPES = (str, Path)


def is_batch_file(path):
    return _get_extension(path) in BATCH_EXTS


class Encoding:
    """Encoding metadata that is used when writing a raster to disk"""

    __slots__ = ("_masked", "_dtype", "_null")

    def __init__(self, masked=False, dtype=F64, null=DEFAULT_NULL):
        self.masked = masked
        self.dtype = dtype
        self.null_value = null

    @property
    def masked(self):
        return self._masked

    @masked.setter
    def masked(self, value):
        self._masked = bool(value)

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = np.dtype(value)

    @property
    def null_value(self):
        return self._null

    @null_value.setter
    def null_value(self, value):
        if not is_scalar(value):
            raise TypeError(f"Null value must be a scalar: {value}")
        self._null = value

    def copy(self):
        return Encoding(self._masked, self._dtype, self._null)

    def __repr__(self):
        return "Encoding {{masked: {}, dtype: {}, null_value: {}}}".format(
            self._masked, self._dtype, self._null
        )


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
    if not ext:
        raise RasterIOError("Could not determine file type")

    # XXX: comments on a few xarray issues mention better performance when
    # using the chunks keyword in open_*(). Consider combining opening and
    # chunking.
    xrs = None
    if ext in TIFF_EXTS:
        xrs = rxr.open_rasterio(path)
    elif ext in NC_EXTS:
        # TODO: this returns a dataset which is invalid. Fix nc handling
        xrs = xr.open_dataset(path, decode_coords="all")
    else:
        raise RasterIOError("Unknown file type")

    nv = xrs.attrs.get("_FillValue", None)
    xrs = chunk(xrs, path)
    mask = None
    mask = create_null_mask(xrs, nv)
    xrs.attrs["res"] = xrs.rio.resolution()
    return xrs, mask, nv


def _write_tif_with_rasterio(
    rs,
    path,
    no_data_value=None,
    compress=False,
    blockwidth=None,
    blockheight=None,
    **kwargs,
):
    # This method uses rasterio to write multi-band tiffs to disk. It does not
    # respect dask and the result raster will be loaded into memory before
    # writing to disk.
    if len(rs.shape) == 3:
        bands, rows, cols = rs.shape
    else:
        rows, cols = rs.shape
        bands = 1
    compress = None if not compress else "lzw"
    if no_data_value is None:
        ndv = (
            rs.rio.nodata
            if rs.rio.encoded_nodata is None
            else rs.rio.encodeded_nodata
        )
        rs.fillna(ndv)
        nodataval = ndv
    else:
        rs.fillna(no_data_value)
        nodataval = no_data_value
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=bands,
        dtype=rs.dtype,
        nodata=nodataval,
        crs=rs.rio.crs,
        transform=rs.rio.transform(),
        tiled=True,
        blockxsize=blockwidth,
        blockysize=blockheight,
        compress=compress,
    ) as dst:
        for band in range(bands):
            if len(rs.shape) == 3:
                values = rs[band].values
            else:
                values = rs.values
            dst.write(values, band + 1)


def write_raster(xrs, path, no_data_value, blockwidth=None, blockheight=None):
    ext = _get_extension(path)
    if (
        is_float(xrs.dtype)
        and no_data_value is not None
        and not np.isnan(no_data_value)
    ):
        xrs = xrs.fillna(no_data_value)
    if xrs.dtype == I64 and ext in TIFF_EXTS:
        # GDAL, and thus rioxarray and rasterio, doesn't support I64 so cast up
        # to float. This avoids to_raster throwing a TypeError.
        xrs = xrs.astype(F64)

    if ext in TIFF_EXTS:
        xrs.rio.to_raster(path, lock=True, compute=True)
    elif ext in NC_EXTS:
        xrs.to_netcdf(path, compute=True)
    else:
        # TODO: populate
        raise NotImplementedError()
