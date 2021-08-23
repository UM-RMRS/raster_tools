import dask
import os
import numpy as np
import rasterio as rio
import rioxarray as rxr
import xarray as xr
from dask.array.core import normalize_chunks as dask_chunks
from pathlib import Path

from ._types import DEFAULT_NULL, F64, I64, is_float, maybe_promote
from ._utils import is_scalar, validate_file


class RasterIOError(BaseException):
    pass


def _is_str(value):
    return isinstance(value, str)


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


def create_encoding_from_xarray(xrs):
    dtype = xrs.dtype
    masked = False
    null = xrs.attrs.get("_FillValue", None)
    if is_float(dtype) or (null is not None and not np.isnan(null)):
        masked = True
    if null is None:
        null = DEFAULT_NULL
    return Encoding(masked, dtype, null)


def open_raster_from_path(path):
    if isinstance(path, Path) or _is_str(path):
        path = str(path)
        path = os.path.abspath(path)
    else:
        raise RasterIOError(f"Could not resolve input to a raster: '{path}'")
    validate_file(path)
    ext = _get_extension(path)
    if not ext:
        raise RasterIOError("Could not determine file type")

    # XXX: comments on a few xarray issues mention better performance when
    # using the chunks keyword in open_*(). Consider combining opening and
    # chunking.
    rs = None
    if ext in TIFF_EXTS:
        rs = rxr.open_rasterio(path)
    elif ext in NC_EXTS:
        # TODO: this returns a dataset which is invalid. Fix nc handling
        rs = xr.open_dataset(path, decode_coords="all")
    else:
        raise RasterIOError("Unknown file type")

    encoding = create_encoding_from_xarray(rs)
    if encoding.masked:
        new_dtype = maybe_promote(rs.dtype)
    # Chunk to start using lazy operations
    rs = chunk(rs, path)
    # Promote to a float type and replace null values with nan
    if encoding.masked:
        if rs.dtype != new_dtype:
            # Rechunk with new data size
            rs = chunk(rs.astype(new_dtype))
        null = encoding.null_value
        if not np.isnan(null):
            rs = rs.where(rs != null, np.nan)
    return rs, encoding


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


def write_raster(
    xrs, encoding, path, no_data_value=None, blockwidth=None, blockheight=None
):
    ext = _get_extension(path)
    if encoding.masked and not np.isnan(encoding.null_value):
        xrs = xrs.fillna(encoding.null_value)
    if encoding.dtype == I64 and ext in TIFF_EXTS:
        # GDAL, and thus rioxarray and rasterio, doesn't support I64 so cast up
        # to float. This avoids to_raster throwing a TypeError.
        encoding.dtype = F64
    if xrs.dtype != encoding.dtype:
        xrs = xrs.astype(encoding.dtype)

    if ext in TIFF_EXTS:
        xrs.rio.to_raster(path, lock=True, compute=True)
    elif ext in NC_EXTS:
        xrs.to_netcdf(path, compute=True)
    else:
        # TODO: populate
        raise NotImplementedError()
