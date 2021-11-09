import dask
import os
import numpy as np
import rasterio as rio
import rioxarray as rxr
import xarray as xr
from dask.array.core import normalize_chunks as dask_chunks
from pathlib import Path

from ._types import DEFAULT_NULL, F64, I64, U8, maybe_promote
from ._utils import (
    create_null_mask,
    is_bool,
    is_float,
    is_scalar,
    is_str,
    validate_file,
)


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


IO_UNDERSTOOD_TYPES = (str, Path)


def is_batch_file(path):
    return _get_extension(path) in BATCH_EXTS


def normalize_xarray_data(xrs):
    # Make sure that x and y are always monotonically increasing
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
    if not ext:
        raise RasterIOError("Could not determine file type")

    # XXX: comments on a few xarray issues mention better performance when
    # using the chunks keyword in open_*(). Consider combining opening and
    # chunking.
    xrs = None
    if ext in TIFF_EXTS:
        xrs = rxr.open_rasterio(path)
    else:
        raise RasterIOError("Unknown file type")
    xrs = chunk(xrs, path)

    xrs = normalize_xarray_data(xrs)

    nv = xrs.attrs.get("_FillValue", None)
    mask = None
    mask = create_null_mask(xrs, nv)
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
        nodataval = ndv
    else:
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
    xrs, path, no_data_value, blockxsize=None, blockysize=None, compress=None
):
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

    if ext in TIFF_EXTS:
        kwargs = {"lock": True, "compute": True}
        if blockxsize is not None:
            kwargs["blockxsize"] = blockxsize
        if blockysize is not None:
            kwargs["blockysize"] = blockysize
        if compress:
            if is_str(compress):
                kwargs["compress"] = compress
            elif is_bool(compress):
                kwargs["compress"] = "lzw"
            else:
                raise TypeError(
                    f"Could not understand compress argument: {compress}"
                )
        if rio_is_bool:
            # Store each entry using a single bit
            kwargs["nbits"] = 1
        xrs.rio.to_raster(path, **kwargs)
    else:
        # TODO: populate
        raise NotImplementedError()
