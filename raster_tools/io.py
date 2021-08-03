import os
import numpy as np
import rasterio as rio
import rioxarray as rxr
import xarray as xr
from dask.array.core import normalize_chunks as dask_chunks
from pathlib import Path

from ._types import F64
from ._utils import validate_file


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
    # mask_and_scale=True causes the null values to masked out and replaced
    # with NaN. This standardizes the rasters. When they are compute()'d or
    # written to disk, the NaNs are replaced with the null value again.
    if ext in TIFF_EXTS:
        rs = rxr.open_rasterio(path, mask_and_scale=True, dtype=F64)
        # XXX: comments on a few xarray issues mention better performance when
        # using the chunks keyword in open_*(). Consider combining opening and
        # chunking.
        rs = chunk(rs, path)
        return rs
    elif ext in NC_EXTS:
        # TODO: chunking logic
        return xr.open_dataset(
            path,
            decode_coords="all",
            mask_and_scale=True,
            dtype=F64,
        )
    else:
        raise RasterIOError("Unknown file type")


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
    rs, path, no_data_value=None, blockwidth=None, blockheight=None
):
    ext = _get_extension(path)
    if ext in TIFF_EXTS:
        # TODO: figure out method for multi-band tiffs that respects dask
        # lazy eval/loading
        nbands = 1
        if len(rs.shape) == 3:
            nbands = rs.shape[0]
        if nbands == 1:
            rs.rio.to_raster(path, compute=True)
        else:
            _write_tif_with_rasterio(
                rs,
                path,
                no_data_value=no_data_value,
                blockwidth=blockwidth,
                blockheight=blockheight,
            )
    elif ext in NC_EXTS:
        rs.to_netcdf(path, compute=True)
    else:
        # TODO: populate
        raise NotImplementedError()
