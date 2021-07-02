import os
import numpy as np
import rasterio as rio
import rioxarray  # noqa: F401; adds ability to save tiffs to xarray
import xarray as xr
from pathlib import Path

from .batch import BatchScript
from ._utils import validate_file


class RasterIOError(BaseException):
    pass


def _is_str(value):
    return isinstance(value, str)


def _get_extension(path):
    return os.path.splitext(path)[-1].lower()


def chunk(xrs):
    # TODO: smarter chunking logic
    return xrs.chunk({"band": 1, "x": 10_000, "y": 10_000})


TIFF_EXTS = frozenset((".tif", ".tiff"))
BATCH_EXTS = frozenset((".bch",))
NC_EXTS = frozenset((".nc",))

FTYPE_TO_EXT = {
    "TIFF": "tif",
}


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
    if ext in TIFF_EXTS:
        rs = xr.open_rasterio(path)
        # XXX: comments on a few xarray issues mention better performance when
        # using the chunks keyword in open_*(). Consider combining opening and
        # chunking.
        rs = chunk(rs)
        return rs
    elif ext in BATCH_EXTS:
        bs = BatchScript(path)
        return bs.parse().final_raster._rs
    elif ext in NC_EXTS:
        # TODO: chunking logic
        return xr.open_dataset(path)
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
        nodatavals = set(rs.nodatavals)
        if rs.dtype.kind in ("u", "i") and np.isnan(list(nodatavals)).any():
            nodatavals.remove(np.nan)
        if len(nodatavals):
            # TODO: add warning if size > 1
            nodataval = nodatavals.pop()
        else:
            nodataval = None
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
        crs=rs.crs,
        transform=rs.transform,
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
