import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox

from raster_tools.dtypes import is_bool, is_scalar
from raster_tools.masking import get_default_null_value
from raster_tools.raster import (
    Raster,
    dataarray_to_xr_raster_ds,
    get_raster,
)
from raster_tools.utils import nan_equal

__all__ = ["pad"]

# rioxarray.pad_box requires a CRS. When the source raster has none, we
# attach a placeholder CRS just for the pad call and strip it afterward.
_DUMMY_CRS = "EPSG:4326"


def _strip_crs(arr):
    return arr.drop_vars("spatial_ref", errors="ignore")


def _is_numeric_scalar(value):
    return is_scalar(value) or is_bool(value)


def _equals_null(value, null_value):
    if null_value is None:
        return False
    return nan_equal(value, null_value)


def _resolve_bounds(raster, target):
    if isinstance(target, GeoBox):
        if (
            raster.crs is not None
            and target.crs is not None
            and target.crs != raster.crs
        ):
            raise ValueError(
                "GeoBox target CRS does not match the raster's CRS"
            )
        bb = target.boundingbox
        return (
            float(bb.left),
            float(bb.bottom),
            float(bb.right),
            float(bb.top),
        )

    if isinstance(target, (Raster, str)):
        other = get_raster(target)
        if (
            raster.crs is not None
            and other.crs is not None
            and other.crs != raster.crs
        ):
            raise ValueError(
                "Raster target CRS does not match the raster's CRS"
            )
        return tuple(float(b) for b in other.bounds)

    try:
        bounds = list(target)
    except TypeError as err:
        raise TypeError(
            "target must be a (minx, miny, maxx, maxy) sequence, Raster, "
            "or GeoBox"
        ) from err
    if len(bounds) != 4 or not all(_is_numeric_scalar(b) for b in bounds):
        raise ValueError(
            "Invalid bounds. Must be a length-4 sequence of numbers."
        )
    return tuple(float(b) for b in bounds)


def _validate_fill_values(raster, fill_values):
    nbands = raster.nbands
    null_value = raster.null_value
    dtype = raster.dtype

    if fill_values is None:
        out_nv = (
            null_value
            if null_value is not None
            else get_default_null_value(dtype)
        )
        return [out_nv] * nbands, [True] * nbands, out_nv

    if _is_numeric_scalar(fill_values):
        per_band = [fill_values] * nbands
        mark = [_equals_null(fill_values, null_value)] * nbands
        return per_band, mark, null_value

    if isinstance(fill_values, (list, tuple)):
        if len(fill_values) != nbands:
            raise ValueError(
                f"fill_values length ({len(fill_values)}) does not match "
                f"the number of bands ({nbands})"
            )
        if not all(_is_numeric_scalar(v) for v in fill_values):
            raise TypeError(
                "fill_values list entries must all be numeric scalars"
            )
        per_band = list(fill_values)
        mark = [_equals_null(v, null_value) for v in per_band]
        return per_band, mark, null_value

    raise TypeError(
        "fill_values must be None, a scalar, or a list/tuple of scalars"
    )


def pad(raster, target, *, fill_values=None):
    """Pad a raster's grid out to cover the given extent.

    The source raster's grid (resolution and origin) is preserved; only
    new cells are added on the edges. If the requested extent is already
    inside the raster, the result is unchanged in shape.

    Parameters
    ----------
    raster : str, Raster
        The raster to pad.
    target : tuple, Raster, or odc.geo.geobox.GeoBox
        The extent to pad to. Accepts:

        - a ``(minx, miny, maxx, maxy)`` tuple in the raster's CRS
          (pixel-edge semantics, matching ``Raster.bounds``)
        - a ``Raster`` -- its ``.bounds`` is used; its resolution and
          origin are ignored
        - a ``GeoBox`` -- its bounding box is used
    fill_values : None, scalar, or list of scalars, optional
        Value(s) used to fill newly added cells.

        - ``None`` (default): use the raster's null value, and mark the
          new cells as null. If the raster has no null value, a
          dtype-appropriate sentinel is chosen and assigned to the
          result.
        - scalar: fill all new cells across all bands with this value.
          If the value equals the raster's null value, behave as
          ``None`` for that case.
        - list/tuple of length ``nbands``: per-band fill. Per-band
          values that equal the raster's null value cause those bands'
          new cells to be marked null.

    Returns
    -------
    Raster
        A new lazy Raster covering at least the requested extent.

    """
    raster = get_raster(raster)
    bounds = _resolve_bounds(raster, target)
    per_band, mark_null, out_nv = _validate_fill_values(raster, fill_values)

    src_crs = raster.crs
    pad_crs = src_crs if src_crs is not None else _DUMMY_CRS

    src_xdata = raster.xdata
    if src_crs is None:
        src_xdata = src_xdata.rio.write_crs(pad_crs)
    xdata_padded = src_xdata.rio.pad_box(*bounds, constant_values=0)

    # is_new: True at newly added cells. Built by padding a False
    # array (same shape as input) with constant_values=True so the
    # original region stays False and the padded region is True.
    zeros = xr.zeros_like(raster.xdata, dtype=bool)
    zeros = zeros.rio.write_nodata(None).rio.write_crs(pad_crs)
    is_new = zeros.rio.pad_box(*bounds, constant_values=True)

    fill_arr = np.asarray(per_band).astype(xdata_padded.dtype)
    fill_da = xr.DataArray(
        fill_arr, dims=("band",), coords={"band": xdata_padded.band}
    )
    xdata = xr.where(is_new, fill_da, xdata_padded)
    if src_crs is not None:
        xdata = xdata.rio.write_crs(src_crs)
    else:
        xdata = _strip_crs(xdata)
    if out_nv is not None:
        xdata = xdata.rio.write_nodata(out_nv)

    if raster._masked:
        src_xmask = raster.xmask
        if src_crs is None:
            src_xmask = src_xmask.rio.write_crs(pad_crs)
        xmask = src_xmask.rio.pad_box(*bounds, constant_values=False)
    else:
        xmask = xr.zeros_like(xdata_padded, dtype=bool)
    mark_da = xr.DataArray(
        np.asarray(mark_null, dtype=bool),
        dims=("band",),
        coords={"band": xdata_padded.band},
    )
    xmask = xmask | (is_new & mark_da)
    if src_crs is None:
        xmask = _strip_crs(xmask)

    ds = dataarray_to_xr_raster_ds(xdata, xmask=xmask, crs=src_crs)
    return Raster(ds, _fast_path=True)
