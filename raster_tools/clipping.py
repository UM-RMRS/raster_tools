import dask
import numpy as np
import rasterio as rio
import rioxarray as rxr
import xarray as xr

from raster_tools._stack import stack_bands
from raster_tools.creation import ones_like, zeros_like
from raster_tools.exceptions import RasterNoDataError
from raster_tools.masking import get_default_null_value
from raster_tools.raster import (
    Raster,
    dataarray_to_xr_raster_ds,
    get_raster,
    grid_transform,
    xr_where_with_meta,
)
from raster_tools.vector import get_vector


def _clip(
    feature,
    data_raster,
    invert=False,
    trim=True,
    bounds=None,
    envelope=False,
):
    feat = get_vector(feature)
    data_raster = get_raster(data_raster)
    if data_raster.crs is None:
        raise ValueError("Data raster has no CRS")
    if bounds is not None and len(bounds) != 4:
        raise ValueError("Invalid bounds. Must be a size 4 array or tuple.")
    if envelope and invert:
        raise ValueError("Envelope and invert cannot both be true")

    crs = data_raster.crs
    feat = feat.to_crs(crs)
    if trim:
        if bounds is None:
            (bounds,) = dask.compute(feat.bounds)
        else:
            bounds = np.atleast_1d(bounds).ravel()
        try:
            data_raster = clip_box(data_raster, bounds)
        except RasterNoDataError as err:
            raise RuntimeError(
                "No data in given bounds. Make sure that the bounds are in the"
                " same CRS as the data raster."
            ) from err

    feature_raster = feat.to_raster(data_raster, mask=True)
    if not envelope:
        if invert:
            clip_mask = feature_raster.to_null_mask()
        else:
            clip_mask = ~feature_raster.to_null_mask()
    else:
        if invert:
            clip_mask = zeros_like(feature_raster, dtype=bool)
        else:
            clip_mask = ones_like(feature_raster, dtype=bool)

    nv = (
        data_raster.null_value
        if data_raster._masked
        else get_default_null_value(data_raster.dtype)
    )
    if data_raster.nbands > 1:
        clip_mask = stack_bands([clip_mask] * data_raster.nbands)
    xdata_out = xr_where_with_meta(
        clip_mask.xdata, data_raster.xdata, nv, crs=crs, nv=nv
    )
    xmask_out = ~clip_mask.xdata
    if data_raster._masked:
        xmask_out |= data_raster.xmask
        xdata_out = xdata_out.rio.write_nodata(nv)
    ds_out = dataarray_to_xr_raster_ds(xdata_out, xmask=xmask_out, crs=crs)
    return Raster(ds_out, _fast_path=True)


def clip(feature, data_raster, bounds=None):
    """Clip the data raster using the given feature.

    Parameters
    ----------
    feature : str, Vector
        The feature to clip with. If a string, this is interpreted as a path.
    data_raster : str, Raster
        The data raster to be clipped. If a string, this is interpreted as a
        path.
    bounds : tuple, list, array, optional
        The bounding box of the clip operation: (minx, miny, maxx, maxy). If
        not provided, the bounds are computed from the feature. This will
        trigger computation of the feature.

    Returns
    -------
    Raster
        The resulting clipped raster with dimensions that match the bounding
        box provided or the bounds of the feature.

    """
    return _clip(
        feature,
        data_raster,
        trim=True,
        bounds=bounds,
    )


def erase(feature, data_raster, bounds=None):
    """Erase the data raster using the given feature. Inverse of :func:`clip`.

    Parameters
    ----------
    feature : str, Vector
        The feature to erase with. If a string, this is interpreted as a path.
    data_raster : str, Raster
        The data raster to be erased. If a string, this is interpreted as a
        path.
    bounds : tuple, list, array, optional
        The bounding box of the clip operation: (minx, miny, maxx, maxy). If
        not provided, the bounds are computed from the feature. This will
        trigger computation of the feature.

    Returns
    -------
    Raster
        The resulting erased raster with dimensions that match the bounding
        box provided or the bounds of the feature. The data inside the feature
        is erased.

    """
    return _clip(
        feature,
        data_raster,
        trim=True,
        invert=True,
        bounds=bounds,
    )


def mask(feature, data_raster, invert=False):
    """Mask the data raster using the given feature.

    Depending on `invert`, the area inside (``True``) or outside (``False``)
    the feature is masked out. The default is to mask the area outside of the
    feature.

    Parameters
    ----------
    feature : str, Vector
        The feature to mask with. If a string, this is interpreted as a path.
    data_raster : str, Raster
        The data raster to be erased. If a string, this is interpreted as a
        path.
    invert : bool, optional
        If ``True``, the mask is inverted and the area inside of the feature is
        set to masked out. Default ``False``.

    Returns
    -------
    Raster
        The resulting masked raster with the same dimensions as the original
        data raster.

    """
    return _clip(
        feature,
        data_raster,
        trim=False,
        invert=invert,
    )


def envelope(feature, data_raster):
    """Clip the data raster using the bounds of the given feature.

    This is the same as :func:`clip` but the bounding box of the feature is
    used instead of the feature itself.

    Parameters
    ----------
    feature : str, Vector
        The feature to erase with. If a string, this is interpreted as a path.
    data_raster : str, Raster
        The data raster to be clipped. If a string, this is interpreted as a
        path.

    Returns
    -------
    Raster
        The resulting clipped raster with dimensions that match the bounding
        box of the feature.

    """
    return _clip(
        feature,
        data_raster,
        trim=True,
        envelope=True,
    )


# Number of times a clip box is grown by half a cell on each side when the
# clipped result is degenerate. Matches rioxarray's auto_expand_limit default.
_CLIP_BOX_EXPAND_LIMIT = 2


def _clip_box_xarray(xarr, bounds):
    """Clip raster xarray data to a box, growing the box if needed.

    If the clip gives no cells, or a single cell along an axis where the
    source has more, the box grows by half a cell on each side and the clip
    is retried, as rioxarray's clip_box does with auto_expand. That
    expansion is done here because rioxarray does not pass
    allow_one_dimensional_raster on to its retries, so its auto_expand
    fails on sources one row or column wide.
    """
    transform = grid_transform(xarr)
    half_x = abs(transform.a) / 2
    half_y = abs(transform.e) / 2
    minx, miny, maxx, maxy = (float(b) for b in bounds)
    for i in range(_CLIP_BOX_EXPAND_LIMIT + 1):
        last = i == _CLIP_BOX_EXPAND_LIMIT
        try:
            result = xarr.rio.clip_box(
                minx, miny, maxx, maxy, allow_one_dimensional_raster=True
            )
        except (rio.errors.WindowError, rxr.exceptions.NoDataInBounds):
            if last:
                raise
            result = None
        if result is not None and (
            last
            or (
                (result.rio.width > 1 or xarr.rio.width == 1)
                and (result.rio.height > 1 or xarr.rio.height == 1)
            )
        ):
            return result
        minx -= half_x
        miny -= half_y
        maxx += half_x
        maxy += half_y


def clip_box(raster, bounds):
    """Clip the raster to the specified box.

    Parameters
    ----------
    raster : str, Raster
        The Raster or raster path string to clip.
    bounds : tuple, list, array
        The bounding box of the clip operation: (minx, miny, maxx, maxy).

    Returns
    -------
    Raster
        The raster clipped to the given bounds.

    """
    raster = get_raster(raster)
    if len(bounds) != 4:
        raise ValueError("Invalid bounds. Must be a size 4 array or tuple.")
    try:
        xdata = _clip_box_xarray(raster.xdata, bounds)
    except rxr.exceptions.NoDataInBounds as err:
        raise RasterNoDataError(
            "No data found within provided bounds"
        ) from err
    if raster._masked:
        xmask = _clip_box_xarray(raster.xmask, bounds)
    else:
        xmask = xr.zeros_like(xdata, dtype=bool)
    ds = dataarray_to_xr_raster_ds(xdata, xmask=xmask, crs=raster.crs)
    return Raster(ds, _fast_path=True)
