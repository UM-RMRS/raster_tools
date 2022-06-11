import dask
import dask.array as da
import numpy as np
import rioxarray as rxr
import xarray as xr

from raster_tools.dtypes import get_default_null_value
from raster_tools.raster import RasterNoDataError, get_raster
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
    rs = get_raster(data_raster)
    if rs.crs is None:
        raise ValueError("Data raster has no CRS")
    if bounds is not None and len(bounds) != 4:
        raise ValueError("Invalid bounds. Must be a size 4 array or tuple.")
    if envelope and invert:
        raise ValueError("Envelope and invert cannot both be true")

    feat = feat.to_crs(rs.crs)

    if trim:
        if bounds is None:
            (bounds,) = dask.compute(feat.bounds)
        else:
            bounds = np.atleast_1d(bounds)
        try:
            rs = clip_box(rs, bounds)
        except RasterNoDataError:
            raise RuntimeError(
                "No data in given bounds. Make sure that the bounds are in the"
                " same CRS as the data raster."
            )

    feat_rs = feat.to_raster(rs)
    if not envelope:
        clip_mask = feat_rs > 0
    else:
        clip_mask = feat_rs >= 0
        clip_mask._mask[:] = False

    if rs._masked:
        nv = rs.null_value
    else:
        nv = get_default_null_value(rs.dtype)

    if invert:
        clip_mask = ~clip_mask
        clip_mask._mask = ~clip_mask._mask
    xrs_out = xr.where(clip_mask.xrs, rs.xrs, nv)
    xrs_out = xrs_out.rio.write_crs(rs.crs)
    mask_out = clip_mask._mask

    if rs._masked:
        mask_out |= rs._mask
    return rs._replace(xrs_out, mask=mask_out, null_value=nv)


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
    rs = get_raster(raster)
    if len(bounds) != 4:
        raise ValueError("Invalid bounds. Must be a size 4 array or tuple.")
    try:
        xrs = rs.xrs.rio.clip_box(*bounds, auto_expand=True)
    except rxr.exceptions.NoDataInBounds:
        raise RasterNoDataError("No data found within provided bounds")
    if rs._masked:
        xmask = xr.DataArray(rs._mask, dims=rs.xrs.dims, coords=rs.xrs.coords)
        mask = xmask.rio.clip_box(*bounds, auto_expand=True).data
    else:
        mask = da.zeros_like(xrs.data, dtype=bool)
    # TODO: This will throw a rioxarray.exceptions.MissingCRS exception if
    # no crs is set. Add code to fall back on
    return rs._replace(xrs, mask=mask)
