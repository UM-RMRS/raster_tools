import dask
import numpy as np
import rioxarray as rxr
import xarray as xr

from raster_tools.creation import ones_like, zeros_like
from raster_tools.masking import get_default_null_value
from raster_tools.raster import Raster, RasterNoDataError, get_raster
from raster_tools.utils import make_raster_ds
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

    feat_rs = feat.to_raster_mask(rs)
    if not envelope:
        if invert:
            clip_mask = feat_rs.to_null_mask()
        else:
            clip_mask = ~feat_rs.to_null_mask()
    else:
        if invert:
            clip_mask = zeros_like(feat_rs, dtype=bool)
        else:
            clip_mask = ones_like(feat_rs, dtype=bool)

    if rs._masked:
        nv = rs.null_value
    else:
        nv = get_default_null_value(rs.dtype)

    xdata_out = xr.where(clip_mask.xdata, rs.xdata, nv)
    xmask_out = ~clip_mask.xdata

    if rs._masked:
        xmask_out |= rs.xmask
        xdata_out = xdata_out.rio.write_nodata(nv)
    ds_out = make_raster_ds(xdata_out, xmask_out)
    if rs.crs is not None:
        ds_out = ds_out.rio.write_crs(rs.crs)
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
        xrs = rs.xdata.rio.clip_box(*bounds, auto_expand=True)
    except rxr.exceptions.NoDataInBounds:
        raise RasterNoDataError("No data found within provided bounds")
    if rs._masked:
        xmask = rs.xmask.rio.clip_box(*bounds, auto_expand=True)
    else:
        xmask = xr.zeros_like(xrs, dtype=bool)
    ds = make_raster_ds(xrs, xmask)
    if rs.crs is not None:
        ds = ds.rio.write_crs(rs.crs)
    return Raster(ds, _fast_path=True)
