import numpy as np
import rasterio as rio
from odc.geo.geobox import GeoBox

from raster_tools._grids import reproject_grid
from raster_tools.dtypes import (
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    get_dtype_min_max,
    is_bool,
    is_float,
)
from raster_tools.masking import get_default_null_value
from raster_tools.raster import (
    Raster,
    dataarray_to_xr_raster_ds,
    get_mask_from_data,
    get_raster,
)

__all__ = [
    "reproject",
]

SUPPORTED_RESAMPLE_METHODS = {m.name for m in rio.warp.SUPPORTED_RESAMPLING}


def reproject(
    raster, crs_or_geobox=None, resample_method="nearest", resolution=None
):
    """Reproject to a new projection or resolution.

    This is a lazy operation.

    Parameters
    ----------
    raster : str, Raster
        The raster to reproject.
    crs_or_geobox : int, str, CRS, GeoBox, optional
        The target grid to reproject the raster to. This can be a projection
        string, EPSG code string or integer, a CRS object, or a GeoBox object.
        When a CRS is given, the output grid is the smallest one that covers
        the raster's footprint in the new CRS. Its origin is the top-left
        corner of the footprint's bounding box, not snapped to a multiple of
        the cell size. `resolution` can also be specified to change the output
        raster's resolution in the new CRS. If `crs_or_geobox` is not
        provided, `resolution` must be specified.
    resample_method : str, optional
        The data resampling method to use. Null pixels are ignored for all
        methods. Some methods require specific versions of GDAL. These are
        noted below. Valid methods are:

        'nearest'
            Nearest neighbor resampling. This is the default.
        'bilinear'
            Bilinear resampling.
        'cubic'
            Cubic resampling.
        'cubic_spline'
            Cubic spline resampling.
        'lanczos'
            Lanczos windowed sinc resampling.
        'average'
            Average resampling, computes the weighted average of all
            contributing pixels.
        'mode'
            Mode resampling, selects the value which appears most often.
        'max'
            Maximum resampling. (GDAL 2.0+)
        'min'
            Minimum resampling. (GDAL 2.0+)
        'med'
            Median resampling. (GDAL 2.0+)
        'q1'
            Q1, first quartile resampling. (GDAL 2.0+)
        'q3'
            Q3, third quartile resampling. (GDAL 2.0+)
        'sum'
            Sum, compute the weighted sum. (GDAL 3.1+)
        'rms'
            RMS, root mean square/quadratic mean. (GDAL 3.3+)
    resolution : int, float, tuple of int or float, optional
        The desired resolution of the reprojected raster. If `crs_or_geobox` is
        unspecified, this is used to reproject to the new resolution while
        maintaining the same CRS. One of `crs_or_geobox` or `resolution` must
        be provided. Both can also be provided. Changing the resolution keeps
        the grid's origin.

    Returns
    -------
    Raster
        The reprojected raster on the new grid. Cells that the source does not
        cover are null. A raster with no null value is given the default null
        value for its dtype. Its valid cells stay valid, even where they hold
        that value, except in a 64-bit integer raster, where those cells are
        null or, with newer GDAL, have their value nudged off it.

    """
    raster = get_raster(raster)
    if resample_method not in SUPPORTED_RESAMPLE_METHODS:
        raise ValueError(
            f"Unsupported resampling method provided: {resample_method!r}. "
            "Supported methods: {}".format(
                ", ".join(sorted(SUPPORTED_RESAMPLE_METHODS))
            )
        )
    if resolution is not None and resolution <= 0:
        raise ValueError("Resolution must be a positive value")
    if crs_or_geobox is None:
        if resolution is None:
            raise ValueError("Must supply either crs_or_geobox or resolution")
        dst_gb = raster.geobox.zoom_to(resolution=resolution)
    elif isinstance(crs_or_geobox, GeoBox):
        dst_gb = crs_or_geobox
        if resolution is not None:
            dst_gb = dst_gb.zoom_to(resolution=resolution)
    else:
        dst_gb = reproject_grid(
            raster.geobox, crs_or_geobox, resolution=resolution
        )
    if dst_gb == raster.geobox:
        return raster.copy()

    nv = (
        raster.null_value
        if raster._masked
        else get_default_null_value(raster.dtype)
    )
    working = _get_working_type(raster.dtype, raster._masked)
    if working is not None:
        reprojected, xmask = _reproject_with_mask(
            raster, dst_gb, resample_method, nv, *working
        )
    else:
        reprojected = raster.xdata.odc.reproject(
            dst_gb, resampling=resample_method, dst_nodata=nv
        )
        xmask = None
    reprojected = reprojected.rio.write_nodata(nv)
    # reproject sets a "nodata" attribute that has type int for whole-number
    # null values (e.g. -3.4028235e+38 becomes a VERY long python int). This
    # can cause all kinds of issues for downstream operations
    # since .rio will get confused. We are already handling null values so it
    # can be dropped.
    reprojected.attrs.pop("nodata", None)
    if "longitude" in reprojected.dims:
        # odc-geo will rename x/y to lon/lat for lon/lat based projections, so
        # revert to x/y
        lonlat_to_xy = {"longitude": "x", "latitude": "y"}
        reprojected = reprojected.rename(lonlat_to_xy)
        if xmask is not None:
            xmask = xmask.rename(lonlat_to_xy)
    ds = dataarray_to_xr_raster_ds(reprojected, xmask=xmask)
    return Raster(ds, _fast_path=True)


# Unmasked integer data is warped in the next wider signed type, whose
# minimum no source value can equal
_WIDER_INT_TYPE = {U8: I16, I8: I16, U16: I32, I16: I32, U32: I64, I32: I64}


def _get_working_type(dtype, masked):
    """Get the type and null value to warp data in, if one is needed.

    Warping marks the cells that the source does not cover, and null source
    cells, with a null value. That value must not occur in the source's
    valid data, or valid cells that hold it come out null or altered. The
    default null value given to an unmasked raster can occur in its data, and
    bool data has no spare value at all. For those, this returns a working
    dtype and null value that cannot collide, so that the mask can be taken
    from the warped data. ``None`` is returned when the raster's own null
    value can be used, or when its dtype has no wider type (64-bit
    integers).
    """
    dtype = np.dtype(dtype)
    if is_bool(dtype):
        # A sum resample can count past 255 cells, so use a type wider than
        # uint8
        return I16, I16.type(np.iinfo(I16).min)
    if masked:
        return None
    if is_float(dtype):
        # Source NaN values are null when warping in any case
        return dtype, dtype.type(np.nan)
    if dtype in _WIDER_INT_TYPE:
        wide = _WIDER_INT_TYPE[dtype]
        return wide, wide.type(np.iinfo(wide).min)
    return None


def _reproject_with_mask(
    raster, dst_gb, resample_method, nv, work_dtype, work_nv
):
    """Reproject data in a working dtype and return it with its mask.

    The data is warped as `work_dtype` with `work_nv` marking null and
    uncovered cells, and the mask is taken from that. The result is cast back
    to the raster's dtype and its null cells get `nv`.
    """
    dtype = raster.dtype
    src = raster.xdata.astype(work_dtype)
    if raster._masked:
        src = src.where(~raster.xmask, work_nv)
    # Unmasked integer data has no null cells, so it gets no source null
    # value, which also keeps GDAL on the kernels it uses for the data in its
    # own dtype. Float sources treat NaN as null, as they always have.
    src_nv = work_nv if raster._masked or is_float(dtype) else None
    warped = src.odc.reproject(
        dst_gb,
        resampling=resample_method,
        src_nodata=src_nv,
        dst_nodata=work_nv,
    )
    xmask = get_mask_from_data(warped, work_nv)
    if is_bool(dtype):
        data = warped != 0
    elif is_float(dtype):
        data = warped
    else:
        # GDAL clamps to the output type's range, so do the same for the
        # source's type
        lo, hi = get_dtype_min_max(dtype)
        data = warped.clip(lo, hi).astype(dtype)
    data = data.where(~xmask, nv)
    data.attrs = {
        k: v
        for k, v in warped.attrs.items()
        if k not in ("nodata", "_FillValue")
    }
    data.encoding = dict(warped.encoding)
    xmask.attrs = {}
    return data, xmask
