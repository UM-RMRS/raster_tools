import rasterio as rio
from odc.geo.geobox import GeoBox

from raster_tools.masking import get_default_null_value
from raster_tools.raster import Raster, dataarray_to_xr_raster_ds, get_raster

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
        `resolution` can also be specified to change the output raster's
        resolution in the new CRS. If `crs_or_geobox` is not provided,
        `resolution` must be specified.
    resample_method : str, optional
        The data resampling method to use. Null pixels are ignored for all
        methods. Some methods require specific versions of GDAL. These are
        noted below. Valid methods are:

        'nearest'
            Nearest neighbor resampling. This is the default.
        'bilinear'
            Bilinear resmapling.
        'cubic'
            Cubic resmapling.
        'cubic_spline'
            Cubic spline resmapling.
        'lanczos'
            Lanczos windowed sinc resmapling.
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
        be provided. Both can also be provided.

    Returns
    -------
    Raster
        The reprojected raster on the new grid.

    """
    raster = get_raster(raster)
    if resample_method not in SUPPORTED_RESAMPLE_METHODS:
        raise ValueError(
            f"Unsupported resampling method provided: {resample_method!r}. "
            "Supported methods: {}".format(
                ", ".join(sorted(SUPPORTED_RESAMPLE_METHODS))
            )
        )
    if crs_or_geobox is None:
        if resolution is None:
            raise ValueError("Must supply either crs_or_geobox or resolution")
        dst_gb = raster.geobox
    elif not isinstance(crs_or_geobox, GeoBox):
        dst_gb = raster.geobox.to_crs(crs_or_geobox)
    else:
        dst_gb = crs_or_geobox
    if resolution is not None:
        if resolution <= 0:
            raise ValueError("Resolution must be a postive value")
        dst_gb = dst_gb.zoom_to(resolution=resolution)
    if dst_gb == raster.geobox:
        return raster.copy()

    nv = (
        raster.null_value
        if raster._masked
        else get_default_null_value(raster.dtype)
    )
    reprojected = raster.xdata.odc.reproject(
        dst_gb, resampling=resample_method, dst_nodata=nv
    ).rio.write_nodata(nv)
    if "longitude" in reprojected.dims:
        # odc-geo will rename x/y to lon/lat for lon/lat based projections, so
        # revert to x/y
        reprojected = reprojected.rename({"longitude": "x", "latitude": "y"})
    ds = dataarray_to_xr_raster_ds(reprojected)
    return Raster(ds, _fast_path=True)
