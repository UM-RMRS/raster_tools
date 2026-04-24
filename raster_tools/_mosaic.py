import numbers

import dask.array as da
import numba as nb
import numpy as np
import rasterio as rio
from odc.geo.geobox import GeoBox

import raster_tools as rts
from raster_tools._grids import (
    _build_empty_raster_from_grid,
    are_all_grids_same,
    combine_grids,
)

__all__ = ["mosaic"]


@nb.jit(nopython=True, nogil=True)
def _nb_push(x, missing_value):
    out = x
    isnan = np.isnan(missing_value)
    for i in range(1, out.shape[0]):
        for j in range(out.shape[1]):
            for k in range(out.shape[2]):
                is_missing = (
                    np.isnan(out[i, j, k])
                    if isnan
                    else out[i, j, k] == missing_value
                )
                if is_missing:
                    out[i, j, k] = out[i - 1, j, k]
    return out


def _push(x, missing_value, axis=None, dtype=None):
    # dask's blelloch forces dtype in the interface so accept and ignore it.
    # Do copy to avoid read-only array issues in _nb_push
    return _nb_push(x.copy(), missing_value)


def _push_binop_last(lhs, rhs, missing_value):
    cond = np.isnan(rhs) if np.isnan(missing_value) else rhs == missing_value
    return np.where(cond, lhs, rhs)


def _push_take_last(
    x, axis=None, dtype=None, keepdims=None, missing_value=None, **kwargs
):
    return _push(x, missing_value)[-1:]


def _dask_push(x, missing_value):
    def push_func(x, axis=None, dtype=None, **kwargs):
        return _push(x, missing_value=missing_value, axis=axis)

    def push_binop(a, b):
        return _push_binop_last(a, b, missing_value=missing_value)

    def push_preop(x, axis=None, dtype=None, keepdims=None, **kwargs):
        return _push_take_last(x, missing_value=missing_value)

    return da.reductions.cumreduction(
        func=push_func,
        binop=push_binop,
        ident=missing_value,
        x=x,
        axis=0,
        dtype=x.dtype,
        method="blelloch",
        preop=push_preop,
    )


def _dask_push_take_last(data, missing_value):
    return _dask_push(data, missing_value)[-1:]


def _paint(stacked_data, nodata, mosaic_method):
    """Paint the stacked data onto a result array.

    This does not operate in place. A new array is returned.

    Parameters
    ----------
    stacked_data : da.Array
        The stacked data as (N, H, W) dask array.
    nodata : scalar
        The nodata value fill missing cells with.
    mosaic_method : str
        The mosaic_method to use when painting `stacked_data` onto the result.
        Possible values: "first", "last", "min", "max", "sum".

    Returns
    -------
    da.Array
        The painted result as a dask array.

    """
    if mosaic_method == "first":
        painted_result = _dask_push_take_last(stacked_data, nodata)
    elif mosaic_method == "last":
        painted_result = _dask_push_take_last(stacked_data[::-1], nodata)
    elif mosaic_method in ("min", "max"):
        mask = rts.raster.get_mask_from_data(stacked_data, nodata)
        if np.issubdtype(stacked_data.dtype, np.floating):
            filled = da.where(mask, np.nan, stacked_data)
            op = da.nanmin if mosaic_method == "min" else da.nanmax
        else:
            info = np.iinfo(stacked_data.dtype)
            sentinel = info.max if mosaic_method == "min" else info.min
            filled = da.where(mask, sentinel, stacked_data)
            op = da.min if mosaic_method == "min" else da.max
        reduced = op(filled, axis=0, keepdims=True)
        all_missing = mask.all(axis=0, keepdims=True)
        painted_result = da.where(all_missing, nodata, reduced)
    elif mosaic_method == "sum":
        mask = rts.raster.get_mask_from_data(stacked_data, nodata)
        stacked_data = da.where(mask, 0, stacked_data)
        # Make sure to set the dtype. Sum will upcast otherwise
        summed_data = da.sum(
            stacked_data, axis=0, keepdims=True, dtype=stacked_data.dtype
        )
        mask = mask.all(axis=0, keepdims=True)
        painted_result = da.where(mask, nodata, summed_data)
    else:
        raise ValueError("Invalid mosaic method")
    return painted_result


def _paint_recursive(stacked_data, dst_nodata, mosaic_method):
    # Use recursion to greatly reduce dask's memory usage. 8 was found to be a
    # good cutoff.
    n = len(stacked_data)
    if n < 8:
        return _paint(stacked_data, dst_nodata, mosaic_method)

    left = _paint_recursive(stacked_data[: n // 2], dst_nodata, mosaic_method)
    right = _paint_recursive(stacked_data[n // 2 :], dst_nodata, mosaic_method)
    return _paint_recursive(
        da.concatenate([left, right], axis=0), dst_nodata, mosaic_method
    )


def _mosaic_single_band(src_rasters, dst_raster, mosaic_method):
    dst_nodata = dst_raster.null_value
    # reverse so that the first raster's data will be just before dst_data
    stacked_data = [sr.data for sr in src_rasters[::-1]]
    dst_data = da.full_like(dst_raster.data, dst_nodata)
    stacked_data.append(dst_data)
    stacked_data = da.concatenate(stacked_data, axis=0)
    return _paint_recursive(stacked_data, dst_nodata, mosaic_method)


def _mosaic(src_rasters, dst_raster, mosaic_method):
    nbands = max(src.nbands for src in src_rasters)

    # Group the src rasters by bands
    src_rasters_as_grouped_bands = [[] for i in range(nbands)]
    for src in src_rasters:
        for i in range(src.nbands):
            src_rasters_as_grouped_bands[i].append(src.get_bands(i + 1))

    mosaiced_bands_data = [
        _mosaic_single_band(
            src_rasters_as_grouped_bands[i], dst_raster, mosaic_method
        )
        for i in range(nbands)
    ]
    data = da.concatenate(mosaiced_bands_data, axis=0)
    return rts.data_to_raster_like(data, dst_raster, nv=dst_raster.null_value)


_MOSAIC_OPS = {"first", "last", "min", "max", "sum"}
_RESAMPLING_METHODS = {v.name for v in rio.warp.Resampling}


def mosaic(
    rasters,
    mosaic_method="last",
    dst_crs=None,
    dst_grid=None,
    resampling_method="nearest",
    dtype=None,
    null_value=None,
):
    """Mosaic multiple rasters into a new, single raster.

    The inputs can have multiple and differing numbers of bands. The number of
    bands in the output will be the same as the input with the largest number
    of bands.

    Parameters
    ----------
    rasters : list of raster_tools.Raster
        A list-like object containing the rasters to be mosaicked. These can
        have differing grids, resolutions, and projections.
    mosaic_method : str, optional
        The method to use when resolving overlap. Valid options are:

        'first'
            The final pixel will take its value from the first raster with a
            valid pixel at the pixel's location.
        'last'
            The final pixel will take its value from the last raster with a
            valid pixel at the pixel's location. Default.
        'min'
            The final pixel will take its value from the minimum valid value
            across all input rasters, at the given location.
        'max'
            The final pixel will take its value from the maximum valid value
            across all input rasters, at the given location.
        'sum'
            The final pixel will be the sum of all valid pixels at the given
            location. Note, this can lead to overflow issues for sufficiently
            large values and small enough dtypes.
    dst_crs : CRS-like, str, int, optional
        The destination CRS to use when building the destination grid. This can
        be anything that can be parsed by
        :py:meth:`rasterio.CRS.from_user_input`. This is only checked if
        `dst_grid` is not provided. The default is to take the CRS from the
        first raster in `rasters`.
    dst_grid : odc.geo.GeoBox, raster_tools.Raster, str, optional
        The definition for the destination grid to mosaic the `rasters` onto.
        This can be a :py:class:`odc.geo.GeoBox`,
        :py:class:`raster_tools.Raster` object, or a path str. If the input is
        a raster object or path, this function does NOT write to the given
        raster, it instead uses the raster as a reference for the grid. The
        default is to check `dst_crs` and construct a grid that encompasses all
        inputs, using the resolution from the first raster in `rasters`.
    resampling_method : str, optional
        Resampling method to use when reprojecting input rasters to the
        destination grid. The default is nearest. Valid options are:

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
    dtype : numpy.dtype, str, optional
        The dtype for the output raster. The default is to use
        :py:func:`numpy.result_type` on the dtypes from the input rasters.
    null_value : scalar, optional
        The nodata/null value to use for the output raster. The default is to
        get a default value based on the output raster's dtype.

    Returns
    -------
    Raster
        The resulting mosaicked raster.

    """
    if len(rasters) == 0:
        raise ValueError("No rasters provided")
    if mosaic_method not in _MOSAIC_OPS:
        raise ValueError("Invalid mosaic operation")

    resampling_method = resampling_method or "nearest"
    if resampling_method not in _RESAMPLING_METHODS:
        raise ValueError("Invalid resampling method")
    nodata = null_value

    src_rasters = [rts.get_raster(r) for r in rasters]

    dtype = (
        np.dtype(dtype)
        if dtype is not None
        else np.result_type(*[r.dtype for r in src_rasters])
    )
    if (nodata is not None) and (not isinstance(nodata, numbers.Number)):
        raise TypeError("nodata must be a scalar or None")
    if nodata is None:
        nodatas = [
            src.null_value for src in src_rasters if src.null_value is not None
        ]
        if nodatas:
            nodata = nodatas[0]
        else:
            nodata = rts.masking.get_default_null_value(dtype)

    if dst_grid is None:
        dst_crs = (
            dst_crs if dst_crs is None else rio.CRS.from_user_input(dst_crs)
        )
        dst_grid = combine_grids(
            [r.geobox for r in src_rasters], "union", dst_crs=dst_crs
        )
    elif isinstance(dst_grid, (str, rts.Raster)):
        dst_grid = rts.get_raster(dst_grid).geobox
    elif not isinstance(dst_grid, GeoBox):
        raise TypeError(
            f"Expected dst_grid to have type GeoBox. Got {type(dst_grid)}"
        )

    # Make sure inputs are on destination grid
    src_rasters_in_dst = [
        (
            r
            if are_all_grids_same([r, dst_grid])
            else r.reproject(dst_grid, resample_method=resampling_method)
        ).astype(dtype, new_null_value=nodata)
        for r in src_rasters
    ]
    dst_raster = _build_empty_raster_from_grid(dst_grid, dtype, nodata)
    # Rechunk all of the reprojected inputs so they are chunk aligned. This
    # greatly boosts the performance of the dask operations down the line, such
    # as da.concatenate.
    tmp = []
    target_chunks_2d = dst_raster.data.chunks[1:]
    for sr in src_rasters_in_dst:
        target_chunks = ((1,) * sr.nbands, *target_chunks_2d)
        tmp.append(sr.chunk(target_chunks))
    src_rasters_in_dst = tmp
    # TODO: check for all boolean
    return _mosaic(src_rasters_in_dst, dst_raster, mosaic_method)
