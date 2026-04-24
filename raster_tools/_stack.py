import numbers

import dask.array as da
import numpy as np
import rasterio as rio
from odc.geo.geobox import GeoBox

import raster_tools as rts
from raster_tools._grids import are_all_grids_same, combine_grids
from raster_tools.utils import null_values_equal
from raster_tools.warp import SUPPORTED_RESAMPLE_METHODS

__all__ = ["stack_bands"]


def _cast_if_needed(raster, dtype, nodata):
    if raster.dtype == dtype and null_values_equal(raster.null_value, nodata):
        return raster
    return raster.astype(dtype, new_null_value=nodata)


def stack_bands(
    rasters,
    dst_crs=None,
    dst_grid=None,
    join="inner",
    resampling_method="nearest",
    resolution=None,
    dtype=None,
    null_value=None,
):
    """Stack input rasters into a multi-band raster.

    All input bands are concatenated along the band axis onto a single
    destination grid. Inputs with differing grids, resolutions, or CRSs are
    reprojected onto the destination grid before stacking. The output has
    ``sum(r.nbands for r in rasters)`` bands.

    Parameters
    ----------
    rasters : list of raster_tools.Raster
        A list-like object containing the rasters to be stacked. These can
        have differing grids, resolutions, and projections.
    dst_crs : CRS-like, str, int, optional
        The destination CRS to use when building the destination grid. This
        can be anything that can be parsed by
        :py:meth:`rasterio.CRS.from_user_input`. Only consulted when
        `dst_grid` is not provided. The default is to take the CRS from the
        first raster in `rasters`.
    dst_grid : odc.geo.GeoBox, raster_tools.Raster, str, optional
        The definition for the destination grid to stack the `rasters` onto.
        This can be a :py:class:`odc.geo.GeoBox`,
        :py:class:`raster_tools.Raster` object, or a path str. If the input
        is a raster object or path, this function does NOT write to the given
        raster, it instead uses the raster as a reference for the grid. The
        default is to build a grid from the inputs according to `join`, using
        the resolution from the first raster in `rasters`. Passing `dst_grid`
        together with `dst_crs` or `resolution` is an error.
    join : str, optional
        How to combine the input grids when building the destination grid.
        Only consulted when `dst_grid` is not provided. Valid options are:

        'inner'
            Use the intersection of the input grid bounds. Only pixels
            covered by every input appear in the output. This is the default.
        'outer'
            Use the union of the input grid bounds. Pixels not covered by a
            given input are filled with that band's null value.
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
    resolution : scalar, optional
        Pixel resolution of the output grid, in units of the output CRS. Only
        consulted when `dst_grid` is not provided. The default is to take the
        resolution from the first raster in `rasters`.
    dtype : numpy.dtype, str, optional
        The dtype for the output raster. The default is to use
        :py:func:`numpy.result_type` on the dtypes from the input rasters.
    null_value : scalar, optional
        The nodata/null value to use for the output raster. The default is to
        take the null value from the first input raster that has one, falling
        back to a default based on the output dtype.

    Returns
    -------
    Raster
        The resulting multi-band raster.

    Notes
    -----
    When `dst_grid` is not provided, the output CRS and resolution default to
    those of the first raster in `rasters`. Reordering the input list can
    therefore change the output grid; pass `dst_crs`, `resolution`, or
    `dst_grid` explicitly when that dependency is undesirable.

    """
    if len(rasters) == 0:
        raise ValueError("No rasters provided")
    if join not in ("inner", "outer"):
        raise ValueError("join must be one of 'inner' or 'outer'")
    resampling_method = resampling_method or "nearest"
    if resampling_method not in SUPPORTED_RESAMPLE_METHODS:
        raise ValueError("Invalid resampling method")
    nodata = null_value
    if nodata is not None and not isinstance(nodata, numbers.Number):
        raise TypeError("null_value must be a scalar or None")

    if dst_grid is not None:
        if isinstance(dst_grid, (str, rts.Raster)):
            dst_grid = rts.get_raster(dst_grid).geobox
        elif not isinstance(dst_grid, GeoBox):
            raise TypeError(
                f"Expected dst_grid to have type GeoBox. Got {type(dst_grid)}"
            )
        if dst_crs is not None:
            parsed = rio.CRS.from_user_input(dst_crs)
            if dst_grid.crs != parsed:
                raise ValueError(
                    "dst_crs does not match dst_grid.crs: "
                    f"{parsed} vs {dst_grid.crs}"
                )
        if resolution is not None:
            raise ValueError(
                "resolution cannot be specified together with dst_grid"
            )

    src_rasters = [rts.get_raster(r) for r in rasters]
    src_grids = [r.geobox for r in src_rasters]
    src_grids_same = are_all_grids_same(src_grids)

    dtype = (
        np.dtype(dtype)
        if dtype is not None
        else np.result_type(*[r.dtype for r in src_rasters])
    )
    if nodata is None:
        nodatas = [
            src.null_value for src in src_rasters if src.null_value is not None
        ]
        if nodatas:
            nodata = nodatas[0]
        else:
            nodata = rts.masking.get_default_null_value(dtype)

    # Single-raster fast path: no output constraints means we can skip grid
    # building, reprojection, and concatenation entirely.
    if (
        len(src_rasters) == 1
        and dst_grid is None
        and dst_crs is None
        and resolution is None
    ):
        return _cast_if_needed(src_rasters[0], dtype, nodata)

    if dst_grid is None:
        dst_crs = (
            dst_crs if dst_crs is None else rio.CRS.from_user_input(dst_crs)
        )
        src_res_matches = resolution is None or np.isclose(
            abs(src_grids[0].resolution.x), resolution
        )
        if (
            src_grids_same
            and (dst_crs is None or src_grids[0].crs == dst_crs)
            and src_res_matches
        ):
            dst_grid = src_grids[0]
        else:
            how = "union" if join == "outer" else "intersection"
            dst_grid = combine_grids(
                src_grids,
                how=how,
                dst_crs=dst_crs,
                resolution=resolution,
            )

    # Fast path: all inputs already share the destination grid, so the
    # per-raster reproject check (and potential reproject) is unnecessary.
    if src_grids_same and are_all_grids_same([src_grids[0], dst_grid]):
        src_rasters_in_dst = [
            _cast_if_needed(r, dtype, nodata) for r in src_rasters
        ]
    else:
        src_rasters_in_dst = [
            _cast_if_needed(
                r
                if are_all_grids_same([r, dst_grid])
                else r.reproject(dst_grid, resample_method=resampling_method),
                dtype,
                nodata,
            )
            for r in src_rasters
        ]
    # Rechunk all of the reprojected inputs so they are chunk aligned. This
    # greatly boosts the performance of da.concatenate.
    tmp = []
    target_chunks_2d = da.empty(list(dst_grid.shape), dtype=dtype).chunks
    for sr in src_rasters_in_dst:
        target_chunks = ((1,) * sr.nbands, *target_chunks_2d)
        tmp.append(sr.chunk(target_chunks))
    src_rasters_in_dst = tmp

    data = da.concatenate([r.data for r in src_rasters_in_dst], axis=0)
    mask = da.concatenate([r.mask for r in src_rasters_in_dst], axis=0)
    y, x = [c.values for c in dst_grid.coordinates.values()]
    return rts.data_to_raster(
        data, mask=mask, x=x, y=y, crs=dst_grid.crs, nv=nodata
    )
