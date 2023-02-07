import dask.array as da
import numba as nb
import numpy as np
import xarray as xr

from raster_tools.dtypes import F32, F64
from raster_tools.masking import get_default_null_value
from raster_tools.raster import Raster, get_raster
from raster_tools.utils import single_band_mappable

__all__ = [
    "pa_allocation",
    "pa_direction",
    "pa_proximity",
    "proximity_analysis",
]

_JIT_KWARGS = {"nopython": True, "nogil": True}
ngjit = nb.jit(**_JIT_KWARGS)


@nb.jit(
    [
        "float64(float64, float64, float64, float64)",
        "int64(int64, int64, int64, int64)",
    ],
    **_JIT_KWARGS,
)
def _euclidean_dist_sqr(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return (dx * dx) + (dy * dy)


@nb.jit(
    [
        "float64(float64, float64, float64, float64)",
        "int64(int64, int64, int64, int64)",
    ],
    **_JIT_KWARGS,
)
def _taxi_cab_dist_sqr(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    dist = np.abs(dx) + np.abs(dy)
    return dist * dist


@nb.jit(
    [
        "float64(float64, float64, float64, float64)",
        "float64(int64, int64, int64, int64)",
    ],
    **_JIT_KWARGS,
)
def _chebyshev_dist_sqr(x1, y1, x2, y2):
    dx = np.abs(x1 - x2)
    dy = np.abs(y1 - y2)
    dist = max(dx, dy)
    return dist * dist


@nb.jit(
    [
        "float64(float64, float64, float64, float64)",
        "float64(int64, int64, int64, int64)",
    ],
    **_JIT_KWARGS,
)
def _haversine_dist_sqr(x1, y1, x2, y2):
    # Great Circle or Haversine distance in meters (squared)
    # ref: https://en.wikipedia.org/wiki/Great-circle_distance
    # ref: https://en.wikipedia.org/wiki/Haversine_formula
    x1 = np.radians(x1)
    y1 = np.radians(y1)
    x2 = np.radians(x2)
    y2 = np.radians(y2)
    sinlat = np.sin((y2 - y1) / 2.0)
    sinlon = np.sin((x2 - x1) / 2.0)
    h = (sinlat * sinlat) + (np.cos(y1) * np.cos(y2) * sinlon * sinlon)
    # Mean Earth radius for WGS84
    r = 6371009.0
    hav = 2 * r * np.arcsin(np.sqrt(h))
    return hav * hav


_DISTANCE_FUNC_MAP = {
    # Euclidean (default)
    None: _euclidean_dist_sqr,
    "euclidean": _euclidean_dist_sqr,
    # Taxi
    "taxi": _taxi_cab_dist_sqr,
    "taxicab": _taxi_cab_dist_sqr,
    "manhatten": _taxi_cab_dist_sqr,
    "city_block": _taxi_cab_dist_sqr,
    # Chebyshev
    "chebyshev": _chebyshev_dist_sqr,
    "chessboard": _chebyshev_dist_sqr,
    # Great circle
    "haversine": _haversine_dist_sqr,
    "great_circle": _haversine_dist_sqr,
}


@nb.jit(
    [
        "float64(float64, float64, float64, float64)",
        "float64(int64, int64, int64, int64)",
    ],
    **_JIT_KWARGS,
)
def _direction(x1, y1, x2, y2):
    # (x1, y1): current cell
    # (x2, y2): nearest target cell
    # From ArcMap Euclidean Direction docs: The output values are based on
    # compass directions (90 to the east, 180 to the south, 270 to the west,
    # and 360 to the north), with 0 reserved for the source cells
    #            y,N,360
    #            ^
    #            |---.
    #            |    \
    #            |     `
    #            |     |
    # W,270 <----+-----v-> x,E,90
    #            |
    #            |
    #            v
    #            S,180
    if x1 == x2 and y1 == y2:
        return 0.0

    dx = x2 - x1
    dy = y2 - y1
    # -dy to flip so theta increases clockwise
    th = np.arctan2(-dy, dx) * 57.29577951308232
    # map to 0-360 range
    # +90 to shift so north is along y axis
    th = (th + 90.0 + 360) % 360
    if th == 0.0:
        th = 360.0
    return th


_MODE_PROXIMITY = 1
_MODE_ALLOCATION = 2
_MODE_DIRECTION = 3


@ngjit
def _process_proximity_line(
    scan_line,
    line_proximity,
    pan_near_xi,
    pan_near_yi,
    pan_nearest_xi,
    pan_nearest_yi,
    xc,
    yc,
    forward,
    iline,
    max_distance,
    targets,
    dist_sqr_func,
    mode,
):
    # Processes a single line in the direction determined by `forward` and
    # updates the proximities for the line. pan_near_{x, y} are buffers that
    # keep track of the closest points seen so far on this pass up or down the
    # raster.
    nx = len(scan_line)
    nt = len(targets)
    start = 0 if forward else nx - 1
    end = nx if forward else -1
    step = 1 if forward else -1

    max_dist_sqr = max_distance * max_distance
    max_dist_sqr_2 = max_dist_sqr * 2.0

    for ipixel in range(start, end, step):
        is_target = False
        if nt > 0:
            for j in range(nt):
                is_target |= scan_line[j] == targets[j]
        else:
            # No targets so assume all finite and non-zero pixels are targets
            is_target = (scan_line[ipixel] != 0) & np.isfinite(
                scan_line[ipixel]
            )
        # Are we on a target?
        if is_target:
            line_proximity[ipixel] = 0
            pan_near_xi[ipixel] = ipixel
            pan_near_yi[ipixel] = iline
            if mode != _MODE_PROXIMITY:
                pan_nearest_xi[ipixel] = ipixel
                pan_nearest_yi[ipixel] = iline
            continue

        # Are we near(er) to the closest target to the above/below pixel?
        near_dist_sqr = max_dist_sqr_2
        if pan_near_xi[ipixel] >= 0:
            dist_sqr = dist_sqr_func(
                xc[pan_near_xi[ipixel]],
                yc[pan_near_yi[ipixel]],
                xc[ipixel],
                yc[iline],
            )
            if dist_sqr < near_dist_sqr:
                near_dist_sqr = dist_sqr
            else:
                pan_near_xi[ipixel] = -1
                pan_near_yi[ipixel] = -1
        # Are we near(er) to the closest target in the backward diagonal
        # direction?
        iprev = ipixel - step
        if ipixel != start and pan_near_xi[iprev] >= 0:
            dist_sqr = dist_sqr_func(
                xc[pan_near_xi[iprev]],
                yc[pan_near_yi[iprev]],
                xc[ipixel],
                yc[iline],
            )
            if dist_sqr < near_dist_sqr:
                near_dist_sqr = dist_sqr
                pan_near_xi[ipixel] = pan_near_xi[iprev]
                pan_near_yi[ipixel] = pan_near_yi[iprev]
        # Are we near(er) to the closest target in the forward diagonal
        # direction?
        inext = ipixel + step
        if inext != end and pan_near_xi[inext] >= 0:
            dist_sqr = dist_sqr_func(
                xc[pan_near_xi[inext]],
                yc[pan_near_yi[inext]],
                xc[ipixel],
                yc[iline],
            )
            if dist_sqr < near_dist_sqr:
                near_dist_sqr = dist_sqr
                pan_near_xi[ipixel] = pan_near_xi[inext]
                pan_near_yi[ipixel] = pan_near_yi[inext]
        # Update proximity
        if (
            pan_near_xi[ipixel] >= 0
            and near_dist_sqr <= max_dist_sqr
            and (
                line_proximity[ipixel] < 0
                or near_dist_sqr < line_proximity[ipixel] ** 2
            )
        ):
            line_proximity[ipixel] = np.sqrt(near_dist_sqr)
            if mode != _MODE_PROXIMITY:
                pan_nearest_xi[ipixel] = pan_near_xi[ipixel]
                pan_nearest_yi[ipixel] = pan_near_yi[ipixel]


@ngjit
def _compute_proximity(
    src,
    proximity,
    secondary,
    target_values,
    xc,
    yc,
    max_distance,
    dist_sqr_func,
    nodata,
    mode,
):
    # This func works by making passes forward and backward (row axis) through
    # the raster, iteretively updating the proximities to targets. It is based
    # on GDAL's proximity implementation.
    #
    # ref: https://github.com/OSGeo/gdal/blob/master/alg/gdalproximity.cpp
    ny, nx = src.shape

    # Buffer for rows from raster
    scan_line = np.empty(nx, dtype=src.dtype)
    # Buffer for proximity values along rows
    line_proximity = np.empty(nx, dtype=proximity.dtype)
    # Arrays to keep track of nearest targets seen so far. Pan means global
    # (raster-chunk-wide), here
    pan_near_xi = np.empty(nx, dtype=np.int64)
    pan_near_yi = np.empty(nx, dtype=np.int64)
    # Only allocate full arrays if needed for secondary calculation
    if mode == _MODE_PROXIMITY:
        pan_nearest_xi = np.empty(0, dtype=np.int64)
        pan_nearest_yi = np.empty(0, dtype=np.int64)
    else:
        pan_nearest_xi = np.empty(nx, dtype=np.int64)
        pan_nearest_yi = np.empty(nx, dtype=np.int64)

    # Top to bottom
    pan_near_xi[:] = -1
    pan_near_yi[:] = -1
    for iline in range(ny):
        scan_line[:] = src[iline]
        line_proximity[:] = -1

        # Left to right
        pan_nearest_xi[:] = -1
        pan_nearest_yi[:] = -1
        _process_proximity_line(
            scan_line,
            line_proximity,
            pan_near_xi,
            pan_near_yi,
            pan_nearest_xi,
            pan_nearest_yi,
            xc,
            yc,
            True,
            iline,
            max_distance,
            target_values,
            dist_sqr_func,
            mode,
        )
        if mode != _MODE_PROXIMITY:
            for i in range(nx):
                if pan_nearest_xi[i] >= 0 and line_proximity[i] >= 0:
                    if mode == _MODE_ALLOCATION:
                        secondary[iline, i] = src[
                            pan_nearest_yi[i], pan_nearest_xi[i]
                        ]
                    else:
                        secondary[iline, i] = _direction(
                            xc[i],
                            yc[iline],
                            xc[pan_nearest_xi[i]],
                            yc[pan_nearest_yi[i]],
                        )
        # Right to left
        pan_nearest_xi[:] = -1
        pan_nearest_yi[:] = -1
        _process_proximity_line(
            scan_line,
            line_proximity,
            pan_near_xi,
            pan_near_yi,
            pan_nearest_xi,
            pan_nearest_yi,
            xc,
            yc,
            False,
            iline,
            max_distance,
            target_values,
            dist_sqr_func,
            mode,
        )
        proximity[iline] = line_proximity
        if mode != _MODE_PROXIMITY:
            for i in range(nx):
                if pan_nearest_xi[i] >= 0 and line_proximity[i] >= 0:
                    if mode == _MODE_ALLOCATION:
                        secondary[iline, i] = src[
                            pan_nearest_yi[i], pan_nearest_xi[i]
                        ]
                    else:
                        secondary[iline, i] = _direction(
                            xc[i],
                            yc[iline],
                            xc[pan_nearest_xi[i]],
                            yc[pan_nearest_yi[i]],
                        )

    # Bottom to top
    pan_near_xi[:] = -1
    pan_near_yi[:] = -1
    for iline in range(ny - 1, -1, -1):
        scan_line[:] = src[iline]
        line_proximity[:] = proximity[iline]

        # Right to left
        pan_nearest_xi[:] = -1
        pan_nearest_yi[:] = -1
        _process_proximity_line(
            scan_line,
            line_proximity,
            pan_near_xi,
            pan_near_yi,
            pan_nearest_xi,
            pan_nearest_yi,
            xc,
            yc,
            False,
            iline,
            max_distance,
            target_values,
            dist_sqr_func,
            mode,
        )
        if mode != _MODE_PROXIMITY:
            for i in range(nx):
                if pan_nearest_xi[i] >= 0 and line_proximity[i] >= 0:
                    if mode == _MODE_ALLOCATION:
                        secondary[iline, i] = src[
                            pan_nearest_yi[i], pan_nearest_xi[i]
                        ]
                    else:
                        secondary[iline, i] = _direction(
                            xc[i],
                            yc[iline],
                            xc[pan_nearest_xi[i]],
                            yc[pan_nearest_yi[i]],
                        )
        # Left to right
        pan_nearest_xi[:] = -1
        pan_nearest_yi[:] = -1
        _process_proximity_line(
            scan_line,
            line_proximity,
            pan_near_xi,
            pan_near_yi,
            pan_nearest_xi,
            pan_nearest_yi,
            xc,
            yc,
            True,
            iline,
            max_distance,
            target_values,
            dist_sqr_func,
            mode,
        )
        if mode != _MODE_PROXIMITY:
            for i in range(nx):
                if pan_nearest_xi[i] >= 0 and line_proximity[i] >= 0:
                    if mode == _MODE_ALLOCATION:
                        secondary[iline, i] = src[
                            pan_nearest_yi[i], pan_nearest_xi[i]
                        ]
                    else:
                        secondary[iline, i] = _direction(
                            xc[i],
                            yc[iline],
                            xc[pan_nearest_xi[i]],
                            yc[pan_nearest_yi[i]],
                        )
        # Fill skipped regions with nodata value
        for i in range(nx):
            if line_proximity[i] < 0:
                line_proximity[i] = nodata
        proximity[iline] = line_proximity


def _get_coords_for_chunk(xc, yc, coords_block_info, block_info):
    # Trim off band dim if present
    chunk_loc = block_info[0]["chunk-location"][-2:]
    ychunks, xchunks = coords_block_info["chunks"]
    ydepth, xdepth = coords_block_info["depth"]
    xc = da.from_array(xc, chunks=xchunks)
    yc = da.from_array(yc, chunks=ychunks)
    xc = da.overlap.overlap(xc, depth=xdepth, boundary=0)
    yc = da.overlap.overlap(yc, depth=ydepth, boundary=0)
    xc = xc.blocks[chunk_loc[1]].compute()
    yc = yc.blocks[chunk_loc[0]].compute()
    return xc, yc


@single_band_mappable(pass_block_info=True)
def _proximity_analysis_chunk(
    src,
    target_values,
    xc,
    yc,
    coords_block_info,
    max_distance,
    dist_sqr_func,
    full_precision,
    alloc_dtype,
    nodata,
    mode,
    block_info=None,
):
    assert mode in {_MODE_PROXIMITY, _MODE_ALLOCATION, _MODE_DIRECTION}

    # Output arrays
    out_dtype = F64 if full_precision else F32
    prox_dst = np.empty(src.shape, dtype=out_dtype)
    # Only fully allocate array if needed for allocation calculation
    if mode == _MODE_PROXIMITY:
        secondary_dst = np.full((1, 1), nodata, dtype=src.dtype)
    else:
        out_dtype = alloc_dtype if mode == _MODE_ALLOCATION else out_dtype
        secondary_dst = np.full(src.shape, nodata, dtype=out_dtype)

    xc, yc = _get_coords_for_chunk(xc, yc, coords_block_info, block_info)

    _compute_proximity(
        src,
        prox_dst,
        secondary_dst,
        target_values,
        xc,
        yc,
        max_distance,
        dist_sqr_func,
        nodata,
        mode,
    )

    if mode == _MODE_PROXIMITY:
        return prox_dst
    else:
        return secondary_dst


def _validate_lonlat_coords(lon, lat):
    if ((lat < -90) | (lat > 90)).any():
        raise ValueError(
            "Invalid longitude values for great circle distance calculation."
            " Values must be in [-90, 90]."
        )
    if np.any(lon > 360) or np.any(lon < -180):
        raise ValueError(
            "Invalid latitude values for great circle distance calculation."
            " Values must be in [-180, 180] or [0, 360]."
        )
    if np.any(lon > 180) and np.any(lon < 0):
        raise ValueError(
            "Invalid latitude values for great circle distance calculation."
            " Values must be in [-180, 180] or [0, 360]."
        )


def _estimate_min_resolution(lon, lat):
    x1 = lon[0]
    y1 = lat.max()
    x2 = lon[1]
    y2 = y1
    xmin = np.sqrt(_haversine_dist_sqr(x1, y1, x2, y2))
    x1 = lon[0]
    x2 = lon[0]
    # Get two largest latitudes
    y1, y2 = np.partition(lat, -2)[-2:]
    ymin = np.sqrt(_haversine_dist_sqr(x1, y1, x2, y2))
    return xmin, ymin


def _proximity_analysis(
    raster,
    mode,
    target_values,
    distance_metric,
    max_distance,
    double_precision,
):
    raster = get_raster(raster)
    if distance_metric not in _DISTANCE_FUNC_MAP:
        raise ValueError(f"Invalid distance metric: {distance_metric!r}")
    dist_sqr_func = _DISTANCE_FUNC_MAP[distance_metric]
    if target_values is None:
        target_values = []
    target_values = np.atleast_1d(target_values)
    if target_values.ndim > 1:
        raise ValueError("target_values must be 1D")
    if not np.isfinite(target_values).all():
        raise ValueError("target_values must be finite")
    if max_distance <= 0:
        raise ValueError(
            f"max_distance must be greater than 0: {max_distance!r}"
        )

    x = raster.x
    y = raster.y
    if distance_metric in ("haversine", "great_circle"):
        _validate_lonlat_coords(x, y)
    if max_distance >= np.sqrt(dist_sqr_func(x[0], y[0], x[-1], y[-1])):
        # If max distance encompasses the entire raster, rechunk to entire
        # raster.
        _, ny, nx = raster.shape
        raster = Raster(
            raster._ds.chunk({"band": 1, "y": ny, "x": nx}), _fast_path=True
        )
        xdepth = 0
        ydepth = 0
    else:
        if distance_metric not in ("haversine", "great_circle"):
            resolution = raster.resolution
        else:
            # When x/y are lon/lat, the resolution in meters decreases with
            # increasing latitude. Estimate the minimum resolution (cell size
            # at highest latitude) and use that for chunk overlap calculation.
            # Use minimum so that we always overestimate and never
            # underestimate the depth.
            resolution = _estimate_min_resolution(x, y)
        xdepth, ydepth = np.ceil(max_distance / np.abs(resolution)).astype(int)
    coords_block_info = {
        "chunks": raster.data.chunks[1:],
        "depth": (ydepth, xdepth),
    }
    if mode in (_MODE_PROXIMITY, _MODE_DIRECTION):
        out_dtype = F64 if double_precision else F32
    else:
        out_dtype = raster.dtype
    nodata = get_default_null_value(out_dtype)
    orig_raster_dtype = raster.dtype
    # If masked, convert to float and replace null values with nan
    raster = get_raster(raster, null_to_nan=True)

    out_data = da.map_overlap(
        _proximity_analysis_chunk,
        raster.data,
        depth=(0, ydepth, xdepth),
        boundary=np.nan,
        meta=np.array((), dtype=out_dtype),
        # func args
        target_values=target_values,
        xc=x,
        yc=y,
        coords_block_info=coords_block_info,
        max_distance=max_distance,
        dist_sqr_func=dist_sqr_func,
        full_precision=double_precision,
        alloc_dtype=orig_raster_dtype if mode == _MODE_ALLOCATION else None,
        nodata=nodata,
        mode=mode,
    )
    xout = xr.DataArray(
        out_data,
        coords=raster.xdata.coords,
        dims=raster.xdata.dims,
        attrs=raster.xdata.attrs,
    ).rio.write_nodata(nodata)
    rs_out = Raster(xout)
    return rs_out


def pa_proximity(
    raster,
    target_values=None,
    max_distance=np.inf,
    distance_metric="euclidean",
    double_precision=False,
):
    """Compute the proximity of each cell to the closest source cell.

    This function takes a source raster and uses it to compute a proximity
    raster. The proximity raster's cells contain the distances, as determined
    by the `distance_metric` input, to the nearest source cell.
    `distance_metric` selects the function to use for calculating distances
    between cells.

    This is very similar to ESRI's Euclidean Distance tool and GDAL's proximity
    function.

    Parameters
    ----------
    raster : Raster, path str
        The input source raster. If `target_values` is not specified or empty,
        any non-null, finite, non-zero cell is considered a source. If
        `target_values` is specified, it will be used to find source cells with
        the given values.
    target_values : array-like, optional
        A 1D sequence of values to use to find source cells in `raster`. If not
        specified, all non-null, finite, non-zero cells are considered sources.
    max_distance : scalar, optional
        The maximum distance (**in georeferenced units**) to consider when
        computing proximities. Distances greater than this value will not be
        computed and the corresponding cells will be skipped. This value also
        determines how much of `raster` is loaded into memory when carrying out
        the computation. A larger value produces a larger memory footprint. The
        default is infinity which causes the entire raster to be loaded at
        computation time. No cells will be skipped in this case.
    distance_metric : str
        The name of the distance metric to use for computing distances. Default
        is `'euclidean'`.

        'euclidean'
            The euclidean distance between two points: ``sqrt((x1 - x2)^2 + (y1
            - y2)^2)``
        'taxi' 'taxicab' 'manhatten' 'city_block'
            The taxicab distance between two points: ``|x1 - x2| + |y1 - y2|``
        'chebyshev' 'chessboard'
            The chebyshev or chessboard distance: ``max(|x1 - x2|, |y1 - y2|)``
        'haversine' 'great_circle'
            The great circle distance between points on a sphere. This
            implementation uses the WGS84 mean earth radius of 6,371,009 m.
            Coordinates are assumed to be in degrees. Latitude must be in the
            range [-90, 90] and longitude must be in the range [-180, 180].
    double_precision : bool
        If ``True``, distances will will be computed with 64-bit precision,
        otherwise 32-bit floats are used. Default is ``False``.

    Returns
    -------
    Raster
        The proximity raster.

    References
    ----------
    * `ESRI: Proximity Analysis <https://desktop.arcgis.com/en/arcmap/latest/analyze/commonly-used-tools/proximity-analysis.htm>`_
    * `ESRI: Euclidean Distance <https://desktop.arcgis.com/en/arcmap/latest/tools/spatial-analyst-toolbox/euclidean-distance.htm>`_
    * `GDAL: Proximity <https://gdal.org/api/gdal_alg.html#_CPPv420GDALComputeProximity15GDALRasterBandH15GDALRasterBandHPPc16GDALProgressFuncPv>`_
    * `Taxicab Distance <https://en.wikipedia.org/wiki/Taxicab_geometry>`_
    * `Chebyshev Distance <https://en.wikipedia.org/wiki/Chebyshev_distance>`_
    * `Great-Circle Distance <https://en.wikipedia.org/wiki/Great-circle_distance>`_

    See also
    --------
    pa_allocation, pa_direction, proximity_analysis

    """  # noqa: E501
    return _proximity_analysis(
        raster,
        mode=_MODE_PROXIMITY,
        target_values=target_values,
        distance_metric=distance_metric,
        max_distance=max_distance,
        double_precision=double_precision,
    )


def pa_allocation(
    raster,
    target_values=None,
    distance_metric="euclidean",
    max_distance=np.inf,
    double_precision=False,
):
    """
    Compute the proximity allocation of each cell to the closest source cell.

    Allocation assigns each raster cell the value of the nearest source cell.
    This function takes a source raster and uses it to compute a proximity
    allocation raster. For large `max_distance` values, this forms the raster
    analogue of a Voronoi diagram. `distance_metric` selects the function to
    use for calculating distances between cells.

    This is very similar to ESRI's Euclidean Allocation tool.

    Parameters
    ----------
    raster : Raster, path str
        The input source raster. If `target_values` is not specified or empty,
        any non-null, finite, non-zero cell is considered a source. If
        `target_values` is specified, it will be used to find source cells with
        the given values.
    target_values : array-like, optional
        A 1D sequence of values to use to find source cells in `raster`. If not
        specified, all non-null, finite, non-zero cells are considered sources.
    max_distance : scalar, optional
        The maximum distance (**in georeferenced units**) to consider when
        computing proximities. Distances greater than this value will not be
        computed and the corresponding cells will be skipped. This value also
        determines how much of `raster` is loaded into memory when carrying out
        the computation. A larger value produces a larger memory footprint. The
        default is infinity which causes the entire raster to be loaded at
        computation time. No cells will be skipped in this case.
    distance_metric : str
        The name of the distance metric to use for computing distances. Default
        is `'euclidean'`.

        'euclidean'
            The euclidean distance between two points: ``sqrt((x1 - x2)^2 + (y1
            - y2)^2)``
        'taxi' 'taxicab' 'manhatten' 'city_block'
            The taxicab distance between two points: ``|x1 - x2| + |y1 - y2|``
        'chebyshev' 'chessboard'
            The chebyshev or chessboard distance: ``max(|x1 - x2|, |y1 - y2|)``
        'haversine' 'great_circle'
            The great circle distance between points on a sphere. This
            implementation uses the WGS84 mean earth radius of 6,371,009 m.
            Coordinates are assumed to be in degrees. Latitude must be in the
            range [-90, 90] and longitude must be in the range [-180, 180].
    double_precision : bool
        If ``True``, distances will will be computed with 64-bit precision,
        otherwise 32-bit floats are used. Default is ``False``.

    Returns
    -------
    Raster
        The allocation raster.

    References
    ----------
    * `ESRI: Proximity Analysis <https://desktop.arcgis.com/en/arcmap/latest/analyze/commonly-used-tools/proximity-analysis.htm>`_
    * `ESRI: Euclidean Allocation <https://desktop.arcgis.com/en/arcmap/latest/tools/spatial-analyst-toolbox/euclidean-allocation.htm>`_
    * `Voronoi Diagram <https://en.wikipedia.org/wiki/Voronoi_diagram>`_
    * `Taxicab Distance <https://en.wikipedia.org/wiki/Taxicab_geometry>`_
    * `Chebyshev Distance <https://en.wikipedia.org/wiki/Chebyshev_distance>`_
    * `Great-Circle Distance <https://en.wikipedia.org/wiki/Great-circle_distance>`_

    See also
    --------
    pa_proximity, pa_direction, proximity_analysis

    """  # noqa: E501
    return _proximity_analysis(
        raster,
        mode=_MODE_ALLOCATION,
        target_values=target_values,
        distance_metric=distance_metric,
        max_distance=max_distance,
        double_precision=double_precision,
    )


def pa_direction(
    raster,
    target_values=None,
    distance_metric="euclidean",
    max_distance=np.inf,
    double_precision=False,
):
    """
    Compute the direction of each cell to the closest source cell.

    This function takes a source raster and uses it to compute a direction
    raster. Each cell contains the direction in degrees to the nearest source
    cell as determined by the distance metric. `distance_metric` selects the
    function to use for calculating distances between cells. Directions are in
    degrees on the range [0, 360]. 0 indicates that a source cell. 360 is
    North, 90 East, 180 South, and 270 indicates West. They indicate the
    direction **to, not from**, the nearest source cell.

    This is very similar to ESRI's Euclidean Direction tool.

    Parameters
    ----------
    raster : Raster, path str
        The input source raster. If `target_values` is not specified or empty,
        any non-null, finite, non-zero cell is considered a source. If
        `target_values` is specified, it will be used to find source cells with
        the given values.
    target_values : array-like, optional
        A 1D sequence of values to use to find source cells in `raster`. If not
        specified, all non-null, finite, non-zero cells are considered sources.
    max_distance : scalar, optional
        The maximum distance (**in georeferenced units**) to consider when
        computing proximities. Distances greater than this value will not be
        computed and the corresponding cells will be skipped. This value also
        determines how much of `raster` is loaded into memory when carrying out
        the computation. A larger value produces a larger memory footprint. The
        default is infinity which causes the entire raster to be loaded at
        computation time. No cells will be skipped in this case.
    distance_metric : str
        The name of the distance metric to use for computing distances. Default
        is `'euclidean'`.

        'euclidean'
            The euclidean distance between two points: ``sqrt((x1 - x2)^2 + (y1
            - y2)^2)``
        'taxi' 'taxicab' 'manhatten' 'city_block'
            The taxicab distance between two points: ``|x1 - x2| + |y1 - y2|``
        'chebyshev' 'chessboard'
            The chebyshev or chessboard distance: ``max(|x1 - x2|, |y1 - y2|)``
        'haversine' 'great_circle'
            The great circle distance between points on a sphere. This
            implementation uses the WGS84 mean earth radius of 6,371,009 m.
            Coordinates are assumed to be in degrees. Latitude must be in the
            range [-90, 90] and longitude must be in the range [-180, 180].
    double_precision : bool
        If ``True``, distances will will be computed with 64-bit precision,
        otherwise 32-bit floats are used. Default is ``False``.

    Returns
    -------
    Raster
        The direction raster.

    References
    ----------
    * `ESRI: Proximity Analysis <https://desktop.arcgis.com/en/arcmap/latest/analyze/commonly-used-tools/proximity-analysis.htm>`_
    * `ESRI: Euclidean Direction <https://desktop.arcgis.com/en/arcmap/latest/tools/spatial-analyst-toolbox/euclidean-direction.htm>`_
    * `Taxicab Distance <https://en.wikipedia.org/wiki/Taxicab_geometry>`_
    * `Chebyshev Distance <https://en.wikipedia.org/wiki/Chebyshev_distance>`_
    * `Great-Circle Distance <https://en.wikipedia.org/wiki/Great-circle_distance>`_

    See also
    --------
    pa_proximity, pa_allocation, proximity_analysis

    """  # noqa: E501
    return _proximity_analysis(
        raster,
        mode=_MODE_DIRECTION,
        target_values=target_values,
        distance_metric=distance_metric,
        max_distance=max_distance,
        double_precision=double_precision,
    )


def proximity_analysis(
    raster,
    target_values=None,
    distance_metric="euclidean",
    max_distance=np.inf,
    double_precision=False,
):
    """
    Compute the proximity, allocation, and direction for a given source raster.

    This function uses proximity analysis to compute the proximity, allocation,
    and direction rasters for a given source raster. See the individual
    functions for more details:

    * :func:`pa_proximity`
    * :func:`pa_allocation`
    * :func:`pa_direction`

    This is very similar to ESRI's Proximity Analysis tools.

    Parameters
    ----------
    raster : Raster, path str
        The input source raster. If `target_values` is not specified or empty,
        any non-null, finite, non-zero cell is considered a source. If
        `target_values` is specified, it will be used to find source cells with
        the given values.
    target_values : array-like, optional
        A 1D sequence of values to use to find source cells in `raster`. If not
        specified, all non-null, finite, non-zero cells are considered sources.
    max_distance : scalar, optional
        The maximum distance (**in georeferenced units**) to consider when
        computing proximities. Distances greater than this value will not be
        computed and the corresponding cells will be skipped. This value also
        determines how much of `raster` is loaded into memory when carrying out
        the computation. A larger value produces a larger memory footprint. The
        default is infinity which causes the entire raster to be loaded at
        computation time. No cells will be skipped in this case.
    distance_metric : str
        The name of the distance metric to use for computing distances. Default
        is `'euclidean'`.

        'euclidean'
            The euclidean distance between two points: ``sqrt((x1 - x2)^2 + (y1
            - y2)^2)``
        'taxi' 'taxicab' 'manhatten' 'city_block'
            The taxicab distance between two points: ``|x1 - x2| + |y1 - y2|``
        'chebyshev' 'chessboard'
            The chebyshev or chessboard distance: ``max(|x1 - x2|, |y1 - y2|)``
        'haversine' 'great_circle'
            The great circle distance between points on a sphere. This
            implementation uses the WGS84 mean earth radius of 6,371,009 m.
            Coordinates are assumed to be in degrees. Latitude must be in the
            range [-90, 90] and longitude must be in the range [-180, 180].
    double_precision : bool
        If ``True``, distances will will be computed with 64-bit precision,
        otherwise 32-bit floats are used. Default is ``False``.

    Returns
    -------
    Raster
        The direction raster.

    References
    ----------
    * `ESRI: Proximity Analysis <https://desktop.arcgis.com/en/arcmap/latest/analyze/commonly-used-tools/proximity-analysis.htm>`_
    * `ESRI: Euclidean Distance <https://desktop.arcgis.com/en/arcmap/latest/tools/spatial-analyst-toolbox/euclidean-distance.htm>`_
    * `ESRI: Euclidean Allocation <https://desktop.arcgis.com/en/arcmap/latest/tools/spatial-analyst-toolbox/euclidean-allocation.htm>`_
    * `ESRI: Euclidean Direction <https://desktop.arcgis.com/en/arcmap/latest/tools/spatial-analyst-toolbox/euclidean-direction.htm>`_
    * `Taxicab Distance <https://en.wikipedia.org/wiki/Taxicab_geometry>`_
    * `Chebyshev Distance <https://en.wikipedia.org/wiki/Chebyshev_distance>`_
    * `Great-Circle Distance <https://en.wikipedia.org/wiki/Great-circle_distance>`_

    See also
    --------
    pa_proximity, pa_allocation, pa_direction

    """  # noqa: E501
    prox = pa_proximity(
        raster=raster,
        target_values=target_values,
        distance_metric=distance_metric,
        max_distance=max_distance,
        double_precision=double_precision,
    )
    alloc = pa_allocation(
        raster=raster,
        target_values=target_values,
        distance_metric=distance_metric,
        max_distance=max_distance,
        double_precision=double_precision,
    )
    direction = pa_direction(
        raster=raster,
        target_values=target_values,
        distance_metric=distance_metric,
        max_distance=max_distance,
        double_precision=double_precision,
    )
    return prox, alloc, direction
