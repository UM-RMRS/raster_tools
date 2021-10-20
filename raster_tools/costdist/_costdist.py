import dask.array as da
import numpy as np
import xarray as xr

from raster_tools import Raster
from raster_tools.raster import is_raster_class, _raster_like
from raster_tools._utils import is_float, is_str
from raster_tools._types import F16, I8, I64
from ._core import cost_distance_analysis_numpy, path_distance_analysis_numpy


__all__ = [
    "allocation",
    "cost_distance",
    "cost_distance_analysis",
    "path_distance_analysis",
    "traceback",
]


def _normalize_raster_data(rs, missing=-1):
    data = rs._rs.data
    # Make sure that null values are skipped
    nv = rs._null_value
    if rs._masked:
        if is_float(rs.dtype):
            if not np.isnan(nv):
                data = da.where(rs._mask, missing, data)
            elif not np.isnan(missing):
                data = da.where(np.isnan(data), missing, data)
        else:
            data = da.where(rs._mask, missing, data)
    # Trim off band dim
    data = data[0]
    return data

# NOTE: this must match the TRACEBACK_NOT_REACHED value in _core.pyx. I can't
# get that variable to import so I'm using this mirror for now.
_TRACEBACK_NOT_REACHED = -2


def cost_distance_analysis(costs, sources):
    """Calculate accumulated cost distance, traceback, and allocation.

    This function uses Dijkstra's algorithm to compute the many-sources
    shortest-paths solution for a given cost surface. Valid moves are from a
    given pixel to its 8 nearest neighbors. This produces 3 rasters. The
    first is the accumulated cost distance, which contains the
    distance-weighted accumulated minimum cost to each pixel. The cost to move
    from one pixel to the next is ``length * mean(costs[i], costs[i+1])``,
    where ``length`` is 1 for horizontal and vertical moves and ``sqrt(2)`` for
    diagonal moves. The `costs` raster's resolution informs the actual scaling
    to use. Source locations have a cost of 0.

    The second raster contains the traceback values for the solution. At each
    pixel, the stored value indicates the neighbor to move to in order to get
    closer to the cost-relative nearest source. The numbering is as follows:
    ::

        6  7  8
        5  X  1
        4  3  2

    Here, X indicates the current pixel and the numbers are the neighbor
    pixel positions. 1 indicates the neighbor immediately to the right and
    7 indicates the neighbor immediately above. In terms of rows and columns,
    these are the neighbor +1 column over and +1 row above, respectively. a
    value of 0 indicates that the current pixel is a source pixel and -1
    indicates that the pixel was not traversed due to a null value.

    The third raster contians the source allocation for each pixel. Each pixel
    is labeled based on the source location that is closest in terms of cost
    distance. The label is the value stored in the sources raster at the
    corresponding source location.

    Parameters
    ----------
    costs : Raster or raster path
        A raster representing a cost surface. Values less than 0 are treated as
        null values.
    sources : Raster or raster path, or sequence
        A raster or sequence of indices. If a raster, pixels that are not null
        are used as source locations. The raster must have a null value set.
        The values at valid locations are used as the corresponding values in
        the allocation output. If a sequence, must have shape (M, 2) where M is
        the number of source pixels. Each item represents an index into `costs`
        to be used as a source. The element number, starting at 0, is used as
        the corresponding allocation value.

    Returns
    -------
    cost_distance : Raster
        The accumulated cost distance solution. This is the same shape as the
        `costs` input Raster and has the same null value.
    traceback : Raster
        The traceback result. This is the same shape as the `costs` input
        Raster.
    allocation : Raster
        The allocation result. This is the same shape as the `costs` input
        Raster.

    """
    if not is_raster_class(costs):
        costs = Raster(costs)
        if costs.shape[0] != 1:
            raise ValueError("Costs raster cannot be multibanded")

    src_idxs = None
    if is_raster_class(sources) or is_str(sources):
        if is_str(sources):
            sources = Raster(sources)
        if sources.shape != costs.shape:
            raise ValueError("Cost and sources raster shapes must match")
        if sources.dtype.kind not in ("u", "i"):
            raise TypeError("Sources raster must be an integer type")
        if not sources._masked:
            raise ValueError("Sources raster must have a null value set")
        sources_null_value = sources._null_value
        srcs = sources.to_dask()[0].astype(I64)
    else:
        try:
            sources = np.asarray(sources).astype(int)
            if len(sources.shape) != 2 or sources.shape[1] != 2:
                raise ValueError("Sources must be an (M, 2) shaped array")
            if np.unique(sources, axis=0).shape != sources.shape:
                raise ValueError("Sources must not contain duplicates")
            sources_null_value = -1
            src_idxs = sources
            srcs = np.full(costs.shape[1:], sources_null_value, dtype=I64)
            srcs[src_idxs[:, 0], src_idxs[:, 1]] = np.arange(
                len(sources), dtype=I64
            )
        except TypeError:
            raise ValueError("Could not understand sources argument")

    data = _normalize_raster_data(costs)

    scaling = np.abs(costs.resolution)
    results = cost_distance_analysis_numpy(
        data, srcs, sources_null_value, scaling
    )
    # Make lazy and add band dim
    cd, tr, al = [da.from_array(r[None]) for r in results]
    # Convert to DataArrays using same coordinate system as costs
    xcosts = costs.to_xarray()
    xcd, xtr, xal = [
        xr.DataArray(
            r, coords=xcosts.coords, dims=xcosts.dims, attrs=xcosts.attrs
        )
        for r in (cd, tr, al)
    ]
    xcd = xcd.where(np.isfinite(xcd), costs.null_value)
    # Add 1 to match ESRI 0-8 scale
    xtr += 1

    cd = _raster_like(costs, xcd, null_value=costs.null_value)
    tr = _raster_like(costs, xtr, null_value=_TRACEBACK_NOT_REACHED + 1)
    al = _raster_like(costs, xal, null_value=sources_null_value)
    return cd, tr, al


def cost_distance(costs, sources):
    """Calculate the cost distance.

    See cost_distance_analysis for a full description.

    Parameters
    ----------
    costs : Raster or raster path
        A raster representing a cost surface. Values less than 0 are treated as
        null values.
    sources : Raster or raster path, or sequence
        A raster or sequence of indices. If a raster, pixels that are not null
        are used as source locations. The raster must have a null value set.
        The values at valid locations are used as the corresponding values in
        the allocation output. If a sequence, must have shape (M, 2) where M is
        the number of source pixels. Each item represents an index into `costs`
        to be used as a source. The element number, starting at 0, is used as
        the corresponding allocation value.

    Returns
    -------
    cost_distance : Raster
        The accumulated cost distance solution. This is the same shape as the
        `costs` input Raster.
    traceback : Raster
        The traceback result. This is the same shape as the `costs` input
        Raster.
    allocation : Raster
        The allocation result. This is the same shape as the `costs` input
        Raster.

    See Also
    --------
    cost_distance_analysis : Full cost distance solution

    """
    cost_dist, _, _ = cost_distance_analysis(costs, sources)
    return cost_dist


def traceback(costs, sources):
    """Calculate the cost distance traceback.

    See cost_distance_analysis for a full description.

    Parameters
    ----------
    costs : Raster or raster path
        A raster representing a cost surface. Values less than 0 are treated as
        null values.
    sources : Raster or raster path, or sequence
        A raster or sequence of indices. If a raster, pixels that are not null
        are used as source locations. The raster must have a null value set.
        The values at valid locations are used as the corresponding values in
        the allocation output. If a sequence, must have shape (M, 2) where M is
        the number of source pixels. Each item represents an index into `costs`
        to be used as a source. The element number, starting at 0, is used as
        the corresponding allocation value.

    Returns
    -------
    cost_distance : Raster
        The accumulated cost distance solution. This is the same shape as the
        `costs` input Raster.
    traceback : Raster
        The traceback result. This is the same shape as the `costs` input
        Raster.
    allocation : Raster
        The allocation result. This is the same shape as the `costs` input
        Raster.

    See Also
    --------
    cost_distance_analysis : Full cost distance solution

    """
    _, trb, _ = cost_distance_analysis(costs, sources)
    return trb


def allocation(costs, sources):
    """Calculate the cost distance allocation.

    See cost_distance_analysis for a full description.

    Parameters
    ----------
    costs : Raster or raster path
        A raster representing a cost surface. Values less than 0 are treated as
        null values.
    sources : Raster or raster path, or sequence
        A raster or sequence of indices. If a raster, pixels that are not null
        are used as source locations. The raster must have a null value set.
        The values at valid locations are used as the corresponding values in
        the allocation output. If a sequence, must have shape (M, 2) where M is
        the number of source pixels. Each item represents an index into `costs`
        to be used as a source. The element number, starting at 0, is used as
        the corresponding allocation value.

    Returns
    -------
    cost_distance : Raster
        The accumulated cost distance solution. This is the same shape as the
        `costs` input Raster.
    traceback : Raster
        The traceback result. This is the same shape as the `costs` input
        Raster.
    allocation : Raster
        The allocation result. This is the same shape as the `costs` input
        Raster.

    See Also
    --------
    cost_distance_analysis : Full cost distance solution

    """
    _, _, alloc = cost_distance_analysis(costs, sources)
    return alloc


def path_distance_analysis(costs, elevation, sources):
    """Calculate accumulated path distance, traceback, and allocation.

    This funtion is very similar to :func:`cost_distance_analysis_numpy` in
    that it finds the many-sources shortest-paths solution for a given cost
    surface. The difference is that the cost function takes into account the
    3D surface distance using an elevation raster and horizontal and vertical
    factors. See :func:`cost_distance_analysis` documentation for
    additional information.

    .. note:: This function is a work in progress and horizontal and vertical
              factors are not yet implemented.

    Parameters
    ----------
    costs : 2D ndarray
        A 2D array representing a cost surface.
    elevation : 2D ndarray
        A 2D array representing the surface elevation. Same shape as `costs`.
    sources : 2D int64 ndarray
        An array of sources. The values at each valid location, as determined
        using `sources_null_value`, are used for the allocation output.
    sources_null_value: int
        The value in `sources` that indicates a null value.
    scaling_2d : scalar or 1D sequence, optional
        The scaling to use in each direction. For a grid with 30m scale, this
        would be 30. Default is 1.

    Returns
    -------
    path_distance : 2D ndarray
        The accumulated path distance solution. This is the same shape as the
        `costs` input array.
    traceback : 2D ndarray
        The traceback result. This is the same shape as the `costs` input
        array.
    allocation : 2D ndarray
        The allocation result. This is the same shape as the `costs` input
        array.

    See Also
    --------
    cost_distance_analysis_numpy

    .. [1] `ESRI: How path distance tools work <https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-the-path-distance-tools-work.htm>`_

    """
    if not is_raster_class(costs):
        costs = Raster(costs)
        if costs.shape[0] != 1:
            raise ValueError("Costs raster cannot be multibanded")
    if not is_raster_class(elevation):
        elevation = Raster(elevation)
        if elevation.shape[0] != 1:
            raise ValueError("Elevation raster cannot be multibanded")
    if costs.shape != elevation.shape:
        raise ValueError(
            "Costs and elevation rasters must have the same shape"
        )

    src_idxs = None
    if is_raster_class(sources) or is_str(sources):
        if is_str(sources):
            sources = Raster(sources)
        if sources.shape != costs.shape:
            raise ValueError("Cost and sources raster shapes must match")
        if sources.dtype.kind not in ("u", "i"):
            raise TypeError("Sources raster must be an integer type")
        if not sources._masked:
            raise ValueError("Sources raster must have a null value set")
        sources_null_value = sources._null_value
        srcs = sources.to_dask()[0].astype(I64)
    else:
        try:
            sources = np.asarray(sources).astype(int)
            if len(sources.shape) != 2 or sources.shape[1] != 2:
                raise ValueError("Sources must be an (M, 2) shaped array")
            if np.unique(sources, axis=0).shape != sources.shape:
                raise ValueError("Sources must not contain duplicates")
            sources_null_value = -1
            src_idxs = sources
            srcs = np.full(costs.shape[1:], sources_null_value, dtype=I64)
            srcs[src_idxs[:, 0], src_idxs[:, 1]] = np.arange(
                len(sources), dtype=I64
            )
        except TypeError:
            raise ValueError("Could not understand sources argument")

    cdata = _normalize_raster_data(costs)
    elevation_null_value = -9999
    edata = _normalize_raster_data(elevation, elevation_null_value)

    scaling = np.abs(costs.resolution)
    results = path_distance_analysis_numpy(
        cdata, edata, srcs, elevation_null_value, sources_null_value, scaling
    )
    # Make lazy and add band dim
    cd, tr, al = [da.from_array(r[None]) for r in results]
    # Convert to DataArrays using same coordinate system as costs
    xcosts = costs.to_xarray()
    xcd, xtr, xal = [
        xr.DataArray(
            r, coords=xcosts.coords, dims=xcosts.dims, attrs=xcosts.attrs
        )
        for r in (cd, tr, al)
    ]
    xcd = xcd.where(np.isfinite(xcd), costs.null_value)
    # Add 1 to match ESRI 0-8 scale
    xtr += 1

    cd = _raster_like(costs, xcd, null_value=costs.null_value)
    tr = _raster_like(costs, xtr, null_value=_TRACEBACK_NOT_REACHED + 1)
    al = _raster_like(costs, xal, null_value=sources_null_value)
    return cd, tr, al
