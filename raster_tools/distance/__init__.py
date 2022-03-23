import dask.array as da
import numpy as np
import xarray as xr

from raster_tools.dtypes import I64, is_float, is_str
from raster_tools.raster import Raster

from ._core import cost_distance_analysis_numpy

__all__ = [
    "cda_allocation",
    "cda_cost_distance",
    "cda_traceback",
    "cost_distance_analysis",
]


def _normalize_raster_data(rs, missing=-1):
    data = rs._data
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


def cost_distance_analysis(costs, sources, elevation=None):
    """Calculate accumulated cost distance, traceback, and allocation.

    This function uses Dijkstra's algorithm to compute the many-sources
    shortest-paths solution for a given cost surface. Valid moves are from a
    given pixel to its 8 nearest neighbors. This produces 3 rasters. The
    first is the accumulated cost distance, which contains the
    distance-weighted accumulated minimum cost to each pixel. The cost to move
    from one pixel to the next is ``length * mean(costs[i], costs[i+1])``,
    where ``length`` is 1 for horizontal and vertical moves and ``sqrt(2)`` for
    diagonal moves. The `costs` raster's resolution informs the actual scaling
    to use. Source locations have a cost of 0. If `elevation` provided, the
    length calculation incorporates the elevation data to make the algorithm 3D
    aware.

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

    The third raster contains the source allocation for each pixel. Each pixel
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
    elevation : Raster or raster path, optional
        A raster containing elevation values on the same grid as `costs`. If
        provided, the elevation values are used when calculating the travel
        distance between pixels. This makes the algorithm 3D aware.

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

    References
    ----------
    * `ESRI: How cost distance tools work <https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-the-cost-distance-tools-work.htm>`_
    * `ESRI: How path distance tools work <https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-the-path-distance-tools-work.htm>`_

    """  # noqa: E501
    if not isinstance(costs, Raster):
        costs = Raster(costs)
        if costs.shape[0] != 1:
            raise ValueError("Costs raster cannot be multibanded")
    if elevation is not None:
        if not isinstance(elevation, Raster):
            elevation = Raster(elevation)
            if elevation.shape[0] != 1:
                raise ValueError("Elevation raster cannot be multibanded")
        if costs.shape != elevation.shape:
            raise ValueError(
                "Costs and elevation rasters must have the same shape"
            )

    src_idxs = None
    if isinstance(sources, Raster) or is_str(sources):
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
    if elevation is not None:
        elevation_null_value = -9999
        edata = _normalize_raster_data(elevation, elevation_null_value)
    else:
        edata = None
        elevation_null_value = 0

    scaling = np.abs(costs.resolution)
    results = cost_distance_analysis_numpy(
        data,
        srcs,
        sources_null_value,
        elevation=edata,
        elevation_null_value=elevation_null_value,
        scaling=scaling,
    )
    # Make lazy and add band dim
    cd, tr, al = [da.from_array(r[None]) for r in results]
    # Convert to DataArrays using same coordinate system as costs
    xcosts = costs.xrs
    xcd, xtr, xal = [
        xr.DataArray(
            r, coords=xcosts.coords, dims=xcosts.dims, attrs=xcosts.attrs
        )
        for r in (cd, tr, al)
    ]
    xcd = xcd.where(np.isfinite(xcd), costs.null_value)
    # Add 1 to match ESRI 0-8 scale
    xtr += 1

    cd = costs._replace(xcd, null_value=costs.null_value)
    tr = costs._replace(xtr, null_value=_TRACEBACK_NOT_REACHED + 1)
    al = costs._replace(xal, null_value=sources_null_value)
    return cd, tr, al


def cda_cost_distance(costs, sources, elevation=None):
    """Calculate just the cost distance for a cost surface

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
    elevation : Raster or raster path, optional
        A raster containing elevation values on the same grid as `costs`. If
        provided, the elevation values are used when calculating the travel
        distance between pixels. This makes the algorithm 3D aware.

    Returns
    -------
    cost_distance : Raster
        The accumulated cost distance solution. This is the same shape as the
        `costs` input Raster.

    See Also
    --------
    cost_distance_analysis : Full cost distance solution

    """
    cost_dist, _, _ = cost_distance_analysis(costs, sources, elevation)
    return cost_dist


def cda_traceback(costs, sources, elevation=None):
    """Calculate just the cost distance traceback for a cost surface.

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
    elevation : Raster or raster path, optional
        A raster containing elevation values on the same grid as `costs`. If
        provided, the elevation values are used when calculating the travel
        distance between pixels. This makes the algorithm 3D aware.

    Returns
    -------
    traceback : Raster
        The traceback result. This is the same shape as the `costs` input
        Raster.

    See Also
    --------
    cost_distance_analysis : Full cost distance solution

    """
    _, trb, _ = cost_distance_analysis(costs, sources, elevation)
    return trb


def cda_allocation(costs, sources, elevation=None):
    """Calculate just the cost distance allocation for a cost surface.

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
    elevation : Raster or raster path, optional
        A raster containing elevation values on the same grid as `costs`. If
        provided, the elevation values are used when calculating the travel
        distance between pixels. This makes the algorithm 3D aware.

    Returns
    -------
    allocation : Raster
        The allocation result. This is the same shape as the `costs` input
        Raster.

    See Also
    --------
    cost_distance_analysis : Full cost distance solution

    """
    _, _, alloc = cost_distance_analysis(costs, sources, elevation)
    return alloc
