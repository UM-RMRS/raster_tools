import dask.array as da
import numpy as np
import xarray as xr

from raster_tools import Raster
from raster_tools.io import Encoding
from raster_tools.raster import is_raster_class, _raster_like
from raster_tools._utils import is_numpy, is_str
from raster_tools._types import F16, F64, I8, I64
from ._core import (
    cost_distance_analysis_numpy,
    cost_distance_numpy,
    cost_traceback_numpy,
    cost_allocation_numpy,
)


__all__ = [
    "allocation",
    "cost_distance",
    "cost_distance_analysis",
    "traceback",
]


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
        A raster representing a cost surface.
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
        if sources.encoding.dtype.kind not in ("u", "i"):
            raise TypeError("Sources raster must be an integer type")
        if not sources.encoding.masked:
            raise ValueError("Sources raster must have a null value set")
        sources_null_value = sources.encoding.null_value
        srcs = sources.as_encoded().to_dask()[0].astype(I64)
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

    scaling = np.abs(costs.resolution)
    results = cost_distance_analysis_numpy(
        costs._rs.data[0], srcs, sources_null_value, scaling
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
    xcd = xcd.where(np.isfinite(xcd), np.nan)
    if costs._masked:
        # Potentially have null values, fill with nan
        xtr = xtr.astype(F16).where(xtr != _TRACEBACK_NOT_REACHED, np.nan)
    # Add 1 to match ESRI 0-8 scale
    xtr += 1

    enc_orig = costs.encoding
    enc_cd = enc_orig.copy()
    enc_cd.dtype = cd.dtype
    cd = _raster_like(costs, xcd, encoding=enc_cd)

    enc_tr = enc_orig.copy()
    enc_tr.dtype = I8
    enc_tr.null_value = _TRACEBACK_NOT_REACHED + 1
    tr = _raster_like(costs, xtr, encoding=enc_tr)
    if costs._masked:
        tr = tr.set_null_value(_TRACEBACK_NOT_REACHED + 1)

    enc_al = enc_orig.copy()
    enc_al.dtype = I64
    enc_al.null_value = sources_null_value
    al = _raster_like(costs, xal, encoding=enc_al)
    if costs._masked:
        al = al.set_null_value(sources_null_value)
    return cd, tr, al


def cost_distance(costs, sources):
    """Calculate the cost distance.

    See cost_distance_analysis for a full description.

    Parameters
    ----------
    costs : Raster or raster path
        A raster representing a cost surface.
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
        A raster representing a cost surface.
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
        A raster representing a cost surface.
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
