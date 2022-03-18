import numba as nb
import numpy as np
from numba.types import float64, int8, int64

from raster_tools.distance._heap import (
    init_heap_data,
    pop,
    push,
    push_if_lower,
)
from raster_tools.dtypes import F16, F32, F64, I8, I64, is_scalar

JIT_KWARGS = {"nopython": True, "nogil": True, "cache": True}
ngjit = nb.jit(**JIT_KWARGS)


@ngjit
def _mult_accumulate(x):
    out = np.empty(len(x), dtype=I64)
    c = 1
    for i in range(len(x)):
        out[i] = c * x[i]
    return out


@nb.jit(
    [
        "int64[:](Tuple((int64, int64)))",
        "int64[:](Tuple((int64, int64, int64)))",
    ],
    **JIT_KWARGS,
)
def _get_strides(shape):
    # Get strides for given nd array shape. Assumes c style strides
    # NOTE: output must have a standardized type because windows has different
    # default int types from linux.
    values = np.empty(len(shape), dtype=I64)
    values[0] = 1
    values[1:] = shape[::-1][:-1]
    strides = _mult_accumulate(values)[::-1]
    return strides


@ngjit
def _ravel_indices(indices, shape):
    # Convert nd c-strided indices to flat indices
    strides = _get_strides(shape)
    flat_indices = [np.sum(strides * idx) for idx in indices]
    return np.array(flat_indices, dtype=I64)


# These are the moves used to explore the cost surface. +1 means to move 1 cell
# in the positive row/col direction. -1 means to move in the negative row/col
# direction. The moves have been ordered so that the index can be mapped to the
# ESRI backtrace value by adding 1.
#
# The move direction is the direction traversed in the array during the
# algorithm's exploration. The bactrace direction is the direction to move when
# traveling back to the nearest source.
#
# +-------+------------+-------------+---------------+
# | Index | ESRI Value |  Move Dir   | Backtrace Dir |
# +-------+------------+-------------+---------------+
# |     0 |          1 | Left        | Right         |
# |     1 |          2 | Upper-Left  | Lower-Right   |
# |     2 |          3 | Up          | Down          |
# |     3 |          4 | Upper-Right | Lower-Left    |
# |     4 |          5 | Right       | Left          |
# |     5 |          6 | Lower-Right | Upper-Left    |
# |     6 |          7 | Down        | Up            |
# |     7 |          8 | Lower-Left  | Upper-Right   |
# +-------+------------+-------------+---------------+
#
# ref: https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/understanding-cost-distance-analysis.htm  # noqa: E501
_MOVES = np.array(
    [
        # 0
        [0, -1],
        # 1
        [-1, -1],
        # 2
        [-1, 0],
        # 3
        [-1, 1],
        # 4
        [0, 1],
        # 5
        [1, 1],
        # 6
        [1, 0],
        # 7
        [1, -1],
    ],
    dtype=I8,
)
TRACEBACK_NOT_REACHED = -2


@ngjit
def _cost_distance_analysis_core(
    flat_costs,
    flat_sources,
    flat_moves,
    move_lengths,
    shape,
    sources_null_value,
    # Elevation args
    flat_elev,
    elevation_null_value,
    use_elevation,
):
    size: int64 = flat_costs.size
    # The cumulative cost distance to every pixel from the sources
    flat_cost_distance = np.full(size, np.inf, dtype=F64)
    # traceback values:
    #    -1: source
    #    -2: not reached, barrier
    #    0-7: index into moves array
    flat_traceback = np.full(size, TRACEBACK_NOT_REACHED, dtype=I8)
    # allocation values:
    #    sources_null_value: not reached, barrier
    #    other: value from sources array
    flat_allocation = np.full(size, sources_null_value, dtype=I64)

    cost: float64
    new_cost: float64
    cumcost: float64
    new_cumcost: float64
    move_length: float64
    dz: float64
    ev: float64
    new_ev: float64
    index: int64
    index: int64
    new_index: int64
    src: int64
    index2d = np.zeros(2, dtype=I64)
    i: int64
    it: int64
    move: int8[:]
    is_at_edge: bool
    bad_move: bool
    left: bool
    right: bool
    top: bool
    bottom: bool
    costs_shape_m1 = np.zeros(2, dtype=I64)
    strides: int64[:] = _get_strides(shape)
    # 1.5 * size
    maxiter: int64 = size + (size // 2)

    # A heap for storing pixels and their accumulated cost as the algorithm
    # explores the cost landscape.
    heap_keys, heap_values, heap_xrefs, heap_state = init_heap_data(
        128, size - 1
    )

    costs_shape_m1[0] = shape[0] - 1
    costs_shape_m1[1] = shape[1] - 1

    for i in range(flat_sources.size):
        src = flat_sources[i]
        if src != sources_null_value:
            flat_traceback[i] = -1
            flat_allocation[i] = src
            heap_keys, heap_values, heap_xrefs, heap_state, _ = push(
                heap_keys, heap_values, heap_xrefs, heap_state, 0, i
            )

    # Main loop for Dijkstra's algorithm
    # The frontier heap contains the current known costs to pixels discovered
    # so far. When a pixel and its cumulative cost are popped, we have found
    # the minimum cost path to it and can store the cumulative cost in our
    # output array. We then add/update the cost to that pixel's neighborhood.
    # If a neighbor has already been popped before, we ignore it. The
    # cumulative cost is used as the priority in the heap. At the start, only
    # the sources are on the heap.
    #
    # ref: https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-the-cost-distance-tools-work.htm  # noqa: E501
    # ref: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
    for it in range(maxiter):
        if heap_state[0].count == 0:
            break

        # Get the cumulative cost at the current pixel of interest and get the
        # flat index of the current pixel and the source that led to it
        heap_keys, heap_values, heap_xrefs, heap_state, cumcost, index = pop(
            heap_keys, heap_values, heap_xrefs, heap_state
        )
        src = flat_allocation[index]
        # We have now found the minimum cost to this pixel so store it
        flat_cost_distance[index] = cumcost

        # Convert to 2d index for bounds checking
        index2d[0] = (index // strides[0]) % shape[0]
        index2d[1] = (index // strides[1]) % shape[1]
        # Compare against the bounds to see if we are at any edges
        top = index2d[0] == 0
        bottom = index2d[0] == costs_shape_m1[0]
        left = index2d[1] == 0
        right = index2d[1] == costs_shape_m1[1]
        is_at_edge = top or bottom or left or right

        # Look at neighborhood
        for i in range(8):
            if is_at_edge:
                move = _MOVES[i]
                bad_move = (
                    (top and move[0] < 0)
                    or (bottom and move[0] > 0)
                    or (left and move[1] < 0)
                    or (right and move[1] > 0)
                )
                if bad_move:
                    continue
            new_index = index + flat_moves[i]
            move_length = move_lengths[i]
            # If a value is already stored in this pixel, we have found an
            # optimal path to it previously and can skip it.
            if flat_cost_distance[new_index] != np.inf:
                continue
            cost = flat_costs[index]
            new_cost = flat_costs[new_index]
            # If the cost at this point is a barrier, skip
            # TODO: may be able consolidate into a single sentinel check if
            #       input data is standardized
            if new_cost < 0 or new_cost == np.inf or np.isnan(new_cost):
                continue

            if use_elevation:
                ev = flat_elev[index]
                new_ev = flat_elev[new_index]
                if new_ev == elevation_null_value:
                    continue
                dz = new_ev - ev
                move_length = np.sqrt(move_length + (dz * dz))

            new_cumcost = cumcost + (move_length * 0.5 * (cost + new_cost))
            if new_cumcost != np.inf:
                (
                    heap_keys,
                    heap_values,
                    heap_xrefs,
                    heap_state,
                    flag,
                ) = push_if_lower(
                    heap_keys,
                    heap_values,
                    heap_xrefs,
                    heap_state,
                    new_cumcost,
                    new_index,
                )
                if flag > 0:
                    flat_traceback[new_index] = i
                    flat_allocation[new_index] = src
    return flat_cost_distance, flat_traceback, flat_allocation


def cost_distance_analysis_numpy(
    costs,
    sources,
    sources_null_value,
    elevation=None,
    elevation_null_value=0,
    scaling=1.0,
):
    """
    Calculate accumulated cost distance, traceback, and allocation.

    This function uses Dijkstra's algorithm to compute the many-sources
    shortest-paths solution for a given cost surface. Valid moves are from a
    given pixel to its 8 nearest neighbors. This produces 3 2D arrays. The
    first is the accumulated cost distance, which contains the
    distance-weighted accumulated minimum cost to each pixel. The cost to move
    from one pixel to the next is ``length * mean(costs[i], costs[i+1])``,
    where ``length`` is 1 for horizontal and vertical moves and ``sqrt(2)`` for
    diagonal moves. The provided scaling factor informs the actual distance
    scaling to use. Source locations have a cost of 0. If `elevation` is
    provided, the length calculation incorporates the elevation data to make
    the algorithm 3D aware.

    The second array contains the traceback values for the solution. At each
    pixel, the stored value indicates the neighbor to move to in order to get
    closer to the cost-relative nearest source. The numbering is as follows:
        5  6  7
        4  X  0
        3  2  1
    Here, X indicates the current pixel and the numbers are the neighbor
    pixel positions. 0 indicates the neighbor immediately to the right and
    6 indicates the neighbor immediately above. In terms of rows and columns,
    these are the neighbor one column over and one row above, respectively. a
    value of -1 indicates that the current pixel is a source pixel and -2
    indicates that the pixel was not traversed (no data or a barrier).

    The third array contians the source allocation for each pixel. Each pixel
    is labeled based on the source location that is closest in terms of cost
    distance. The label is the value stored in the sources array at the
    corresponding source location.

    Parameters
    ----------
    costs : 2D ndarray
        A 2D array representing a cost surface.
    sources : 2D int64 ndarray
        An array of sources. The values at each valid location, as determined
        using `sources_null_value`, are used for the allocation output.
    sources_null_value: int
        The value in `sources` that indicates a null value.
    elevation : 2D ndarray, optional
        An array of elevation values. Same shape as `costs`. If provided, the
        elevation is used when determining the move length between pixels.
    elevation_null_value : scalar, optional
        The null value for `elevation` data. Default is 0.
    scaling : scalar or 1D sequence, optional
        The scaling to use in each direction. For a grid with 30m scale, this
        would be 30. Default is 1.

    Returns
    -------
    cost_distance : 2D ndarray
        The accumulated cost distance solution. This is the same shape as the
        `costs` input array.
    traceback : 2D ndarray
        The traceback result. This is the same shape as the `costs` input
        array.
    allocation : 2D ndarray
        The allocation result. This is the same shape as the `costs` input
        array.

    """
    costs = np.asarray(costs)
    if costs.dtype == F16:
        # Promote because our fused type doesn't define a float16 type
        costs = costs.astype(F32)
    sources = np.asarray(sources).astype(I64)
    shape = costs.shape
    if sources.shape != shape:
        raise ValueError("Costs and sources array shapes must match")

    if elevation is None:
        elevation = np.array([])
        elevation_null_value = 0
        use_elevation = False
    else:
        elevation = np.asarray(elevation)
        if elevation.shape != shape:
            raise ValueError(
                "Elevation must have the same shape as costs array"
            )
        if elevation.dtype == F16:
            # Promote because our fused type doesn't define a float16 type
            elevation = elevation.astype(F32)
        use_elevation = True

    if is_scalar(scaling):
        scaling = np.array([scaling for _ in shape], dtype=F64)
    elif isinstance(scaling, (np.ndarray, list, tuple)):
        scaling = np.asarray(scaling).astype(F64)
        if scaling.size != len(shape) or len(scaling.shape) != 1:
            raise ValueError(f"Invalid scaling shape: {scaling.shape}")
    if any(scaling <= 0):
        raise ValueError("Scaling values must be greater than 0")

    flat_costs = costs.ravel()
    flat_elev = elevation.ravel()
    flat_sources = sources.ravel()
    flat_moves = _ravel_indices(_MOVES, shape)
    # Compute squared move lengths
    move_lengths = np.sum((scaling * _MOVES) ** 2, axis=1).astype(F64)
    if not use_elevation:
        # No elevation data provided so convert to actual euclidean lengths
        for i in range(move_lengths.size):
            move_lengths[i] = np.sqrt(move_lengths[i])

    (
        flat_cost_distance,
        flat_traceback,
        flat_allocation,
    ) = _cost_distance_analysis_core(
        flat_costs,
        flat_sources,
        flat_moves,
        move_lengths,
        shape,
        sources_null_value,
        # Optional elevation args
        flat_elev,
        elevation_null_value,
        use_elevation,
    )
    cost_distance = flat_cost_distance.reshape(shape)
    traceback = flat_traceback.reshape(shape)
    allocation = flat_allocation.reshape(shape)
    return cost_distance, traceback, allocation
