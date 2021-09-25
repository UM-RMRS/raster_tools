#cython: language_level=3
#cython: cdivision=True

import cython
import numpy as np

from raster_tools._utils import is_scalar
from raster_tools._types import F16, F32, F64

from libc.math cimport isnan
cimport numpy as cnp

from ._heap cimport (
    HEAP_t, init_heap_data, free_heap_data, push, pop, push_if_lower
)


__all__ = [
    "cost_allocation_numpy",
    "cost_distance_analysis_numpy",
    "cost_distance_numpy",
    "cost_traceback_numpy",
]


cnp.import_array()


ctypedef fused COST_t:
    cnp.uint8_t
    cnp.int8_t
    cnp.uint16_t
    cnp.int16_t
    cnp.uint32_t
    cnp.int32_t
    cnp.int64_t
    cnp.float32_t
    cnp.float64_t

ctypedef cnp.int8_t INT8_t
ctypedef cnp.int64_t INT64_t
ctypedef cnp.float64_t FLOAT_t

cdef struct index2d_t:
    INT64_t i
    INT64_t j
ctypedef index2d_t INDEX2D_t


@cython.boundscheck(True)
@cython.wraparound(True)
cdef _get_strides(shape):
    # Get strides for given nd array shape. Assumes c style strides
    return np.array(np.multiply.accumulate([1] + list(shape[::-1][:-1]))[::-1])


@cython.boundscheck(True)
@cython.wraparound(True)
cdef _ravel_indices(indices, shape):
    # Convert nd c-strided indices to flat indices
    strides = _get_strides(shape)
    flat_indices = [np.sum(strides * idx) for idx in indices]
    return np.array(flat_indices, dtype=np.int64)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _normalize_indices(indices, shape):
    # Make indices positive and check bounds
    normd_indices = np.zeros((len(indices), len(shape)), dtype=np.int64)
    for i, index in enumerate(indices):
        if len(index) != len(shape):
            raise ValueError("Index shapes must match data shape")
        for j, (idx, d) in enumerate(zip(index, shape)):
            idx = int(idx)
            idx = idx + d if idx < 0 else idx
            if not (0 <= idx < d):
                raise ValueError("Index value out of shape bounds")
            normd_indices[i, j] = idx
    return normd_indices


# Add 1 to the index move index to get the ESRI value
cdef INT8_t[:,::1] _MOVES = np.array(
    [
        # right: 0
        [0, -1],
        # lower-right: 1
        [-1, -1],
        # Down: 2
        [-1, 0],
        # lower-left: 3
        [-1, 1],
        # left: 4
        [0, 1],
        # upper-left: 5
        [1, 1],
        # up: 6
        [1, 0],
        # upper-right: 7
        [1, -1],
    ],
    dtype=np.int8,
)
cdef INT8_t TRACEBACK_NOT_REACHED = -2


@cython.boundscheck(False)
@cython.wraparound(False)
def _cost_distance_analysis_core(
        COST_t[::1] flat_costs,
        INT64_t[::1] flat_sources,
        FLOAT_t[::1] scaling,
        INT64_t[::1] flat_moves,
        FLOAT_t[::1] move_lengths,
        (INT64_t, INT64_t) shape,
        INT64_t sources_null_value,
):
    cdef INT64_t size = flat_costs.size
    # The cumulative cost distance to every pixel from the sources
    cdef FLOAT_t[::1] flat_cost_distance = np.full(
        size, np.inf, dtype=np.float64
    )
    # traceback values:
    #    -1: source
    #    -2: not reached, barrier
    #    0-7: index into moves array
    cdef INT8_t[::1] flat_traceback = np.full(
        size, TRACEBACK_NOT_REACHED, dtype=np.int8
    )
    # allocation values:
    #    sources_null_value: not reached, barrier
    #    other: value from sources array
    cdef INT64_t[::1] flat_allocation = np.full(
        size, sources_null_value, dtype=np.int64
    )

    cdef FLOAT_t cost, new_cost, cumcost, new_cumcost, move_length
    cdef INT64_t index, new_index, src
    cdef INDEX2D_t index2d
    cdef INT64_t i, it
    cdef INT8_t[::1] move
    cdef unsigned char is_at_edge, bad_move, left, right, top, bottom
    cdef INDEX2D_t costs_shape_m1,
    cdef INT64_t[::1] strides = _get_strides(shape)
    # 1.5 * size
    cdef INT64_t maxiter = size + (size // 2)
    cdef FLOAT_t inf = np.inf

    # A heap for storing pixels and their accumulated cost as the algorithm
    # explores the cost landscape.
    cdef HEAP_t frontier
    init_heap_data(&frontier, 128, size - 1)

    costs_shape_m1.i = shape[0] - 1
    costs_shape_m1.j = shape[1] - 1

    for i in range(flat_sources.size):
        src = flat_sources[i]
        if src != sources_null_value:
            flat_traceback[i] = -1
            flat_allocation[i] = src
            push(&frontier, 0, i)

    ## Main loop for Dijkstra's algorithm
    # The frontier heap contains the current known costs to pixels discovered
    # so far. When a pixel and its cumulative cost are popped, we have found
    # the minimum cost path to it and can store the cumulative cost in our
    # output array. We then add/update the cost to that pixel's neighborhood.
    # If a neighbor has already been popped before, we ignore it. The
    # cumulative cost is used as the priority in the heap. At the start, only
    # the sources are on the heap.
    for it in range(maxiter):
        if frontier.count == 0:
            break

        # Get the cumulative cost at the current pixel of interest
        cumcost = pop(&frontier)
        # Get the flat index of the current pixel and the source that led
        # to it
        index = frontier.popped_value
        src = flat_allocation[index]
        # We have now found the minimum cost to this pixel so store it
        flat_cost_distance[index] = cumcost

        # Convert to 2d index for bound checking
        index2d.i = (index // strides[0]) % shape[0]
        index2d.j = (index // strides[1]) % shape[1]
        # Compare against the bounds to see if we are at any edges
        top = index2d.i == 0
        bottom = index2d.i == costs_shape_m1.i
        left = index2d.j == 0
        right = index2d.j == costs_shape_m1.j
        is_at_edge = top or bottom or left or right

        # Look at neighborhood
        for i in range(8):
            bad_move = 0
            if is_at_edge:
                move = _MOVES[i]
                bad_move = (top and move[0] < 0) \
                        or (bottom and move[0] > 0) \
                        or (left and move[1] < 0) \
                        or (right and move[1] > 0)
            if bad_move:
                continue
            new_index = index + flat_moves[i]
            move_length = move_lengths[i]
            # If a value is already stored in this pixel, we have found an
            # optimal path to it previously and can skip it.
            if flat_cost_distance[new_index] != inf:
                continue
            cost = flat_costs[index]
            new_cost = flat_costs[new_index]
            # If the cost at this point is a barrier, skip
            # TODO: may be able consolidate into a single sentinel check if
            #       input data is standardized
            if new_cost < 0 or new_cost == inf or isnan(new_cost):
                continue

            new_cumcost = cumcost + (move_length * 0.5 * (cost + new_cost))
            if new_cumcost != inf:
                if push_if_lower(&frontier, new_cumcost, new_index) > 0:
                    flat_traceback[new_index] = i
                    flat_allocation[new_index] = src
    free_heap_data(&frontier)
    return flat_cost_distance, flat_traceback, flat_allocation


def cost_distance_analysis_numpy(
    costs, sources, sources_null_value, scaling=1.0
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
    scaling to use. Source locations have a cost of 0.

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
    sources = np.asarray(sources).astype(np.int64)
    if sources.shape != costs.shape:
        raise ValueError("Costs and sources array shapes must match")
    shape = costs.shape

    if is_scalar(scaling):
        scaling = np.array([scaling for _ in shape], dtype=F64)
    elif isinstance(scaling, (np.ndarray, list, tuple)):
        scaling = np.asarray(scaling).astype(np.float64)
        if scaling.size != len(shape) or len(scaling.shape) != 1:
            raise ValueError(f"Invalid scaling shape: {scaling.shape}")
    if any(scaling <= 0):
        raise ValueError("Scaling values must be greater than 0")

    flat_costs = costs.ravel()
    cdef INT64_t[::1] flat_sources = sources.ravel()
    cdef INT64_t[::1] flat_moves = _ravel_indices(_MOVES, shape)
    cdef FLOAT_t[::1] move_lengths = np.sqrt(
        np.sum((scaling * _MOVES)**2, axis=1)
    ).astype(np.float64)

    flat_cost_distance, flat_traceback, flat_allocation = \
        _cost_distance_analysis_core(
            flat_costs,
            flat_sources,
            scaling,
            flat_moves,
            move_lengths,
            shape,
            sources_null_value,
        )
    cost_distance = np.asarray(flat_cost_distance).reshape(shape)
    traceback = np.asarray(flat_traceback).reshape(shape)
    allocation = np.asarray(flat_allocation).reshape(shape)
    return cost_distance, traceback, allocation


def cost_distance_numpy(costs, sources, sources_null_value, scaling):
    """
    Calculate the accumulated cost distance for the cost surface.

    See cost_distance_analysis_numpy for a full description.

    Parameters
    ----------
    costs : 2D ndarray
        A 2D array representing a cost surface.
    sources : 2D int64 ndarray
        An array of sources. The values at each valid location, as determined
        using `sources_null_value`, are used for the allocation output.
    sources_null_value: int
        The value in `sources` that indicates a null value.
    scaling : scalar or 1D sequence, optional
        The scaling to use in each direction. For a grid with 30m scale, this
        would be 30. Default is 1.

    Returns
    -------
    cost_distance : 2D ndarray
        The accumulated cost distance solution. This is the same shape as the
        `costs` input array.

    See Also
    --------
    cost_distance_analysis_numpy : Full cost distance solution

    """
    cost_dist, _, _ = cost_distance_analysis_numpy(
        costs, sources, sources_null_value, scaling
    )
    return cost_dist


def cost_traceback_numpy(costs, sources, sources_null_value, scaling):
    """
    Calculate the traceback for the cost surface.

    See cost_distance_analysis_numpy for a full description.

    Parameters
    ----------
    costs : 2D ndarray
        A 2D array representing a cost surface.
    sources : 2D int64 ndarray
        An array of sources. The values at each valid location, as determined
        using `sources_null_value`, are used for the allocation output.
    sources_null_value: int
        The value in `sources` that indicates a null value.
    scaling : scalar or 1D sequence, optional
        The scaling to use in each direction. For a grid with 30m scale, this
        would be 30. Default is 1.

    Returns
    -------
    traceback : 2D ndarray
        The traceback result. This is the same shape as the `costs` input
        array.

    See Also
    --------
    cost_distance_analysis_numpy : Full cost distance solution

    """
    _, traceback, _ = cost_distance_analysis_numpy(
        costs, sources, sources_null_value, scaling
    )
    return traceback


def cost_allocation_numpy(costs, sources, sources_null_value, scaling):
    """
    Calculate the cost allocation for the cost surface.

    See cost_distance_analysis_numpy for a full description.

    Parameters
    ----------
    costs : 2D ndarray
        A 2D array representing a cost surface.
    sources : 2D int64 ndarray
        An array of sources. The values at each valid location, as determined
        using `sources_null_value`, are used for the allocation output.
    sources_null_value: int
        The value in `sources` that indicates a null value.
    scaling : scalar or 1D sequence, optional
        The scaling to use in each direction. For a grid with 30m scale, this
        would be 30. Default is 1.

    Returns
    -------
    allocation : 2D ndarray
        The allocation result. This is the same shape as the `costs` input
        array.

    See Also
    --------
    cost_distance_analysis_numpy : Full cost distance solution

    """
    _, _, allocation = cost_distance_analysis_numpy(
        costs, sources, sources_null_value, scaling
    )
    return allocation
