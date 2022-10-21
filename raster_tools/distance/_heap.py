import numba as nb
import numpy as np
from numba.types import Tuple

from raster_tools.dtypes import F64, I64

__all__ = ["HEAP_TYPE", "init_heap_data", "pop", "push", "push_if_lower"]

JIT_KWARGS = {"nopython": True, "nogil": True}

HEAP_SPEC = [
    ("max_value", np.int64),
    ("levels", np.int8),
    ("min_levels", np.int8),
    ("count", np.int64),
]
HEAP_DT = np.dtype(HEAP_SPEC)
HEAP_TYPE = nb.from_dtype(HEAP_DT)

HEAP_STATE_SIG = Tuple(
    (
        nb.float64[:],
        nb.int64[:],
        nb.int64[:],
        HEAP_TYPE[:],
    )
)
(
    KEYS_TYPE_SIG,
    VALUES_TYPE_SIG,
    CROSSREFS_TYPE_SIG,
    HEAP_TYPE_SIG,
) = HEAP_STATE_SIG


@nb.jit(HEAP_STATE_SIG(nb.int64, nb.int64), **JIT_KWARGS)
def init_heap_data(capacity: I64, max_value: I64):
    if capacity < 1:
        raise ValueError("Capacity must be greater than 0")
    if max_value < 1:
        raise ValueError("Max value must be greater than 0")
    # Have to use single element array. Numba throws an error if you try to
    # return a record dtype. Using the array as a wrapper allows the record to
    # be passed around in these functions.
    heap = np.zeros(1, dtype=HEAP_DT)
    heap[0].max_value = max_value
    heap[0].levels = 0
    while 2 ** heap[0].levels < capacity:
        heap[0].levels += 1
    heap[0].min_levels = heap[0].levels
    heap[0].count = 0

    n = 2 ** heap[0].levels
    keys = np.full(2 * n, np.inf, dtype=F64)
    values = np.full(n, -1, dtype=I64)
    crossrefs = np.full(heap[0].max_value + 1, -1, dtype=I64)
    return keys, values, crossrefs, heap


@nb.jit(nb.void(KEYS_TYPE_SIG, HEAP_TYPE_SIG, nb.int64), **JIT_KWARGS)
def _sift(keys, heap, i):
    # Get first index in the pair. Pairs always start at uneven indices
    i -= i % 2 == 0
    # sift the minimum value back through the heap
    for _ in range(heap[0].levels, 1, -1):
        # Index into previous level
        iplvl = (i - 1) // 2
        keys[iplvl] = min(keys[i], keys[i + 1])
        i = iplvl - (iplvl % 2 == 0)


@nb.jit(nb.void(KEYS_TYPE_SIG, HEAP_TYPE_SIG), **JIT_KWARGS)
def _sift_all(keys, heap):
    # index at start of final level
    ilvl = (1 << heap[0].levels) - 1
    # Length of final level
    n = ilvl + 1
    for i in range(ilvl, ilvl + n, 2):
        _sift(keys, heap, i)


@nb.jit(
    Tuple((KEYS_TYPE_SIG, VALUES_TYPE_SIG))(
        KEYS_TYPE_SIG, VALUES_TYPE_SIG, HEAP_TYPE_SIG, nb.int8
    ),
    **JIT_KWARGS,
)
def _resize_levels(keys, values, heap, inc_dec):
    new_levels = heap[0].levels + inc_dec
    n = 1 << new_levels
    # We don't need to resize the crossrefs array
    new_keys = np.full(2 * n, np.inf, dtype=F64)
    new_values = np.full(n, -1, dtype=I64)
    if heap[0].count:
        inew = (1 << new_levels) - 1
        iold = (1 << heap[0].levels) - 1
        n = min(inew, iold) + 1
        new_keys[inew : inew + n] = keys[iold : iold + n]
        new_values[:n] = values[:n]
    keys = new_keys
    values = new_values
    heap[0].levels = new_levels
    _sift_all(keys, heap)
    return keys, values


@nb.jit(
    Tuple((KEYS_TYPE_SIG, VALUES_TYPE_SIG))(*HEAP_STATE_SIG, nb.int64),
    **JIT_KWARGS,
)
def _remove(keys, values, crossrefs, heap, i):
    lvl_start = (1 << heap[0].levels) - 1
    ilast = lvl_start + heap[0].count - 1
    i_val = i - lvl_start
    i_val_last = heap[0].count - 1

    # Get the last value
    value = values[i_val_last]
    # and update the crossref to point to its new location
    crossrefs[value] = i_val
    # Get the value to be removed
    value = values[i_val]
    # and delete its crossref
    crossrefs[value] = -1
    # Swap key to be removed with the last key
    keys[i] = keys[ilast]
    # Swap corresponding values
    values[i_val] = values[i_val_last]
    # Remove the last key since it is now invalid
    keys[ilast] = np.inf

    heap[0].count -= 1
    if (heap[0].levels > heap[0].min_levels) & (
        heap[0].count < (1 << (heap[0].levels - 2))
    ):
        keys, values = _resize_levels(keys, values, heap, -1)
    else:
        _sift(keys, heap, i)
        _sift(keys, heap, ilast)
    return keys, values


@nb.jit(
    HEAP_STATE_SIG(*HEAP_STATE_SIG, nb.float64, nb.int64),
    **JIT_KWARGS,
)
def _simple_push(keys, values, crossrefs, heap, key, value):
    lvl_size = 1 << heap[0].levels

    if heap[0].count >= lvl_size:
        keys, values = _resize_levels(keys, values, heap, 1)
        lvl_size = lvl_size << 1
    i = lvl_size - 1 + heap[0].count
    keys[i] = key
    values[heap[0].count] = value
    crossrefs[value] = heap[0].count
    heap[0].count += 1
    _sift(keys, heap, i)
    return keys, values, crossrefs, heap


@nb.jit(
    nb.types.Tuple((*HEAP_STATE_SIG, nb.int8))(
        *HEAP_STATE_SIG, nb.float64, nb.int64
    ),
    **JIT_KWARGS,
)
def push(keys, values, crossrefs, heap, key, value):
    if not (0 <= value <= heap[0].max_value):
        return keys, values, crossrefs, heap, -1

    lvl_size = 1 << heap[0].levels
    ii = crossrefs[value]
    if ii != -1:
        # Update the key value
        i = lvl_size - 1 + ii
        keys[i] = key
        _sift(keys, heap, i)
        return keys, values, crossrefs, heap, 0

    keys, values, crossrefs, heap = _simple_push(
        keys, values, crossrefs, heap, key, value
    )
    return keys, values, crossrefs, heap, 1


@nb.jit(
    nb.types.Tuple((*HEAP_STATE_SIG, nb.int8))(
        *HEAP_STATE_SIG, nb.float64, nb.int64
    ),
    **JIT_KWARGS,
)
def push_if_lower(keys, values, crossrefs, heap, key, value):
    if not (0 <= value <= heap[0].max_value):
        return keys, values, crossrefs, heap, -1

    icr = crossrefs[value]
    if icr != -1:
        i = (1 << heap[0].levels) - 1 + icr
        if keys[i] > key:
            keys[i] = key
            _sift(keys, heap, i)
            return keys, values, crossrefs, heap, 1
        return keys, values, crossrefs, heap, 0

    keys, values, crossrefs, heap = _simple_push(
        keys, values, crossrefs, heap, key, value
    )
    return keys, values, crossrefs, heap, 1


@nb.jit(
    nb.types.Tuple((*HEAP_STATE_SIG, nb.float64, nb.int64))(*HEAP_STATE_SIG),
    **JIT_KWARGS,
)
def pop(keys, values, crossrefs, heap):
    # The minimum key sits at position 1
    i = 1
    # Trace min key through the heap levels
    for lvl in range(1, heap[0].levels):
        if keys[i] <= keys[i + 1]:
            i = (i * 2) + 1
        else:
            i = ((i + 1) * 2) + 1
    # Find it in the last level
    if keys[i] > keys[i + 1]:
        i += 1
    # Get corresponding index into values
    ii = i - ((1 << heap[0].levels) - 1)
    popped_key = keys[i]
    popped_value = values[ii]
    if heap.count:
        keys, values = _remove(keys, values, crossrefs, heap, i)
    return keys, values, crossrefs, heap, popped_key, popped_value
