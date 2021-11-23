#cython: language_level=3
#cython: cdivision=True

import cython
import numpy as np

from libc.stdlib cimport free, malloc
from libc.string cimport memcpy


cdef KEY_t inf = np.inf


cdef void init_heap_data(
    HEAP_t* heap, INDEX_t capacity, INDEX_t max_value
) nogil:
    heap._max_value = max_value
    heap._levels = 0
    while 2 ** heap._levels < capacity:
        heap._levels += 1
    heap._min_levels = heap._levels
    heap.count = 0

    cdef INDEX_t n = 2 ** heap._levels
    heap._keys = <KEY_t*>malloc(2 * n * sizeof(KEY_t))
    heap._values = <INDEX_t*>malloc(n * sizeof(INDEX_t))
    heap._crossrefs = <INDEX_t*>malloc(
        (heap._max_value + 1) * sizeof(INDEX_t)
    )
    heap.popped_value = -1
    clear(heap)


cdef void free_heap_data(HEAP_t* heap) nogil:
    free(heap._keys)
    free(heap._values)
    free(heap._crossrefs)


cdef void clear(HEAP_t* heap) nogil:
    cdef INDEX_t n = 2 ** heap._levels
    cdef INDEX_t i
    for i in range(n * 2):
        heap._keys[i] = inf
    for i in range(n):
        heap._values[i] = -1
    for i in range(heap._max_value + 1):
        heap._crossrefs[i] = -1

cdef void _resize_levels(HEAP_t* heap, LEVEL_t inc_dec) nogil:
    cdef INDEX_t i, inew, iold
    cdef LEVEL_t new_levels = heap._levels + inc_dec
    cdef INDEX_t n = (1 << new_levels)
    # We don't need to resize the crossrefs array
    cdef KEY_t* keys = <KEY_t*>malloc(2 * n * sizeof(KEY_t))
    cdef INDEX_t* values = <INDEX_t*>malloc(n * sizeof(INDEX_t))
    if keys is NULL or values is NULL:
        free(keys)
        free(values)
        with gil:
            raise MemoryError()

    for i in range(n * 2):
        keys[i] = inf
    for i in range(n):
        values[i] = -1
    if heap.count:
        inew = (1 << new_levels) - 1
        iold = (1 << heap._levels) - 1
        n = min(inew, iold) + 1
        memcpy(&(keys[inew]), &(heap._keys[iold]), n * sizeof(KEY_t))
        memcpy(values, heap._values, n * sizeof(INDEX_t))
    free(heap._keys)
    free(heap._values)
    heap._keys = keys
    heap._values = values
    heap._levels = new_levels
    _sift_all(heap)

cdef void _sift_all(HEAP_t* heap) nogil:
    # index at start of final level
    cdef INDEX_t ilvl = (1 << heap._levels) - 1
    # Length of final level
    cdef INDEX_t n = ilvl + 1
    cdef INDEX_t i
    for i in range(ilvl, ilvl + n, 2):
        _sift(heap, i)

cdef void _sift(HEAP_t* heap, INDEX_t i) nogil:
    # Index into previous level
    cdef INDEX_t iplvl
    cdef LEVEL_t _

    # Get first index in the pair. Pairs always start at uneven indices
    i -= i % 2 == 0
    # sift the minimum value back through the heap
    for _ in range(heap._levels, 1, -1):
        iplvl = (i - 1) // 2
        heap._keys[iplvl] = min(heap._keys[i], heap._keys[i + 1])
        i = iplvl - (iplvl % 2 == 0)

cdef void _remove(HEAP_t* heap, INDEX_t i) nogil:
    cdef INDEX_t lvl_start = (1 << heap._levels) - 1
    cdef INDEX_t ilast = lvl_start + heap.count - 1
    cdef INDEX_t i_val = i - lvl_start
    cdef INDEX_t i_val_last = heap.count - 1
    cdef INDEX_t value

    # Get the last value
    value = heap._values[i_val_last]
    # and update the crossref to point to its new location
    heap._crossrefs[value] = i_val
    # Get the value to be removed
    value = heap._values[i_val]
    # and delete its crossref
    heap._crossrefs[value] = -1
    # Swap key to be removed with the last key
    heap._keys[i] = heap._keys[ilast]
    # Swap corresponding values
    heap._values[i_val] = heap._values[i_val_last]
    # Remove the last key since it is now invalid
    heap._keys[ilast] = inf

    heap.count -= 1
    if (
        (heap._levels > heap._min_levels)
        & (heap.count < (1 << (heap._levels - 2)))
    ):
        _resize_levels(heap, -1)
    else:
        _sift(heap, i)
        _sift(heap, ilast)

cdef void _simple_push(HEAP_t* heap, KEY_t key, INDEX_t value) nogil:
    cdef INDEX_t i
    cdef INDEX_t lvl_size = 1 << heap._levels

    if heap.count >= lvl_size:
        _resize_levels(heap, 1)
        lvl_size = lvl_size << 1
    i = lvl_size - 1 + heap.count
    heap._keys[i] = key
    heap._values[heap.count] = value
    heap._crossrefs[value] = heap.count
    heap.count += 1
    _sift(heap, i)

cdef INDEX_t push(HEAP_t* heap, KEY_t key, INDEX_t value) nogil:
    if not (0 <= value <= heap._max_value):
        return -1

    cdef INDEX_t i
    cdef INDEX_t lvl_size = 1 << heap._levels
    cdef INDEX_t ii = heap._crossrefs[value]
    if ii != -1:
        # Update the key value
        i = lvl_size - 1 + ii
        heap._keys[i] = key
        _sift(heap, i)
        return 0

    _simple_push(heap, key, value)
    return 1

cdef INDEX_t push_if_lower(HEAP_t* heap, KEY_t key, INDEX_t value) nogil:
    if not (0 <= value <= heap._max_value):
        return -1

    cdef INDEX_t i
    cdef INDEX_t icr = heap._crossrefs[value]
    if icr != -1:
        i = (1 << heap._levels) - 1 + icr
        if heap._keys[i] > key:
            heap._keys[i] = key
            _sift(heap, i)
            return 1
        return 0

    _simple_push(heap, key, value)
    return 1

cdef KEY_t pop(HEAP_t* heap) nogil:
    cdef LEVEL_t lvl
    cdef INDEX_t i
    cdef INDEX_t ii
    cdef KEY_t popped_key

    # The minimum key sits at position 1
    i = 1
    # Trace min key through the heap levels
    for lvl in range(1, heap._levels):
        if heap._keys[i] <= heap._keys[i + 1]:
            i = (i * 2) + 1
        else:
            i = ((i + 1) * 2) + 1
    # Find it in the last level
    if heap._keys[i] > heap._keys[i + 1]:
        i += 1
    # Get corresponding index into values
    ii = i - ((1 << heap._levels) - 1)
    popped_key = heap._keys[i]
    heap.popped_value = heap._values[ii]
    if heap.count:
        _remove(heap, i)
    return popped_key
