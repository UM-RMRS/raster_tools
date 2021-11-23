#cython: language_level=3

cimport numpy as cnp

ctypedef cnp.float64_t KEY_t
ctypedef cnp.int64_t INDEX_t
# We don't need a very big number for the level count since it is used as
# a power
ctypedef cnp.int8_t LEVEL_t


cdef struct heap_t:
    ## Internal use members
    # Ideally we would use numpy arrays and memoryviews for our data arrays
    # but that would require calling numpy allocations functions when resizing.
    # This would greatly slow down the resize operation since the numpy
    # functions are python objects. Instead, we use raw c arrays and malloc :(.
    # Used as the heap, stores the priority keys
    KEY_t* _keys
    # extra information stored with a priority key
    INDEX_t* _values
    # Cross references used for fast lookup and updating of priorities.
    # Stores the relative index into the final heap level of a value's
    # corresponding key.
    INDEX_t* _crossrefs
    INDEX_t _max_value
    LEVEL_t _levels
    LEVEL_t _min_levels
    INDEX_t count
    INDEX_t popped_value
ctypedef heap_t HEAP_t

## Heap maintenance methods
cdef void _resize_levels(HEAP_t* heap, LEVEL_t inc_dec) nogil
cdef void _sift_all(HEAP_t* heap) nogil
cdef void _sift(HEAP_t* heap, INDEX_t i) nogil
cdef void _remove(HEAP_t* heap, INDEX_t i) nogil
cdef void _simple_push(HEAP_t* heap, KEY_t key, INDEX_t item) nogil
## API
cdef void init_heap_data(HEAP_t* heap, INDEX_t capacity, INDEX_t max_value) nogil
cdef void free_heap_data(HEAP_t* heap) nogil
cdef void clear(HEAP_t* heap) nogil
cdef INDEX_t push(HEAP_t* heap, KEY_t key, INDEX_t item) nogil
cdef KEY_t pop(HEAP_t* heap) nogil
cdef INDEX_t push_if_lower(HEAP_t* heap, KEY_t key, INDEX_t item) nogil
