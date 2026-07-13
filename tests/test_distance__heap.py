import numpy as np
import pytest

from raster_tools.distance import _heap as hp


def test_init_heap_data():
    keys, values, xrefs, heap_state = hp.init_heap_data(10, 100)
    assert keys.size == 32
    assert values.size == 16
    assert xrefs.size == 101
    assert (keys == np.inf).all()
    assert (values == -1).all()
    assert (xrefs == -1).all()
    assert heap_state[0]["count"] == 0

    keys, values, xrefs, heap_state = hp.init_heap_data(128, 1100)
    assert keys.size == 256
    assert values.size == 128
    assert xrefs.size == 1101
    assert (keys == np.inf).all()
    assert (values == -1).all()
    assert (xrefs == -1).all()
    assert heap_state[0]["count"] == 0

    keys, values, xrefs, heap_state = hp.init_heap_data(257, 1100)
    assert keys.size == 1024
    assert values.size == 512
    assert xrefs.size == 1101
    assert (keys == np.inf).all()
    assert (values == -1).all()
    assert (xrefs == -1).all()
    assert heap_state[0]["count"] == 0


@pytest.mark.parametrize(
    "capacity,max_value", [(-1, 100), (0, 100), (12, -1), (12, 0)]
)
def test_init_heap_data_errors(capacity, max_value):
    with pytest.raises(ValueError):
        hp.init_heap_data(capacity, max_value)
        hp.init_heap_data(capacity, max_value)


def test_init_heap_data_capacity_one():
    # A capacity-1 request must be bumped up to a capacity-2 (1 level) heap so
    # that pop does not read past the end of the keys array.
    keys, values, xrefs, heap_state = hp.init_heap_data(1, 10)
    assert keys.size == 4
    assert values.size == 2
    assert xrefs.size == 11
    assert (keys == np.inf).all()
    assert (values == -1).all()
    assert (xrefs == -1).all()
    assert heap_state[0]["levels"] == 1
    assert heap_state[0]["count"] == 0


def test_capacity_one_heap_push_pop_single():
    # A single push then pop on a capacity-1 heap must return the pushed
    # key/value without an out-of-bounds read.
    keys, values, xrefs, heap_state = hp.init_heap_data(1, 10)
    keys, values, xrefs, heap_state, flag = hp.push(
        keys, values, xrefs, heap_state, 3.5, 7
    )
    assert flag == 1
    assert heap_state[0]["count"] == 1
    keys, values, xrefs, heap_state, key, value = hp.pop(
        keys, values, xrefs, heap_state
    )
    assert key == 3.5
    assert value == 7
    assert heap_state[0]["count"] == 0


def test_capacity_one_heap_push_pop_sorted():
    # A capacity-1 heap must still push/pop several items in sorted order,
    # resizing itself as needed.
    keys, values, xrefs, heap_state = hp.init_heap_data(1, 10)
    items = [(5.0, 0), (1.0, 1), (3.0, 2), (2.0, 3), (4.0, 4)]
    for key, value in items:
        keys, values, xrefs, heap_state, _ = hp.push(
            keys, values, xrefs, heap_state, key, value
        )
    popped = []
    while heap_state[0]["count"] > 0:
        keys, values, xrefs, heap_state, key, value = hp.pop(
            keys, values, xrefs, heap_state
        )
        popped.append((key, value))
    assert popped == sorted(items)


# TODO: Test the rest of the funcs
