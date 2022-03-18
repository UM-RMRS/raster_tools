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


# TODO: Test the rest of the funcs
