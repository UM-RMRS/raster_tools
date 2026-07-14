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


# ---------------------------------------------------------------------------
# Helpers for the full heap suite
# ---------------------------------------------------------------------------


def _assert_heap_invariants(keys, values, crossrefs, heap_state):
    """Structural invariants that must hold after every heap operation."""
    levels = int(heap_state[0]["levels"])
    count = int(heap_state[0]["count"])
    n = 1 << levels
    assert keys.size == 2 * n
    assert values.size == n
    lvl_start = n - 1
    # Tournament property on internal nodes. Index 0 is a dead slot that
    # _sift never writes (it stays inf), so it is excluded. Exact == is safe
    # because keys are copied through the heap, never combined arithmetically.
    for p in range(1, n - 1):
        assert keys[p] == min(keys[2 * p + 1], keys[2 * p + 2])
    # Crossrefs of live entries point at their value slots.
    live_values = values[:count]
    for slot, value in enumerate(live_values):
        assert crossrefs[value] == slot
    # Exactly `count` live crossrefs.
    assert int((crossrefs != -1).sum()) == count
    # Leaf keys beyond the live count are inf. (Stale values[count:] are left
    # dirty by design and are NOT checked.)
    assert np.all(np.isinf(keys[lvl_start + count :]))


def _drain(keys, values, crossrefs, heap_state):
    popped = []
    while int(heap_state[0]["count"]) > 0:
        keys, values, crossrefs, heap_state, key, value = hp.pop(
            keys, values, crossrefs, heap_state
        )
        popped.append((key, value))
    return keys, values, crossrefs, heap_state, popped


class _ReferenceHeap:
    """A dict value->key model mirroring the heap's push/push_if_lower flags.

    Popping is checked against the real heap's choice to sidestep tie-break
    ambiguity: ``pop`` takes the heap's popped ``(value, key)`` and asserts the
    key equals the model minimum and the value currently holds it.
    """

    def __init__(self, max_value):
        self.max_value = max_value
        self.data = {}

    def push(self, key, value):
        if not (0 <= value <= self.max_value):
            return -1
        existed = value in self.data
        # push updates the key unconditionally (it can raise or lower it).
        self.data[value] = key
        return 0 if existed else 1

    def push_if_lower(self, key, value):
        if not (0 <= value <= self.max_value):
            return -1
        if value in self.data:
            if key < self.data[value]:
                self.data[value] = key
                return 1
            return 0
        self.data[value] = key
        return 1

    def min_key(self):
        return min(self.data.values())

    def pop(self, value, key):
        assert self.data, "model is empty"
        assert key == self.min_key()
        assert value in self.data
        assert self.data[value] == key
        del self.data[value]

    def __len__(self):
        return len(self.data)


# ---------------------------------------------------------------------------
# push / push_if_lower flag semantics
# ---------------------------------------------------------------------------


def test_push_flags():
    keys, values, xrefs, hs = hp.init_heap_data(10, 100)
    # New value -> flag 1.
    keys, values, xrefs, hs, flag = hp.push(keys, values, xrefs, hs, 5.0, 3)
    assert flag == 1
    # Existing value -> flag 0, and the key is updated unconditionally. Here
    # we RAISE it from 5.0 to 9.0 (verified via the drained pop order below).
    keys, values, xrefs, hs, flag = hp.push(keys, values, xrefs, hs, 9.0, 3)
    assert flag == 0
    # Boundary values 0 and max_value are accepted.
    keys, values, xrefs, hs, flag = hp.push(keys, values, xrefs, hs, 1.0, 0)
    assert flag == 1
    keys, values, xrefs, hs, flag = hp.push(keys, values, xrefs, hs, 2.0, 100)
    assert flag == 1
    _assert_heap_invariants(keys, values, xrefs, hs)
    keys, values, xrefs, hs, popped = _drain(keys, values, xrefs, hs)
    assert popped == [(1.0, 0), (2.0, 100), (9.0, 3)]


@pytest.mark.parametrize("value", [-1, 11, 100])
def test_push_out_of_range(value):
    keys, values, xrefs, hs = hp.init_heap_data(5, 10)
    k0, v0, x0 = keys.copy(), values.copy(), xrefs.copy()
    count0 = int(hs[0]["count"])
    keys, values, xrefs, hs, flag = hp.push(
        keys, values, xrefs, hs, 3.0, value
    )
    assert flag == -1
    assert int(hs[0]["count"]) == count0
    assert np.array_equal(keys, k0)
    assert np.array_equal(values, v0)
    assert np.array_equal(xrefs, x0)


def test_push_if_lower_flags():
    keys, values, xrefs, hs = hp.init_heap_data(10, 100)
    # New -> 1.
    keys, values, xrefs, hs, flag = hp.push_if_lower(
        keys, values, xrefs, hs, 5.0, 7
    )
    assert flag == 1
    # Strictly lower -> 1.
    keys, values, xrefs, hs, flag = hp.push_if_lower(
        keys, values, xrefs, hs, 3.0, 7
    )
    assert flag == 1
    # Equal -> 0.
    keys, values, xrefs, hs, flag = hp.push_if_lower(
        keys, values, xrefs, hs, 3.0, 7
    )
    assert flag == 0
    # Higher -> 0 (key must stay 3.0).
    keys, values, xrefs, hs, flag = hp.push_if_lower(
        keys, values, xrefs, hs, 8.0, 7
    )
    assert flag == 0
    keys, values, xrefs, hs, key, value = hp.pop(keys, values, xrefs, hs)
    assert (key, value) == (3.0, 7)


@pytest.mark.parametrize("value", [-1, 11, 100])
def test_push_if_lower_out_of_range(value):
    keys, values, xrefs, hs = hp.init_heap_data(5, 10)
    k0, v0, x0 = keys.copy(), values.copy(), xrefs.copy()
    count0 = int(hs[0]["count"])
    keys, values, xrefs, hs, flag = hp.push_if_lower(
        keys, values, xrefs, hs, 3.0, value
    )
    assert flag == -1
    assert int(hs[0]["count"]) == count0
    assert np.array_equal(keys, k0)
    assert np.array_equal(values, v0)
    assert np.array_equal(xrefs, x0)


def test_push_updates_key_changes_pop_order():
    keys, values, xrefs, hs = hp.init_heap_data(10, 100)
    items = {0: 5.0, 1: 1.0, 2: 3.0, 3: 2.0, 4: 4.0}
    for value, key in items.items():
        keys, values, xrefs, hs, _ = hp.push(
            keys, values, xrefs, hs, key, value
        )
    # Raise value 1 and lower value 0 via push.
    keys, values, xrefs, hs, _ = hp.push(keys, values, xrefs, hs, 10.0, 1)
    keys, values, xrefs, hs, _ = hp.push(keys, values, xrefs, hs, 0.5, 0)
    items[1] = 10.0
    items[0] = 0.5
    _assert_heap_invariants(keys, values, xrefs, hs)
    keys, values, xrefs, hs, popped = _drain(keys, values, xrefs, hs)
    assert popped == sorted((k, v) for v, k in items.items())


def test_push_if_lower_only_lowers_pop_order():
    keys, values, xrefs, hs = hp.init_heap_data(10, 100)
    items = {0: 5.0, 1: 1.0, 2: 3.0, 3: 2.0, 4: 4.0}
    for value, key in items.items():
        keys, values, xrefs, hs, _ = hp.push_if_lower(
            keys, values, xrefs, hs, key, value
        )
    # Attempt to raise value 1 (ignored); lower value 0 (applied).
    keys, values, xrefs, hs, flag = hp.push_if_lower(
        keys, values, xrefs, hs, 10.0, 1
    )
    assert flag == 0
    keys, values, xrefs, hs, flag = hp.push_if_lower(
        keys, values, xrefs, hs, 0.5, 0
    )
    assert flag == 1
    items[0] = 0.5
    _assert_heap_invariants(keys, values, xrefs, hs)
    keys, values, xrefs, hs, popped = _drain(keys, values, xrefs, hs)
    assert popped == sorted((k, v) for v, k in items.items())


@pytest.mark.parametrize("capacity", [1, 5])
def test_pop_empty_heap(capacity):
    # Characterization: popping a FRESH (never-used) heap returns (inf, -1)
    # because values are initialized to -1. A used-then-drained heap would
    # instead return a stale value, since pop reads values[ii] before the
    # count guard.
    keys, values, xrefs, hs = hp.init_heap_data(capacity, 10)
    keys, values, xrefs, hs, key, value = hp.pop(keys, values, xrefs, hs)
    assert np.isinf(key)
    assert value == -1
    assert int(hs[0]["count"]) == 0


def test_resize_grow_and_shrink():
    keys, values, xrefs, hs = hp.init_heap_data(1, 100)
    assert int(hs[0]["levels"]) == 1
    assert keys.size == 4
    assert values.size == 2
    items = [(float(k), k) for k in range(1, 10)]  # 9 distinct items
    for key, value in items:
        keys, values, xrefs, hs, flag = hp.push(
            keys, values, xrefs, hs, key, value
        )
        assert flag == 1
        _assert_heap_invariants(keys, values, xrefs, hs)
    # Grown from 1 level to 4 levels.
    assert int(hs[0]["levels"]) == 4
    assert keys.size == 32
    assert values.size == 16
    popped = []
    while int(hs[0]["count"]) > 0:
        keys, values, xrefs, hs, key, value = hp.pop(keys, values, xrefs, hs)
        popped.append((key, value))
        _assert_heap_invariants(keys, values, xrefs, hs)
        # Shrink threshold is count < 2**(levels - 2): after the 6th pop the
        # count drops to 3 and the heap shrinks from 4 levels to 3.
        if len(popped) == 6:
            assert int(hs[0]["levels"]) == 3
    assert popped == sorted(items)
    # Fully drained back down to the minimum level and original sizes.
    assert int(hs[0]["levels"]) == int(hs[0]["min_levels"]) == 1
    assert keys.size == 4
    assert values.size == 2


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_heap_matches_reference_model(seed):
    rng = np.random.default_rng(seed)
    max_value = 15
    keys, values, xrefs, hs = hp.init_heap_data(4, max_value)
    model = _ReferenceHeap(max_value)
    for _ in range(300):
        op = int(rng.integers(0, 3))  # 0 push, 1 push_if_lower, 2 pop
        if op == 2:
            keys, values, xrefs, hs, key, value = hp.pop(
                keys, values, xrefs, hs
            )
            if len(model) == 0:
                # Empty-heap pop: only the key and count are defined. The
                # returned value is a stale read and is ignored.
                assert np.isinf(key)
                assert int(hs[0]["count"]) == 0
            else:
                model.pop(value, key)
        else:
            # Small key range forces ties; the value range spans out-of-bounds
            # to exercise the flag -1 path.
            key = float(rng.integers(0, 6))
            value = int(rng.integers(-2, max_value + 3))
            if op == 0:
                keys, values, xrefs, hs, flag = hp.push(
                    keys, values, xrefs, hs, key, value
                )
                assert flag == model.push(key, value)
            else:
                keys, values, xrefs, hs, flag = hp.push_if_lower(
                    keys, values, xrefs, hs, key, value
                )
                assert flag == model.push_if_lower(key, value)
        _assert_heap_invariants(keys, values, xrefs, hs)
        assert int(hs[0]["count"]) == len(model)
