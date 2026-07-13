import unittest

import numpy as np
import pytest
from affine import Affine
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from raster_tools import Raster, distance
from raster_tools.distance.cost_distance import (
    _get_strides,
    cost_distance_analysis_numpy,
)
from tests import testdata
from tests.utils import assert_valid_raster

# Example taken from ESRI docs
SOURCES = np.array(
    [
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0],
    ]
)
SOURCES_IDXS = np.argwhere(SOURCES != 0)
COST_SURF = np.array(
    [
        [1, 3, 4, 4, 3, 2],
        [7, 3, 2, 6, 4, 6],
        [5, 8, 7, 5, 6, 6],
        [1, 4, 5, -1, 5, 1],
        [4, 7, 5, -1, 2, 6],
        [1, 2, 2, 1, 3, 4],
    ]
)
# Cost dist truths
CD_TRUTH_SCALE_1 = np.array(
    [
        [
            [2.0, 0.0, 0.0, 4.0, 7.5, 10.0],
            [6.0, 2.5, 0.0, 4.0, 9.0, 13.86396103],
            [8.0, 7.07106781, 4.5, 4.94974747, 10.44974747, 12.74264069],
            [5.0, 7.5, 10.5, -1.0, 10.62132034, 9.24264069],
            [2.5, 5.65685425, 6.44974747, -1.0, 7.12132034, 11.12132034],
            [0.0, 1.5, 3.5, 5.0, 7.0, 10.5],
        ]
    ]
)
CD_TRUTH_SCALE_5 = np.array(
    [
        [
            [10.0, 0.0, 0.0, 20.0, 37.5, 50.0],
            [30.0, 12.5, 0.0, 20.0, 45.0, 69.31980515],
            [40.0, 35.35533906, 22.5, 24.74873734, 52.24873734, 63.71320344],
            [25.0, 37.5, 52.5, -1.0, 53.10660172, 46.21320344],
            [12.5, 28.28427125, 32.24873734, -1.0, 35.60660172, 55.60660172],
            [0.0, 7.5, 17.5, 25.0, 35.0, 52.5],
        ]
    ]
)
# Traceback Truths
TR_TRUTH_SCALE_1 = np.array(
    [
        [
            [1, 0, 0, 5, 5, 5],
            [7, 1, 0, 5, 5, 6],
            [3, 8, 7, 6, 5, 3],
            [3, 5, 7, -1, 3, 4],
            [3, 4, 4, -1, 4, 5],
            [0, 5, 5, 5, 5, 5],
        ]
    ],
)
# Allocation truths
AL_TRUTH_SCALE_1 = np.array(
    [
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [2, 1, 1, 1, 1, 2],
            [2, 2, 1, 0, 2, 2],
            [2, 2, 2, 0, 2, 2],
            [2, 2, 2, 2, 2, 2],
        ]
    ]
)
AL_TRUTH_IDXS_SCALE_1 = np.array(
    [
        [
            [0, 0, 1, 1, 1, 1],
            [0, 2, 2, 2, 2, 1],
            [3, 2, 2, 2, 2, 3],
            [3, 3, 2, -1, 3, 3],
            [3, 3, 3, -1, 3, 3],
            [3, 3, 3, 3, 3, 3],
        ]
    ]
)


class TestCostDist(unittest.TestCase):
    def setUp(self):
        self.cs = Raster(COST_SURF).set_null_value(-1)
        self.srcs = Raster(SOURCES).set_null_value(0)
        self.srcs_idx = SOURCES_IDXS

    def test_cost_distance_analysis(self):
        # ** Using srcs raster **
        cost_dist, traceback, allocation = distance.cost_distance_analysis(
            self.cs, self.srcs
        )
        assert_valid_raster(cost_dist)
        assert_valid_raster(traceback)
        assert_valid_raster(allocation)

        # Cost dist
        self.assertTrue(
            np.allclose(cost_dist.to_numpy(), CD_TRUTH_SCALE_1, equal_nan=True)
        )
        self.assertTrue(cost_dist._masked)
        self.assertTrue(cost_dist.dtype == np.dtype(np.float64))
        self.assertTrue(cost_dist.null_value == -1)
        # traceback
        self.assertTrue(
            np.allclose(traceback.to_numpy(), TR_TRUTH_SCALE_1, equal_nan=True)
        )
        self.assertTrue(traceback._masked)
        self.assertTrue(traceback.dtype == np.dtype(np.int8))
        self.assertTrue(traceback.null_value == -1)
        # Allocation
        self.assertTrue(
            np.allclose(
                allocation.to_numpy(), AL_TRUTH_SCALE_1, equal_nan=True
            )
        )
        self.assertTrue(allocation._masked)
        self.assertTrue(allocation.dtype == np.dtype(np.int64))
        self.assertTrue(allocation.null_value == self.srcs.null_value)

        # ** Using srcs indices **
        cost_dist, traceback, allocation = distance.cost_distance_analysis(
            self.cs, self.srcs_idx
        )
        assert_valid_raster(cost_dist)
        assert_valid_raster(traceback)
        assert_valid_raster(allocation)

        # Cost dist
        self.assertTrue(
            np.allclose(cost_dist.to_numpy(), CD_TRUTH_SCALE_1, equal_nan=True)
        )
        self.assertTrue(cost_dist._masked)
        self.assertTrue(cost_dist.dtype == np.dtype(np.float64))
        self.assertTrue(cost_dist.null_value == -1)
        # traceback
        self.assertTrue(
            np.allclose(traceback.to_numpy(), TR_TRUTH_SCALE_1, equal_nan=True)
        )
        self.assertTrue(traceback._masked)
        self.assertTrue(traceback.dtype == np.dtype(np.int8))
        self.assertTrue(traceback.null_value == -1)
        # Allocation
        self.assertTrue(
            np.allclose(
                allocation.to_numpy(), AL_TRUTH_IDXS_SCALE_1, equal_nan=True
            )
        )
        self.assertTrue(allocation._masked)
        self.assertTrue(allocation.dtype == np.dtype(np.int64))
        self.assertTrue(allocation.null_value == -1)

    def test_cost_distance_analysis_errors(self):
        with self.assertRaises(ValueError):
            # Must be single band
            distance.cost_distance_analysis(
                "tests/data/raster/multiband_small.tif", self.srcs
            )
        with self.assertRaises(ValueError):
            # Must have same shape
            distance.cost_distance_analysis(
                self.cs, "tests/data/raster/dem_small.tif"
            )
        with self.assertRaises(TypeError):
            # source raster must be int
            distance.cost_distance_analysis(
                "tests/data/raster/dem_small.tif",
                "tests/data/raster/dem_small.tif",
            )
        with self.assertRaises(ValueError):
            # source raster must have null value
            distance.cost_distance_analysis(
                "tests/data/raster/dem_small.tif",
                testdata.raster.dem_small.astype(int, False).set_null_value(
                    None
                ),
            )
        with self.assertRaises(ValueError):
            # sources array must have shape (M, 2)
            distance.cost_distance_analysis(
                self.cs, np.zeros((5, 3), dtype=int)
            )
        with self.assertRaises(ValueError):
            # sources array must not have duplicates
            distance.cost_distance_analysis(
                self.cs, np.zeros((5, 2), dtype=int)
            )

    def test_cost_distance_analysis_scale(self):
        af = self.cs.affine
        coefs = [5, af.b, af.c, af.d, 5, af.f]
        trans = Affine(*coefs)
        cs = self.cs.copy()
        ds = self.cs._ds.rio.write_transform(trans)
        ds["x"] = ds.x * 5
        ds["y"] = ds.y * 5
        cs._ds = ds
        cost_dist, traceback, allocation = distance.cost_distance_analysis(
            cs, self.srcs
        )
        assert_valid_raster(cost_dist)
        assert_valid_raster(traceback)
        assert_valid_raster(allocation)

        self.assertTrue(
            np.allclose(cost_dist.to_numpy(), CD_TRUTH_SCALE_5, equal_nan=True)
        )


def test_cost_distance_sources_float_error_message():
    dem_path = "tests/data/raster/dem_small.tif"
    with pytest.raises(TypeError) as exc:
        distance.cost_distance_analysis(dem_path, dem_path)
    msg = str(exc.value)
    assert "integer type" in msg
    assert str(Raster(dem_path).dtype) in msg
    assert "GeoTIFF" in msg
    assert "astype" in msg


def test_cost_distance_analysis_crs():
    rs = testdata.raster.dem_small
    srcs = np.array([[1, 1], [20, 30]])
    cd, tr, al = distance.cost_distance_analysis(rs, srcs)
    assert_valid_raster(cd)
    assert_valid_raster(tr)
    assert_valid_raster(al)
    assert cd.crs == rs.crs
    assert tr.crs == rs.crs
    assert al.crs == rs.crs


# ---------------------------------------------------------------------------
# Regression tests for findings F1, F3, F5, and F5b
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(6, 6), (3, 5), (7, 13), (2, 3, 4), (5, 6, 7)]
)
def test_get_strides_matches_numpy(shape):
    # F1: _mult_accumulate must build a true cumulative product so that
    # _get_strides returns numpy's own C-contiguous element strides. Covers
    # both the 2D and 3D declared signatures.
    itemsize = np.dtype(np.int64).itemsize
    expected = tuple(
        s // itemsize for s in np.empty(shape, dtype=np.int64).strides
    )
    assert tuple(_get_strides(shape)) == expected


def test_cost_distance_1x1_with_source():
    # F3: a 1x1 grid whose single cell is the source. The heap capacity clamp
    # (max(size - 1, 1)) must let this run and return a trivial solution.
    costs = np.array([[3.0]])
    sources = np.array([[7]], dtype=np.int64)
    cd, tb, al = cost_distance_analysis_numpy(costs, sources, -1)
    assert np.array_equal(cd, np.array([[0.0]]))
    assert np.array_equal(tb, np.array([[-1]]))
    assert np.array_equal(al, np.array([[7]]))


def test_cost_distance_1x1_no_source():
    # F3: a 1x1 grid with no source (all-null sources) must not raise and
    # should report the cell as unreached.
    costs = np.array([[3.0]])
    sources = np.array([[-1]], dtype=np.int64)
    cd, tb, al = cost_distance_analysis_numpy(costs, sources, -1)
    assert np.isinf(cd[0, 0])
    assert tb[0, 0] == -2
    assert al[0, 0] == -1


def test_source_on_barrier_cost_does_not_propagate():
    # F5b: a source placed on a barrier-cost (-1) cell must not propagate. The
    # buggy version used the -1 fill as the popped cell's own cost, producing
    # edge weights L * 0.5 * (-1 + new_cost) that were zero or negative. Here
    # the neighbors have cost 0.5, so the buggy weight was -0.25 * L < 0.
    costs = np.full((3, 3), 0.5)
    costs[1, 1] = -1.0
    sources = np.full((3, 3), -1, dtype=np.int64)
    sources[1, 1] = 0
    cd, tb, al = cost_distance_analysis_numpy(costs, sources, -1)
    # Only the source cell is reached; neighbors stay unreached.
    assert cd[1, 1] == 0.0
    neighbors = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ]
    for r, c in neighbors:
        assert np.isinf(cd[r, c])
    # No negative cost distances anywhere.
    assert not (cd[np.isfinite(cd)] < 0).any()


def test_barrier_source_alongside_valid_source_no_negatives():
    # F5b: when a barrier-cost source coexists with a valid source, the valid
    # source drives propagation and no negative cost distances appear.
    costs = np.full((3, 3), 1.0)
    costs[0, 0] = -1.0
    sources = np.full((3, 3), -1, dtype=np.int64)
    sources[0, 0] = 0  # source sitting on the barrier cell
    sources[2, 2] = 1  # valid source
    cd, tb, al = cost_distance_analysis_numpy(costs, sources, -1)
    finite = cd[np.isfinite(cd)]
    assert not (finite < 0).any()
    # The barrier source keeps its own zero cumcost but does not seed
    # neighbors; those are reached from the valid source instead.
    assert cd[0, 0] == 0.0
    assert al[0, 1] == 1
    assert al[1, 0] == 1


def test_source_on_null_elevation_does_not_propagate():
    # F5: with elevation, a source on a null-elevation cell must not
    # propagate. The buggy version used the elevation null fill in dz,
    # inflating outgoing edge lengths instead of skipping the cell.
    costs = np.full((3, 3), 1.0)
    elevation = np.full((3, 3), 10.0)
    elev_null = -9999.0
    elevation[1, 1] = elev_null
    sources = np.full((3, 3), -1, dtype=np.int64)
    sources[1, 1] = 0
    cd, tb, al = cost_distance_analysis_numpy(
        costs,
        sources,
        -1,
        elevation=elevation,
        elevation_null_value=elev_null,
    )
    assert cd[1, 1] == 0.0
    neighbors = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ]
    for r, c in neighbors:
        assert np.isinf(cd[r, c])
    assert not (cd[np.isfinite(cd)] < 0).any()


def test_cost_distance_source_on_masked_cost_no_negatives():
    # F5b at the Raster level: a source located on a masked (null) cost cell
    # must not yield negative cost distances in the unmasked output.
    costs_arr = np.full((5, 5), 1.0)
    costs_arr[2, 2] = -1.0  # becomes a masked / barrier cell
    src_arr = np.zeros((5, 5), dtype=np.int64)
    src_arr[2, 2] = 5  # source on the masked cost cell
    src_arr[0, 0] = 1  # a valid source
    costs = Raster(costs_arr).set_null_value(-1)
    srcs = Raster(src_arr).set_null_value(0)
    cd, tb, al = distance.cost_distance_analysis(costs, srcs)
    # The masked cost cell is masked out of the output (it holds the source's
    # own zero cumcost), so only the unmasked cells are inspected here.
    data = cd.to_numpy()
    mask = cd.mask.compute()
    valid = data[~mask]
    assert not (valid < 0).any()


# ---------------------------------------------------------------------------
# scipy-oracle property test for cost_distance_analysis_numpy
# ---------------------------------------------------------------------------

# Move order matching _MOVES in cost_distance.py: (drow, dcol).
_ORACLE_MOVES = [
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
]


def _oracle_cost_distance(costs, sources, scaling, elevation, elev_null):
    """Ground-truth cost distance via scipy Dijkstra on the grid graph.

    Nodes are non-barrier (and, with elevation, non-null) cells. The
    undirected edge weight between neighbors i and j is
    ``L * 0.5 * (cost_i + cost_j)`` where ``L`` is the scaled, optionally
    elevation-aware, move length.
    """
    rows, cols = costs.shape
    if np.isscalar(scaling):
        sr = sc = float(scaling)
    else:
        sr, sc = float(scaling[0]), float(scaling[1])
    n = rows * cols

    def valid(r, c):
        if costs[r, c] < 0 or not np.isfinite(costs[r, c]):
            return False
        return not (elevation is not None and elevation[r, c] == elev_null)

    edge_rows, edge_cols, data = [], [], []
    for r in range(rows):
        for c in range(cols):
            if not valid(r, c):
                continue
            i = r * cols + c
            for dr, dc in _ORACLE_MOVES:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if not valid(nr, nc):
                    continue
                planar_sq = (sr * dr) ** 2 + (sc * dc) ** 2
                if elevation is not None:
                    dz = elevation[nr, nc] - elevation[r, c]
                    length = np.sqrt(planar_sq + dz * dz)
                else:
                    length = np.sqrt(planar_sq)
                edge_rows.append(i)
                edge_cols.append(nr * cols + nc)
                data.append(length * 0.5 * (costs[r, c] + costs[nr, nc]))
    graph = coo_matrix((data, (edge_rows, edge_cols)), shape=(n, n)).tocsr()
    src_cells = np.flatnonzero(sources.ravel() != -1)
    dist = dijkstra(graph, directed=False, indices=src_cells, min_only=True)
    return dist.reshape(rows, cols)


def _random_case(seed, barrier_frac, use_elevation, n_sources):
    rng = np.random.default_rng(seed)
    rows = int(rng.integers(12, 17))
    cols = int(rng.integers(12, 17))
    costs = rng.uniform(0.5, 10.0, (rows, cols))
    if barrier_frac > 0:
        costs[rng.random((rows, cols)) < barrier_frac] = -1.0
    elevation = None
    elev_null = -9999.0
    if use_elevation:
        elevation = rng.uniform(0.0, 50.0, (rows, cols))
        elevation[rng.random((rows, cols)) < 0.1] = elev_null
    valid_mask = costs >= 0
    if use_elevation:
        valid_mask &= elevation != elev_null
    valid_cells = np.argwhere(valid_mask)
    rng.shuffle(valid_cells)
    sources = np.full((rows, cols), -1, dtype=np.int64)
    for k in range(min(n_sources, len(valid_cells))):
        r, c = valid_cells[k]
        sources[r, c] = k
    return costs, sources, elevation, elev_null


# category -> (barrier_frac, scaling, use_elevation, n_sources)
_ORACLE_CATEGORIES = {
    "no_barriers_isotropic": (0.0, 1.0, False, 1),
    "barriers_isotropic": (0.15, 1.0, False, 1),
    "barriers_anisotropic": (0.15, (2.0, 0.5), False, 1),
    "multisource_barriers": (0.15, 1.0, False, 4),
    "elevation_isotropic": (0.15, 1.0, True, 1),
    "elevation_anisotropic": (0.15, (2.0, 0.5), True, 2),
}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("category", list(_ORACLE_CATEGORIES))
def test_cost_distance_matches_scipy_oracle(category, seed):
    barrier_frac, scaling, use_elevation, n_sources = _ORACLE_CATEGORIES[
        category
    ]
    costs, sources, elevation, elev_null = _random_case(
        seed, barrier_frac, use_elevation, n_sources
    )
    cd, _, _ = cost_distance_analysis_numpy(
        costs,
        sources,
        -1,
        elevation=elevation,
        elevation_null_value=(elev_null if use_elevation else 0),
        scaling=scaling,
    )
    ref = _oracle_cost_distance(costs, sources, scaling, elevation, elev_null)
    reached = np.isfinite(cd)
    assert np.array_equal(reached, np.isfinite(ref))
    assert np.allclose(cd[reached], ref[reached], rtol=1e-12, atol=0.0)
