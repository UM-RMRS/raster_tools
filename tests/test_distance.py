import unittest

import numpy as np
from affine import Affine

from raster_tools import Raster, distance
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
            np.allclose(cost_dist.values, CD_TRUTH_SCALE_1, equal_nan=True)
        )
        self.assertTrue(cost_dist._masked)
        self.assertTrue(cost_dist.dtype == np.dtype(np.float64))
        self.assertTrue(cost_dist.null_value == -1)
        # traceback
        self.assertTrue(
            np.allclose(traceback.values, TR_TRUTH_SCALE_1, equal_nan=True)
        )
        self.assertTrue(traceback._masked)
        self.assertTrue(traceback.dtype == np.dtype(np.int8))
        self.assertTrue(traceback.null_value == -1)
        # Allocation
        self.assertTrue(
            np.allclose(allocation.values, AL_TRUTH_SCALE_1, equal_nan=True)
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
            np.allclose(cost_dist.values, CD_TRUTH_SCALE_1, equal_nan=True)
        )
        self.assertTrue(cost_dist._masked)
        self.assertTrue(cost_dist.dtype == np.dtype(np.float64))
        self.assertTrue(cost_dist.null_value == -1)
        # traceback
        self.assertTrue(
            np.allclose(traceback.values, TR_TRUTH_SCALE_1, equal_nan=True)
        )
        self.assertTrue(traceback._masked)
        self.assertTrue(traceback.dtype == np.dtype(np.int8))
        self.assertTrue(traceback.null_value == -1)
        # Allocation
        self.assertTrue(
            np.allclose(
                allocation.values, AL_TRUTH_IDXS_SCALE_1, equal_nan=True
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
                self.cs, "tests/data/raster/elevation_small.tif"
            )
        with self.assertRaises(TypeError):
            # source raster must be int
            distance.cost_distance_analysis(
                "tests/data/raster/elevation_small.tif",
                "tests/data/raster/elevation_small.tif",
            )
        with self.assertRaises(ValueError):
            # source raster must have null value
            distance.cost_distance_analysis(
                "tests/data/raster/elevation_small.tif",
                Raster("tests/data/raster/elevation_small.tif")
                .astype(int, False)
                .set_null_value(None),
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
            np.allclose(cost_dist.values, CD_TRUTH_SCALE_5, equal_nan=True)
        )


def test_cost_distance_analysis_crs():
    rs = Raster("tests/data/raster/elevation_small.tif")
    srcs = np.array([[1, 1], [20, 30]])
    cd, tr, al = distance.cost_distance_analysis(rs, srcs)
    assert_valid_raster(cd)
    assert_valid_raster(tr)
    assert_valid_raster(al)
    assert cd.crs == rs.crs
    assert tr.crs == rs.crs
    assert al.crs == rs.crs
