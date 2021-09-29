import numpy as np
import unittest

from raster_tools import costdist, Raster


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
            [5.0, 7.5, 10.5, np.nan, 10.62132034, 9.24264069],
            [2.5, 5.65685425, 6.44974747, np.nan, 7.12132034, 11.12132034],
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
            [25.0, 37.5, 52.5, np.nan, 53.10660172, 46.21320344],
            [12.5, 28.28427125, 32.24873734, np.nan, 35.60660172, 55.60660172],
            [0.0, 7.5, 17.5, 25.0, 35.0, 52.5],
        ]
    ]
)
# Traceback Truths
TR_TRUTH_SCALE_1 = np.array(
    [
        [
            [1.0, 0.0, 0.0, 5.0, 5.0, 5.0],
            [7.0, 1.0, 0.0, 5.0, 5.0, 6.0],
            [3.0, 8.0, 7.0, 6.0, 5.0, 3.0],
            [3.0, 5.0, 7.0, np.nan, 3.0, 4.0],
            [3.0, 4.0, 4.0, np.nan, 4.0, 5.0],
            [0.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        ]
    ],
)
# Allocation truths
AL_TRUTH_SCALE_1 = np.array(
    [
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0, 1.0, 1.0, 2.0],
            [2.0, 2.0, 1.0, np.nan, 2.0, 2.0],
            [2.0, 2.0, 2.0, np.nan, 2.0, 2.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ]
    ]
)
AL_TRUTH_IDXS_SCALE_1 = np.array(
    [
        [
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 2.0, 2.0, 2.0, 2.0, 1.0],
            [3.0, 2.0, 2.0, 2.0, 2.0, 3.0],
            [3.0, 3.0, 2.0, np.nan, 3.0, 3.0],
            [3.0, 3.0, 3.0, np.nan, 3.0, 3.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        ]
    ]
)


class TestCostDist(unittest.TestCase):
    def setUp(self):
        self.cs = Raster(COST_SURF).set_null_value(-1)
        self.srcs = Raster(SOURCES).set_null_value(0)
        self.srcs_idx = SOURCES_IDXS

    def test_cost_distance_analysis(self):
        ## Using srcs raster ##
        cost_dist, traceback, allocation = costdist.cost_distance_analysis(
            self.cs, self.srcs
        )

        # Cost dist
        self.assertTrue(
            np.allclose(
                cost_dist.to_dask().compute(),
                CD_TRUTH_SCALE_1,
                equal_nan=True,
            ),
        )
        self.assertTrue(cost_dist.encoding.masked)
        self.assertTrue(cost_dist.encoding.dtype == np.dtype(np.float64))
        self.assertTrue(cost_dist.encoding.null_value == -1)
        # traceback
        self.assertTrue(
            np.allclose(
                traceback.to_dask().compute(),
                TR_TRUTH_SCALE_1,
                equal_nan=True,
            ),
        )
        self.assertTrue(traceback.encoding.masked)
        self.assertTrue(traceback.encoding.dtype == np.dtype(np.int8))
        self.assertTrue(traceback.encoding.null_value == -1)
        # Allocation
        self.assertTrue(
            np.allclose(
                allocation.to_dask().compute(),
                AL_TRUTH_SCALE_1,
                equal_nan=True,
            ),
        )
        self.assertTrue(allocation.encoding.masked)
        self.assertTrue(allocation.encoding.dtype == np.dtype(np.int64))
        self.assertTrue(
            allocation.encoding.null_value == self.srcs.encoding.null_value
        )

        ## Using srcs indices ##
        cost_dist, traceback, allocation = costdist.cost_distance_analysis(
            self.cs, self.srcs_idx
        )

        # Cost dist
        self.assertTrue(
            np.allclose(
                cost_dist.to_dask().compute(),
                CD_TRUTH_SCALE_1,
                equal_nan=True,
            ),
        )
        self.assertTrue(cost_dist.encoding.masked)
        self.assertTrue(cost_dist.encoding.dtype == np.dtype(np.float64))
        self.assertTrue(cost_dist.encoding.null_value == -1)
        # traceback
        self.assertTrue(
            np.allclose(
                traceback.to_dask().compute(),
                TR_TRUTH_SCALE_1,
                equal_nan=True,
            ),
        )
        self.assertTrue(traceback.encoding.masked)
        self.assertTrue(traceback.encoding.dtype == np.dtype(np.int8))
        self.assertTrue(traceback.encoding.null_value == -1)
        # Allocation
        self.assertTrue(
            np.allclose(
                allocation.to_dask().compute(),
                AL_TRUTH_IDXS_SCALE_1,
                equal_nan=True,
            ),
        )
        self.assertTrue(allocation.encoding.masked)
        self.assertTrue(allocation.encoding.dtype == np.dtype(np.int64))
        self.assertTrue(allocation.encoding.null_value == -1)

    def test_cost_distance_analysis_errors(self):
        with self.assertRaises(ValueError):
            # Must be single band
            costdist.cost_distance_analysis(
                "test/data/multiband_small.tif", self.srcs
            )
        with self.assertRaises(ValueError):
            # Must have same shape
            costdist.cost_distance_analysis(
                self.cs, "test/data/elevation_small.tif"
            )
        with self.assertRaises(TypeError):
            # source raster must be int
            costdist.cost_distance_analysis(
                "test/data/elevation_small.tif",
                "test/data/elevation_small.tif",
            )
        with self.assertRaises(ValueError):
            # source raster must have null value
            costdist.cost_distance_analysis(
                "test/data/elevation_small.tif",
                Raster("test/data/elevation_small.tif").astype(np.int64),
            )
        with self.assertRaises(ValueError):
            # sources array must have shape (M, 2)
            costdist.cost_distance_analysis(
                self.cs, np.zeros((5, 3), dtype=int)
            )
        with self.assertRaises(ValueError):
            # sources array must not have duplicates
            costdist.cost_distance_analysis(
                self.cs, np.zeros((5, 2), dtype=int)
            )

    def test_cost_distance_analysis_scale(self):
        self.cs._rs.attrs["res"] = (5, 5)
        cost_dist, traceback, allocation = costdist.cost_distance_analysis(
            self.cs, self.srcs
        )

        self.assertTrue(
            np.allclose(
                cost_dist.to_dask().compute(),
                CD_TRUTH_SCALE_5,
                equal_nan=True,
            ),
        )


class TestCostDistAttrsPropagation(unittest.TestCase):
    def test_costdist_attrs(self):
        rs = Raster("test/data/elevation_small.tif")
        srcs = np.array([[1, 1], [20, 30]])
        attrs = rs._attrs
        cd, tr, al = costdist.cost_distance_analysis(rs, srcs)
        self.assertEqual(cd._attrs, attrs)
        # Null values may not match costs raster for traceback and allocation
        attrs.pop("_FillValue")
        tr._rs.attrs.pop("_FillValue")
        self.assertEqual(tr._attrs, attrs)
        al._rs.attrs.pop("_FillValue")
        self.assertEqual(al._attrs, attrs)
