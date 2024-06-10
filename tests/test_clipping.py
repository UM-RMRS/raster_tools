from unittest import TestCase

import dask
import numpy as np

import raster_tools as rts
from raster_tools import clipping
from raster_tools.exceptions import RasterNoDataError
from raster_tools.raster import Raster
from tests import testdata
from tests.utils import assert_rasters_similar, assert_valid_raster


class TestClipping(TestCase):
    def setUp(self):
        self.dem = testdata.raster.dem
        self.pods = testdata.vector.pods
        self.v10 = self.pods[10]
        self.v10_bounds = dask.compute(self.v10.to_crs(self.dem.crs).bounds)[0]

    def test_core_clip_out_dtype(self):
        result = clipping._clip(self.pods, self.dem)
        assert_valid_raster(result)
        self.assertTrue(result.dtype == self.dem.dtype)
        self.assertTrue(result.eval().dtype == self.dem.dtype)

    def test_clip(self):
        res = clipping.clip(self.v10, self.dem)
        truth = testdata.raster.clipping_clip_pods_10
        assert_valid_raster(res)
        self.assertTrue(np.allclose(res, truth))
        assert res.crs == self.dem.crs

        res = clipping.clip(self.v10, self.dem, bounds=self.v10_bounds)
        self.assertTrue(np.allclose(res, truth))
        assert res.crs == self.dem.crs

    def test_erase(self):
        res = clipping.erase(self.v10, self.dem)
        truth = testdata.raster.clipping_erase_pods_10
        assert_valid_raster(res)
        self.assertTrue(np.allclose(res, truth))
        assert res.crs == self.dem.crs

        res = clipping.erase(self.v10, self.dem, bounds=self.v10_bounds)
        self.assertTrue(np.allclose(res, truth))
        assert res.crs == self.dem.crs

    def test_mask(self):
        res = clipping.mask(self.v10, self.dem)
        truth = testdata.raster.clipping_mask_pods_10
        assert_valid_raster(res)
        self.assertTrue(np.allclose(res, truth))
        assert res.crs == self.dem.crs

        res = clipping.mask(self.v10, self.dem, invert=True)
        truth = testdata.raster.clipping_mask_inverted_pods_10
        assert_valid_raster(res)
        self.assertTrue(np.allclose(res, truth))
        assert res.crs == self.dem.crs

    def test_envelope(self):
        res = clipping.envelope(self.v10, self.dem)
        truth = testdata.raster.clipping_envelope_pods_10
        assert_valid_raster(res)
        self.assertTrue(np.allclose(res, truth))
        assert res.crs == self.dem.crs

    def test_errors(self):
        with self.assertRaises(ValueError):
            rs = Raster(np.ones((4, 4)))
            clipping.clip(self.v10, rs)

        with self.assertRaises(ValueError):
            clipping._clip(self.v10, self.dem, bounds=(0, 3))

        with self.assertRaises(ValueError):
            clipping._clip(self.v10, self.dem, invert=True, envelope=True)

        with self.assertRaises(RuntimeError):
            clipping._clip(self.v10, self.dem, bounds=self.v10.bounds)

    def test_clip_box(self):
        self.dem = testdata.raster.dem
        rs_clipped = testdata.raster.dem_small
        bounds = [
            rs_clipped.xdata.x.min().item(),
            rs_clipped.xdata.y.min().item(),
            rs_clipped.xdata.x.max().item(),
            rs_clipped.xdata.y.max().item(),
        ]
        test = clipping.clip_box(self.dem, bounds)
        self.assertTrue(test.shape == rs_clipped.shape)
        self.assertTrue(np.allclose(test.to_numpy(), rs_clipped.to_numpy()))

        # Test that the mask is also clipped
        x = np.arange(25).reshape((1, 5, 5))
        x[x < 12] = 0
        rs = Raster(x).set_null_value(0).set_crs("epsg:3857")
        self.assertTrue(np.allclose(x == 0, rs.mask))
        rs_clipped = clipping.clip_box(rs, (1, 1, 4, 4))
        mask_truth = np.array([[[1, 1, 1], [1, 0, 0], [0, 0, 0]]], dtype=bool)
        self.assertTrue(np.allclose(rs_clipped.mask, mask_truth))

    def test_clip_out_of_bounds(self):
        with self.assertRaises(RasterNoDataError):
            clipping.clip_box(self.dem, (9e6, 9e6, 10e6, 10e6))


def test_clip_multiband():
    raster = testdata.raster.dem_small
    raster2 = rts.band_concat([raster, raster])
    vector = testdata.vector.pods_small
    expected = rts.band_concat([clipping.clip(vector, raster)] * 2)

    result = clipping.clip(vector, raster2)
    assert_valid_raster(result)
    assert_rasters_similar(result, expected)
    assert np.allclose(result, expected)
