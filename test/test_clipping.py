from unittest import TestCase

import dask
import numpy as np

from raster_tools import clipping
from raster_tools.raster import Raster
from raster_tools.vector import open_vectors


class TestClipping(TestCase):
    def setUp(self):
        self.dem = Raster("test/data/elevation.tif")
        self.pods = open_vectors("test/data/vector/pods.shp")
        self.v10 = self.pods[10]
        self.v10_bounds = dask.compute(self.v10.to_crs(self.dem.crs).bounds)[0]

    def test_core_clip_out_dtype(self):
        result = clipping._clip(self.pods, self.dem)
        self.assertTrue(result.dtype == self.dem.dtype)
        self.assertTrue(result.eval().dtype == self.dem.dtype)

    def test_clip(self):
        res = clipping.clip(self.v10, self.dem)
        truth = Raster("test/data/clipping_clip_pods_10.tif")
        self.assertTrue(np.allclose(res, truth))

        res = clipping.clip(self.v10, self.dem, bounds=self.v10_bounds)
        self.assertTrue(np.allclose(res, truth))

    def test_erase(self):
        res = clipping.erase(self.v10, self.dem)
        truth = Raster("test/data/clipping_erase_pods_10.tif")
        self.assertTrue(np.allclose(res, truth))

        res = clipping.erase(self.v10, self.dem, bounds=self.v10_bounds)
        self.assertTrue(np.allclose(res, truth))

    def test_mask(self):
        res = clipping.mask(self.v10, self.dem)
        truth = Raster("test/data/clipping_mask_pods_10.tif")
        self.assertTrue(np.allclose(res, truth))

        res = clipping.mask(self.v10, self.dem, invert=True)
        truth = Raster("test/data/clipping_mask_inverted_pods_10.tif")
        self.assertTrue(np.allclose(res, truth))

    def test_envelope(self):
        res = clipping.envelope(self.v10, self.dem)
        truth = Raster("test/data/clipping_envelope_pods_10.tif")
        self.assertTrue(np.allclose(res, truth))

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
