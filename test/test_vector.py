import dask
import geopandas as gpd
import numpy as np
import rasterio as rio
import unittest
import xarray as xr

from raster_tools import Raster
from raster_tools.vector import open_vectors, Vector


class TestOpenVectors(unittest.TestCase):
    def test_open_vectors(self):
        vs = open_vectors("test/data/vector/Zones.gdb")

        self.assertIsInstance(vs, list)
        self.assertTrue(len(vs) == 2)
        self.assertIsInstance(vs[0], Vector)
        self.assertIsInstance(vs[1], Vector)
        self.assertTrue(len(vs[0]) == 10642)
        self.assertTrue(len(vs[1]) == 184)

        pods1 = open_vectors("test/data/vector/Zones.gdb", layers="PODs")
        self.assertIsInstance(pods1, Vector)
        pods2 = open_vectors("test/data/vector/Zones.gdb", layers=1)
        pods3 = open_vectors("test/data/vector/Zones.gdb", layers=["PODs"])
        pods4 = open_vectors("test/data/vector/Zones.gdb", layers=[1])
        for v in [pods2, pods3, pods4]:
            self.assertTrue(pods1.data.equals(v.data))

    def test_open_vectors_errors(self):
        with self.assertRaises(ValueError):
            open_vectors("test/data/vector/Zones.gdb", layers="dummy")
        with self.assertRaises(ValueError):
            open_vectors("test/data/vector/Zones.gdb", layers=2)
        with self.assertRaises(ValueError):
            open_vectors("test/data/vector/Zones.gdb", layers=-1)
        with self.assertRaises(ValueError):
            open_vectors("test/data/vector/Zones.gdb", layers=[-1])

        with self.assertRaises(TypeError):
            open_vectors("test/data/vector/Zones.gdb", layers=[0, "PODs"])
        with self.assertRaises(TypeError):
            open_vectors("test/data/vector/Zones.gdb", layers={})


class TestVectorProperties(unittest.TestCase):
    def setUp(self):
        self.v = open_vectors("test/data/vector/pods.shp")

    def test_table(self):
        self.assertTrue(hasattr(self.v, "table"))
        self.assertIsInstance(self.v.table, gpd.GeoDataFrame)
        self.assertTrue(len(self.v) == len(self.v.table))
        # Table doesn't contain geometry. It is only vector attributes
        self.assertTrue("geometry" not in self.v.table.columns)
        self.assertTrue(
            self.v.table.columns.size == (self.v.data.columns.size - 1)
        )

    def test_size(self):
        self.assertTrue(hasattr(self.v, "size"))
        self.assertTrue(self.v.size == 184)

    def test_shape(self):
        self.assertTrue(hasattr(self.v, "shape"))
        self.assertIsInstance(self.v.shape, tuple)
        self.assertTrue(self.v.shape == self.v.table.shape)

    def test_crs(self):
        self.assertTrue(hasattr(self.v, "crs"))
        self.assertIsInstance(self.v.crs, rio.crs.CRS)
        self.assertTrue(self.v.crs == rio.crs.CRS.from_epsg(5070))

    def test_field_schema(self):
        self.assertTrue(hasattr(self.v, "field_schema"))
        self.assertIsInstance(self.v.field_schema, dict)
        self.assertTrue(self.v.field_schema == self.v.table.dtypes.to_dict())

    def test_field_names(self):
        self.assertTrue(hasattr(self.v, "field_names"))
        self.assertIsInstance(self.v.field_names, list)
        self.assertTrue(self.v.field_names == self.v.table.columns.to_list())

    def test_field_dtypes(self):
        self.assertTrue(hasattr(self.v, "field_dtypes"))
        self.assertIsInstance(self.v.field_dtypes, list)
        self.assertTrue(
            self.v.field_dtypes == list(self.v.table.dtypes.to_dict().values())
        )

    def test_geometry(self):
        self.assertTrue(hasattr(self.v, "geometry"))
        self.assertIsInstance(self.v.geometry, gpd.GeoSeries)
        self.assertTrue((self.v.geometry == self.v.data.geometry).all())

    def test_tasks(self):
        self.assertTrue(hasattr(self.v, "tasks"))
        self.assertFalse(dask.is_dask_collection(self.v.data))
        self.assertTrue(self.v.tasks == 0)
        v = self.v.to_lazy()
        self.assertTrue(v.tasks == 1)


class TestSpecialMethods(unittest.TestCase):
    def setUp(self):
        self.v = open_vectors("test/data/vector/pods.shp")

    def test_len(self):
        self.assertTrue(hasattr(self.v, "__len__"))
        self.assertTrue(len(self.v) == self.v.size)
        self.assertTrue(len(self.v) == len(self.v.data))

    def test_getitem(self):
        self.assertTrue(hasattr(self.v, "__getitem__"))
        self.assertIsInstance(self.v[0], Vector)
        self.assertTrue(self.v[0].data.equals(self.v.data.loc[[0]]))
        self.assertTrue(
            self.v[-1].data.equals(self.v.data.loc[[self.v.size - 1]])
        )
        with self.assertRaises(NotImplementedError):
            self.v[0:3]
        with self.assertRaises(TypeError):
            self.v[9.0]
        with self.assertRaises(IndexError):
            self.v[self.v.size]
        with self.assertRaises(IndexError):
            self.v[-200]
        with self.assertRaises(IndexError):
            self.v[999]
