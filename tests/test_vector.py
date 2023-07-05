import unittest

import dask
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import rasterio as rio

from raster_tools.vector import Vector, open_vectors


class TestOpenVectors(unittest.TestCase):
    def test_open_vectors(self):
        vs = open_vectors("tests/data/vector/Zones.gdb")

        self.assertIsInstance(vs, list)
        self.assertTrue(len(vs) == 2)
        self.assertIsInstance(vs[0], Vector)
        self.assertIsInstance(vs[1], Vector)
        self.assertTrue(len(vs[0]) == 10642)
        self.assertTrue(len(vs[1]) == 184)

        pods1 = open_vectors("tests/data/vector/Zones.gdb", layers="PODs")
        self.assertIsInstance(pods1, Vector)
        pods2 = open_vectors("tests/data/vector/Zones.gdb", layers=1)
        pods3 = open_vectors("tests/data/vector/Zones.gdb", layers=["PODs"])
        pods4 = open_vectors("tests/data/vector/Zones.gdb", layers=[1])
        for v in [pods2, pods3, pods4]:
            self.assertTrue(pods1.data.compute().equals(v.data.compute()))

    def test_open_vectors_errors(self):
        with self.assertRaises(ValueError):
            open_vectors("tests/data/vector/Zones.gdb", layers="dummy")
        with self.assertRaises(ValueError):
            open_vectors("tests/data/vector/Zones.gdb", layers=2)
        with self.assertRaises(ValueError):
            open_vectors("tests/data/vector/Zones.gdb", layers=-1)
        with self.assertRaises(ValueError):
            open_vectors("tests/data/vector/Zones.gdb", layers=[-1])

        with self.assertRaises(TypeError):
            open_vectors("tests/data/vector/Zones.gdb", layers=[0, "PODs"])
        with self.assertRaises(TypeError):
            open_vectors("tests/data/vector/Zones.gdb", layers={})


class TestVectorProperties(unittest.TestCase):
    def setUp(self):
        self.v = open_vectors("tests/data/vector/pods.shp")

    def test_table(self):
        self.assertTrue(hasattr(self.v, "table"))
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
        self.assertTrue(self.v.shape == dask.compute(self.v.table.shape)[0])

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
        self.assertIsInstance(self.v.geometry, dgpd.GeoSeries)
        self.assertTrue(
            (self.v.geometry == self.v.data.geometry).all().compute()
        )

    def test_tasks(self):
        assert hasattr(self.v, "tasks")
        v = self.v.copy()
        v._geo = v._geo.compute()
        assert not dask.is_dask_collection(v.data)
        assert v.tasks == 0
        v = self.v.to_lazy()
        assert v.tasks >= 1

    def test_bounds(self):
        self.assertTrue(hasattr(self.v, "bounds"))
        df = gpd.read_file("tests/data/vector/pods.shp")
        v = open_vectors("tests/data/vector/pods.shp")
        self.assertTrue(all(v.bounds == df.total_bounds))
        self.assertTrue(dask.is_dask_collection(v.to_lazy().bounds))
        self.assertTrue(all(v.to_lazy().bounds == df.total_bounds))


class TestSpecialMethods(unittest.TestCase):
    def setUp(self):
        self.v = open_vectors("tests/data/vector/pods.shp")

    def test_len(self):
        self.assertTrue(hasattr(self.v, "__len__"))
        self.assertTrue(len(self.v) == self.v.size)
        self.assertTrue(len(self.v) == len(self.v.data))

    def test_getitem(self):
        self.assertTrue(hasattr(self.v, "__getitem__"))
        self.assertIsInstance(self.v[0], Vector)
        self.assertTrue(
            self.v[0].data.compute().equals(self.v.data.loc[[0]].compute())
        )
        last = self.v.data.loc[[self.v.size - 1]]
        last.index = dask.array.from_array([0]).to_dask_dataframe()
        self.assertTrue(self.v[-1].data.compute().equals(last.compute()))
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


class TestCopy(unittest.TestCase):
    def test_copy(self):
        v = open_vectors("tests/data/vector/pods.shp")
        vc = v.copy()
        self.assertIsNot(v, vc)
        self.assertIsNot(v.data, vc.data)
        self.assertTrue(v.data.compute().equals(vc.data.compute()))


class TestLazyByDefault(unittest.TestCase):
    def test_lazy_by_default(self):
        v = open_vectors("tests/data/vector/pods.shp")
        self.assertTrue(dask.is_dask_collection(v._geo))


class TestEval(unittest.TestCase):
    def test_eval(self):
        v = open_vectors("tests/data/vector/Zones.gdb", 0)
        self.assertTrue(dask.is_dask_collection(v.data))
        ve = v.eval()
        self.assertTrue(dask.is_dask_collection(ve.data))
        self.assertTrue(ve.data.npartitions == 1)


class TestConversions(unittest.TestCase):
    def setUp(self):
        self.v = open_vectors("tests/data/vector/pods.shp")

    def test_to_lazy(self):
        vc = self.v.copy()
        vc._geo = vc._geo.compute()
        self.assertFalse(dask.is_dask_collection(vc.data))
        self.assertTrue(dask.is_dask_collection(vc.to_lazy().data))

    def test_to_dataframe(self):
        self.assertIsInstance(self.v.to_dataframe(), dgpd.GeoDataFrame)
        self.assertIsInstance(
            self.v.to_lazy().to_dataframe(), dgpd.GeoDataFrame
        )
        self.assertTrue(
            self.v.data.compute().equals(self.v.to_dataframe().compute())
        )

    def test_to_crs(self):
        crs = rio.crs.CRS.from_epsg(4326)
        self.assertTrue(self.v.crs == rio.crs.CRS.from_epsg(5070))
        self.assertTrue(self.v.to_crs(crs).crs == crs)
        self.assertTrue(self.v.to_crs(4326).crs == crs)
        self.assertTrue(self.v.to_crs("epsg:4326").crs == crs)


class TestCastField(unittest.TestCase):
    def test_cast_field(self):
        v = open_vectors("tests/data/vector/pods.shp")
        dtypes = [int, "int16", np.int32, float, "float"]
        for d in dtypes:
            dt = np.dtype(d)
            vc = v.cast_field("POD_Num", d)
            self.assertTrue(vc.data.POD_Num.dtype == dt)
