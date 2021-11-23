import unittest

import dask
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import rasterio as rio

from raster_tools import Raster
from raster_tools.vector import Vector, open_vectors


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

    def test_bounds(self):
        self.assertTrue(hasattr(self.v, "bounds"))
        df = gpd.read_file("test/data/vector/pods.shp")
        v = open_vectors("test/data/vector/pods.shp")
        self.assertTrue(all(v.bounds == df.total_bounds))
        self.assertTrue(dask.is_dask_collection(v.to_lazy().bounds))
        self.assertTrue(all(v.to_lazy().bounds == df.total_bounds))


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


class TestCopy(unittest.TestCase):
    def test_copy(self):
        v = open_vectors("test/data/vector/pods.shp")
        vc = v.copy()
        self.assertIsNot(v, vc)
        self.assertIsNot(v.data, vc.data)
        self.assertTrue(v.data.equals(vc.data))


class TestEval(unittest.TestCase):
    def test_eva(self):
        v = open_vectors("test/data/vector/pods.shp")
        vd = v.to_lazy()
        ve = vd.eval()

        self.assertIs(v, v.eval())
        self.assertTrue(dask.is_dask_collection(vd.data))
        self.assertIsNot(vd, ve)
        self.assertFalse(dask.is_dask_collection(ve.data))


class TestConversions(unittest.TestCase):
    def setUp(self):
        self.v = open_vectors("test/data/vector/pods.shp")

    def test_to_lazy(self):
        self.assertIsNot(self.v, self.v.to_lazy())
        self.assertFalse(dask.is_dask_collection(self.v.data))
        self.assertTrue(dask.is_dask_collection(self.v.to_lazy().data))

    def test_to_dataframe(self):
        self.assertIsInstance(self.v.to_dataframe(), gpd.GeoDataFrame)
        self.assertIsInstance(
            self.v.to_lazy().to_dataframe(), dgpd.GeoDataFrame
        )
        self.assertTrue(self.v.data.equals(self.v.to_dataframe()))

    def test_to_crs(self):
        crs = rio.crs.CRS.from_epsg(4326)
        self.assertTrue(self.v.crs == rio.crs.CRS.from_epsg(5070))
        self.assertTrue(self.v.to_crs(crs).crs == crs)
        self.assertTrue(self.v.to_crs(4326).crs == crs)
        self.assertTrue(self.v.to_crs("epsg:4326").crs == crs)

    def test_to_raster_many(self):
        like = Raster("test/data/elevation.tif")
        truth = Raster("test/data/pods_like_elevation.tif")
        result = self.v.to_raster(like)

        self.assertTrue(result.null_value == 0)
        self.assertTrue(result.dtype == np.dtype("uint8"))
        self.assertTrue(result.shape[0] == 1)
        self.assertTrue(np.allclose(result, truth))
        self.assertTrue(np.allclose(result._rs.x, truth._rs.x))
        self.assertTrue(np.allclose(result._rs.y, truth._rs.y))
        self.assertTrue(np.allclose(result._rs.band, truth._rs.band))

    def test_to_raster_single(self):
        like = Raster("test/data/elevation.tif")
        truth = Raster("test/data/pods0_like_elevation.tif")
        result = self.v[0].to_raster(like)

        self.assertTrue(result.null_value == 0)
        self.assertTrue(result.dtype == np.dtype("uint8"))
        self.assertTrue(result.shape[0] == 1)
        self.assertTrue(np.allclose(result, truth))
        self.assertTrue(np.allclose(result._rs.x, truth._rs.x))
        self.assertTrue(np.allclose(result._rs.y, truth._rs.y))
        self.assertTrue(np.allclose(result._rs.band, truth._rs.band))

        self.assertTrue(all(np.unique(result) == [0, 1]))

    def test_to_raster_lazy_one_partition(self):
        like = Raster("test/data/elevation.tif")
        truth = Raster("test/data/pods_like_elevation.tif")
        result = self.v.to_lazy().to_raster(like)

        self.assertTrue(result.null_value == 0)
        self.assertTrue(result.dtype == np.dtype("uint8"))
        self.assertTrue(result.shape[0] == 1)
        self.assertTrue(np.allclose(result, truth))
        self.assertTrue(np.allclose(result._rs.x, truth._rs.x))
        self.assertTrue(np.allclose(result._rs.y, truth._rs.y))
        self.assertTrue(np.allclose(result._rs.band, truth._rs.band))

    def test_to_raster_lazy_many_partitions(self):
        like = Raster("test/data/elevation.tif")
        truth = Raster("test/data/pods_like_elevation.tif")
        v = Vector(self.v.to_lazy().data.repartition(10))
        result = v.to_raster(like)

        self.assertTrue(result.null_value == 0)
        self.assertTrue(result.dtype == np.dtype("uint8"))
        self.assertTrue(result.shape[0] == 1)
        self.assertTrue(np.allclose(result, truth))
        self.assertTrue(np.allclose(result._rs.x, truth._rs.x))
        self.assertTrue(np.allclose(result._rs.y, truth._rs.y))
        self.assertTrue(np.allclose(result._rs.band, truth._rs.band))


class TestCastField(unittest.TestCase):
    def test_cast_field(self):
        v = open_vectors("test/data/vector/pods.shp")
        dtypes = [int, "int16", np.int32, float, "float"]
        for d in dtypes:
            dt = np.dtype(d)
            vc = v.cast_field("POD_Num", d)
            self.assertTrue(vc.data.POD_Num.dtype == dt)
