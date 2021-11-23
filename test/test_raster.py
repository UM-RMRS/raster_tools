import unittest

import affine
import dask
import numpy as np
import rasterio as rio
import xarray as xr

import raster_tools.focal as focal
from raster_tools import Raster, band_concat
from raster_tools._types import DTYPE_INPUT_TO_DTYPE
from raster_tools.raster import _BINARY_ARITHMETIC_OPS, _BINARY_LOGICAL_OPS

TEST_ARRAY = np.array(
    [
        [1, 3, 4, 4, 3, 2],
        [7, 3, 2, 6, 4, 6],
        [5, 8, 7, 5, 6, 6],
        [1, 4, 5, -1, 5, 1],
        [4, 7, 5, -1, 2, 6],
        [1, 2, 2, 1, 3, 4],
    ]
)


def rs_eq_array(rs, ar):
    return (rs._rs.values == ar).all()


class TestRasterCreation(unittest.TestCase):
    def test_ctor_errors(self):
        with self.assertRaises(ValueError):
            Raster(np.ones(4))
        with self.assertRaises(ValueError):
            Raster(np.ones((1, 3, 4, 4)))

    def test_increasing_coords(self):
        # This raster has an inverted y axis
        rs = Raster("test/data/elevation_small.tif")
        x, y = rs._rs.x.values, rs._rs.y.values
        self.assertTrue((np.diff(x) > 0).all())
        self.assertTrue((np.diff(y) > 0).all())

        rs = Raster(TEST_ARRAY)
        x, y = rs._rs.x.values, rs._rs.y.values
        self.assertTrue((np.diff(x) > 0).all())
        self.assertTrue((np.diff(y) > 0).all())

    def test_creation_from_numpy(self):
        for nprs in [np.ones((6, 6)), np.ones((1, 6, 6)), np.ones((4, 5, 5))]:
            rs = Raster(nprs)
            shape = nprs.shape if len(nprs.shape) == 3 else (1, *nprs.shape)
            self.assertEqual(rs.shape, shape)
            self.assertTrue(rs_eq_array(rs, nprs))

        rs = Raster(TEST_ARRAY)
        # Band dim has been added
        self.assertTrue(rs.shape == (1, 6, 6))
        # Band dim starts at 1
        self.assertTrue((rs._rs.band == [1]).all())
        # x/y dims start at 0 and increase
        self.assertTrue((rs._rs.x == np.arange(0, 6)).all())
        self.assertTrue((rs._rs.y == np.arange(0, 6)).all())
        # No null value determined for int type
        self.assertIsNone(rs.null_value)

        rs = Raster(TEST_ARRAY.astype(float))
        self.assertTrue(np.isnan(rs.null_value))


class TestProperties(unittest.TestCase):
    def test__attrs(self):
        rs = Raster("test/data/elevation_small.tif")
        self.assertDictEqual(rs._attrs, rs._rs.attrs)
        rs._attrs = {}
        self.assertDictEqual(rs._attrs, {})

    def test__masked(self):
        rs = Raster("test/data/elevation_small.tif")
        self.assertTrue(rs._masked)
        rs = Raster("test/data/null_values.tiff")
        self.assertTrue(rs._masked)
        x = np.ones((1, 3, 3))
        rs = Raster(x)
        self.assertTrue(rs._masked)
        rs = Raster(x.astype(int))
        self.assertFalse(rs._masked)

    def test__values(self):
        rs = Raster("test/data/elevation_small.tif")
        self.assertTrue((rs._values == rs._rs.values).all())

    def test__null_value(self):
        rs = Raster("test/data/elevation_small.tif")
        self.assertEqual(rs._null_value, rs._rs.attrs["_FillValue"])
        rs._null_value = 1
        self.assertEqual(rs._null_value, 1)
        self.assertEqual(rs._rs.attrs["_FillValue"], 1)

    def test_null_value(self):
        rs = Raster("test/data/elevation_small.tif")
        self.assertEqual(rs.null_value, rs._rs.attrs["_FillValue"])

    def test_dtype(self):
        rs = Raster("test/data/elevation_small.tif")
        self.assertTrue(rs.dtype == rs._rs.dtype)

    def test_shape(self):
        rs = Raster("test/data/elevation_small.tif")
        self.assertTrue(rs.shape == rs._rs.shape)
        self.assertIsInstance(rs.shape, tuple)

    def test_crs(self):
        rs = Raster("test/data/elevation_small.tif")
        self.assertIsInstance(rs.crs, rio.crs.CRS)
        self.assertTrue(rs.crs == rs._rs.rio.crs)

        x = np.arange(25).reshape((5, 5))
        rs = Raster(x)
        self.assertIsNone(rs.crs)

    def test_affine(self):
        rs = Raster("test/data/elevation_small.tif")
        self.assertIsInstance(rs.affine, affine.Affine)
        self.assertTrue(rs.affine == rs._rs.rio.transform())

    def test_resolution(self):
        rs = Raster("test/data/elevation_small.tif")
        self.assertTupleEqual(rs.resolution, rs._rs.rio.resolution(True))

        r = np.arange(25).reshape((5, 5))
        rs = Raster(r)
        self.assertTupleEqual(rs.resolution, rs._rs.rio.resolution(True))


class TestRasterMath(unittest.TestCase):
    def setUp(self):
        self.rs1 = Raster("test/data/elevation_small.tif")
        self.rs1_np = self.rs1._rs.values
        self.rs2 = Raster("test/data/elevation2_small.tif")
        self.rs2_np = self.rs2._rs.values

    def tearDown(self):
        self.rs1.close()
        self.rs2.close()

    def test_add(self):
        # Raster + raster
        truth = self.rs1_np + self.rs2_np
        rst = self.rs1.add(self.rs2)
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs2.add(self.rs1)
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs1 + self.rs2
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs2 + self.rs1
        self.assertTrue(rs_eq_array(rst, truth))
        # Raster + scalar
        for v in [-23, 0, 1, 2, 321]:
            truth = self.rs1_np + v
            rst = self.rs1.add(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 + v
            self.assertTrue(rs_eq_array(rst, truth))
            rst = v + self.rs1
            self.assertTrue(rs_eq_array(rst, truth))
        for v in [-23.3, 0.0, 1.0, 2.0, 321.4]:
            truth = self.rs1_np + v
            rst = self.rs1.add(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 + v
            self.assertTrue(rs_eq_array(rst, truth))
            rst = v + self.rs1
            self.assertTrue(rs_eq_array(rst, truth))

    def test_subtract(self):
        # Raster - raster
        truth = self.rs1_np - self.rs2_np
        rst = self.rs1.subtract(self.rs2)
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs2.subtract(self.rs1)
        self.assertTrue(rs_eq_array(rst, -truth))
        rst = self.rs1 - self.rs2
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs2 - self.rs1
        self.assertTrue(rs_eq_array(rst, -truth))
        # Raster - scalar
        for v in [-1359, 0, 1, 2, 42]:
            truth = self.rs1_np - v
            rst = self.rs1.subtract(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 - v
            self.assertTrue(rs_eq_array(rst, truth))
            rst = v - self.rs1
            self.assertTrue(rs_eq_array(rst, -truth))
        for v in [-1359.2, 0.0, 1.0, 2.0, 42.5]:
            truth = self.rs1_np - v
            rst = self.rs1.subtract(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 - v
            self.assertTrue(rs_eq_array(rst, truth))
            rst = v - self.rs1
            self.assertTrue(rs_eq_array(rst, -truth))

    def test_mult(self):
        # Raster * raster
        truth = self.rs1_np * self.rs2_np
        rst = self.rs1.multiply(self.rs2)
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs2.multiply(self.rs1)
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs1 * self.rs2
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs2 * self.rs1
        self.assertTrue(rs_eq_array(rst, truth))
        # Raster * scalar
        for v in [-123, 0, 1, 2, 345]:
            truth = self.rs1_np * v
            rst = self.rs1.multiply(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 * v
            self.assertTrue(rs_eq_array(rst, truth))
            rst = v * self.rs1
            self.assertTrue(rs_eq_array(rst, truth))
        for v in [-123.9, 0.0, 1.0, 2.0, 345.3]:
            truth = self.rs1_np * v
            rst = self.rs1.multiply(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 * v
            self.assertTrue(rs_eq_array(rst, truth))
            rst = v * self.rs1
            self.assertTrue(rs_eq_array(rst, truth))

    def test_div(self):
        # Raster / raster
        truth = self.rs1_np / self.rs2_np
        rst = self.rs1.divide(self.rs2)
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs2.divide(self.rs1)
        self.assertTrue(rs_eq_array(rst, 1 / truth))
        rst = self.rs1 / self.rs2
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs2 / self.rs1
        self.assertTrue(rs_eq_array(rst, 1 / truth))
        # Raster / scalar, scalar / raster
        for v in [-123, -1, 1, 2, 345]:
            truth = self.rs1_np / v
            rst = self.rs1.divide(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 / v
            self.assertTrue(rs_eq_array(rst, truth))
            rst = v / self.rs1
            np.testing.assert_array_almost_equal(rst._rs.values, 1 / truth)
        for v in [-123.8, -1.0, 1.0, 2.0, 345.6]:
            truth = self.rs1_np / v
            rst = self.rs1.divide(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 / v
            self.assertTrue(rs_eq_array(rst, truth))
            rst = v / self.rs1
            np.testing.assert_array_almost_equal(rst._rs.values, 1 / truth)

    def test_mod(self):
        # Raster % raster
        truth = self.rs1_np % self.rs2_np
        rst = self.rs1.mod(self.rs2)
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs1 % self.rs2
        self.assertTrue(rs_eq_array(rst, truth))
        truth = self.rs2_np % self.rs1_np
        rst = self.rs2.mod(self.rs1)
        self.assertTrue(rs_eq_array(rst, truth))
        rst = self.rs2 % self.rs1
        self.assertTrue(rs_eq_array(rst, truth))
        # Raster % scalar, scalar % raster
        for v in [-123, -1, 1, 2, 345]:
            truth = self.rs1_np % v
            rst = self.rs1.mod(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 % v
            self.assertTrue(rs_eq_array(rst, truth))
            truth = v % self.rs1_np
            rst = v % self.rs1
            self.assertTrue(rs_eq_array(rst, truth))
        for v in [-123.8, -1.0, 1.0, 2.0, 345.6]:
            truth = self.rs1_np % v
            rst = self.rs1.mod(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 % v
            self.assertTrue(rs_eq_array(rst, truth))
            truth = v % self.rs1_np
            rst = v % self.rs1
            self.assertTrue(rs_eq_array(rst, truth))

    def test_power(self):
        # Raster ** raster
        rs1 = self.rs1 / self.rs1._rs.max().values.item() * 2
        rs2 = self.rs2 / self.rs2._rs.max().values.item() * 2
        rs1_np = self.rs1_np / self.rs1_np.max() * 2
        rs2_np = self.rs2_np / self.rs2_np.max() * 2
        truth = rs1_np ** rs2_np
        rst = rs1.pow(rs2)
        self.assertTrue(rs_eq_array(rst, truth))
        rst = rs2.pow(rs1)
        self.assertTrue(rs_eq_array(rst, truth))
        rst = rs1 ** rs2
        self.assertTrue(rs_eq_array(rst, truth))
        truth = rs2_np ** rs1_np
        rst = rs2 ** rs1
        self.assertTrue(rs_eq_array(rst, truth))
        # Raster ** scalar, scalar ** raster
        for v in [-10, -1, 1, 2, 11]:
            truth = rs1_np ** v
            rst = rs1.pow(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = rs1 ** v
            self.assertTrue(rs_eq_array(rst, truth))
            # Avoid complex numbers issues
            if v >= 0:
                truth = v ** rs1_np
                rst = v ** rs1
                self.assertTrue(rs_eq_array(rst, truth))
        for v in [-10.5, -1.0, 1.0, 2.0, 11.3]:
            truth = rs1_np ** v
            rst = rs1.pow(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = rs1 ** v
            self.assertTrue(rs_eq_array(rst, truth))
            # Avoid complex numbers issues
            if v >= 0:
                truth = v ** rs1_np
                rst = v ** rs1
                self.assertTrue(rs_eq_array(rst, truth))

    def test_sqrt(self):
        rs = self.rs1 + np.abs(self.rs1_np.min())
        rsnp = rs._rs.values
        truth = np.sqrt(rsnp)
        self.assertTrue(rs_eq_array(rs.sqrt(), truth))


class TestLogicalOps(unittest.TestCase):
    def setUp(self):
        self.rs1 = Raster("test/data/elevation_small.tif")
        self.rs1_np = self.rs1._rs.values
        self.rs2 = Raster("test/data/elevation2_small.tif")
        self.rs2_np = self.rs2._rs.values

    def tearDown(self):
        self.rs1.close()
        self.rs2.close()

    def test_eq(self):
        for v, vnp in [
            (self.rs2, self.rs2_np),
            (0, 0),
            (self.rs1_np[0, 10, 10], self.rs1_np[0, 10, 10]),
        ]:
            truth = self.rs1_np == vnp
            rst = self.rs1.eq(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 == v
            self.assertTrue(rs_eq_array(rst, truth))

    def test_ne(self):
        for v, vnp in [
            (self.rs2, self.rs2_np),
            (0, 0),
            (self.rs1_np[0, 10, 10], self.rs1_np[0, 10, 10]),
        ]:
            truth = self.rs1_np != vnp
            rst = self.rs1.ne(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 != v
            self.assertTrue(rs_eq_array(rst, truth))

    def test_le(self):
        for v, vnp in [
            (self.rs2, self.rs2_np),
            (0, 0),
            (self.rs1_np[0, 10, 10], self.rs1_np[0, 10, 10]),
        ]:
            truth = self.rs1_np <= vnp
            rst = self.rs1.le(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 <= v
            self.assertTrue(rs_eq_array(rst, truth))

    def test_ge(self):
        for v, vnp in [
            (self.rs2, self.rs2_np),
            (0, 0),
            (self.rs1_np[0, 10, 10], self.rs1_np[0, 10, 10]),
        ]:
            truth = self.rs1_np >= vnp
            rst = self.rs1.ge(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 >= v
            self.assertTrue(rs_eq_array(rst, truth))

    def test_lt(self):
        for v, vnp in [
            (self.rs2, self.rs2_np),
            (0, 0),
            (self.rs1_np[0, 10, 10], self.rs1_np[0, 10, 10]),
        ]:
            truth = self.rs1_np < vnp
            rst = self.rs1.lt(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 < v
            self.assertTrue(rs_eq_array(rst, truth))

    def test_gt(self):
        for v, vnp in [
            (self.rs2, self.rs2_np),
            (0, 0),
            (self.rs1_np[0, 10, 10], self.rs1_np[0, 10, 10]),
        ]:
            truth = self.rs1_np > vnp
            rst = self.rs1.gt(v)
            self.assertTrue(rs_eq_array(rst, truth))
            rst = self.rs1 > v
            self.assertTrue(rs_eq_array(rst, truth))


class TestAstype(unittest.TestCase):
    def test_astype(self):
        rs = Raster("test/data/elevation_small.tif")
        for type_code, dtype in DTYPE_INPUT_TO_DTYPE.items():
            self.assertEqual(rs.astype(type_code).dtype, dtype)
            self.assertEqual(rs.astype(type_code).eval().dtype, dtype)
            self.assertEqual(rs.astype(dtype).dtype, dtype)
            self.assertEqual(rs.astype(dtype).eval().dtype, dtype)

    def test_wrong_type_codes(self):
        rs = Raster("test/data/elevation_small.tif")
        with self.assertRaises(ValueError):
            rs.astype("not float32")
        with self.assertRaises(ValueError):
            rs.astype("other")

    def test_dtype_property(self):
        rs = Raster("test/data/elevation_small.tif")
        self.assertEqual(rs.dtype, rs._rs.dtype)

    def test_astype_str_uppercase(self):
        rs = Raster("test/data/elevation_small.tif")
        for type_code, dtype in DTYPE_INPUT_TO_DTYPE.items():
            if isinstance(type_code, str):
                type_code = type_code.upper()
                self.assertEqual(rs.astype(type_code).eval().dtype, dtype)


class TestRasterAttrsPropagation(unittest.TestCase):
    def test_arithmetic_attrs(self):
        r1 = Raster("test/data/elevation_small.tif")
        true_attrs = r1._attrs
        v = 2.1
        for op in _BINARY_ARITHMETIC_OPS.keys():
            r2 = r1._binary_arithmetic(v, op).eval()
            self.assertEqual(r2._rs.attrs, true_attrs)
            self.assertEqual(r2._attrs, true_attrs)
        for r in [+r1, -r1]:
            self.assertEqual(r._rs.attrs, true_attrs)
            self.assertEqual(r._attrs, true_attrs)

    def test_logical_attrs(self):
        r1 = Raster("test/data/elevation_small.tif")
        true_attrs = r1._attrs
        v = 1.0
        for op in _BINARY_LOGICAL_OPS.keys():
            r2 = r1._binary_logical(v, op).eval()
            self.assertEqual(r2._rs.attrs, true_attrs)
            self.assertEqual(r2._attrs, true_attrs)

    def test_ctor_attrs(self):
        r1 = Raster("test/data/elevation_small.tif")
        true_attrs = r1._attrs.copy()
        r2 = Raster(Raster("test/data/elevation_small.tif"))
        test_attrs = {"test": 0}
        r3 = Raster("test/data/elevation_small.tif")
        r3._attrs = test_attrs
        self.assertEqual(r2._attrs, true_attrs)
        self.assertEqual(r3._attrs, test_attrs)

    def test_astype_attrs(self):
        rs = Raster("test/data/elevation_small.tif")
        attrs = rs._attrs
        self.assertEqual(rs.astype(int)._attrs, attrs)

    def test_sqrt_attrs(self):
        rs = Raster("test/data/elevation_small.tif")
        rs += np.abs(rs._rs.values.min())
        attrs = rs._attrs
        self.assertEqual(rs.sqrt()._attrs, attrs)

    def test_log_attrs(self):
        rs = Raster("test/data/elevation_small.tif")
        attrs = rs._attrs
        self.assertEqual(rs.log()._attrs, attrs)
        self.assertEqual(rs.log10()._attrs, attrs)

    def test_convolve_attrs(self):
        rs = Raster("test/data/elevation_small.tif")
        attrs = rs._attrs
        self.assertEqual(focal.convolve(rs, np.ones((3, 3)))._attrs, attrs)

    def test_focal_attrs(self):
        rs = Raster("test/data/elevation_small.tif")
        attrs = rs._attrs
        self.assertEqual(focal.focal(rs, "max", 3)._attrs, attrs)

    def test_band_concat_attrs(self):
        rs = Raster("test/data/elevation_small.tif")
        attrs = rs._attrs
        rs2 = Raster("test/data/elevation2_small.tif")
        self.assertEqual(band_concat([rs, rs2])._attrs, attrs)


class TestCopy(unittest.TestCase):
    def test_copy(self):
        rs = Raster("test/data/elevation_small.tif")
        copy = rs.copy()
        self.assertIsNot(rs, copy)
        self.assertIsNot(rs._rs, copy._rs)
        self.assertIsNot(rs._attrs, copy._attrs)
        self.assertTrue((rs._rs == copy._rs).all())
        self.assertEqual(rs._attrs, copy._attrs)


class TestSetCrs(unittest.TestCase):
    def test_set_crs(self):
        rs = Raster("test/data/elevation_small.tif")
        self.assertTrue(rs.crs != 4326)

        rs4326 = rs.set_crs(4326)
        self.assertTrue(rs4326.crs != rs.crs)
        self.assertTrue(rs4326.crs == 4326)
        self.assertTrue(np.allclose(rs._values, rs4326._values))


class TestSetNullValue(unittest.TestCase):
    def test_set_null_value(self):
        rs = Raster("test/data/null_values.tiff")
        ndv = rs.null_value
        rs2 = rs.set_null_value(0)
        self.assertEqual(rs.null_value, ndv)
        self.assertEqual(rs._attrs["_FillValue"], ndv)
        self.assertEqual(rs2._attrs["_FillValue"], 0)

        rs = Raster("test/data/elevation_small.tif")
        nv = rs.null_value
        rs2 = rs.set_null_value(None)
        self.assertEqual(rs.null_value, nv)
        self.assertEqual(rs._attrs["_FillValue"], nv)
        self.assertIsNone(rs2.null_value)
        self.assertIsNone(rs2._attrs["_FillValue"])


class TestReplaceNull(unittest.TestCase):
    def test_replace_null(self):
        fill_value = 0
        rs = Raster("test/data/null_values.tiff")
        nv = rs.null_value
        rsnp = rs._values
        rsnp_replaced = rsnp.copy()
        rsnp_replaced[rsnp == rs.null_value] = fill_value
        rs = rs.replace_null(fill_value)
        self.assertTrue(np.allclose(rs._values, rsnp_replaced, equal_nan=True))
        self.assertEqual(rs.null_value, nv)


class TestClipBox(unittest.TestCase):
    def test_clip_box(self):
        rs = Raster("test/data/elevation.tif")
        rs_clipped = Raster("test/data/elevation_small.tif")
        bounds = [
            rs_clipped._rs.x.min().item(),
            rs_clipped._rs.y.min().item(),
            rs_clipped._rs.x.max().item(),
            rs_clipped._rs.y.max().item(),
        ]
        test = rs.clip_box(*bounds)
        self.assertTrue(test.shape == rs_clipped.shape)
        self.assertTrue(np.allclose(test._values, rs_clipped._values))

        # Test that the mask is also clipped
        x = np.arange(25).reshape((1, 5, 5))
        x[x < 12] = 0
        rs = Raster(x).set_null_value(0).set_crs("epsg:3857")
        self.assertTrue(np.allclose(x == 0, rs._mask))
        rs_clipped = rs.clip_box(1, 1, 3, 3)
        mask_truth = np.array([[[1, 1, 1], [1, 0, 0], [0, 0, 0]]], dtype=bool)
        self.assertTrue(np.allclose(rs_clipped._mask, mask_truth))


class TestToNullMask(unittest.TestCase):
    def test_to_null_mask(self):
        rs = Raster("test/data/null_values.tiff")
        nv = rs.null_value
        rsnp = rs._values
        truth = rsnp == nv
        self.assertTrue(rs_eq_array(rs.to_null_mask(), truth))
        # Test case where no null values
        rs = Raster("test/data/elevation_small.tif")
        truth = np.full(rs.shape, False, dtype=bool)
        self.assertTrue(rs_eq_array(rs.to_null_mask(), truth))


class TestRemapRange(unittest.TestCase):
    def test_remap_range(self):
        rs = Raster("test/data/elevation_small.tif")
        rsnp = rs._rs.values
        min, max, new_value = rs._rs.values.min(), rs._rs.values.max(), 0
        rng = (min, min + (0.2 * (max - min)))
        match = rsnp >= rng[0]
        match &= rsnp < rng[1]
        rsnp[match] = new_value
        rs = rs.remap_range(rng[0], rng[1], new_value)
        self.assertTrue(rs_eq_array(rs, rsnp))

    def test_remap_range_multi(self):
        rs = Raster("test/data/elevation_small.tif")
        min, max = rs._rs.values.min(), rs._rs.values.max()
        rsnp = rs._rs.values
        span = max - min
        rng1 = (min, min + (0.2 * span))
        match = rsnp >= rng1[0]
        match &= rsnp < rng1[1]
        nv1 = 0
        rsnp[match] = nv1
        rng2 = (min + (0.2 * span), min + (0.3 * span))
        match = rsnp >= rng2[0]
        match &= rsnp < rng2[1]
        nv2 = 1
        rsnp[match] = nv2
        rs = rs.remap_range(rng1[0], rng1[1], nv1, rng2[0], rng2[1], nv2)
        self.assertTrue(rs_eq_array(rs, rsnp))
        rsnp = rs._rs.values
        span = max - min
        rng1 = (min, min + (0.2 * span))
        match = rsnp >= rng1[0]
        match &= rsnp < rng1[1]
        nv1 = 0
        rsnp[match] = nv1
        rng2 = (min + (0.2 * span), min + (0.3 * span))
        match = rsnp >= rng2[0]
        match &= rsnp < rng2[1]
        nv2 = 1
        rsnp[match] = nv2
        rs = rs.remap_range(rng1[0], rng1[1], nv1, rng2[0], rng2[1], nv2)
        rng3 = (min + (0.3 * span), min + (0.4 * span))
        match = rsnp >= rng3[0]
        match &= rsnp < rng3[1]
        nv3 = 2
        rsnp[match] = nv3
        rs = rs.remap_range(*rng1, nv1, *rng2, nv2, *rng3, nv3)
        self.assertTrue(rs_eq_array(rs, rsnp))

    def test_remap_range_errors(self):
        rs = Raster("test/data/elevation_small.tif")
        # TypeError if not scalars
        with self.assertRaises(TypeError):
            rs.remap_range(None, 2, 4)
        with self.assertRaises(TypeError):
            rs.remap_range(0, "2", 4)
        with self.assertRaises(TypeError):
            rs.remap_range(0, 2, None)
        with self.assertRaises(TypeError):
            rs.remap_range(0, 2, 1, 2, 3, None)
        # ValueError if nan
        with self.assertRaises(ValueError):
            rs.remap_range(np.nan, 2, 4)
        with self.assertRaises(ValueError):
            rs.remap_range(0, np.nan, 4)
        # ValueError if range reversed
        with self.assertRaises(ValueError):
            rs.remap_range(0, -1, 6)
        # RuntimeError if not enough values to form group
        with self.assertRaises(RuntimeError):
            rs.remap_range(0, 1, 2, 0, 3)
        with self.assertRaises(RuntimeError):
            rs.remap_range(0, 1, 2, 2, 3, 4, 9)


class TestEval(unittest.TestCase):
    def test_eval(self):
        rs = Raster("test/data/elevation_small.tif")
        rsnp = rs._rs.values
        rs += 2
        rsnp += 2
        rs -= rs
        rsnp -= rsnp
        rs *= -1
        rsnp *= -1
        result = rs.eval()
        # Make sure new raster returned
        self.assertIsNot(rs, result)
        self.assertIsNot(rs._rs, result._rs)
        # Make sure that original raster is still lazy
        self.assertTrue(dask.is_dask_collection(rs._rs))
        self.assertTrue(rs_eq_array(result, rsnp))
        self.assertFalse(dask.is_dask_collection(result._rs))


class TestToXarray(unittest.TestCase):
    def test_to_xarray(self):
        rs = Raster("test/data/elevation2_small.tif")
        self.assertTrue(isinstance(rs.to_xarray(), xr.DataArray))
        self.assertIs(rs.to_xarray(), rs._rs)


class TestToDask(unittest.TestCase):
    def test_to_dask(self):
        rs = Raster("test/data/elevation2_small.tif")
        self.assertTrue(isinstance(rs.to_dask(), dask.array.Array))
        self.assertIs(rs.to_dask(), rs._rs.data)
        self.assertTrue(isinstance(rs.eval().to_dask(), dask.array.Array))


class TestToLazy(unittest.TestCase):
    def test_to_lazy(self):
        rs = Raster("test/data/elevation2_small.tif")
        rs += rs
        rs_nonlazy = rs.eval()
        rs_lazy = rs_nonlazy.to_lazy()
        self.assertFalse(dask.is_dask_collection(rs_nonlazy._rs))
        self.assertTrue(dask.is_dask_collection(rs_lazy._rs))


class TestAndOr(unittest.TestCase):
    def test_and(self):
        rs1 = Raster("test/data/elevation_small.tif")
        rsnp1 = rs1._rs.values
        rs2 = Raster("test/data/elevation2_small.tif")
        rsnp2 = rs2._rs.values
        rsnp2 -= rsnp2.max() / 2
        truth = (rsnp1 > 0) & (rsnp2 > 0)
        self.assertTrue(rs_eq_array(rs1 & rs2, truth))
        self.assertTrue(rs_eq_array(rs1.and_(rs2), truth))
        truth = rsnp1.astype(bool) & rsnp2.astype(bool)
        self.assertTrue(rs_eq_array(rs1.and_(rs2, "cast"), truth))
        for v in [-22.0, -20, 0, 1, 1.0, 23.1, 30]:
            truth = (rsnp1 > 0) & (v > 0)
            self.assertTrue(rs_eq_array(rs1 & v, truth))
            self.assertTrue(rs_eq_array(rs1.and_(v), truth))
            truth = rsnp1.astype(bool) & bool(v)
            self.assertTrue(rs_eq_array(rs1.and_(v, "cast"), truth))
        for v in [False, True]:
            truth = (rsnp1 > 0) & v
            self.assertTrue(rs_eq_array(rs1 & v, truth))
            self.assertTrue(rs_eq_array(rs1.and_(v), truth))
            truth = rsnp1.astype(bool) & v
            self.assertTrue(rs_eq_array(rs1.and_(v, "cast"), truth))

    def test_or(self):
        rs1 = Raster("test/data/elevation_small.tif")
        rsnp1 = rs1._rs.values
        rs2 = Raster("test/data/elevation2_small.tif")
        rsnp2 = rs2._rs.values
        rsnp2 -= rsnp2.max() / 2
        truth = (rsnp1 > 0) | (rsnp2 > 0)
        self.assertTrue(rs_eq_array(rs1 | rs2, truth))
        self.assertTrue(rs_eq_array(rs1.or_(rs2), truth))
        truth = rsnp1.astype(bool) | rsnp2.astype(bool)
        self.assertTrue(rs_eq_array(rs1.or_(rs2, "cast"), truth))
        for v in [-22.0, -20, 0, 1, 1.0, 23.1, 30]:
            truth = (rsnp1 > 0) | (v > 0)
            self.assertTrue(rs_eq_array(rs1 | v, truth))
            self.assertTrue(rs_eq_array(rs1.or_(v), truth))
            truth = rsnp1.astype(bool) | bool(v)
            self.assertTrue(rs_eq_array(rs1.or_(v, "cast"), truth))
        for v in [False, True]:
            truth = (rsnp1 > 0) | v
            self.assertTrue(rs_eq_array(rs1 | v, truth))
            self.assertTrue(rs_eq_array(rs1.or_(v), truth))
            truth = rsnp1.astype(bool) | v
            self.assertTrue(rs_eq_array(rs1.or_(v, "cast"), truth))


class TestBitwiseComplement(unittest.TestCase):
    def test_invert(self):
        ar = np.array([[0, 1], [1, 0]])
        bool_ar = ar.astype(bool)
        inv_bool_ar = np.array([[1, 0], [0, 1]], dtype=bool)

        rs = Raster(bool_ar)
        rs_inv = Raster(inv_bool_ar)
        self.assertTrue(rs_eq_array(~rs, inv_bool_ar))
        self.assertTrue(rs_eq_array(~rs_inv, bool_ar))
        self.assertTrue(rs_eq_array(~Raster(ar), ~ar))

    def test_invert_errors(self):
        ar = np.array([[0, 1], [1, 0]], dtype=float)
        rs = Raster(ar)
        with self.assertRaises(TypeError):
            ~rs


class TestGetBands(unittest.TestCase):
    def test_get_bands(self):
        rs = Raster("test/data/multiband_small.tif")
        rsnp = rs._rs.values
        self.assertTrue(rs_eq_array(rs.get_bands(1), rsnp[:1]))
        self.assertTrue(rs_eq_array(rs.get_bands(2), rsnp[1:2]))
        self.assertTrue(rs_eq_array(rs.get_bands(3), rsnp[2:3]))
        self.assertTrue(rs_eq_array(rs.get_bands(4), rsnp[3:4]))
        for bands in [[1], [1, 2], [1, 1], [3, 1, 2], [4, 3, 2, 1]]:
            np_bands = [i - 1 for i in bands]
            self.assertTrue(rs_eq_array(rs.get_bands(bands), rsnp[np_bands]))

        self.assertTrue(len(rs.get_bands(1).shape) == 3)

        for bands in [0, 5, [1, 5], [0]]:
            with self.assertRaises(IndexError):
                rs.get_bands(bands)
        with self.assertRaises(ValueError):
            rs.get_bands([])


if __name__ == "__main__":
    unittest.main()
